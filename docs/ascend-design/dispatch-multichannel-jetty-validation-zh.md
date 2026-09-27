# Dispatch 多 channel / jetty 验证

日期：2026-09-27。基线：`5bc937d`（生产代码与 `1d6ba92` 相同）。

## 测试目标与口径

验证当前典型用例的 1、2、4 channel 是否有稳定的端到端收益，并核查
channel 是否对应独立 jetty、payload 是否真正分流。暂不同时引入 chunk
overlap、多个 service worker 或共享 jetty，避免混合变量。

- NPU8P-ALT，`task-submit` 显式申请设备 0–7，8 rank。
- CANN/HCOMM：`/data/disk2/cann_version/0916/use_cann/cann-9.3.0` 同树。
- 8192 tokens/rank、hidden 7168、top-k 8、256 experts、num_sms 64。
- `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`，五操作校验开启。
- 正式性能沿用原 benchmark：每进程 30 warmups / 30 iterations，
  max-rank NPU Event 延迟，sum logical bytes；不混入诊断探针数据。
- 当前正式扩展 SHA256：
  `aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0`。
- 包内 `libhcomm.so` SHA256：
  `3cab033f9624774816c5aca8ab77a469c54b40a83cdc73c148c16acbd424fc16`。

## 首先发现：direct producer 丢失 channel 数量

`make_hybrid_dispatch_context` 和 `make_hybrid_combine_context` 从展开参数
重建设备 context 时，没有填充 `DeviceTransportContext::channel_count`，
因此它保持默认值 0。`device::channel_count` 在该字段为 0 时返回 0，
`put_staged_records_striped` 随后走普通 `put`，使用 facade 的 channel 0。

因此，host 请求多条 channel，不代表当前 direct producer 真正分片提交。
必须同时观察资源创建和队列使用，不能只对比环境变量。

本地最小复现直接调用这两个生产 context helper，给定 channel 表中
1–4 条 channel，原实现全部返回 0；补齐后检查通过。实验补丁仅在
`.scratch/multichannel/source/` 中，从现有 device channel 表通过
`simt::load_observed` 读取数量，不修改参数 ABI，不修改完成确认或协议。
该补丁并未通过实机验收，后续情况见下文；本地检查不能替代设备测试。

原二进制初次测试任务 `task_20260927_073401_319505527603` 在发现上述
问题后主动终止，exit=130。已完成的第一组 2/4 channel ABBA 分别显示
Dispatch +3.196%/+1.859%，但这只是配置变化的结果，不能作为 payload
已分流的性能证据，也不用于决定默认配置。中断的 run 不纳入统计。

## 实际资源和命令的观测方法

独立初始化探针拦截 `HcclChannelAcquire`，调用原接口后回读 channel、
SQ/CQ context，记录 peer、channel index、SQ/CQ id、buffer/doorbell 和
head/tail 地址。五操作执行完成并同步后，再读取每条 SQ 的累计请求数
与 CQ 完成数，检查资源是否独立、是否实际使用、是否全部完成。

`cann_compat.hpp::SqContext::queue_id` 对应 CANN `ubJfs.jfsID`；
仓库的 `cann_abi_probe.cpp` 明确校验该映射。远端 HCOMM 源码
`src/legacy/ascend950/unified_platform/resource/connection/dev_ub_connection.cc`
的 `SetSqContextInfo` 将 `jettyId` 写入此字段。资源唯一性同时用 SQ/CQ
地址与 id 交叉检查，不只比较一个数字。

诊断数据不参与正式带宽统计。原正式二进制的完整 stage profile 被
`invalid_block_count` 拒绝，Combine profile 返回 `operation_unavailable`；
这些失败没有当作有效完整 profile。临时诊断脚本单独保存原始返回值，
仅以 Dispatch 命令计数和实际 SQ/CQ 使用状态确认分流，不对缺失 stage
做补零或修改正式 benchmark 的校验。

初始化探针最初遇到 loader 环境和 `RTLD_NEXT` 无法找到局部加载符号
的问题，修正后从已加载的同树 `libhcomm.so` 解析原函数；失败尝试不计入
正确性或性能结果。生产扩展未加入诊断 hook。

实验脚本与原始结果本地保留在 `.scratch/multichannel/`，远端工作区为
`/home/pyptouser/yuqitao/deepep-official-simt.AaR0IW`。

### 原实现实测：资源创建成功，额外队列未使用

任务 `task_20260927_075016_32527061977` 正常结束，以下结果在全部
8 rank 一致。每个 rank 对另外 7 个 peer 创建资源。

| 每 peer channel 数 | 每 rank 独立 SQ/CQ 套数 | 实际有请求的队列数 | 普通 Dispatch payload put 数 |
| --- | ---: | ---: | ---: |
| 1 | 7 | 7 | 7 |
| 2 | 14 | 7 | 7 |
| 4 | 28 | 7 | 7 |

以 4 channel 的 rank 0 → peer 1 为例，SQ/jetty id 分别为
9815、9814、9816、9817，CQ id 为 6940、6941、6942、6943；全部地址
独立，但执行完五操作后累计请求数分别为 1561、0、0、0。其他 peer/rank
同样仅 channel 0 有请求，且 SQ 提交数与 CQ 完成数相等。

这确认当前缺失的是 producer 对已创建独立队列的使用。单纯再增加
jetty 资源不能绕过这个问题。原始数据：
`results/multichannel/queue-v7-c{1,2,4}/{rank*.jsonl,raw-rank*.json}`；
汇总：`.scratch/multichannel/summarize_queues.py queue-v7`。

## 实验候选与功能门槛

第一版在两个共用 context helper 中补齐字段，构建成功，但任务
`task_20260927_075454_327889140` 在单 channel preparation 阶段即失败：
设备报告 340、VEC 访问 UB 未对齐，host 返回 507035。没有进入正式性能
验收，该候选不保留。退出脚本恢复了正式扩展。

第二版缩小到 `device::channel_count`：context 中已有数量时保持原行为；
字段为 0 时从现有 device channel 表读取。这样避免改动共用 context
helper 的返回值和所有调用它的 kernel。实验代码保存在
`.scratch/multichannel/source-v2/`，验收后按下文保留范围合入生产源码。

第二版构建任务 `task_20260927_075636_32846102465`，扩展 SHA256：
`db547c29d1106516a157733e74b61aa0b9e8f52c421bd513ac958836e126ca48`。
只有通过实际队列使用检查和功能校验，才继续同二进制的 1/2/4 channel
性能对照；所有实验退出时均恢复原正式扩展。

### 第二版的实际分流

第二版 1/2/4 channel 的五操作诊断全部完成，8 rank 的 profile 均正常
返回。每 rank 独立队列、实际使用队列和普通 Dispatch payload put 数量
分别都为 7、14、28；每条队列的 SQ 请求数与 CQ 完成数相等。
同一 rank 的 payload 字节数不随 channel 数改变。

以 rank 0 → peer 1 的 4 channel 为例，四个 SQ/jetty id 为
9902–9905，累计请求数为 1561、314、314、314。channel 0 同时承载
控制操作，所以其累计请求数更多；其他 channel 已实际承载 payload。
这组独立 channel 的测试确实使用了多个独立 jetty，不需要再用另一个
名称重复相同的资源配置实验。

诊断中的 service active mean 随 1/2/4 channel 为
1219107/1773172/3378892 cycles，CQ wait 为 794834/675881/286900 cycles。
**不能把这些差值直接当作生产通信收益或退化**：开启 stage profile 时，
`update_queue_profile` 在提交前后、drain 后遍历所有 peer/channel。
多 channel 增加提交数量，也放大了这个诊断开销；提交较晚还可能让后续
CQ 观察时已经就绪。因此正式验收必须关闭 profile 和初始化探针。

数据：`results/multichannel/fixed-v2-queue-c{1,2,4}/`；汇总命令：
`.scratch/multichannel/summarize_queues.py fixed-v2-queue`。

### 分流回归与补充功能

本地 `.scratch/multichannel/channel_selection_probe.cpp` 使用生产的两个
context helper、实际 `device::channel_count` 查询以及实际 record 分片函数，
仅把设备内存读写模拟为 host 读写。覆盖 1–4 channel、17 条不整除记录、
连续且无重叠的地址/字节范围、本地和越界 peer，以及非零 context 显式
数量的保留。原代码报告 6 个失败（两个 helper 的 2/3/4 channel），
第二版为 0 失败。它验证选择和分片逻辑，设备执行由上述实机诊断验证。

2 和 4 channel 各通过三种补充功能 case：BF16 sync、FP8 async、
FP8 previous-event + async + allocate，每 case 五操作，2 warmups /
5 iterations。结果分别为 `channels-fixed-functional-c{2,4}.json`，
各 3 cases passed、0 failed。

第二版设备任务：`task_20260927_075733_329306424942`。正式 ABBA 使用
同一第二版二进制，A=1 channel、B=2 或 4 channel；这样只比较实际
channel 数量，不混入重新编译与修正分流逻辑的影响。不能直接把第二版
测得的带宽与此前 1990 GB/s 的历史批次相减作为新收益。

## 正式性能结果

上述任务正常结束，exit=0。两种候选各三组 A1/B1/B2/A2，共 24 次
完整五操作正式运行；每种候选及其配对基线各 180 个 max-rank 样本。
少数进程在完整结果写出后发生已知 teardown SIGSEGV，由现有完整结果
验证器确认后继续；这不等于所有进程均干净退出。没有把失败诊断或
中断的旧批次加入统计。

下表的正数为延迟改善，负数为退化。两种候选的 A 是各自配对的
1 channel 运行，不是复用同一份历史基线。

| channel | 组 | A mean / ms | B mean / ms | mean 改善 | B logical GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | 1 | 3.980532 | 3.912600 | 1.707% | 1990.004 |
| 2 | 2 | 4.006808 | 3.893787 | 2.821% | 1999.619 |
| 2 | 3 | 3.755677 | 3.803939 | -1.285% | 2046.849 |
| 2 | 合并 | 3.914339 | 3.870109 | 1.130% | 2011.853 |
| 4 | 1 | 3.765100 | 3.981536 | -5.748% | 1955.549 |
| 4 | 2 | 3.776733 | 4.139853 | -9.615% | 1880.765 |
| 4 | 3 | 3.729748 | 4.052398 | -8.651% | 1921.354 |
| 4 | 合并 | 3.757194 | 4.057929 | -8.004% | 1918.735 |

2 channel 合并 p50/p95 改善 2.453%/0.869%；4 channel 合并 p50/p95
退化 9.019%/8.784%。2 channel 合并均值为正，但第三组方向反转，尚不能
认定获得稳定端到端收益。第 3 组 B 的绝对带宽最高，却比配对 A 更慢，
说明不能单凭达到约 2 TB/s 就宣布优化成功。

| 三组合并 mean 改善 | 2 channel | 4 channel |
| --- | ---: | ---: |
| Dispatch | 1.130% | -8.004% |
| Expanded Dispatch | -0.180% | -1.144% |
| Cached Dispatch | 0.087% | 0.026% |
| Combine | -0.721% | -0.626% |
| Reduced Combine | -0.102% | -1.061% |

原始数据：`results/dispatch-tail/channels-fixed-20260927-c{2,4}-g{1,2,3}-{A1,B1,B2,A2}.json`。
汇总命令：`.scratch/multichannel/summarize.py channels-fixed-20260927`。
脚本校验 workload、计时协议、设备、五操作数量、每 run 30 个样本及
每操作逻辑字节数一致，再由合并延迟计算带宽，不平均各 run 的 GB/s。

## 结论和保留范围

1. 保持正式默认 1 channel。2 channel 尚未证明稳定收益；4 channel
   持续退化，不作为当前典型用例的性能优化保留。
2. 发现了独立的实现缺陷：direct context 没有传递 channel 数量，导致
   原二进制创建了额外队列却未使用。第二版实验修正后，资源和实际请求
   都验证为多队列分流。保留第二版 `device::channel_count` 的查询回退修复，
   使显式配置的多 channel 生效；默认仍为 1。这是功能修复，不宣称性能收益。
3. 本次独立 channel 实际对应独立 jetty，已经覆盖每 peer 1/2/4 jetty
   的分流测试。“再增加 jetty 数量”不能作为尚未测试的全新方向。
   同一 peer 的多个 jetty 还观察到相同 transport path id，例如
   rank 0 → peer 1 的四条队列均为 8388615；不能将队列增加等同于
   物理链路增加，也没有据此证明链路饱和。
4. 共享 jetty 的 one-to-many 队列组织是另一种实验，本轮没有测试。
   已核查 CANN 9.3 `HcclChannelAcquireWithConfig` 的 shared-queue
   约束：共享 channel 调用需要串行化。当前 service、完成游标和 drain
   按独立 channel 维护，因此不能只打开 shared-queue 开关就视作兼容。
   后续若验证该模式，需先建立共享 SQ/CQ 的所有权与完成模型。
5. 现有结果仍支持把主要性能工作转向分块 overlap；这里没有新增可直接
   默认启用的稳定多 channel 收益，也不宣称所有多 jetty 调度方法均无效。

实验结束时核查：无 pending/running 任务；远端扩展恢复 SHA256
`aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0`。
临时替换的两个 context helper 和 `device_transport_commands.hpp`
均已恢复，当时哈希与本地生产源码相同。

随后按用户要求，将已测试的第二版查询回退修复和本文档一起提交。
生产默认值仍为 1，诊断代码、被否决的第一版及实验产物留在 `.scratch`，
不随提交发布。正式 ABBA 对比的是同一修复后二进制的 channel 配置，
没有额外做原二进制与修复后二进制在 1 channel 下的 ABBA 对比。
远端已安装扩展仍为实验后恢复的原版本，本次提交不重新部署远端。

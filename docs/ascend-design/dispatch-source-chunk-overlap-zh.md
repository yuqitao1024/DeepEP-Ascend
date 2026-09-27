# Dispatch 源 token 分块 overlap

日期：2026-09-27。基线：`09b075c`，默认 1 channel。

## 目标

让普通 Dispatch 的下一块 record/hidden 打包与上一块 URMA 通信重叠。
保持输入、逻辑字节数、完成确认、错误检查和五操作校验不变。
最终性能用 NPU8P，8 ranks，`--num-tokens 8192`、hidden 7168、top-k 8、
256 experts、num_sms 64；每进程 30 warmups / 30 iterations，以配对
ABBA 的 max-rank NPU Event 时间验收。正确性先过，再判断稳定收益。
沿用既有 manifest 生成逻辑，各 rank 实际 tokens 为 `8192-rank`，
即 8192 至 8185；A/B 使用相同输入和实际逻辑字节数。

## 旧实现核查

两个环境变量默认未设置，互斥：

| 开关 | 分块单位 | 既有状态 |
| --- | --- | --- |
| `DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_SLOTS` | 每目标 rank 的 slot | 旧版本有性能退化；不是本轮主要目标 |
| `DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES` | 源 token tile，每 tile 4 tokens | 实验功能，历史 prefix/count 错误；当前版停滞 |

历史设计和结果见 `epv2-ascend-p7-source-pipeline-deadlock-design.md`。
历史观察不能直接作为当前版本的结果：VF 拆分后，`dispatch.asc` 的
persistent producer/release manager 中若干原 VF 调用已被 `(void)0`
替代，独立 VF 文件虽存在，却未从这些 manager 接回调用。producer
仍等待 scalar progress，release 也无法正常完成原来的发布流程。

当前复现：CANN/HCOMM 同树
`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`；设备 0、1；
典型输入，2 warmups / 3 iterations，完整五操作校验。
扩展 SHA256 为 `db547c29d1106516a157733e74b61aa0b9e8f52c421bd513ac958836e126ca48`，
即上轮已测试、随后合入基线的 channel 查询修复版本。

- 关闭 pipeline：完整报告通过，进程 exit=0。
- `CHUNK_TILES=1024`（2 chunks）：首轮准备未返回，150 秒超时，exit=124。
- `CHUNK_TILES=512`（4 chunks）：相同停滞，150 秒超时，exit=124。
- 超时实验任务 `task_20260927_083830_392197211829` 的包装脚本 exit=0
  只表示复现采集结束，不表示设备用例通过。

首次基线任务 `task_20260927_083750_39199016673` 的报告成功；采集脚本
随后误用仅针对 teardown SIGSEGV 的验证器而 exit=1。修正脚本后继续
采集启用路径。没有把该脚本错误当作设备故障。

## 第一版：有限 chunk kernel + 设备事件

不直接恢复常驻 kernel 内的全部 VF 调用：这还会重新引入原设计的
manager/VF 等待环和常驻资源约束。复用已有 source tile 边界、record
打包、hidden 搬运、release 和 request 完成函数，先建立可验证基线。

1. ProducerControl/Group/Prefix 在 producer stream 完成。
2. chunk n 的 record 和 hidden 写完后记录 ready[n]。
3. communication stream 等 ready[n]，重置已完成的命令队列，发布该
   chunk 的命令，执行 service，等待 request 完成，记录 done[n]。
4. producer stream 可同时处理 n+1；复用两槽 request 状态前等待
   done[n-2]。payload 使用完整 staging 中互不重叠的最终 slot 区间。
5. 最后一块才发布整体 count/generation/release signal。producer
   stream 等最终 done，再走普通 acquire、验证、prefix 和输出路径。

队列 reset 必须位于 command VF **之前**。旧 slot 分支的 reset 在
VF 之后的 service wrapper 中，可能拒绝或清空刚提交的 batch。
每次 reset 仍检查上一批已完成，不能放宽 queue ownership 检查。

epilogue 清除 chunk 坐标和 persistent 标志，恢复普通 consumed generation
与远端 release 校验。按 stage 类型选择 epilogue，避免 profile 数组
硬编码下标把 release barrier 再执行一次。

第一版暂保留既有事件创建、最终 stream 同步和事件销毁。它们可能抵消
overlap 收益，应先测量，再决定是否缓存事件或改设备调度；不先删除
完成边界。source pipeline 仍显式启用，不默认打开。

## 验证与状态

实验脚本和原始结果分别位于 `.scratch/overlap/`、远端
`results/overlap/`。构建及设备任务结束后恢复原远端源码和安装扩展。

第一版编译任务 `task_20260927_084424_39397115559`，二进制 SHA256
`9c4cf49f56c5546d10961f6a0baf6c272e166236c88683bff5d10b7513dca279`。
双卡任务 `task_20260927_084522_39482362461`、8-rank 任务
`task_20260927_084658_395403327707` 的 baseline 和 2/4/8 chunks 均通过
完整五操作。每配置 2 warmups / 3 iterations；个别报告后的已知 teardown
SIGSEGV 由既有完整结果验证器确认，不计作干净进程退出。

第一版 8-rank Dispatch 短测：关闭/2/4/8 chunks 分别为
3.6583/8.3279/12.4730/22.0657 ms。不能保留为性能优化。
旧 host selector 同时关闭了 token fan-out，而有限 source-chunk kernel
已有相应 token range 支持。因此第二版允许 source chunk 继续使用
token fan-out；slot pipeline 仍排除。非 7168-byte 等不满足 fan-out 条件
的输入仍使用原有向量/标量搬运策略。

第二版构建 `task_20260927_084948_39661211370`，SHA256
`712d73ff289f813debbe86bb2ca991386bd7cedbe6068835accfa10914d61572`。
任务 `task_20260927_085055_396899819018` 再次通过 8-rank baseline 和
2/4/8 chunks 五操作。短测分别为 3.8797/3.7315/4.0738/4.8036 ms；
只有 2 chunks 出现改善信号，不能据此声明稳定收益。

同任务使用 Torch-NPU profiler，沿用 30 warmups / 12 captures，排除
首个 capture 后每配置保留 8 ranks × 11 = 88 个诊断样本。基线的
producer record/hidden 与通信 service 重叠为 0；2 chunks 全部 88 个
样本重叠为正，物理 stream 61/26，平均 71.891 μs、p50 65.352 μs。
它证明 kernel 区间存在真实并发，不直接证明链路数据传输始终与计算
同时发生，也不等同于端到端节省的时间。

| profiler 区间均值 / μs | baseline | 2 chunks |
| --- | ---: | ---: |
| producer record + hidden 总时长 | 285.616 | 358.923 |
| transport service 总时长 | 944.897 | 987.664 |
| 两者重叠 | 0 | 71.891 |
| DeepEP kernel 首尾跨度 | 2780.687 | 2876.336 |
| capture（含 host 与 profiler） | 3945.858 | 4202.499 |

分块本身增加总工作和提交开销；因此只有不带 profiler 的正式 ABBA 能
判断净收益。原始 trace：`results/dispatch-tail/overlap-event-v2-{baseline,source1024}/`；
分析：`.scratch/overlap/summarize_trace.py`。

正式 ABBA 首次任务 `task_20260927_085430_39894405869` 在第一个关闭
pipeline 的基线准备阶段停滞，240 秒 timeout，未生成报告。GDB 显示
rank 6 在 Torch-NPU `_local_scalar_dense` / `item()` 引发的
`aclrtSynchronizeStreamWithTimeout` 中等待。该栈不能单独判定是哪个
设备任务未完成，也不能归因于尚未启用的 overlap 分支；本次不计入性能
统计，保留日志并用独立进程重跑。

重跑 `task_20260927_085947_400728017391` 同样在关闭 pipeline 的
准备阶段超时。控制实验 `task_20260927_090423_40225762218` 分别装载
原生产扩展（SHA256 `aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0`）
和第二版候选，均关闭 pipeline，均未生成报告并超时。包装任务 exit=0
只表示两组诊断采集完成。因此目前不能把该阻塞归因于 overlap 修改，
也不能据此认定是硬件故障。

诊断任务 `task_20260927_090804_404142731809` 在测试框架汇总异常
之前捕获到第一个本地错误：rank 6，`completion_timeout`，
`command_index=7 opcode=5 peer=4 world_peer=4 team=0 channel=0`，
`backend_status=0 reserved=0 generation=1`。opcode 5 是 flush，即
rank 6 向 rank 4 的 payload 完成等待超时。测试框架随后执行错误
all-reduce；其他 rank 尚在 Dispatch，导致原始错误被集体等待遮住。
这是关闭 pipeline 的诊断结果，不是启用分块时的性能结果。

下一步使用原生产扩展在物理设备 4、6 上做双卡隔离，区分特定卡对和
8-rank 并发条件；恢复可重复通过的基线后再跑正式 ABBA。

双卡隔离 `task_20260927_091310_40631918062` 使用原生产扩展、
`ASCEND_RT_VISIBLE_DEVICES=4,6`，典型输入和 2/3 次预热/测量，
五操作校验全部通过、exit=0。随后原生产 8-rank 对照
`task_20260927_091417_406938513928` 再次捕获完全相同的
rank 6 → rank 4 / channel 0 / generation 1 flush 超时。因此简单的
双卡路径可用，但 8-rank 并发完成路径仍有阻塞；尚不能细分为实现、
运行库或设备状态原因。

4-rank 隔离任务 `task_20260927_091623_40827619077` 指定物理设备
4、5、6、7、原生产扩展及关闭 pipeline；等待设备释放后已执行完毕，
五操作完整报告通过，随后出现已知 teardown SIGSEGV，由完整结果
验证器确认。不能将包装任务 exit=0 表述为设备进程正常退出。
诊断脚本 `.scratch/overlap/run_queue_failure.sh` 已准备，用既有只读
初始化探针记录 channel/SQ/CQ 地址，在本地错误汇总前采集 SQ head、
CQ tail 和待消费 CQE word0。首次两次启动（任务
`task_20260927_092402_42116414831`、`task_20260927_092431_42378917663`）
因 preload 与 CANN set_env 的库路径清理顺序冲突而 exit=127，未运行
设备用例。改为仅向 Python 子进程传入 preload 后重新提交诊断。

队列诊断 `task_20260927_092532_4309781881` 随后通过 8-rank 五操作，
exit=0，没有触发超时，因此没有错误时的计数快照。探针在 channel
初始化时增加同步回读，可能改变时序；该通过结果不能证明故障根因
已消除，也不进入性能统计。接着移除 preload 和 Python 诊断包装，
分别重复原生产及候选的普通路径，确认无探针基线。

无探针对照 `task_20260927_092658_43645716488` 已完成：原生产与
第二版候选各跑两次独立的 8-rank 进程，pipeline 关闭，全部五操作
通过，四次均 exit=0。采用 2 warmups / 3 iterations，仅用于验证
基线可运行。没有执行设备 reset、重启或系统配置修改；此前超时
根因尚未确认，不能声称已修复。随后恢复边界测试和正式性能对照。

新增 host 调度回归直接编译真实 launcher（仅替换 ACL/stage 执行边界），
检查 2/4/8 chunks、profile 开关、slot 复用事件、完整 epilogue、失败
后的事件清理。原 launcher 失败，新 launcher 通过；它不替代设备可见性
和实机并发验证。

本地相关调度/layout/runtime/selector 回归 5 项通过。扩展到旧 fan-out
源码文本测试时另有 3 项失败：它们仍在 `dispatch.asc` 查找 VF 拆分前
的函数位置和展平 kernel 参数；这些位置在本轮基线已改变。本轮不将
它们记录为通过，也不据此判断设备功能；尚未迁移这批旧断言。
使用 `git archive HEAD` 的独立目录重跑这 3 项，确认基线同样失败。

## 边界验证发现：跨代接收区覆盖

第二版边界任务 `task_20260927_092922_44804210644` 的仅本地和仅远端
路由通过，但部分 masked 路由在 Combine 权重校验失败。输入为
8 ranks、257-rank 个 tokens、hidden 7168、top-k 8；每个 token 的
有效 expert 全部位于下一 rank，交替屏蔽一半 lane，有效权重 0.125。
`CHUNK_TILES=16`，各 rank 为 4/5 chunks，五操作完整检查。

对照任务 `task_20260927_093149_45764429072`：同一 manifest 下原生产
和第二版候选关闭 pipeline 均通过。诊断任务
`task_20260927_093330_46407627668`：启用 source chunks 后，接收
Dispatch 权重不一致，前一发送 rank 的 Combine 有效权重整批为 0。
这不是只发生在 masked lane 上的无效值差异。

两项针对性实验确认下一代覆盖：

- `task_20260927_093616_4710596022`：仅在正常 Dispatch 返回后、
  Cached Dispatch 开始前加入诊断 HCCL barrier，同一用例通过。
- `task_20260927_093802_47631527280`：不加 barrier，给下一次 Cached
  Dispatch 注入权重标记 0.375；上一轮期望 0.125 的有效权重读到了
  0.375。原本不传权重的 Cached Dispatch 写入 0，解释了之前的整批 0。

本地 chunk ready/done 只约束本机 producer 和 communication stream；
最终 release 说明数据已发出，不能证明远端 epilogue 已读完接收区。
快 rank 可以进入下一次 Cached Dispatch，复用并覆盖慢 rank 尚在
读取的同一接收区。分块改变了跨 rank 时序，使该生命周期缺口可复现。
本发现不能直接解释更早的普通路径 flush 超时，二者仍分别记录。

## 第三版：设备侧消费完成边界

source 开关启用且模式受支持时，完整 epilogue 后追加一次设备 world
barrier，复用已有 URMA barrier 执行器和本代 generation。此时所有
rank 已从接收区复制输出，才允许任一 rank 进入下一操作。诊断 HCCL
barrier 不进入正式实现，也不加入性能测试脚本。

是否参加完成确认独立于本 rank 的 token 数。需要 0/1 chunk 或超出
分块数量上限而回退普通路径的 rank，同样参加；否则会出现部分 rank
进入 barrier、其他 rank 未进入的死锁。开关未设置时不追加设备任务。
同时把 source 模式限定为现有分组实现支持的 FP8、top-k≤8、world≤8。

完成确认 reset 队列前先检查现有诊断和 workspace status，保留原
Dispatch 错误，不能用新 batch 清掉上游失败。完整错误仍由 host 原有
回读检查。额外 barrier 的成本包含在正式 NPU Event 测量中。

构建 `task_20260927_094330_48752130272` 暴露 GM 地址空间强转编译错误，
未运行设备测试；修正后构建 `task_20260927_094501_49409419260` 成功，
SHA256 `568a44c2e03fdf14f43a5101a6e037354155b5a5be93adeb37713480c554e3d9`。
host 回归覆盖完成确认必须在全部 epilogue 之后、失败时不得追加，以及
单块回退 rank 仍参加确认。

第三版设备回归（设备 0–7，同一 CANN/HCOMM，五操作完整校验）：

| 场景 | 结果 | 任务 |
| --- | --- | --- |
| 下一代权重标记 0.375，前一代应保持 0.125，无诊断 HCCL barrier | 通过 | `task_20260927_094614_50093323117` |
| 仅本地、仅远端、部分 masked 路由 | 均通过，每配置 2 warmups / 5 iterations | 同上 |
| rank 0 分块，其余 rank 普通回退 | 通过，未出现部分参与死锁 | `task_20260927_094943_51628311770` |
| 8193 token 设置、hidden 7184、masked ratio 0.25 | 通过，覆盖 token/hidden 尾块 | 同上 |
| 典型输入，关闭、2/4/8 chunks | 均通过，每配置 2 warmups / 3 iterations，均 exit=0 | 同上 |

全 masked 路由未通过完整五操作：正常 Dispatch 返回后，Cached Dispatch
在 host 参数检查报 `dispatch scale factors do not match element kind`，
零接收的 scale-factor 指针被拒绝。原生产扩展关闭 pipeline 的同 manifest
对照也复现这一错误（后一任务）。不能把任务包装脚本成功或其余用例通过
表述为全 masked 场景通过；该 Cached Dispatch 边界问题本轮未修改。

第三版典型输入短测，关闭/2/4/8 chunks 的 Dispatch 均值分别为
3.8814/4.1355/4.1143/4.7537 ms；短测未见净收益，需正式对照确认。
完成边界保证正确性，不能为获取收益删除它。

正式 3 组 ABBA 已完成：`task_20260927_095358_5670531303`。
A 为第三版关闭 pipeline，B 为 `CHUNK_TILES=1024`；每轮独立进程，
30 warmups / 30 iterations，五操作校验，完成边界计入时间。
12 轮报告均通过，没有超时或数值失败。G1-B1 在完整报告后出现已知
teardown SIGSEGV，由既有验证器确认；其余 11 轮未出现该退出错误。
因此不能把包装任务 exit=0 表述为全部子进程干净退出。

## 第三版正式性能结论

下表为 Dispatch max-rank NPU Event 均值；每组 A/B 各 60 个样本，
合并各 180 个样本。带宽沿用全 rank 逻辑字节求和除以时间，单位为
十进制 GB/s，不表示单链路物理带宽。统计逻辑未修改。

| ABBA 组 | A 关闭 / ms | B 两分块 / ms | B 耗时增加 | A / GB/s | B / GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.88325 | 3.98628 | 2.65% | 2005.05 | 1953.22 |
| 2 | 3.92146 | 4.04671 | 3.19% | 1985.51 | 1924.06 |
| 3 | 3.88205 | 4.28620 | 10.41% | 2005.67 | 1816.55 |
| 合并 | 3.89558 | 4.10640 | 5.41% | 1998.70 | 1896.09 |

合并 p50：3.93070 → 4.09430 ms；p95：4.24491 → 4.47639 ms。
三组均值均退化，p50/p95 也没有改善，不满足“稳定复现收益才保留”的
性能验收要求。第三组退化幅度更大，不能把合并 5.41% 当作固定开销。

同一批次另外四个操作的合并均值如下。它们没有启用 source overlap，
列出用于完整记录，不把小幅波动解释成优化收益。

| 操作 | A / ms | B / ms |
| --- | ---: | ---: |
| Expanded Dispatch | 15.02531 | 15.01393 |
| Cached Dispatch | 64.02130 | 63.98380 |
| Combine | 13.97812 | 14.00700 |
| Reduced Combine | 14.33026 | 14.35540 |

原始文件：远端 `results/dispatch-tail/overlap-event-v3b-g{1,2,3}-{A1,B1,B2,A2}.{json,log}`；
本地副本位于 `.scratch/overlap/results/dispatch-tail/`，汇总文件为
`overlap-event-v3b-summary.json`，由 `.scratch/overlap/summarize_abba.py` 生成。
脚本核对 workload、设备、预热数、每 rank 样本数、五操作和逻辑字节一致。

本轮结论：有限分块和消费完成确认修复了已复现的停滞/跨代覆盖问题，
保留为默认关闭的实验实现，不能作为已验收的性能优化。
此前普通路径 generation 1 flush 超时在本批 6 个 A 轮次未复现，根因仍未
确认；全 masked 的 Cached Dispatch 边界问题仍未解决。

后续若继续优化，先对第三版拆分测量 event 创建/销毁、额外 kernel 提交、
stream 同步和消费完成 barrier 的成本，再选择事件复用或减少 chunk 提交
开销。第二版 trace 已证明存在 kernel 重叠，但不能据此断言第三版的额外
耗时全由 barrier 引起。任何后续方案都必须保留等价的远端消费完成保证。

任务结束后，远端源码与构建前归档逐文件比较一致，安装扩展恢复为原
SHA256 `aaf762e60005f108d500a73d6ff46962dda6f5db9ff1e321a09461879fec57c0`。
本任务无 pending/running，设备锁已释放。代码、回归测试和本文一并归档；
`.scratch/` 下的原始日志、二进制和临时诊断脚本保留在实验工作区。

## 其他输入规模的性能筛选

补充任务 `task_20260927_103135_6691305733` 使用同一第三版扩展、环境和
设备 0–7。保持 hidden 7168、256 experts、num_sms 64、FP8 五操作用例，
每轮 30 warmups / 30 iterations。每种形状先做一组 ABBA；若出现正收益，
再加组数检查复现性，不把单组改善判为稳定收益。

| num-tokens 设置 | top-k | B 的 CHUNK_TILES | 每 rank chunks |
| --- | ---: | ---: | ---: |
| 2048 | 8 | 256 | 2 |
| 4096 | 8 | 512 | 2 |
| 16384 | 8 | 2048 | 2 |
| 8192 | 4 | 1024 | 2 |
| 10240（补充） | 8 | 1280 | 2 |

每 rank 实际 tokens 仍为 `num-tokens-rank`。B 的分块大小随输入规模变化，
用于比较同样两分块的调度方案，不是固定 1024 tiles 的跨形状测试。
脚本 `.scratch/overlap/run_shape_abba.sh`；原始文件前缀
`results/dispatch-tail/overlap-shape-v3b-t{tokens}-k{topk}-g1-`。
首任务完成了 2048/4096 两组，但在 16384 的 A1 准备阶段失败：正常
Dispatch 的 host 检查报 `dispatch capacity exceeds runtime storage`。
此时 pipeline 关闭，尚未运行 B，也没有该形状的可用性能结果。

容量原因通过真实 `build_core_tiling` 的 host probe 核查：
`.scratch/overlap/storage_probe.cpp`。运行时固定分配 4,194,304 bytes
workspace（4 MiB）；16384/top-k 8 的普通 Dispatch 需要 6,503,232 bytes，
两分块需要 6,545,120 bytes。这里不是系统显存耗尽；对外通信 buffer
大小提示也不能扩大独立的 workspace。该固定分配和容量检查在 `09b075c`
已有，本轮未修改，且没有现成环境开关。

因此不把扩大 workspace 混入本轮性能实验。8192/top-k 4 单独续跑于
`task_20260927_103710_8024941610`，完整通过；再补充 10240/top-k 8
（任务 `task_20260927_104012_9450675470`），其 A/B workspace 分别为
4,073,280 / 4,115,168 bytes，均在既有分配内。

四个可运行形状现已完成，各一组 ABBA，A/B 各 60 个正式样本：

| tokens / top-k | A 普通 / ms | B 两分块 / ms | B 耗时增加 | A / GB/s | B / GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 / 8 | 2.47952 | 3.08413 | 24.38% | 785.30 | 631.35 |
| 4096 / 8 | 2.84071 | 3.36309 | 18.39% | 1370.48 | 1157.61 |
| 8192 / 4 | 3.33436 | 3.49660 | 4.87% | 1457.85 | 1390.20 |
| 10240 / 8 | 4.42641 | 4.57968 | 3.46% | 2199.25 | 2125.65 |

16 轮完整报告的五操作均通过，没有超时或数值错误；其中 2048 的 A2、
8192/top-k 4 的 A1/B1、10240 的 B2/A2 在完整报告后出现已知 teardown
SIGSEGV，由原验证器确认。这不包含 16384 的失败尝试。

四个新增形状的均值和 p50 均退化，因此没有进入追加两组的正收益复验。
10240 的 p95 单组下降 2.29%（4.93587 → 4.82266 ms），但均值和 p50
分别变慢 3.46% / 3.95%；这一单组尾部波动不足以判定稳定尾延迟收益，
也不能当作逻辑带宽提升。其余三个形状的 p95 同样退化。

这说明本轮已测的大小输入和较低 top-k 都未发现两分块的平均性能收益，
但不是对所有形状、分块数或 rank 数的穷举结论。无需因本批数据开启默认
pipeline；仍需先解决提交、事件和同步成本，再重新验收。

结果和日志均已收取到 `.scratch/overlap/results/dispatch-tail/`，对应
`overlap-shape-v3b-t{tokens}-k{topk}-summary.json` 为汇总。实验后远端
安装扩展 hash 恢复为原 `aaf762e...`，源码与构建前归档一致，本 agent
无待执行或运行中的任务。该轮只增加实验脚本和文档记录，未修改生产实现。

## 第三版 Profiling：重叠收益被哪些成本抵消

任务 `task_20260927_105256_114598014955`，NPU8P 设备 0–7，同一 CANN/HCOMM
和第三版二进制。典型 8192/top-k 8，A1/B1/B2/A2，各独立进程预热 30 次，
采集 12 次，排除第一次；每配置 2 × 8 × 11 = 176 个 rank 样本。
Torch-NPU profiler 使用 Level2、关闭 AIC metrics，同时捕获 kernel 和
AscendCL API。没有启用改变 kernel 划分的 `DEEP_EP_ASCEND_PROFILE_STAGES`。
采集脚本沿用 `.scratch/dispatch-tail/probe.py`，仅增加可选 runtime API
采样。诊断会改变时序，不替代前述无 profiler 的性能和正确性验收。

结论：真实 overlap 已发生，退化来自当前分块调度的额外成本，消费完成
同步是其中一部分；不能归因于“stream synchronize 把两个 stream 全串行”。

### 设备时间的闭合分解

176 个 B 样本均有 producer record/hidden 与通信阶段的交集，A 全部为 0。
下表按相同口径给出所有 rank 的均值，单位 μs；B 的分块 kernel 先求和。

| 区间 | A 普通 | B 两分块 | 增量 |
| --- | ---: | ---: | ---: |
| prepare/group/prefix | 117.73 | 117.81 | +0.07 |
| producer record | 119.46 | 141.66 | +22.19 |
| producer hidden | 166.52 | 221.32 | +54.80 |
| 显式 chunk queue reset | 0 | 54.74 | +54.74 |
| release 命令生成 | 168.74 | 251.32 | +82.59 |
| transport service | 942.05 | 988.68 | +46.63 |
| chunk request wait kernel | 0 | 8.39 | +8.39 |
| epilogue | 1318.38 | 1363.78 | +45.40 |
| 末尾消费完成 barrier | 0 | 75.56 | +75.56 |
| kernel 累计时间 | 2832.88 | 3223.26 | +390.38 |
| kernel 并发交集（应扣除） | 0 | 214.73 | +214.73 |
| kernel 首尾之间的空档 | 15.50 | 17.32 | +1.82 |
| kernel 首尾跨度 | 2848.39 | 3025.85 | +177.47 |

对每个样本核对 `跨度 = kernel 累计时间 - 并发交集 + 空档`，再汇总：
`390.38 - 214.73 + 1.82 = 177.47 μs`。这里累计时间的增加既可来自
重复调度/协议工作，也可来自并发时资源争用，不能等同于指令数同比增加。

完整通信阶段包含 reset、release 命令生成、service 和 request wait。
若只统计 producer 与 service 的交集，则为 81.61 μs；此前第二版约
71.89 μs 使用的正是这一较窄口径。第二版按完整通信阶段重算为 210.59 μs，
其 kernel 累计时间增加 303.89 μs，空档增加 2.34 μs，跨度仍增加
95.65 μs。第二版尚无末尾消费 barrier，因此已有独立证据说明分块成本
本身就能抵消重叠，不能把全部退化归到第三版新增的 barrier。

第三版 barrier 占本批平均设备跨度增量的一部分（约 76 μs）。在仅做
算术分解、假设其他时间不变时，余下仍约 102 μs；这不是删除 barrier
后的性能预测。该 barrier 解决已复现的跨代接收区覆盖，不能直接删除。
epilogue 的 +45 μs 主要体现 acquire 等待波动：B1 与 A1 基本持平，
B2 较 A2 更长。即便不把这部分波动算作固定退化，额外分块成本仍存在。

### Host 同步不是另外一笔 2.8 ms

只统计 `cpp_dispatch` 区间内、调用线程上的 ACL API，排除 Torch 异步
工作线程销毁上一次计时事件的活动；不把 ACL 与其嵌套 runtime API 重复相加。

| API | A 每次调用 | B 每次调用 | B 累计耗时 / μs |
| --- | ---: | ---: | ---: |
| CreateEventWithFlag | 0 | 4 | 65.89 |
| DestroyEvent | 0 | 4 | 48.57 |
| RecordEvent | 0 | 4 | 11.39 |
| StreamWaitEvent | 0 | 3 | 10.95 |
| SynchronizeStream | 4 | 6 | 2854.63 |

事件在提交 chunk kernel 前创建，4 次销毁全部位于最后一个 DeepEP kernel
完成之后。它们是真实的 host 管理成本。相比之下，stream synchronize
的 A 累计时间已有 2789.27 μs，B 为 2854.63 μs：大部分在等待同一条
设备关键路径。B 多出的两次调用位于完整调度之后，并没有逐 chunk 阻塞
host 提交，也没有消除已观察到的双 stream 并发。设备完成后的同步 API
残余部分，A/B 分别为 115.66 / 90.26 μs，并未随调用数一起增加。

因此不能把 2854.63 μs 加到 kernel 时间上，更不能声称删除同步能省下
2.8 ms。事件创建/销毁也不能简单与上述跨度差直接相加当作端到端预测：
CPU/NPU 并发、原有同步回读、rank 间等待和 profiler 都会影响所在区间。
28 → 37 个 kernel 的 host launch API 累计时间约 97.93 → 130.27 μs；
这同样有可优化部分，但大多与设备执行重叠。

### 核对 max-rank 口径及下一步

每个 capture 选择 NPU Event 最慢的 rank 再汇总（每配置 22 个样本）：
Event 均值 3925.53 → 4135.06 μs，增加 209.53 μs；方向和量级与无 profiler
三组 ABBA 的 +210.81 μs 一致，但不把两批样本当作同一次测量。
所选 rank 的 kernel 跨度 2774.68 → 2967.47 μs（+192.79 μs），并发交集
213.75 μs，消费 barrier 71.41 μs。全部 rank 的均值并未掩盖关键 rank
上的退化。

当前不优先删除消费同步。后续应按以下顺序做独立候选验证：

1. 缓存/复用 ready、done 事件，在 buffer 生命周期内管理；减少为了销毁
   临时事件而设置的完成边界，保留事件复用和接收区消费的安全约束。
2. 减少每 chunk 的协议/提交成本：本轮新增 9 个 kernel，包括重复
   record/hidden、release/service、reset、request wait 和最终 barrier。
   优先研究合并 reset/命令准备、复用已准备的 chunk 描述符，而非单纯增大
   分块数量。
3. 单独定位 hidden 的并发退化：两块 hidden 分别约 80.26 / 141.06 μs，
   合计大于未分块的 166.52 μs。源码按互不相交的 token 范围搬运，没有
   每块重搬整个输入；第二块与通信重叠时变慢，需进一步区分核调度、缓存和
   内存带宽争用，当前 trace 不能直接证明具体硬件瓶颈。

当前 epilogue 仍在最后一个 chunk 完成后整体执行；已有 pipeline 只隐藏
producer 打包的一部分，未覆盖整个 Dispatch 的计算阶段。若上述成本压低后
仍不足以产生净收益，再评估按接收块提前执行 epilogue 的更深层流水设计。

原始目录 `results/dispatch-tail/overlap-profile-v3b-{A1,B1,B2,A2}/` 已完整
收取到本地 `.scratch/overlap/results/dispatch-tail/`。分析脚本为
`.scratch/overlap/analyze_profile.py`、`summarize_profile_v3.py`，汇总为
`overlap-profile-v3b-summary.json`。四轮全部采集成功，任务 exit=0，实验
扩展已恢复、源码未改。本轮只做诊断和文档记录，没有实现新的优化。

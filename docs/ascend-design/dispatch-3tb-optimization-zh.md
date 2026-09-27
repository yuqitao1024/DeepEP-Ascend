# 普通 Dispatch 3 TB/s 优化实验

日期：2026-09-27。基线 `0fe8976`，当前约 3.745–3.896 ms。
固定 8 rank / 8192 tokens / hidden 7168 / top-k 8 / 256 experts，FP8。
保持 7,786,089,792 logical bytes，3 TB/s 对应 max-rank Event mean ≤2.595363 ms。
此前详细时间线见 `five-operation-profile-zh.md`。

当前决策：第一项合批实现初筛不保留，代码已撤回；第二项通过独立设备
oracle 和三组完整 ABBA，保留并默认开启；第三项完成三组 ABBA 与时间线分析，
收益方向不稳定，已撤回生产改动，实验源码和结果留在 `.scratch`。
另确认普通路径存在跨代接收区覆盖，作为正确性修复单独处理，不算性能收益。

上一轮第二项已提交为 `6626795`，三组 ABBA 合计缩短 Dispatch 耗时
4.804%。这不是当前新一轮的完成项。当前新一轮针对 stage profile 的
三个方向是：隐藏 flush/CQ 等待、合并或复用 completion 边界、处理跨 rank
启动偏斜。截至 2026-09-28，新一轮三项均已完成：第一项三组收益方向
不一致并撤回；第二项三组稳定正向，保留并默认开启；第三项证明共同 host
截止时刻可消除 Event 指标中的启动偏斜，但 rendezvous 位于计时区间外，
只保留为诊断工具，不作为生产提速。最终记录见文末。
下文第一至第三项指上一轮实验；新一轮进度见文末续接记录。

用户授权按顺序实施三个方向，每项独立正确性和三组 30/30 ABBA 验收，
稳定正向才保留；不要求最低百分比。其他形状与 BF16、2/4/8 rank 做功能
及适当性能对照，不把典型用例收益泛化成所有路径收益。

1. **完成回读合批**：diagnostic 和 count bridge 在同一 stream 排队，
   一次等待后读取。错误和 generation 校验不变；超过固定页容量时有界回退。
   假设是减少独立同步后缩短 host 尾部，实际设备工作和逻辑字节不变。
2. **consumer 重复遍历**：普通路径尝试合并 metadata 与 SF/weights 输出，
   再判断按 source 分配搬运是否值得保留；不直接恢复此前无稳定收益的
   metadata lane 并行候选。假设是减少重复地址定位和 record 读取。
3. **接收端与通信 overlap**：根据真实到达边界提前处理已到达 source，
   保留跨代接收区消费约束。需要解决 count/输出偏移、验证与发布顺序。
   假设是隐藏部分 epilogue，而不只隐藏现有约 0.286 ms 的 producer。

## 第一项：合批回读候选（不保留）

已撤回的实验开关为 `DEEP_EP_ASCEND_DISPATCH_BATCH_READBACK=0/1`，初始默认 0，
同二进制 ABBA。非 cached Dispatch 使用；cached 和 Combine 保持原路径。

在 runtime 固定页缓冲中按 64-byte 对齐分配多个回读区段，一起入队后仅
等待一次。只有所有提交与等待成功，才向 caller 复制结果；等待失败保留
pending stream，禁止复用及提前释放。大于 4096-byte 容量时保留原有有界
顺序回读。Host 只在检查 diagnostic 后使用 count，不忽略错误。

新回归直接测试真实 `SmallHostTransfer`，替换 ACL 边界，覆盖延迟完成、
第二次提交失败、等待失败、不发布半成品、失败后禁止复用和 drain 后释放。
新增测试先因缺少 batch API 失败，实现后与已有单次回归共 2 项通过。

实验文件和源代码快照位于 `.scratch/dispatch-3tb/`。一组完整 ABBA，
每次 30 warmups / 30 iterations，每变体共 60 个 max-rank Event 样本：

| 指标 | A：关闭 | B：开启 |
| --- | ---: | ---: |
| Dispatch mean ms | 3.918218 | 4.005410 |
| Dispatch p50 ms | 3.953402 | 4.030860 |
| Dispatch p95 ms | 4.380525 | 4.277784 |
| Dispatch logical GB/s | 1987.151 | 1943.894 |

主指标平均耗时退化 2.225%，没有保留依据；不宣称一组即可证明稳定退化。
四次五操作完整正确性均通过，B2/A2 在完整结果后触发已知 teardown
SIGSEGV，经已有结果校验器确认。任务 `task_20260927_124020_36779951167`，
设备固定 `0,1,2,3,4,5,6,7`。汇总与原始 JSON：
`.scratch/dispatch-3tb/results/dispatch-3tb/readback-full-{g1-*,summary}.json`。

已撤回 host 合批、runtime batch API 和对应测试改动；实验测试源移至
`.scratch/dispatch-3tb/rejected-small-host-readback-batch-probe.cpp`，可恢复。
目前没有达到 3 TB/s 的证据。

### 早期故障对照记录

- NPU8P-ALT，`yuqitao`，配对 CANN/HCOMM 来自
  `/data/disk2/cann_version/0916/use_cann/cann-9.3.0`，Python 3.10。
- 第一项增量构建和全量 66 个目标构建均通过，产物 SHA256 一致：
  `e9cb7e864358ce2dea64beda4b56d8635cf5650c73a69d37f6e02266eb4c1c93`。
  全量构建任务 `task_20260927_122707_30971202998`。
- 小用例为 2 rank / 32 tokens / hidden 128 / top-k 2 / experts 8 / BF16。
  首次候选开启失败；第一次隔离对照中原版通过、候选关闭失败、开启通过。
  重复任务 `task_20260927_123133_313242720216` 固定设备 `0,1`，其第一轮
  原版与候选关闭均在权重比较处失败，开启通过。原版是原 HEAD 已测产物，
  SHA256 `568a44c2e03fdf14f43a5101a6e037354155b5a5be93adeb37713480c554e3d9`。
  原版复现排除了“仅新增合批逻辑引入”的解释；不能据偶发通过宣称修复。
- 当时尚未执行性能 ABBA，后续结论见上表。原始日志在远端
  `results/dispatch-3tb/`，对照脚本失败后会继续，不能把包装脚本 exit 0
  当作所有用例通过。
- 新回读单元测试以及 runtime 生命周期、纯 C++ runtime 和 VF ABI 相关检查
  共 6 项通过。额外旧检查 `test_direct_epilogues_keep_shape_iteration_32_bit`
  在寻找已从 `dispatch.asc` 拆出的函数时失败；其源文件定位与 HEAD 现状
  不符，不作为新内核通过证据。

## 第二项：metadata 与 SF/weights 合并（保留）

开关 `DEEP_EP_ASCEND_DISPATCH_FUSED_METADATA_COPY=0/1`，验收后默认 1。
仅普通非 cached、非 expanded、非 hybrid、非 stream 模式启用。
metadata 内核每线程处理一条接收 record，同时写 metadata、top-k、SF 和
weights；保留向量 hidden 搬运，hidden 非 32-byte 对齐时保留原标量尾部。
原路径开关关闭时仍可对照。

新增参数使用 POD 包装，并静态核查大小和关键偏移。生产 launcher 的独立
oracle 覆盖 SF 连续/转置布局、top-k 1/2/8/32、空输入、错误状态保持、
多 block/grid stride 以及输出尾部 guard。适配器本地 C++ 语法检查通过；
全量设备构建 `task_20260927_123907_366667029184` 通过，产物 SHA256
`b6ee8920f8734dadef853904c59baeb1e0bcfb2a949677b982a0decdadbf8ab2`。
设备任务 `task_20260927_124020_36779951167` 的独立 oracle 在 device 0
通过 120 个用例。

`metadata-fence` 产物的端到端检查覆盖 2-rank BF16 小输入、4-rank FP8
hidden 7184 的非 32-byte 对齐尾部、8-rank 典型输入，以及 2-rank / 1024
experts，五操作 oracle 全部通过。随后三组 ABBA 的 12 次运行也全部有
完整有效结果。任务 `task_20260927_124924_374189331480`，固定设备 0–7。

每次 30 warmups / 30 iterations，合并每变体 180 个 max-rank Event 样本；
A/B 均启用必要的设备消费完成 barrier，readback 合批均关闭：

| ABBA 组 | A 关闭 ms | B 开启 ms | 耗时缩短 |
| --- | ---: | ---: | ---: |
| 1 | 3.957470 | 3.761581 | 4.950% |
| 2 | 4.064096 | 3.858877 | 5.050% |
| 3 | 4.011035 | 3.834060 | 4.412% |
| 合并 | 4.010867 | 3.818173 | 4.804% |

合并 logical bandwidth 1941.249→2039.219 GB/s；p50 4.031486→3.802002 ms，
p95 4.347199→4.216235 ms。其余四项平均耗时变化在 +0.16%～+0.54%，
它们不走融合路径，不能把这些波动解释成优化收益。
原始数据与汇总在 `.scratch/dispatch-3tb/results/dispatch-3tb/metadata-fence-*`。
收益在三组中方向一致，按用户标准保留；3 TB/s 目标仍未达到。

### 原路径小用例错误的进一步证据

上述三轮对照完整结果：原版通过 1/3、候选关闭 0/3、开启 2/3。追加诊断
三者均失败。原版 rank 0 接收权重有 38 个有效值变为 0，全部位于来自
rank 1 的部分；rank 1 Combine 返回同样 38 个错误值。小数没有近似
误差，实际值是 0，期望为输入随机权重（例如 0.8474337458610535）。

现象与此前 source pipeline 文档的“下一次 Cached Dispatch 覆盖上一代
接收区”一致。原 HEAD 的 `completion_fence` 仅在 source pipeline 开关启用且
模式受支持时生效；普通路径开关未设置时不参加消费完成 barrier。
任务 `task_20260927_124924_374189331480` 已验证因果：原版仅加普通 Dispatch
返回后的诊断 barrier，3/3 通过；不加 barrier、给后续 Cached Dispatch
的输入权重填 0.375 后，上一轮的 38 个错误值变为 0.375。
诊断脚本仅位于 `.scratch`，不能用其加 barrier 的时间作性能结论。

同一任务中，将已有设备消费完成 barrier 扩展至非 cached、非 hybrid、
全设备 epilogue 的多 rank Dispatch，3/3 小用例通过。实验二进制
`metadata-fence` SHA256 为
`2110e93ad27b8e31d78eb089b557e990f4d817cea4064f6901e9f7c55ae98ff3`，
构建任务 `task_20260927_124726_372854028363`。实验 fence 开关仅存在于
该快照；后续本地代码将此正确性边界设为该路径的必要步骤，不提供关闭开关。
此开销同时计入后续优化的 A/B；不与旧的无 fence 时间混称单项收益。

新增生产回归 `tests/ascend/production/run_dispatch_receive_reuse.py`，连续
20 次真实的五操作准备与 oracle 检查，后续 Cached Dispatch 使用 0.375
标记，保留所有断言且不插入 host barrier。`receiver-v4` 的普通路径和
overlap 路径分别通过 20 次检查。最终代码还把消费完成 barrier 放到
CPU-count split 的实际 epilogue 之后，使异步 rank 与普通 rank 使用
同一跨代边界；新增 `--reuse-mixed-streams` 回归验证该组合。

## 第三项：接收侧 overlap（不保留）

候选设备构建和功能检查通过，但三组性能测试方向不稳定，未验收。
以下描述归档候选的实现，当前生产代码中已移除该开关和实现。
候选让本 rank 的发送服务与接收 epilogue 并发，而非直接
重启已退化的 source chunk pipeline。当前 acquire 首先校验本地
`consumed_generation == generation`，且最终完成与跨代 consumed barrier
依赖发送服务结束。因此不能只换 stream：需要明确开始事件、发送服务完成
事件，以及错误/代际校验所在的完成边界，保证函数返回前两条 stream 均完成。
开关为 `DEEP_EP_ASCEND_DISPATCH_RECEIVER_OVERLAP=0/1`。producer 和 release
命令生成均在第一条 stream 上完成后才记录开始事件，第二条 stream 仅
执行发送服务。命令生成读取的 outbound counts 与 acquire 写入的 inbound
counts 复用 workspace，不能让二者并发。第一条 stream 按现有
远端 ready 校验执行 acquire、metadata 和 payload epilogue。发布完成前
等待发送服务事件，再验证本地 `consumed_generation`，最后参加跨代
消费完成 barrier。两个事件销毁前排空两条 stream，失败路径也保留排空。
这些成本全部纳入 Event 测量。普通单 stream 路径保持原检查位置。

阶段 profiling 开启时仍使用一个完整 release command batch，避免后续
control 命令读取已改写的计数；该 service 区间包含 payload/control/barrier。
不开阶段 profiling 的正式 Event ABBA 不受此标注差异影响。

`receiver-v4` 全量构建任务 `task_20260927_130345_380697824747`，SHA256
`a90dfe6312fb9812eb5816830c122aeeb85a650af4c7cf7d32266d88a8eaf552`。
设备任务 `task_20260927_130523_381713517759` 固定设备 0–7：普通路径与
receiver overlap 路径各通过 20 次带 0.375 权重标记的接收区复用回归；
overlap 开启后的 2-rank BF16 小输入、4-rank FP8 非对齐尾部、8-rank
典型输入、2-rank 1024 experts 均通过完整五操作 oracle。随后做一组 ABBA
初筛，双方默认融合 metadata 且保留设备消费完成 barrier。

后续任务 `task_20260927_131139_384131011515` 完成第 2/3 组 ABBA、
开关两侧 Profiling 和 descriptor snapshot 边界回归，任务 exit 0。
固定设备 0–7，每次 30 warmups / 30 iterations，A/B 均使用相同二进制，
第二项和正确性 fence 均启用；每变体合计 180 个 max-rank Event 样本：

| ABBA 组 | A overlap 关闭 ms | B overlap 开启 ms | 耗时缩短 |
| --- | ---: | ---: | ---: |
| 1 | 4.034537 | 3.751517 | 7.015% |
| 2 | 4.093467 | 3.843964 | 6.095% |
| 3 | 3.807987 | 3.964447 | -4.109% |
| 合并 | 3.978664 | 3.853309 | 3.151% |

合并逻辑带宽 1956.961→2020.624 GB/s，但第 3 组 mean、p50、p95 均退化。
因此不把合并的正值当作稳定收益，也不宣称 overlap 必然退化。
g2-B1/g3-B1 在完整有效五操作报告后出现已知 teardown SIGSEGV，
经既有校验器确认；其余本任务 ABBA 正常退出。数据与汇总为
`.scratch/dispatch-3tb/results/dispatch-3tb/receiver-v4-{g*,summary}.json`。

### 时间线：确有并发，但端到端收益不稳定

使用现有五操作 profiler；每侧 8 rank，丢弃 capture 0 后分析 88 个
rank/capture，并按每次 max-Event rank 汇总 11 个关键路径。单位 µs：

| 指标 | 关闭 | 开启 |
| --- | ---: | ---: |
| kernel span | 2910.52 | 2643.66 |
| kernel 并发区间（sum − union） | 0 | 946.10 |
| acquire kernel | 359.18 | 741.09 |
| consumed barrier kernel | 74.15 | 292.93 |
| host stream sync 次数 | 4 | 6 |
| 新建事件总耗时 | 0 | 69.41 |
| 销毁事件总耗时 | 0 | 30.52 |
| profiler 中 max-rank Event | 3881.66 | 4142.89 |

确有设备并发，kernel span 减少约 267 µs。acquire 提前启动后等待更久，
末尾 barrier 也更长；候选另外引入事件与 stream 管理成本。
Profiler 下 host 入口、Python 收尾亦有变化，不能把全部差额归因于某个
同步 API，也不能把 host 等待与设备耗时重复相加。两次独立 profiler run
前的无采集 Event 均值为 3749.64/3696.83 µs，仅作为诊断，不取代 ABBA。
当前证据支持“实现了并发，但尚无稳定端到端收益”。

若后续继续，先验证事件复用/减少必要边界外的 host 同步，再判断是否需要
按 source 的细粒度接收就绪协议；不能删除跨代消费完成 fence 换取分数。
本轮不继续扩大实现范围。候选快照为
`.scratch/dispatch-3tb/receiver-with-split-fence-source.tgz`；实测二进制的
源快照为 `receiver-v4-source.tgz`，两者区别是前者扩展了 split fence。
原始 trace 和 `analysis.json` 位于
`.scratch/dispatch-3tb/results/receiver-v4-{off,on}-dispatch/`。

## 最终保留版本验证

撤回第一/三项后，仅保留第二项融合与独立的接收区复用正确性修复。
全量设备构建 `task_20260927_132006_388008019187` 通过，产物 SHA256：
`740d7e0c66b574797670d69f089d972834271ca32305cce845141c97bcbdac10`。
源码归档 `.scratch/dispatch-3tb/retained-source.tgz`。

设备任务 `task_20260927_132151_389037328233` 使用 NPU8P 设备 0–7 与同一
配对 CANN/HCOMM 环境。已通过 metadata 独立 oracle 120 项；普通模式、
混合普通/异步 stream 模式各通过 20 次接收区复用检查，两 rank 均报告成功。
descriptor snapshot 的篡改、旧 generation、异步 event 复制与销毁边界
回归两 rank 通过。未加入绕过问题的 host barrier，也未跳过 oracle。

同一任务已正常完成（exit 0），2-rank BF16 小输入、4-rank FP8 hidden
7184 尾部、8-rank 典型输入、2-rank 1024 experts 均通过五操作检查。
最终默认配置（不设置 metadata selector）的一轮 30 warmups / 30 iterations：

| 操作 | mean ms | p50 ms | p95 ms | logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | 3.664034 | 3.613646 | 3.972025 | 2125.005 |
| Expanded Dispatch | 15.289504 | 15.284659 | 15.703263 | 605.087 |
| Cached Dispatch | 64.116342 | 64.025253 | 65.044058 | 121.437 |
| Combine | 14.004960 | 13.946628 | 14.410011 | 778.382 |
| Reduced Combine | 14.566444 | 14.521817 | 15.194867 | 748.378 |

这是最终代码的一次确认采样，不代替第二项三组 ABBA 的稳定收益依据，
也不与其他时段数字拼接计算收益。Dispatch 约 2.125 TB/s，仍未达到
3 TB/s。原始报告为
`.scratch/dispatch-3tb/results/dispatch-3tb/retained-final-performance.json`；
所有最终功能结果在同目录 `retained-*`。远端运行后已恢复测试前安装的
扩展，实验产物单独归档，不将旧的安装产物误称为最终代码。

本地 small-host-transfer、runtime 生命周期、纯 C++ runtime、SIMT VF ABI
相关回归共 5 项通过；`git diff --check` 通过。前述 HEAD 已存在的旧结构
检查失败不计为设备验证通过，本轮没有修改该无关测试。

## 2026-09-27 下一阶段：Producer 路径定位

当前普通 Dispatch 的最大单点是 producer 侧。最终代码的 profiler 中，
`dispatch_kernel<false>` 合计约 1.45 ms，占关键路径设备 kernel 总量的
约 50%；acquire 约 0.36 ms，metadata 约 0.19 ms，barrier 约 0.07 ms。
下一阶段先做定位，不做投机性优化。

### 反馈回路与验收

1. 以 8-rank / 8192 tokens / hidden 7168 / top-k 8 / 256 experts 的
   representative case 为固定 workload，使用现有五操作 profiler 采集
   8 rank × 11 capture，丢弃首个 capture。
2. 解析 `dispatch_kernel<false>` 的每次调用：调用次数、顺序、stream、
   grid/block、耗时分布，并与 producer control/group/prefix/record/release
   及 epilogue stage 对齐，确认 1.45 ms 是单个 kernel、重复调用还是聚合误差。
3. 只对可独立开关的候选做同二进制 ABBA；每个候选至少 3 组
   30 warmups / 30 iterations，mean 方向一致才保留。正确性门禁包含
   2/4/8 rank 功能、metadata oracle、接收区复用和 snapshot 边界。
4. 不得删除跨代 consumed barrier；不得为了 profiling 或性能改变逻辑字节、
   rank 聚合或计时公式。

### 候选优先级

1. **Producer record/release 遍历合并**：检查 grouping、record 构造和
   release 命令生成是否重复扫描 token/top-k；若存在可合并的重复遍历，
   先量化扫描次数和内存访问量，再实现。
2. **边界与状态更新下沉**：检查 producer kernel 中的 expert 归属计算、
   capacity 校验、workspace 状态写和 command queue 更新是否被重复执行；
   只下沉到正确边界，不能把错误检查延迟到数据被覆盖之后。
3. **命令生成与 payload 服务边界**：区分记录扫描、命令 materialization
   和实际 URMA service。第三项 overlap 已证明服务可与 epilogue 并发约
   0.95 ms，但简单双 stream 端到端不稳定；后续只考虑事件/队列复用或
   细粒度 source-ready，不重启同一候选。
4. **acquire/barrier 成本**：二者是必要同步边界，只研究降低实现成本或
   与已有边界合并，不通过移除等待换性能。

若 producer 拆解后没有可稳定复现的通用收益，再评估 cached/expanded
Dispatch 的独立路径；不把普通 Dispatch 的结论外推到其他操作。以下数据
仅作为下一轮诊断输入，不作为新优化收益。

### 首次拆解结果

重新分析最终代码的关闭 overlap trace（8 rank × 11 capture，共 88 个
rank/capture）后，此前“`dispatch_kernel` 合计 1.45 ms”不是单个 producer
kernel，而是同一符号在多个 stage 的 service wrapper 聚合。按 stage 顺序：

| Stage | kernel / wrapper | mean µs | p50 µs |
| ---: | --- | ---: | ---: |
| 7 | producer release 前的 service wrapper | 166.26 | 165.88 |
| 8 | producer release | 169.28 | 167.51 |
| 9 | release 后完整 payload/control service | 942.54 | 942.31 |
| 11 | acquire | 314.64 | 179.71 |
| 17 | parallel prefix | 233.06 | 235.00 |
| 20 | fused metadata | 188.88 | 188.87 |
| 24 | hidden vector copy | 317.03 | 316.86 |
| 27 | consumed barrier | 71.60 | 71.94 |

同符号还有多次 1–4 µs 的轻量 wrapper。rank0 单个 capture 的完整顺序显示：
producer record 119.43 µs、release 169.28 µs、随后 stage 9 service
935.01 µs；acquire 在该 capture 中 340.11 µs。因此下一阶段的第一优先级
修正为 stage 9 的 command/service 拆分，而不是泛称 producer kernel。

初步假设（待用独立开关验证）：

1. stage 9 同时执行 payload 写、control/barrier command 和 transport
   service，942 µs 中至少一部分可与本地 epilogue 并发；但第三项已证明
   简单双 stream 端到端不稳定，需要先拆出 command 数量与各类命令耗时。
2. stage 7 与 stage 8 分别执行 service wrapper 和 release，二者相邻且合计
   约 336 µs，可能存在重复状态 reset/queue flush；若 command queue 可复用或
   可合并 flush，才有优化空间。
3. hidden vector copy 317 µs 与 fused metadata 189 µs 是接收侧最大计算段；
   除非重新设计 copy lane，否则它们与通信 service 的 overlap 是主要方向。
4. acquire p50 179.71 µs、max 1410.57 µs，均值受慢 rank 拖动，说明跨 rank
   到达不均是重要长尾来源；先做 rank 粒度到达分布，再决定是否做
   source-ready 细粒度处理。

下一步先给 stage 9 增加可选 profile 拆分，区分 payload/control/barrier 与
实际 transport service；同时统计每 rank 的 acquire 等待区间和接收 source
数量。该诊断必须能用现有 trace 复现，并在功能测试中不改变默认行为。

### Stage profile 复测

使用最终保留二进制（SHA256 `740d7e...bdac10`）在 NPU8P 设备 0–7 运行
`bench_ep --profile-stages`，任务
`task_20260927_185539_15248406634` 正常完成。该运行是诊断采样，不代表
正式 Event 性能。Dispatch max-rank Event 为 4.104 ms；主要 stage：

| Stage | max-rank span |
| --- | ---: |
| producer record | 0.172 ms |
| release payload | 0.917 ms |
| release control | 0.283 ms |
| release barrier | 0.045 ms |
| epilogue copy | 0.320 ms |
| consumed barrier | 0.072 ms |

Service 内部归因（按 1 GHz cycle 近似为 µs）：

| 指标 | 值 |
| --- | ---: |
| payload command cycles | 0.072 ms |
| control command cycles | 0.205 ms |
| flush command cycles | 0.891 ms |
| CQ drain/wait cycles | 0.803 ms |
| barrier command cycles | 0.253 ms |
| service active cycles | 1.481 ms |

这说明 0.917 ms 的 release payload 不是“生成 payload 命令”慢，而是
Flush/CQ completion 等待占主导。生产 payload 命令本身只有约 72 µs。
因此下一轮优化不应泛化成“减少 producer 计算”，而应聚焦：

1. 减少 flush/CQ 等待的暴露时间，特别是与本地 epilogue 的 overlap；
2. 合并或复用必要的 completion 边界，避免重复 drain；
3. 处理跨 rank 启动偏斜。该 stage-profile run 中 rank 7 的 host entry
   比多数 rank 早约 1.1–1.9 ms，device envelope 达 4.65 ms；其他 rank
   多为 2.8–3.1 ms。这个偏斜是测量/启动同步问题，不能直接算作 kernel
   计算差异。

## 2026-09-27 20:30 会话续接：新一轮 overlap/事件复用正确性门禁

从 provider `mycodex` 的会话 `01a0a2a5-e655-7413-bbf0-6a8dcd1780f6`
恢复实验进度。会话已完成 v10 一次性事件对照，随后尝试缓存事件、reset
和 `aclrtCreateEventExWithFlag`；最后停在 v14 构建完成、未跑设备回归。
本节数据是正确性验证，不是性能 ABBA，不能用来计算优化收益。

### 本轮结果

| 候选/检查 | 结果 |
| --- | --- |
| v14：Ex 缓存事件、不 reset，普通复用 | 失败：首次准备阶段 normal/cached payload 不一致，rank 0 有 6873/7040 个元素不匹配 |
| v10：一次性事件，2-rank BF16 小输入 | 通过五操作校验 |
| v10：4-rank FP8 hidden 7184 尾部 | 通过五操作校验 |
| v10：8-rank 8192/7168/top-k 8/256 experts | 通过五操作校验 |
| v10：2-rank 1024 experts | 通过五操作校验 |
| v10：metadata oracle | 120 项通过 |
| v10：snapshot 边界 | 两 rank 通过 |
| v10：首次 overlap 开启、混合 stream 复用 | 失败：combine 输出 2040/4096 个元素不匹配 |
| v10：随后 overlap 关闭/开启各一次混合复用 | 均通过，每次两 rank 各 20 次 |
| v10：再做三组关闭/开启配对混合复用 | 六次均通过，每次两 rank 各 20 次 |

后续通过没有消除首次失败，不能将 v10 标记为正确性验收通过，也不能
将混合 stream 故障直接归因于缓存事件：v10 使用一次性事件。当前还没有
确定首次 combine 输出错误的根因，不以一次开关对照证明因果。
另发现首次混合 stream 检查与误以无设备任务提交的 metadata oracle 有时间
重叠，因此该次失败还有实验隔离方面的干扰因素，不能据此认定 overlap
引入了 combine 错误。后续三组配对为独占设备运行。
因此本轮未启动性能 ABBA，未启用第三项为默认路径。

v10 二进制 SHA256：
`308abf8194f2d9909edccfd747019e0a385f42308a65823d72dd7c1054f48e32`。
v14 二进制 SHA256：
`a9cb0debd9fb47bb6a00c86ea49d7543619e516c9faa335dd96852faa8030f40`。
这些是已有实验快照，不是本轮本地接口修复后的构建产物。

设备任务：v14 `task_20260927_202309_205802018673`；v10 多形状
`task_20260927_202401_206044714448`；首次混合复用
`task_20260927_202520_206754632041`；关闭对照及 snapshot
`task_20260927_202602_207096027219`；开启复测
`task_20260927_202703_20742176472`；三组配对
`task_20260927_202748_207630411375`。metadata oracle 首次误以无设备任务
`task_20260927_202532_206871023051` 提交；检查脚本确认它实际使用 NPU，
因此该次结果不作为隔离验证依据，另按设备队列独占 device 0 复测。
独占复测任务 `task_20260927_203111_208682428668` 的 120 项全部通过。

原始 JSON 和日志（含失败）已取回
`.scratch/dispatch-3tb/handoff-20260927/results/dispatch-3tb/`，归档为同目录上层
`device-results.tgz`。远端任务已结束，测试安装的扩展已由脚本恢复。

### 修复候选事件 API 对已有 host 回归的影响

续接时 `create_event_ex` 被插入 `StreamEventApi` 聚合字段中间，并被普通
`valid_api` 强制要求，导致原有 callback table 初始化无法编译。
将该可选回调移到末尾，普通事件路径继续接受旧表；只有显式创建 Ex
事件时校验该回调。新增回归覆盖缺少扩展回调、设备不匹配、后端失败、
空句柄和成功创建/销毁。普通事件测试仍使用原有 callback table。

异步事件（testing=0/1）、runtime 生命周期、纯 C++ runtime 和小传输回归
通过。扩大检查时还遇到已有测试夹具缺失 `pybind11/pytypes.h`，以及
combine 源码结构检查寻找已拆走符号的失败；这些不计为通过。
本地修复尚未设备重建，不将已有 v10/v14 的设备结果归给它。

下一步应先定位 mixed-stream 首次 combine 错误的跨操作数据所有权边界，
保留全部失败记录；不能仅因后续复跑通过便进入“稳定性能收益”的结论。

## 2026-09-28 新一轮三项结论

本轮固定 8-rank / 8192 tokens / hidden 7168 / top-k 8 / 256 experts，
每项使用同二进制三组 30 warmups / 30 iterations ABBA。mean 三组方向一致
才保留。所有百分比均由每组两次 A 与两次 B 的 60 个 max-rank Event 样本
计算，不与其他时段采样拼接。

### 1. 隐藏 flush/CQ 暴露时间：撤回

receiver overlap 使用独立 service stream，把 producer release 后的 transport
service 与本地 epilogue 并发；可复用 CANN timeline event 的普通与混合
stream 接收区复用各 20 次通过，snapshot、2/4/8-rank 和 1024 experts
边界通过。产物 `receiver-v15` SHA256：
`8c598e5cb86f8028a27b1a42bd89824f4cc32e31ac845291ff3e47825d3849d4`。

三组 Dispatch mean 收益分别为 -7.112%、+1.155%、+5.954%，合并仅
+0.117%。方向不一致，按门槛撤回双 stream、event cache 和扩展 event API；
不能因设备 trace 中存在并发就宣称端到端提速。ABBA 任务
`task_20260927_204021_231693827352`。

### 2. 融合 completion 边界：保留并默认开启

`DEEP_EP_ASCEND_DISPATCH_FUSED_CONSUMED_BARRIER=0/1` 把原独立
consumed-barrier kernel 合入 epilogue-complete kernel。融合路径仍执行
diagnostic/scratch 检查、queue reset、dispatch generation 发布、barrier
generation 发布、system fence、world barrier 和 service execute；没有删除
跨代接收区保护。只在多 rank 普通非 cached、非 expanded、非 hybrid、
非 stream、非 pipeline、非 CPU-sync、非 stage-profile 路径生效，未设置时
默认开启。

v18 三组结果：

| ABBA 组 | A 关闭 ms | B 开启 ms | 耗时缩短 |
| --- | ---: | ---: | ---: |
| 1 | 3.938236 | 3.831452 | 2.711% |
| 2 | 3.937021 | 3.829758 | 2.724% |
| 3 | 3.933472 | 3.873716 | 1.519% |
| 合并 | 3.936243 | 3.844975 | 2.319% |

合并 logical bandwidth 1978.051→2025.004 GB/s，p50 改善 2.456%，p95
改善 3.219%。ABBA 任务 `task_20260928_012617_351864118233`；v18 产物
SHA256 `883b98bd13b1e89d6afe348a4d5b7f2bacf7da0e1dbcbc7336ac2949c782c3fc`。
普通/混合接收区复用、snapshot、2-rank BF16、4-rank FP8 hidden 7184
尾部、8-rank 典型输入和 2-rank 1024 experts 全部通过。

清除第一项代码后的最终源码由任务 `task_20260928_014621_362429426534`
构建成功，产物 SHA256：
`033b1ffb40a71a05348e0438194debf5eaddbe58aeb127e3b90005d748b15fdc`。
最终设备任务 `task_20260928_014711_363254718627` 在默认开启状态重复通过
上述正确性边界。最终一次 30/30 确认采样 Dispatch 为 3.576366 ms、
2177.095 GB/s；这是单次状态确认，稳定收益依据仍是三组 ABBA。当前仍未
达到 3 TB/s。

### 3. 处理跨 rank 启动偏斜：诊断完成，不并入生产

正式 benchmark 参数 `--rank-launch-deadline-us 2000` 在每个 NPU Event
start 前用 all-reduce 取得共同 host 时钟上界，再等待到未来 2 ms 的共同截止时刻。
rendezvous 与等待均位于 Event 区间外，操作、逻辑字节、rank 聚合和计时
公式不变。第二项固定开启时，三组 Dispatch Event mean 分别改善 7.763%、
13.122%、16.476%，合并 3.730373→3.260960 ms（12.584%）；任务
`task_20260928_013529_357153030449`。

该结果证明当前 Event 数字显著受 host launch skew 影响，也说明单次 3.576 ms
不能完全归因于设备 kernel。但方案把同步成本移到计时外，应用端总延迟没有
得到同等改善，不能计为生产性能收益，也不用于声称达到 3 TB/s。该能力已
进入受版本管理的 benchmark，默认关闭，并将值写入 `timing_protocol`；原始
wrapper 仍随实验归档保留。生产源码不包含 host collective、busy-wait 或
额外入口 barrier。

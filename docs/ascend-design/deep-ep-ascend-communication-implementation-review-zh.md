# DeepEP-Ascend 通信实现现状讨论稿

日期：2026-09-22

## 1. 当前结论

当前实现已经在 CANN 9.3.0、NPU8P-ALT 的 8 rank 典型 case 上通过功能验收。通信栈分成两层：host 控制面负责创建 communicator、注册内存、获取真实链路和 AIV channel；device 数据面由 SIMT VF 生成命令，AICore service 把命令翻译成 URMA WQE 并驱动 SQ/CQ。

Normal Dispatch 约 22 ms，Normal Combine 约 31 ms。同一固定 workload 下，payload 数据面本身的 transport-only 时间只有 0.88 ms，说明大块 HCCS 传输能力不是当前主要瓶颈，端到端时间主要花在打包、控制、同步和 consumer 侧工作上。

之前说的"rank 差 50 倍"需要重新表述。固定顺序串行 acquire 时，第一个未 ready 的 source 会挡住后面的 source，profile 会把等待累计到这个 source 上；反向遍历后慢 source 也跟着反向，说明这主要是遍历顺序造成的队头阻塞和归因偏置，不是某个固定 rank 的物理链路慢。改成 ready-first 非阻塞轮询后，rank 均值差降到约 5%。早期 acquire 到 validate 的 2.1M 到 2.4M cycles 差距，后来确认是把通用 no-op kernel 当成真实专用 VF 边界导致的测量错误；真实专用 VF 边界差只有约 5.4k 到 7.5k cycles。

剩余约 5% 的 rank 差异还没有完全定位。通用 no-op kernel 的调度间隙在 profile 里可以从 73k 波动到 5.9M cycles，但跳过 no-op 后端到端没有明显收益，22.099 ms 和 9 月 20 日的 21.913 ms 基线同量级。所以只能说 50 倍现象的主体已经拆开：一部分是串行 acquire 的队头阻塞，一部分是 profiling 边界错误，剩余约 5% 仍要继续查。不能把 no-op 调度写成最终根因。

2026-09-22 晚间又修正了一处诊断采样：之前的 first-ready 字段记录的是前 16 次 ready 事件，不是每个 source 的第一次 ready，ready-first 轮询会让重复 source 占满槽位。修正后重新跑 8-rank，case 通过，Dispatch mean/P95 为 21.841/23.315 ms。结果显示 rank 3、4、7 的真实 acquire VF 分别等待约 9.64M、10.04M、9.41M cycles；rank 5、6 等同一批 source 约 1.2M 到 2.2M；rank 0、1、2 基本在 30k 内完成。所有 rank 的 producer record、release payload/control/barrier 和 service span 都在同一量级。

当晚继续补了两层证据。第一层是 producer 侧 per-destination publication timestamp：8 个 producer 对 7 个远端 destination 的 control 发布调用都在约 0.079 到 0.083 ms 内完成，单个 producer 内部 destination 间差异约 0.063 到 0.066 ms，且这个顺序只是固定遍历顺序，没有数毫秒级差异。第二层是 host 侧绝对时间：8 个 Python rank 进入同一次 Dispatch 的时刻可以相差约 5.4 到 5.7 ms；最早进入的 rank acquire 等待最长，最晚进入的 rank acquire 几乎立即完成。把每个 rank 的 barrier 完成时间按 entry skew 归一后，最晚 ready 的公共时间落在约 5.2 到 6.5 ms，基本收敛。

当前更合理的解释是：profile 采样前调用 barrier(with_cpu_sync=True)，barrier 本身依赖 device 同步和 host 调度，退出后各 rank 进入 Dispatch 的时刻没有对齐。先进入的 rank 先跑到 acquire，等后进入的 producer 完成 release；后进入的 rank 到达时大部分控制面已 ready。因此之前的“特定 source-destination 路径慢”主要是观测窗口被启动错位放大，不能直接归因到 HCCS 路径或 HCOMM channel。这个问题首先出现在 benchmark/profile 编排层，通信库内还有少量约 0.1 ms 量级的发布顺序差异，但没有数毫秒级路径长尾。

补过一个简单的 A/B：barrier 后再追加一次 dist.all_reduce，希望 host 进程再次汇合。结果 entry spread 仍是 5.175 ms，acquire 等待仍是 0.051 到 4.038 ms。单次 all_reduce 不能消除这个错位，说明差异发生在 collective 返回后的 host 调度阶段，或者需要更严格的进入控制。profile 分析时必须保留 entry/prelaunch/synchronize 绝对时间并做公共时间换算。

## 2. 通信栈分层

### 2.1 host 控制面

进程启动后复用框架传入的 HcclComm，校验 rank 和 world size。CANN 9.2 和 9.3 在资源创建路径上差别很大。

先对齐几个对象。HcclComm 说明参与通信的进程集合和本进程 rank。rank graph 给出某个 src/dst pair 实际可用的通信链路。MR 描述本端和远端哪些内存可以被通信路径访问。channel 是到某个 peer 的传输上下文，里面绑定队列、token 和地址表；SQ/CQ 是 channel 内的提交和完成队列，不等于 channel 本身。

9.2 的路径是：

```text
HcclWorldTeamCreate / HcclTeamCreate
  -> HcclTeamWindowRegister
  -> HcclTeamCreateChannelsDescInit
  -> HcclTeamChannelsCreate
```

这条路给 DeepEP 两个设备可见句柄：HcommTeam* 和 HcommWindow*。Team 里有连续的 ChannelEntity 表；window 负责 symmetric address 到远端实际地址的转换。endpoint、MR 交换、queue 建立大多被 HCCL/HCOMM 藏起来了。

9.3 在当前 950PR 组合包上，旧的 symmetric window 路径没有得到可用结果：HcclCommSymWinGet 返回成功但 window 为空，HcclTeamCreate 得到的 team 中 channel 表也为空。已验证可用的是 communication-domain 路径：

```text
HcclCommMemReg
  -> HcclRankGraphGetLayers / HcclRankGraphGetLinks
  -> HcclChannelDescInit
  -> HcclChannelAcquire(COMM_ENGINE_AIV)
  -> HcclChannelGetRemoteMems
```

DeepEP 自己维护设备可见的 DeviceChannelTable 和 DeviceWindowTable。远端地址解析不再依赖旧 HcommWindow*，而是根据注册的远端 MR 实体和 token 信息查表。也试过显式 endpoint 路径，但 channel 一直停在 CONNECTING，原因是手工选 endpoint 未必是 rank graph 为该 rank pair 选择的真实路径，所以这条路没有被采用。

9.3 的 AIV channel 不调用 HcclChannelDestroy。当前包对该接口返回 backend error，头文件也说明暂只支持 CCU。清理顺序是先停 DeepEP 自己的 service 和 staged 资源，再销毁 communicator，AIV channel 随 communicator 释放。

对应源码入口：

| 步骤 | 文件 | 函数入口 |
| --- | --- | --- |
| 读取 channel 数量配置 | `csrc/backends/ascend/transport/channel_config.hpp` | `configure_transport_channels_from_environment` |
| 注册 payload/sync/command MR | `csrc/backends/ascend/transport/cann_transport.cpp` | `cann_register_memory`、`CannHostTransport::register_*` |
| 查询 rank graph | `csrc/backends/ascend/transport/cann_transport.cpp` | `cann_acquire_channels` |
| 建立 AIV channel | `csrc/backends/ascend/transport/cann_transport.cpp` | `cann_channel_desc_init`、`cann_channel_acquire` |
| 建立设备侧 peer/channel 表 | `csrc/backends/ascend/transport/cann_transport.cpp` | `CannHostTransport::build_device_tables` |
| 设备侧解析 channel | `csrc/backends/ascend/transport/device_transport_commands.hpp` | `channel_count`、`resolve_channel` |
| AICore service 消费命令 | `csrc/backends/ascend/transport/aicore_transport_service.hpp` | `execute_body`、`resolve_channel`、`resolve_peer` |
| 释放自建资源 | `csrc/backends/ascend/transport/cann_transport.cpp` | `CannHostTransport::release_backend_resources`、`release_backend_channels` |

### 2.2 device 数据面

数据面没有让 SIMT VF 直接写 SQ/CQ doorbell。原因很具体：CANN 9.2 时公开 st_dev 不能从 __simt_vf__ 调用，直接调 builtin 会触发编译器崩溃；__stg 虽然能编译，但它和 st_dev 的设备语义不同，不能当 doorbell 用。因此当前实现保留了这个分层：

```text
SIMT producer
  -> GM command queue 中的固定大小命令
  -> 单个 AICore transport service 逐条解释
  -> 构造 URMA WQE，写 SQ，ring doorbell
  -> poll CQ，处理 completion
```

支持的操作包括远端写、inline value 写、FAA、drain/flush、signal 读和等待、barrier。这个方案的代价是 command queue 编码、AICore 解析和一层 service 边界；好处是语义可控，特别是 flush、barrier、generation 完成和错误处理都能按 DeepEP 的协议实现。

release 协议先发 payload，drain 相关 SQ/CQ，再发布 count、generation 和 signal。barrier 对 peer 做 FAA，drain 后用 pending bitmap 轮询 generation counter。terminal completion 只有在所有队列 outstanding 清零且 diagnostic clean 时才发布 generation，active generation 不允许 reset 复用。这套规则修掉了早期 consumed_generation 提前发布的问题。

官方 asc-comm 后来提供了 SIMT 直接提交 Hcomm 的实现，历史阻塞点已经消失。但它要求一个 channel 由单个 lane 驱动，DeepEP 当前 producer 是多 block、多 lane 形态，不能直接替换。官方 SIMT 和当前 staged 实现的差异是数据面调度模型不同，不是同一套接口的包装差异。长期可以做一个最小 backend probe，先在单 lane、单 channel 上覆盖 put、put_value、remote_add、flush，再评估 producer 改造。

## 3. Dispatch 当前方案

direct scale-up Normal Dispatch 的主流程：

```text
producer control
  -> producer grouping
  -> producer prefix / slot
  -> producer record
  -> release payload / control / signal
  -> epilogue acquire
  -> validate records
  -> error reduce
  -> expert count / prefix
  -> metadata / destination assignment
  -> copy outputs
  -> completion
```

producer 侧按 expert 和目标 rank 分组，生成 record 和远端 payload 命令。release 阶段先保证 payload 可见，再发布 count、generation 和 signal。consumer 侧先等所有 source ready，再 validate、统计、分配输出并 copy。

关键源码入口：

| 阶段 | 文件 | 函数入口 |
| --- | --- | --- |
| host 发起 Dispatch | `csrc/backends/ascend/elastic_buffer.hpp` | `ElasticBuffer::dispatch` |
| host 选择普通/流水线 launch | `csrc/backends/ascend/elastic/runtime.cpp` | `launch_internal_dispatch`、`launch_internal_dispatch_pipeline` |
| producer 编排 | `csrc/backends/ascend/elastic/dispatch.asc` | `dispatch_kernel`、`direct_dispatch_producer_*` 调用链 |
| control/prefix/record | `csrc/backends/ascend/elastic/dispatch_vf_host_calls.hpp` | `launch_direct_dispatch_producer_control_variant_0`、`launch_direct_dispatch_producer_plan_variant_0`、`launch_direct_dispatch_producer_record_variant_0` |
| payload 写入与发布 | `csrc/backends/ascend/elastic/dispatch_device_common.hpp` | `direct_dispatch_producer_vector_payload_impl`、`direct_dispatch_producer_record_body`、`direct_dispatch_producer_release_body` |
| release 分片 | `csrc/backends/ascend/elastic/release_protocol.hpp` | `put_staged_records_striped`、`publish_control_and_release` |
| consumer acquire | `csrc/backends/ascend/elastic/direct_dispatch_epilogue_acquire.asc` | `direct_dispatch_epilogue_acquire_vf` |
| record 校验 | `csrc/backends/ascend/elastic/direct_dispatch_epilogue_validate_records.asc` | `direct_dispatch_epilogue_validate_records_vf` |
| metadata/destination/copy | `csrc/backends/ascend/elastic/dispatch_vf_host_calls.hpp` | `launch_direct_dispatch_epilogue_metadata_variant_0`、`launch_direct_dispatch_epilogue_assign_destinations_variant_0`、`launch_direct_dispatch_epilogue_copy_outputs_variant_0` |

acquire 原来按 source 0 到 7 固定顺序串行等待。诊断发现它会放大第一个晚到 source 的影响：早 ready 的 source 也会被它挡住。现在 direct Dispatch 的 acquire 已改成 ready-first 非阻塞轮询，每轮扫描所有 source，control ready 后再检查 release signal，全部 ready 后进入原有 final acquisition 和 validation。Hybrid Dispatch 和 Combine 还没有做同样改造。

ready-first 连续三轮的 Dispatch mean 为 20.381、20.807、20.920 ms，平均 20.703 ms；当时未修改基线为 22.104 ms。方向上约有 6% 收益，但样本少，不能当稳定收益。9 月 20 日的同环境代表基线是 21.913 ms；9 月 22 日跳过 no-op 的版本是 22.099 ms，二者仍在同一区间。Direct Combine acquire 也已切换为先轮询所有 contributor、再做最终读取的非阻塞模式；Hybrid 路径暂不改动。

## 4. Combine 当前方案

direct Combine 的主流程：

```text
producer control / plan
  -> producer record 写入目标 staging
  -> 本地直接落位，或本地 staging copy
  -> release payload / control / signal
  -> acquire 并 validate contributor slot
  -> reduce by source token
  -> weights
  -> completion
```

旧实现的本地 staging copy 很重，profile 中 producer_local_copy 可以占到 66% 左右。当前 direct path 已经做了 direct local placement：本 rank 作为目标时，record 直接写 receive shard，远端 rank 仍走 staging 和 HCOMM put，避免一次完整的本地 staging 到 receive copy。另一个保留下来的优化是更大的 producer tile，Normal Combine 因此有比较明显的下降。

当前 Combine 的瓶颈更偏 reduce、发布和 arrival 行为，而不是旧路径的本地 copy。Direct Combine acquire 已改为非阻塞 ready-first；Hybrid Combine 仍保留原有等待路径，不能把 direct 结果外推到 Hybrid。

Combine 的关键入口与 Dispatch 分开记录如下：

| 阶段 | 文件 | 函数入口 |
| --- | --- | --- |
| host 发起 Combine | `csrc/backends/ascend/elastic_buffer.hpp` | `ElasticBuffer::combine` |
| host launch | `csrc/backends/ascend/elastic/runtime.cpp` | `launch_internal_combine` |
| producer record/local copy | `csrc/backends/ascend/elastic/direct_combine_producer_record.asc`、`direct_combine_producer_local_copy.asc` | `direct_combine_producer_record_vf`、`direct_combine_producer_local_copy_vf` |
| payload/control 发布 | `csrc/backends/ascend/elastic/direct_combine_producer_release.asc` | `direct_combine_producer_release_vf`、`release_protocol::put_staged_records_striped`、`publish_control_and_release` |
| consumer acquire/validate/reduce | `csrc/backends/ascend/elastic/direct_combine_epilogue_acquire.asc`、`direct_combine_epilogue_validate.asc`、`combine_epilogue_reduce.asc` | `direct_combine_epilogue_acquire_vf`、`direct_combine_epilogue_validate_vf`、`direct_combine_epilogue_reduce_vf` |

## 5. 当前性能口径

### 5.1 AIV block/core 配置口径

benchmark 的 `--num-sms` 是逻辑 AIV block 数，不是“固定启动多少个物理核”。入口在
`tests/ascend/benchmark/bench_ep.py`，运行时由
`tests/ascend/benchmark/runtime.py::run_benchmark` 调用
`get_ascend_aiv_count(local_rank)`；未显式传参时使用设备返回值，当前 NPU8P
典型设备为 28 AICore、56 AIV，因此基线使用 `num_sms=56`。显式值只能不超过设备
AIV 数；它不是写死的 56。

Dispatch/Combine 的不同阶段使用不同 block 数：producer grouping、record、metadata
和 copy 等数据并行阶段通常按 `num_sms` 启动，典型 profile 中为 56；control、prefix、
release、acquire、validate、complete 等协议控制阶段通常是 1 个 block。因而“启动满核”
只适用于数据并行阶段，不能套用到整个操作；单 block 控制阶段也不是空跑，而是协议上
必须由一个执行者维护 generation、队列和跨 rank 状态。若要判断是否有空转，应同时看
profile 的 `block_count`、`work_count` 和 stage span，不能只看 `num_sms`。

CUDA legacy 也不是固定 8 blocks。`csrc/kernels/legacy/intranode.cu` 中 dispatch 和
combine kernel 读取 `gridDim.x` 作为 `num_sms`，并要求为偶数；`num_channels =
num_sms / 2`。host 侧由 `SETUP_LAUNCH_CONFIG(num_sms, ...)` 设置 grid，所以 CUDA 的
block 数由调用方配置，8 只是某些小规模实验可能采用的值，不是实现约束。

固定 workload：

| 项 | 值 |
| --- | --- |
| 平台 | NPU8P-ALT，Ascend950DT，8 rank |
| CANN | 9.3.0 |
| tokens/rank | 8192 |
| hidden | 7168 |
| top-k | 8 |
| experts | 256 |
| data blocks | 56 |
| warmup / samples | 30 / 30 |
| seed | 0 |

9 月 20 日代表结果：

| 操作 | mean | P95 | 逻辑带宽 |
| --- | ---: | ---: | ---: |
| Normal Dispatch | 21.913 ms | 23.080 ms | 355.31 GB/s |
| Expanded Dispatch | 38.593 ms | 39.062 ms | 239.72 GB/s |
| Cached Dispatch | 87.575 ms | 88.924 ms | 88.91 GB/s |
| Normal Combine | 30.327 ms | 30.755 ms | 359.46 GB/s |
| Reduced Combine | 30.065 ms | 31.787 ms | 362.59 GB/s |

9 月 22 日跳过 epilogue no-op 的结果：Normal Dispatch 22.099 ms，P95 24.691 ms；Normal Combine 30.983 ms，P95 32.930 ms。case 通过，八卡 rank 均值 spread 分别为 5.57% 和 0.35%。

这里的逻辑带宽是 benchmark 的 logical bytes 除以最慢 rank 时间，不是物理链路带宽。CANN 9.3.0 固定环境的链路测量结果是：

| 测项 | 结果 |
| --- | ---: |
| 双向 64 MiB P2P 平均 | 52.434 GB/s |
| 32 MiB/peer 八卡 all-to-all | 2594.173 GB/s |
| 独立链路外推 | 2936.281 GB/s |
| 八卡争用系数 | 88.35% |
| representative transport-only | 0.880 ms，2601.679 GB/s |

不能用 8 乘 7 的独立链路外推来设目标，因为典型 router manifest 中每个 token 的平均唯一远端目标 rank 是 4.63，不是 7。按这个路由矩阵修正，无争用参考约 1941.916 GB/s，再乘 88.35% 后约 1715.662 GB/s。这个值是当前固定 workload 的链路级规划参考；transport-only 实测 2601.679 GB/s 更高，是因为非均匀路由允许短传输填充争用空隙。

当前端到端 Dispatch 约 22 ms，而只做 representative payload put/flush 的 transport-only 是 0.88 ms。这个对比不能理解成 HCOMM 只占 4%，因为 transport-only 排除了 grouping、record 打包、scale、控制发布、consumer 等待、validate 和 copy。它说明的是：大块 payload 的链路能力足够，优化优先级应放在生产协议和 consumer 路径。

## 6. rank 差异 50 倍的证据链

先给状态表：

| 问题层 | 状态 |
| --- | --- |
| 固定顺序串行 acquire 造成队头阻塞 | 已确认，direct Dispatch 已改 ready-first |
| acquire 到 validate 的数百万 cycle gap | 已确认是 profiling 边界错误，真实专用 VF gap 只有 5.4k 到 7.5k cycles |
| 通用 no-op stage 调度间隙 73k 到 5.9M cycles | profile 中存在；跳过 no-op 后端到端无收益，不能定为根因 |
| acquire 轮询本身的 destination 相关等待 | 已确认存在，具体根因未定位 |
| producer 侧 control 发布顺序慢 | 已基本排除：每个 producer 7 个远端 destination 的发布调用约 0.08 ms 内完成 |
| host 侧 rank 启动错位被 acquire 放大 | 已确认，当前最优先解释 |

### 6.1 原始现象为什么像 50 倍

固定顺序串行 acquire 中，lane 0 依次等 source 0、1、2 直到 7。假设 source 0 晚到，那么后面 7 个 source 的等待时间都会累计到 source 0 这个观测点上。两轮正向 profile 中，7 个非 0 rank 的最大阻塞 source 都落在 source 0；反向遍历后，最大阻塞 source 立即变成 source 7 和 source 4。

这个对照说明 source 0 不是固定慢的物理 peer，而是遍历顺序造成的队头阻塞。被怀疑的 rank 0 自身 producer record、release payload/control 和 service 周期都与其他 rank 同量级。

### 6.2 ready-first 改造后的结果

direct Dispatch acquire 改为非阻塞 ready-first 后，三轮干净样本的 rank 均值平均：最快 rank 约 19.20 ms，最慢约 20.20 ms，差约 1.0 ms，即 5.1%。operation CV 约 5.3%，P95/mean 约 1.075。

9 月 22 日跳过 no-op 的运行中，Dispatch rank 均值 min/max 为 20.352/21.485 ms，spread 5.57%；Combine 只有 0.35%。因此 50 倍级别的 acquire 观测不再是当前主表现。

### 6.3 profiling 边界错误

早期看到 epilogue_acquire 结束到 epilogue_validate 开始有 2.1M 到 2.4M cycles 差距，一度被当成真实等待。后来给 acquire 和 validate 两个专用 VF 分别加了真实 start/end cycle。真实结果是：

| rank | 最晚 source ready | 真实 VF gap |
| ---: | ---: | ---: |
| 0 | 55,559 | 5,729 |
| 1 | 52,826 | 6,000 |
| 2 | 28,031 | 5,598 |
| 3 | 69,052 | 6,466 |
| 4 | 56,532 | 6,293 |
| 5 | 55,094 | 6,083 |
| 6 | 62,234 | 5,740 |
| 7 | 60,253 | 6,956 |

普通、Expanded、Cached Dispatch 都在 5.4k 到 7.1k cycles。原因很直接：stage profile 里的 kEpilogueAcquire/kEpilogueValidate 是通用 dispatch_kernel 的 no-op stage，真正工作由随后单独提交的专用 VF kernel 完成。把两个 no-op stage 相减，得到的是专用 VF launch 和提交间隔，不是数据面等待。注意，这张表里的“最晚 source ready”来自旧版采样字段，不能作为 acquire 等待上界；后面 6.5 的修正版数据才是有效归因。

### 6.4 no-op 调度假设的 A/B

同一轮 profile 里，通用 no-op stage 的 gap 确实有 rank 差异：

```text
rank 0: 636,671 cycles
rank 1: 349,297
rank 2: 73,438
rank 3: 5,899,795
rank 4: 542,299
rank 5: 690,221
rank 6: 2,847,105
rank 7: 1,373,154
```

所以做了一个 DEEP_EP_ASCEND_SKIP_EPILOGUE_NOOP A/B：非 profile direct path 跳过这两个 no-op kernel，profile 模式保留以便 stage 统计。结果 case 通过，但 Dispatch mean 22.099 ms，P95 24.691 ms，和 21.913 ms 的 9 月 20 日基线同量级；Combine 30.983 ms，也和 30.327 ms 同量级。跳过 no-op 没有带来端到端收益。

当前判断是：no-op 调度间隙是 profile 下真实存在的现象，但它不是端到端剩余 5% rank spread 的充分解释。默认移除这两个 no-op 需要再看 profile 兼容和更多同二进制 ABBA 数据，暂时不应作为修复结论。

### 6.5 还没关掉的问题

修正 per-source 采样后，剩余差异已经不是抽象的 5% spread，而是明确的 destination 相关等待。当前数据如下：

| destination | 明显晚到的 source | 量级 |
| ---: | --- | ---: |
| 3 | 0、1、2、5、6 | 7.2M 到 9.5M cycles |
| 4 | 0、1、2、5、6 | 7.6M 到 9.9M cycles |
| 7 | 0、1、2、5、6 | 6.9M 到 9.3M cycles |
| 5 | 0、1、2 | 1.25M 到 2.02M cycles |
| 6 | 0、1、2 | 1.42M 到 2.19M cycles |
| 0、1、2 | 无数百万级晚到 | 最大约 328k cycles |

后续又补了 producer per-destination publication timestamp 和 host entry/prelaunch/synchronize 绝对时间。两层数据把结论改成了 6.6：release 发布本身约 0.08 ms 内完成，几毫秒 acquire 等待主要由各 rank 进入 Dispatch 的时刻相差 5 ms 以上造成。这个现象出现在 benchmark/profile 编排层，特别是采样前的 barrier 和 host 调度，不是某个固定 rank 的 HCCS 路径慢。

这些诊断都应该用宏控字段，结论记录后再决定是否保留。Combine 是否有独立形态，仍需要单独测，不能直接继承 Dispatch 结论。

### 6.6 host 启动错位的证据

v5/v7 三轮 profile 的形态一致。以 v7 为例：

| rank | Dispatch entry 相对最早值 | acquire 等待 | device envelope |
| ---: | ---: | ---: | ---: |
| 0 | 5.417 ms | 0.051 ms | 2.105 ms |
| 1 | 0.388 ms | 3.790 ms | 4.377 ms |
| 2 | 3.811 ms | 1.103 ms | 2.749 ms |
| 3 | 3.802 ms | 1.140 ms | 2.774 ms |
| 4 | 0.000 ms | 4.117 ms | 4.598 ms |
| 5 | 0.870 ms | 3.487 ms | 4.201 ms |
| 6 | 0.252 ms | 4.012 ms | 4.506 ms |
| 7 | 0.037 ms | 4.170 ms | 4.624 ms |

entry spread 是 5.417 ms，synchronize end spread 只有 0.346 ms。把每个 rank 的 release barrier 完成时间换算到公共 host 时间，再加 first-ready 等待，得到的最晚 ready 公共时间约为 5.165 到 6.498 ms，量级收敛。这个关系在另外两轮中也成立：launch 相对晚的 rank，ready 等待短；launch 相对早的 rank，ready 等待长。

## 7. 当前阻塞和风险

1. HCOMM teardown 偶发 SIGSEGV。根因在 HCOMM 的 rank-info server 线程使用已析构对象的成员函数，修复方式是静态线程入口加异步 server，避免悬垂 this；client 侧 DeInit() 保持同步。自编 HCOMM 已通过反复构造/销毁验证。这个问题与 rank spread 无关。
2. source pipeline 多 chunk 死锁问题仍未完全解决。当前典型 direct case 关闭该 pipeline 宏，不能把典型 case 的通过外推到 pipeline 开启路径。
3. 多 channel 的资源模型和代码路径已经有设计，但 CANN 9.3.0 下的 1/2/4 channel 8 rank 功能和性能 A/B 还没完成。
4. ready-first 目前只覆盖 direct Dispatch。Hybrid 和 Combine 需要单独实验。
5. teardown 后的清理稳定性和重复 generation 复用需要继续作为回归项。

## 8. 建议的讨论问题

1. CANN 9.3 communication-domain 路径是否确认为主路径，特别是 AIV channel 只能随 communicator 销毁这一点。
2. SymWin 在 9.3 当前组合包返回空 window，是否是版本限制还是使用方式变化。
3. 官方 SIMT 数据面是否值得做最小 probe；单 lane channel 约束下，DeepEP producer 应该怎么拆。
4. batch 接口和 AICore service 命令合并能减少多少固定开销，是否有官方推荐的使用方式。
5. no-op stage 是否应该在 profile 模式保留、非 profile 默认跳过。
6. Combine 是否需要做 Dispatch 同样的 ready-first 改造。
7. 多 channel 的收益预期要按 payload 字节自适应，而不是固定 4 channel。
8. rank spread 剩余 5% 的优先级：如果端到端收益有限，应先从整体 critical path 入手，而不是继续单点追 spread。

## 9. Profile 复盘脚本

profile 不再依赖手工查看 JSON。`tools/ascend/analyze_profile.py` 读取
`bench_ep.py` 生成的 benchmark report，输出以下几类数据：

- 每个 rank 的 stage、block_count、stage span 和 work_counts；
- acquire 的 per-source first-ready 矩阵；
- producer 到各 destination 的 publication 时间；
- `dispatch_entry_ns` 的 rank spread；
- 把 acquire wait 加回 host entry skew 后的 normalized ready 时间。

示例：

```bash
python3 tools/ascend/analyze_profile.py /tmp/profile.json \
  --operation dispatch --format markdown
python3 tools/ascend/analyze_profile.py /tmp/profile-v1.json /tmp/profile-v2.json \
  --operation dispatch --format json
```

脚本只使用标准库，不连接 NPU，也不改变 benchmark 运行方式。比较多轮 profile 时，
应优先看 `entry_spread_ms`、`normalized_ready_ms` 和 stage 的 `block_count/work_counts`
是否同时变化；不能只比较某个 rank 的 acquire wait。

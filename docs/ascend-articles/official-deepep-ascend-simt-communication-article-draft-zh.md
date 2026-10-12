# DeepEP-Ascend：SIMT 为什么适合 MoE 通信？

## 一分钟版

MoE 通信的难点不在数据传输，而在确定数据去向。

Dispatch/Combine 由 Router 的 top-k 结果决定：每个 token 去哪个 rank、写到哪个 slot、携带多少 payload，都要在运行时才知道。

DeepEP-Ascend 在 Device 侧将动态路由结果转换为固定格式的通信描述符：SIMT 负责路由统计、去重、slot 分配和 WQE 构造；URMA 按描述符完成远端写入。

SIMT 的价值不是替代 memcpy，而是擅长处理每个 token 都不同的分支、去重、归约和字段并行构造；URMA 则继续承担大块 payload 传输。

## 阅读说明

本文面向做 MoE、分布式通信或 Ascend 后端的工程师，介绍 DeepEP-Ascend 的设备侧通信路径。先明确三个概念：

- Host / Device 执行模型：Host 负责 Python/C++ 控制流、资源初始化和 kernel launch，Device 上的 AICore/AIV 执行算子；
- 专家并行（Expert Parallelism，EP）：expert 分布在不同 rank 上，token 需要跨 rank 流动；
- one-sided write：发送端描述将本地数据写入 peer 某个地址的操作，接收端不需要同时调用匹配的 receive。

## 1. MoE 通信的动态路由问题

MoE 模型将每个 token 交给少量 expert 处理，而不是使用一个稠密 FFN。Router 为每个 token 选择 top-k expert，被选中的 expert 分布在不同 rank 上。

EP 下有两个必要通信操作：

- Dispatch：把 token 发送到 expert 所在 rank；
- Combine：把 expert 输出写回 token 原始 rank，并聚合 top-k 结果。

这两个操作都由 Router 的 top-k 结果决定。通信的难点正在这里：路由在每一步都变化。

假设 256 个 expert 分布在 8 个 rank 上，每个 token 选 6 个 expert。token 0 可能去 rank 1 的 expert 19，也可能同时去 rank 5 的 expert 180；token 1 又是另一组目标。发给谁、每个 rank 收多少、每个 token 写到哪个 slot，都无法在运行前固定。

这类通信有三个特点：

- 动态：路由每一步都变；
- 稀疏：不是每个 peer 都有等量数据；
- 不均衡：热门 expert 所在 rank 可能成为长尾。

若将这段逻辑放回 Host，Device 需先同步路由结果，Host 生成通信计划后再由 Device 执行后处理。路由信息本来就在 Device 上，该往返主要引入同步和 launch 开销。

DeepEP-Ascend 的方案是：SIMT 直接读取路由结果，生成 slot 和 WQE 字段；每个 AIV 拥有自己的 Jetty/SQ，多个 AIV 并行构造通信描述符。

![MoE routing：每个 token 的目标由 top-k 动态决定](../ascend-assets/deepep-ascend-simt-moe-routing.png)

图 1：MoE routing。同一个 token 可能去多个 rank，同一个 rank 上也可能有多个目标 expert；通信目标来自每一步的 top-k 结果，不是静态 broadcast。

## 2. 总体路径：Host 建资源，Device 构造描述符

实现边界清晰。

Host 完成初始化：复用 HCCL communicator，选择 netlayer 1 的 SuperPoD 外部 Clos 通路，发现远端地址，按 AIV 数量创建 Jetty endpoint，并将 channel 表交给 Device。

Device 执行每步路由和提交：SIMT worker 读取 top-k 和 metadata，完成 histogram、去重、prefix 和 slot 分配；将 WQE 写入本 AIV 的 SQ；scalar 更新 head 并 ring doorbell；URMA 根据 SGE 和远端地址写 payload。

![Host 与 Device 的分工：资源准备与每步通信路径](../ascend-assets/deepep-ascend-simt-host-device-architecture.png)

图 2：Host 与 Device 的分工。Host 负责资源初始化；Device 侧将动态路由、通信描述符和硬件传输组织为一条路径。

该图强调职责边界：SIMT 处理动态路由，scalar 发布提交边界，URMA 执行硬件传输。

另一个边界是提交模型：DeepEP-Ascend 并非由每个 lane 单独 ring doorbell。SIMT 负责并行构造 WQE；WQE 写入 SQ 后，由该 AIV 的 scalar 侧更新 head 并发布 doorbell。lane 只参与字段构造，不发布提交边界。

## 3. 关键设计：每个 AIV 一个 Jetty

Jetty 是本地提交 endpoint，对应一个 AIV 的 SQ/CQ 和 doorbell；Channel 是建立在 Jetty 上的 connection context，由 ChannelEntity 描述远端 endpoint、远端内存和本队列的 SQ/CQ context。二者为层级关系：

    Jetty endpoint
      └── Channel
            └── ChannelEntity
                  ├── remoteBuffer
                  ├── sqContext
                  └── cqContext

DeepEP-Ascend 为每个 AIV 创建一个 Jetty，而不是为每个 peer 创建一个 Jetty。AIV0 使用 Jetty0/SQ0，AIV1 使用 Jetty1/SQ1，直到 AIV63 使用 Jetty63/SQ63。Jetty 在 AIV 维度私有，因此各 AIV 不会写同一个 SQ；同一 Jetty 可以服务多个 peer。

Host 侧的 acquire_jetties 创建：

1. num_jetties 个 Jetty endpoint，数量来自 runtime 的设备 AIV 数；
2. max(num_jetties, num_peers) 个 Channel，形成 Device 侧的 channel 表。

在 8 rank / 64 AIV 下，Host 创建 64 个 Channel。Device 侧的 jetty_ptrs 表保存 ChannelHandle，用法分两类：

1. AIV i 用 jetty_ptrs[i] 找到自己的 ChannelEntity，再解析出本 AIV 的 SQ/CQ；
2. AIV 用前 num_peers 个 Channel 解析远端 peer 的 Jetty metadata、远端内存地址和 token。

AIV 的提交队列来自自己的 Jetty，远端地址信息来自 peer Channel。per-AIV Jetty 因此消除了多 AIV 写同一个 SQ 时的 slot 预留、head 推进、owner bit 和 doorbell 顺序问题。

Host 还要求所有 peer 使用同一个 local UBC_CTP endpoint。Device 因此可以按 AIV index 选 Jetty，按 peer index 选远端 metadata。

![per-AIV Jetty 的队列所有权：AIV 私有，peer 共享](../ascend-assets/deepep-ascend-simt-per-aiv-jetty.png)

图 3：per-AIV Jetty。Jetty 在 AIV 维度私有，在 peer 维度共享；每个 AIV 推进自己的 SQ，避免多个 producer 争用同一个队列。

## 4. Dispatch：把 token 发给 expert 所在 rank

Dispatch 把 Router 选出的 top-k token 发送到对应 expert 所在 rank，并整理成 expert 可消费的布局。实现上，它将动态 top-k 结果转换为四类固定格式信息：目标 rank、接收 slot、hidden/metadata payload，以及描述这些 payload 的 WQE。

### 4.1 工作划分：AIV 分片与 512 SIMT thread

DeepEP-Ascend Dispatch 将连续 token range 分配给 AIV，每个 AIV 启动 512 个 SIMT thread。该划分有利于本地 copy 和 metadata prefetch，也将通信分散到多个 Jetty/SQ。

### 4.2 路由转换：top-k 到 rank、slot 与 SGE

SIMT worker 读取 top-k 后，先做三件事：

1. 统计 expert histogram，得到每个 expert 将接收的 token 数；
2. 统计 rank histogram，得到每个 peer 将接收的 token 数；
3. 对同一个 token 的重复目标 rank 去重。如果一个 token 命中同一 rank 上的多个 expert，只向该 rank 发送一份 payload，接收端再按 expert 展开。

top-k 选择结果可映射到 lane；ballot、popc、shuffle 和 warp reduction 分别完成重复判断、位置选择和前缀计算。这类分支、比较和归约符合 SIMT 的执行特点。

去重后，worker 为远端 entry 分配 SGE index，建立 sgep_to_entry 映射，再计算 SQE 区间和接收 slot。到这里，token 路由已经变成硬件布局。

![Dispatch 数据流：从动态路由到固定格式 WQE 描述符](../ascend-assets/deepep-ascend-simt-dispatch-dataflow.png)

图 4：Dispatch 数据流。左半段是动态路由控制流，右半段是硬件按固定格式消费的 WQE 描述符。

### 4.3 通信描述：WQE 与 WQEBB 布局

WQE 是逻辑提交描述符；WQEBB 是 SQ 中的 64B 物理块，一个 WQE 可跨多个 WQEBB。Dispatch 的一个 SQE slot 固定占 4 个 WQEBB，即 256B。有效内容最多为 48B SQE header 加 12 个 16B SGE，共 240B，剩余空间用 padding/NOP 补齐。

WQE 按字段映射到 lane，而非由单线程串行填写。lane 是 warp 内的线程编号。512 个 SIMT thread 分为 16 个 warp：路由扫描按 top-k 分组处理 token，WQE 构造阶段将 SQE 分片分配给这些 warp。

字段布局按 u64 word 组织：SQE header 使用 6 个 word；每个 SGE pair 使用 4 个 word，即两个 16B SGE。lane 并行计算这些 word：

| lane | 字段 |
| --- | --- |
| 0 | SQE header / opcode / owner / index |
| 1-5 | peer metadata / token id / remote address |
| 6-7 | SGE 0：hidden length / source address |
| 8-9 | SGE 1：metadata length / source address |
| 10-13 | 后续 SGE pair，每个 pair 4 个 lane |
| 14 以后 | 未参与本 SQE 有效字段，物理块内补 padding/NOP |

SIMT worker 计算 sqe_word 后，用 asc_stcg 写入 SQ。写入地址由 SQE slot 的起始 WQEBB 和 lane 所属 word 决定；lane14 之后不填充有效字段，slot 仍按 4 个 WQEBB 推进。图 5 展示字段来源，而非完整 SQ 内存布局。

![lane 到 WQE 字段的映射：一个 warp 并行构造一条 WQE](../ascend-assets/deepep-ascend-simt-lane-to-wqe-mapping.png)

图 5：lane 到 WQE 字段的映射。图中重点是 lane0 到 lane13 分别提供 header、peer metadata 和 SGE 字段；完整 Dispatch SQE slot 会扩展到 4 个 WQEBB，并用 padding/NOP 补齐。

### 4.4 Payload 与完成语义

WQE 中的 SGE 直接指向本地 payload：

- hidden payload：指向输入 x；
- metadata payload：指向 metadata_send_buffer；
- FP8 场景还会携带 scale factors。

URMA 根据远端地址和 SGE 中的本地地址，直接写入远端接收窗口。跨 rank token 不经过 UB 或 Host。

本地 rank 例外：kernel 使用 MTE 将本地 token 从 GM 读入 UB，再写回本地接收窗口。因此“不先进 UB”仅指远端 URMA payload。

最后一个 SQE 设置 strong order、fence 和 CQE，覆盖前面的写入。SIMT 完成字段构造后，scalar 更新 SQ head 并 ring doorbell。benchmark 再用 barrier 等待 URMA traffic，得到包含 issue 和 drain 的时间。

## 5. Combine：将 expert 输出写回原 rank 并聚合

Combine 与 Dispatch 相反：它将 expert 输出写回 token 原始 rank，并将同一 token 的 top-k 结果加权聚合。

### 5.1 聚合输入：source rank、token index 与 slot 列表

每个 token 需要三类信息：source rank 决定目标 rank，原始 token index 决定目标位置，slot 列表决定参与 reduce 的 expert 输出。

### 5.2 工作划分：source rank task

Combine 按 task 组织工作。一个 task 覆盖同一 source rank 上最多 8 个 token，包含 source rank、token 数和起始接收 slot。接收 buffer 按 source rank 分区，task 通过起始 slot 定位连续 token；8 的上限由 UB metadata buffer 和 VF 形态决定：一个 VF 使用 8 个 warp，每个 warp 处理一个 token。

### 5.3 通信描述：WQE 与两个 SGE

对 remote task，load_meta_issue_wqe 读取 token metadata、写入 top-k weights，并为每个 token 构造一条远端写 WQE。本地 task 不构造 WQE，仅执行本地聚合或写入目标位置。

每个 token 的 WQE 固定带两个 SGE，因为一个 token 的远端写内容有两段：

| SGE | 内容 | 来源 |
| --- | --- | --- |
| SGE 0 | expert 输出的 hidden payload | 输入 x 中该 token 的 slot |
| SGE 1 | top-k weights 区域 | send buffer 中为该 token 预留的 weights 区域 |

hidden 是主要 payload，weights 是 Combine 必需的附加数据。两个 SGE 使一个 WQE 可描述两个数据段：hidden 来自 expert 输出，weights 来自 send buffer；URMA 按 SGE 分别取地址和长度，避免重新打包 hidden。

WQE 字段仍按 lane 展开：lane0 填 header，lane1-5 填 peer metadata 和远端地址，lane6-9 填两个 SGE。lane 分工解决字段并行构造，两个 SGE 定义一个 token 的传输内容。

### 5.4 提交与流水：批量 doorbell 与两级 stage

doorbell 按批提交：每 8 个 remote task 更新一次 SQ head 并 ring doorbell，以摊薄每个 token 或 task 的发布开销。结尾追加 strong-order NOP 并请求 CQE，用最后一个完成事件覆盖前面的远端写。

Stage 按 task 的软件流水阶段划分，而非按 token 或 expert 划分。当前实现固定 kNumStages = 2：stage 0 和 stage 1 各保存一组 task metadata，两个 stage 交替复用。执行时的重叠为：

    stage A: load metadata -> 写 weights -> 构造 WQE
                    |
                    v
    stage B:                        local reduce / 写入目标位置

当前 stage 执行 reduce 或写入目标位置时，下一个 task 的 metadata 已进入另一个 stage buffer，其 WQE 也已构造完成。通信准备与本地聚合因此重叠；当前 stage 完成后，可立即处理下一 task。图 6 的两个重叠窗口对应相邻 task stage 的交叠时间。

![Combine 任务流水：通信构造与本地 reduce 重叠执行](../ascend-assets/deepep-ascend-simt-combine-pipeline.png)

图 6：Combine 的两级 task 流水。Stage 表示相邻 task 的软件流水阶段，而非模型层：当前 task 执行 reduce 时，下一 task 已在加载 metadata 并构造 WQE；批量 doorbell 每 8 个 remote task 发布一次。

## 6. SIMT 的优势是什么

分工可以整理成下表：

| 工作类型 | 执行路径 | 原因 |
| --- | --- | --- |
| top-k 扫描 | SIMT | 每个 token 目标不同 |
| rank 去重 | SIMT warp primitive | ballot/popc/shuffle 对应这类比较和归约 |
| expert/rank histogram | SIMT | 分支多，需要归约 |
| slot/SGE 映射 | SIMT | 每个 token 的映射不同 |
| WQE 字段构造 | SIMT | 字段可按 lane 展开 |
| SQ head / doorbell | scalar | 需要明确的发布边界 |
| bulk payload 传输 | URMA / MTE | 硬件路径更高效 |

SIMT 的价值在于将动态路由转换为硬件可消费的描述符；URMA 的价值在于按描述符执行远端写入。

MoE 通信之所以适合 SIMT，不是因为 SIMT 是更快的 memcpy，而是因为它擅长分支、去重、归约和字段展开。

## 7. DeepEP-Ascend 性能数据

DeepEP-Ascend 官方性能报告给出以下带宽数据：

| EP size | Dispatch GB/s | Combine GB/s |
| ---: | ---: | ---: |
| EP8 | 373-375 | 345-347 |
| EP16 | 348-352 | 338-341 |
| EP32 | 335-340 | 320-324 |
| EP64 | 323-327 | 294-298 |
| EP128 | 313-320 | 272-278 |

统计口径为包含 issue 和 drain、排除最终 epilogue；采样 dispatch_impl / combine_impl 的 dur_ns，50 次取平均，字节按 per-rank 逻辑 URMA 字节计算。

Dispatch 在 EP32 以内达到物理 payload 带宽的约 90-95%。

## 8. 延伸阅读：更高层 API 与 transport service

第一条路线是提供更高层的通信 API。

本文解释了 SQE/WQE、SGE 和 doorbell 等底层机制。GitCode 的 ASC-COMM 报告展示了另一个层次：Hcomm 提供 WriteNbi、MakeBatchHandle、BatchCommit 和 Drain，应用只需描述写入内容、目标地址以及提交和等待时机。

这与 DeepEP-Ascend 源码不在同一抽象层：仓库直接操作 HcommUrmaSqeCtx / HcommUrmaSgeCtx；报告描述应用在更高层 API 下如何组织 Dispatch/Combine。报告中的 API 名不能直接对应到仓库源码。两个视角互补：前者解释 WQE 构造，后者描述这些细节的封装方式。

下面这组数据来自 ASC-COMM 技术报告，与上一节数据不是同一实现和统计口径。报告中的案例环境为：Ascend 950DT、EP16、4096 tokens/rank、hidden 7168、top-k 6、384 experts。

| 操作 | 前处理 | URMA | 后处理 | 带宽 |
| --- | ---: | ---: | ---: | ---: |
| Dispatch BF16 | 58.021 μs | 649.602 μs | 298.877 μs | 508.467 GB/s |
| Dispatch FP8 | 71.031 μs | 318.665 μs | 239.513 μs | 518.258 GB/s |
| Combine BF16 | 37.861 μs | 769.748 μs | 151.046 μs | 429.103 GB/s |

这组数据把通信拆成前处理、URMA 和后处理。它说明 MoE 通信优化不只压缩 URMA 时间，也要控制路由、metadata、展开和聚合。

第二条路线是在设备侧保留一个专用 transport service kernel。

DeepEP-Ascend 将 WQE 构造放回 producer：路由结果生成后，通信描述符即可直接构造。该路径依赖资源模型配合，例如每个 AIV 拥有自己的 Jetty/SQ，避免多个 producer 争用同一个队列。

另一种工程路线是让 producer 和队列所有权解耦：SIMT producer 只生成语义化 command，例如 put、signal、flush；专用 transport service kernel 独占 channel/Jetty，将 command 解析为 WQE/SQE，并统一维护 head/tail、completion 和错误诊断状态。

service 分层与 direct WQE 是两种工程取舍。它更容易封装资源所有权、completion 协议和诊断信息，也让不同 producer 复用同一套通信语义；代价是 command 编码、全局队列传递和额外调度开销。若沿这条路继续发展，需要回答 command 粒度、channel 独占方式、跨 generation 的 completion 语义，以及 producer 与 service 能否形成足够深的流水线。

## 9. 小结

DeepEP-Ascend 提供了一个结构清晰的 SIMT 通信实现样例：

1. 动态路由适合 SIMT：top-k、去重、histogram、slot 分配都是动态分支和归约；
2. WQE 字段适合 lane 映射：header、peer metadata、SGE 地址和长度都可以并行构造；
3. per-AIV Jetty 支持多 producer 并发：每个 AIV 推进自己的 SQ，避免共享队列所有权问题；
4. URMA 继续负责 payload：SIMT 不替代硬件传输，而是生成硬件能直接消费的描述符。

通信不仅包含 payload 传输，也包含将动态通信意图转换为硬件描述符。DeepEP-Ascend 说明 SIMT 适合承担这项工作。

## 参考入口

- [DeepEP-Ascend 仓库](https://github.com/deepseek-ai/DeepEP-Ascend)
- [本文对应源码快照](https://github.com/deepseek-ai/DeepEP-Ascend/commit/3b25377d04b24fc6154698ded78a2bcb2c59afff)
- [ASC-COMM 技术报告](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek_v4_1/deepseek_v4.1_asc_comm_tech_report.md)
- [通信对象与 WQE/SQE/CQE 布局](../ascend-reference/deep-ep-ascend-communication-layouts-zh.md)

## 附录：术语表

| 术语 | 含义 | 本文中的作用 |
| --- | --- | --- |
| Host | CPU 侧控制程序 | 创建通信资源 |
| Device | NPU 侧执行环境 | 执行 Dispatch / Combine |
| AICore / AIV | Ascend AI Core 与 Vector 执行路径 | 每个 AIV 是一个通信 producer 分片 |
| SIMT | 单指令多线程执行模式 | 并行处理 top-k、去重、slot 和 WQE 字段 |
| SIMT VF | 以 SIMT 方式执行的 vector function | 承载 Combine 中的 metadata 读取和 WQE 构造 |
| lane | warp 内线程编号 | 描述 WQE 字段在 warp 内的并行分工 |
| top-k | Router 为 token 选择的 expert 集合 | 决定目标 rank 和 slot |
| slot | 接收 buffer 中的 token 记录索引 | 定位远端写入位置 |
| payload | 被传输的 token 数据 | 包括 hidden、metadata 和 scale factors |
| Jetty | HCOMM/URMA endpoint 与队列资源 | 每个 AIV 一个，服务多个 peer |
| Channel | 建立在 Jetty 上的 connection context | 提供远端 endpoint、远端内存和队列 context |
| SQ / CQ | 提交队列 / 完成队列 | 承载 WQE 提交与 CQE 完成 |
| WQE / SQE | Work Queue Entry / Submission Queue Entry | WQE 表示一次提交；SQE 指 WQE 的 header/context 部分 |
| SGE | Scatter-Gather Element | 描述本地 payload 的地址和长度 |
| doorbell | 通知硬件有新 WQE 的机制 | 由 AIV 的 scalar 侧发布 |
| URMA | Unified Remote Memory Access | 执行远端写入 |
| MTE | Memory Transfer Engine | 执行 bulk 数据传输 |
| CQE | Completion Queue Entry | 记录 WQE 完成 |

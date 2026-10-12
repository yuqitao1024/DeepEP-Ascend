# DeepEP-Ascend：SIMT为什么适合MoE通信？

## 一分钟版

MoE通信的难点不在数据传输，而在确定数据去向。

Dispatch/Combine由Router的top-k结果决定：每个token去哪个rank、写到哪个slot、携带多少payload，都要在运行时才知道。

[DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)在`Device`侧将动态路由结果转换为固定格式的通信描述符：`SIMT`负责路由统计、去重、slot分配和`WQE`构造；`URMA`按描述符完成远端写入。

`SIMT`的价值不是替代`memcpy`，而是擅长处理每个token都不同的分支、去重、归约和字段并行构造；`URMA`则继续承担大块payload传输。

## 阅读说明

本文面向做MoE、分布式通信或Ascend后端的工程师，介绍DeepEP-Ascend的设备侧通信路径。先明确三个概念：

- `Host` / `Device`执行模型：`Host`负责Python/C++控制流、资源初始化和`kernel launch`，`Device`上的`AICore`/`AIV`执行算子；
- 专家并行（Expert Parallelism，`EP`）：expert分布在不同rank上，token需要跨rank流动；
- `one-sided write`：发送端描述将本地数据写入peer某个地址的操作，接收端不需要同时调用匹配的receive。

## 1. MoE通信的动态路由问题

MoE模型将每个token交给少量expert处理，而不是使用一个稠密FFN。Router为每个token选择top-k expert，被选中的expert分布在不同rank上。

`EP`下有两个必要通信操作：

- Dispatch：把token发送到expert所在rank；
- Combine：把expert输出写回token原始rank，并聚合top-k结果。

这两个操作都由Router的top-k结果决定。通信的难点正在这里：路由在每一步都变化。

假设256个expert分布在8个rank上，每个token选6个expert。token 0可能去rank 1的expert 19，也可能同时去rank 5的expert 180；token 1又是另一组目标。发给谁、每个rank收多少、每个token写到哪个slot，都无法在运行前固定。

这类通信有三个特点：

- 动态：路由每一步都变；
- 稀疏：不是每个peer都有等量数据；
- 不均衡：热门expert所在rank可能成为长尾。

若将这段逻辑放回`Host`，`Device`需先同步路由结果，`Host`生成通信计划后再由`Device`执行后处理。路由信息本来就在`Device`上，该往返主要引入同步和launch开销。

DeepEP-Ascend的方案是：`SIMT`直接读取路由结果，生成slot和`WQE`字段；每个`AIV`拥有自己的`Jetty`/`SQ`，多个`AIV`并行构造通信描述符。

![MoE routing：每个token的目标由top-k动态决定](../ascend-assets/deepep-ascend-simt-moe-routing.png)

图1：MoE routing。同一个token可能去多个rank，同一个rank上也可能有多个目标expert；通信目标来自每一步的top-k结果，不是静态broadcast。

## 2. 总体路径：Host建资源，Device构造描述符

实现边界清晰。

`Host`完成初始化：复用`HCCL` communicator，选择netlayer 1的SuperPoD外部Clos通路，发现远端地址，按`AIV`数量创建`Jetty` endpoint，并将channel表交给`Device`。

`Device`执行每步路由和提交：`SIMT` worker读取top-k和metadata，完成histogram、去重、prefix和slot分配；将`WQE`写入本`AIV`的`SQ`；scalar更新head并ring `doorbell`；`URMA`根据`SGE`和远端地址写payload。

![Host与Device的分工：资源准备与每步通信路径](../ascend-assets/deepep-ascend-simt-host-device-architecture.png)

图2：Host与Device的分工。Host负责资源初始化；Device侧将动态路由、通信描述符和硬件传输组织为一条路径。

该图强调职责边界：`SIMT`处理动态路由，scalar发布提交边界，`URMA`执行硬件传输。

另一个边界是提交模型：DeepEP-Ascend并非由每个lane单独ring `doorbell`。`SIMT`负责并行构造`WQE`；`WQE`写入`SQ`后，由该`AIV`的scalar侧更新head并发布`doorbell`。lane只参与字段构造，不发布提交边界。

## 3. 关键设计：每个AIV一个Jetty

`Jetty`是本地提交endpoint，对应一个`AIV`的`SQ`/`CQ`和`doorbell`；`Channel`是建立在`Jetty`上的connection context，由`ChannelEntity`描述远端endpoint、远端内存和本队列的`SQ`/`CQ` context。二者为层级关系：

    Jetty endpoint
      └── Channel
            └── ChannelEntity
                  ├── remoteBuffer
                  ├── sqContext
                  └── cqContext

DeepEP-Ascend为每个`AIV`创建一个`Jetty`，而不是为每个peer创建一个`Jetty`。`AIV0`使用`Jetty0`/`SQ0`，`AIV1`使用`Jetty1`/`SQ1`，直到`AIV63`使用`Jetty63`/`SQ63`。`Jetty`在`AIV`维度私有，因此各`AIV`不会写同一个`SQ`；同一`Jetty`可以服务多个peer。

`Host`侧的`acquire_jetties`创建：

1. num_jetties个`Jetty` endpoint，数量来自runtime的设备`AIV`数；
2. max(num_jetties, num_peers) 个`Channel`，形成`Device`侧的channel表。

在8 rank / 64 `AIV`下，`Host`创建64个`Channel`。`Device`侧的`jetty_ptrs`表保存`ChannelHandle`，用法分两类：

1. `AIV` i用`jetty_ptrs`[i] 找到自己的`ChannelEntity`，再解析出本`AIV`的`SQ`/`CQ`；
2. `AIV`用前num_peers个`Channel`解析远端peer的`Jetty` metadata、远端内存地址和token。

`AIV`的提交队列来自自己的`Jetty`，远端地址信息来自peer `Channel`。per-AIV `Jetty`因此消除了多`AIV`写同一个`SQ`时的slot预留、head推进、owner bit和`doorbell`顺序问题。

`Host`还要求所有peer使用同一个local `UBC_CTP` endpoint。`Device`因此可以按`AIV` index选`Jetty`，按peer index选远端metadata。

![per-AIV Jetty的队列所有权：AIV私有，peer共享](../ascend-assets/deepep-ascend-simt-per-aiv-jetty.png)

图3：per-AIV Jetty。Jetty在AIV维度私有，在peer维度共享；每个AIV推进自己的SQ，避免多个producer争用同一个队列。

## 4. Dispatch：把token发给expert所在rank

Dispatch把Router选出的top-k token发送到对应expert所在rank，并整理成expert可消费的布局。实现上，它将动态top-k结果转换为四类固定格式信息：目标rank、接收slot、hidden/metadata payload，以及描述这些payload的`WQE`。

### 4.1 工作划分：AIV分片与512 SIMT thread

DeepEP-Ascend Dispatch将连续token range分配给`AIV`，每个`AIV`启动512个`SIMT` thread。该划分有利于本地copy和metadata prefetch，也将通信分散到多个`Jetty`/`SQ`。

### 4.2 路由转换：top-k到rank、slot与SGE

`SIMT` worker读取top-k后，先做三件事：

1. 统计expert histogram，得到每个expert将接收的token数；
2. 统计rank histogram，得到每个peer将接收的token数；
3. 对同一个token的重复目标rank去重。如果一个token命中同一rank上的多个expert，只向该rank发送一份payload，接收端再按expert展开。

top-k选择结果可映射到lane；ballot、popc、shuffle和warp reduction分别完成重复判断、位置选择和前缀计算。这类分支、比较和归约符合`SIMT`的执行特点。

去重后，worker为远端entry分配`SGE` index，建立`sgep_to_entry`映射，再计算`SQE`区间和接收slot。到这里，token路由已经变成硬件布局。

![Dispatch数据流：从动态路由到固定格式WQE描述符](../ascend-assets/deepep-ascend-simt-dispatch-dataflow.png)

图4：Dispatch数据流。左半段是动态路由控制流，右半段是硬件按固定格式消费的WQE描述符。

### 4.3 通信描述：WQE与WQEBB布局

`WQE`是逻辑提交描述符；`WQEBB`是`SQ`中的64B物理块，一个`WQE`可跨多个`WQEBB`。Dispatch的一个`SQE` slot固定占4个`WQEBB`，即256B。有效内容最多为48B `SQE` header加12个16B `SGE`，共240B，剩余空间用padding/NOP补齐。

`WQE`按字段映射到lane，而非由单线程串行填写。lane是warp内的线程编号。512个`SIMT` thread分为16个warp：路由扫描按top-k分组处理token，`WQE`构造阶段将`SQE`分片分配给这些warp。

字段布局按u64 word组织：`SQE` header使用6个word；每个`SGE` pair使用4个word，即两个16B `SGE`。lane并行计算这些word：

| lane | 字段|
| --- | --- |
| 0 | SQE header / opcode / owner / index |
| 1-5 | peer metadata / token id / remote address |
| 6-7 | SGE 0：hidden length / source address |
| 8-9 | SGE 1：metadata length / source address |
| 10-13 | 后续SGE pair，每个pair 4个lane |
| 14以后| 未参与本SQE有效字段，物理块内补padding/NOP |

`SIMT` worker计算`sqe_word`后，用`asc_stcg`写入`SQ`。写入地址由`SQE` slot的起始`WQEBB`和lane所属word决定；lane14之后不填充有效字段，slot仍按4个`WQEBB`推进。图5展示字段来源，而非完整`SQ`内存布局。

![lane到WQE字段的映射：一个warp并行构造一条WQE](../ascend-assets/deepep-ascend-simt-lane-to-wqe-mapping.png)

图5：lane到WQE字段的映射。图中重点是lane0到lane13分别提供header、peer metadata和SGE字段；完整Dispatch SQE slot会扩展到4个WQEBB，并用padding/NOP补齐。

### 4.4 Payload与完成语义

`WQE`中的`SGE`直接指向本地payload：

- hidden payload：指向输入x；
- metadata payload：指向metadata_send_buffer；
- FP8场景还会携带scale factors。

`URMA`根据远端地址和`SGE`中的本地地址，直接写入远端接收窗口。跨rank token不经过`UB`或`Host`。

本地rank例外：kernel使用`MTE`将本地token从`GM`读入`UB`，再写回本地接收窗口。因此“不先进`UB`”仅指远端`URMA` payload。

最后一个`SQE`设置strong order、fence和`CQE`，覆盖前面的写入。`SIMT`完成字段构造后，scalar更新`SQ` head并ring `doorbell`。benchmark再用barrier等待`URMA` traffic，得到包含issue和drain的时间。

## 5. Combine：将expert输出写回原rank并聚合

Combine与Dispatch相反：它将expert输出写回token原始rank，并将同一token的top-k结果加权聚合。

### 5.1 聚合输入：source rank、token index与slot列表

每个token需要三类信息：source rank决定目标rank，原始token index决定目标位置，slot列表决定参与reduce的expert输出。

### 5.2 工作划分：source rank task

Combine按task组织工作。一个task覆盖同一source rank上最多8个token，包含source rank、token数和起始接收slot。接收buffer按source rank分区，task通过起始slot定位连续token；8的上限由`UB` metadata buffer和VF形态决定：一个VF使用8个warp，每个warp处理一个token。

### 5.3 通信描述：WQE与两个SGE

对remote task，`load_meta_issue_wqe`读取token metadata、写入top-k weights，并为每个token构造一条远端写`WQE`。本地task不构造`WQE`，仅执行本地聚合或写入目标位置。

每个token的`WQE`固定带两个`SGE`，因为一个token的远端写内容有两段：

| SGE | 内容| 来源|
| --- | --- | --- |
| SGE 0 | expert输出的hidden payload | 输入x中该token的slot |
| SGE 1 | top-k weights区域| send buffer中为该token预留的weights区域|

hidden是主要payload，weights是Combine必需的附加数据。两个`SGE`使一个`WQE`可描述两个数据段：hidden来自expert输出，weights来自send buffer；`URMA`按`SGE`分别取地址和长度，避免重新打包hidden。

`WQE`字段仍按lane展开：lane0填header，lane1-5填peer metadata和远端地址，lane6-9填两个`SGE`。lane分工解决字段并行构造，两个`SGE`定义一个token的传输内容。

### 5.4 提交与流水：批量doorbell与两级stage

`doorbell`按批提交：每8个remote task更新一次`SQ` head并ring `doorbell`，以摊薄每个token或task的发布开销。结尾追加strong-order NOP并请求`CQE`，用最后一个完成事件覆盖前面的远端写。

Stage按task的软件流水阶段划分，而非按token或expert划分。当前实现固定`kNumStages` = 2：stage 0和stage 1各保存一组task metadata，两个stage交替复用。执行时的重叠为：

    stage A: load metadata -> 写 weights -> 构造 WQE
                    |
                    v
    stage B:                        local reduce / 写入目标位置

当前stage执行reduce或写入目标位置时，下一个task的metadata已进入另一个stage buffer，其`WQE`也已构造完成。通信准备与本地聚合因此重叠；当前stage完成后，可立即处理下一task。图6的两个重叠窗口对应相邻task stage的交叠时间。

![Combine任务流水：通信构造与本地reduce重叠执行](../ascend-assets/deepep-ascend-simt-combine-pipeline.png)

图6：Combine的两级task流水。Stage表示相邻task的软件流水阶段，而非模型层：当前task执行reduce时，下一task已在加载metadata并构造WQE；批量doorbell每8个remote task发布一次。

## 6. SIMT的优势是什么

分工可以整理成下表：

| 工作类型| 执行路径| 原因|
| --- | --- | --- |
| top-k扫描| SIMT | 每个token目标不同|
| rank去重| SIMT warp primitive | ballot/popc/shuffle对应这类比较和归约|
| expert/rank histogram | SIMT | 分支多，需要归约|
| slot/SGE映射| SIMT | 每个token的映射不同|
| WQE字段构造| SIMT | 字段可按lane展开|
| SQ head / doorbell | scalar | 需要明确的发布边界|
| bulk payload传输| URMA / MTE | 硬件路径更高效|

`SIMT`的价值在于将动态路由转换为硬件可消费的描述符；`URMA`的价值在于按描述符执行远端写入。

MoE通信之所以适合`SIMT`，不是因为`SIMT`是更快的`memcpy`，而是因为它擅长分支、去重、归约和字段展开。

## 7. DeepEP-Ascend性能数据

DeepEP-Ascend官方性能报告给出以下带宽数据：

| EP size | Dispatch GB/s | Combine GB/s |
| ---: | ---: | ---: |
| EP8 | 373-375 | 345-347 |
| EP16 | 348-352 | 338-341 |
| EP32 | 335-340 | 320-324 |
| EP64 | 323-327 | 294-298 |
| EP128 | 313-320 | 272-278 |

统计口径为包含issue和drain、排除最终epilogue；采样dispatch_impl / combine_impl的dur_ns，50次取平均，字节按per-rank逻辑`URMA`字节计算。

Dispatch在EP32以内达到物理payload带宽的约90-95%。

## 8. 延伸阅读：更高层API与transport service

第一条路线是提供更高层的通信API。

本文解释了`SQE`/`WQE`、`SGE`和`doorbell`等底层机制。GitCode的`ASC-COMM`报告展示了另一个层次：`Hcomm`提供`WriteNbi`、`MakeBatchHandle`、`BatchCommit`和`Drain`，应用只需描述写入内容、目标地址以及提交和等待时机。

这与DeepEP-Ascend源码不在同一抽象层：仓库直接操作`HcommUrmaSqeCtx` / `HcommUrmaSgeCtx`；报告描述应用在更高层API下如何组织Dispatch/Combine。报告中的API名不能直接对应到仓库源码。两个视角互补：前者解释`WQE`构造，后者描述这些细节的封装方式。

下面这组数据来自`ASC-COMM`技术报告，与上一节数据不是同一实现和统计口径。报告中的案例环境为：Ascend 950DT、EP16、4096 tokens/rank、hidden 7168、top-k 6、384 experts。

| 操作| 前处理| URMA | 后处理| 带宽|
| --- | ---: | ---: | ---: | ---: |
| Dispatch BF16 | 58.021 μs | 649.602 μs | 298.877 μs | 508.467 GB/s |
| Dispatch FP8 | 71.031 μs | 318.665 μs | 239.513 μs | 518.258 GB/s |
| Combine BF16 | 37.861 μs | 769.748 μs | 151.046 μs | 429.103 GB/s |

这组数据把通信拆成前处理、`URMA`和后处理。它说明MoE通信优化不只压缩`URMA`时间，也要控制路由、metadata、展开和聚合。

第二条路线是在设备侧保留一个专用transport service kernel。

DeepEP-Ascend将`WQE`构造放回producer：路由结果生成后，通信描述符即可直接构造。该路径依赖资源模型配合，例如每个`AIV`拥有自己的`Jetty`/`SQ`，避免多个producer争用同一个队列。

另一种工程路线是让producer和队列所有权解耦：`SIMT` producer只生成语义化command，例如put、signal、flush；专用transport service kernel独占channel/`Jetty`，将command解析为`WQE`/`SQE`，并统一维护head/tail、completion和错误诊断状态。

service分层与direct `WQE`是两种工程取舍。它更容易封装资源所有权、completion协议和诊断信息，也让不同producer复用同一套通信语义；代价是command编码、全局队列传递和额外调度开销。若沿这条路继续发展，需要回答command粒度、channel独占方式、跨generation的completion语义，以及producer与service能否形成足够深的流水线。

## 9. 小结

DeepEP-Ascend提供了一个结构清晰的`SIMT`通信实现样例：

1. 动态路由适合`SIMT`：top-k、去重、histogram、slot分配都是动态分支和归约；
2. `WQE`字段适合lane映射：header、peer metadata、`SGE`地址和长度都可以并行构造；
3. per-AIV `Jetty`支持多producer并发：每个`AIV`推进自己的`SQ`，避免共享队列所有权问题；
4. `URMA`继续负责payload：`SIMT`不替代硬件传输，而是生成硬件能直接消费的描述符。

通信不仅包含payload传输，也包含将动态通信意图转换为硬件描述符。DeepEP-Ascend说明`SIMT`适合承担这项工作。

## 参考入口

- [DeepEP-Ascend仓库](https://github.com/deepseek-ai/DeepEP-Ascend)
- [本文对应源码快照](https://github.com/deepseek-ai/DeepEP-Ascend/commit/3b25377d04b24fc6154698ded78a2bcb2c59afff)
- [`ASC-COMM`技术报告](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek_v4_1/deepseek_v4.1_asc_comm_tech_report.md)
- [通信对象与`WQE`/`SQE`/`CQE`布局](../ascend-reference/deep-ep-ascend-communication-layouts-zh.md)

## 附录：术语表

| 术语 | 含义 | 本文中的作用 |
| --- | --- | --- |
| Host | CPU侧控制程序 | 创建通信资源 |
| Device | NPU侧执行环境 | 执行Dispatch / Combine |
| AICore / AIV | Ascend AI Core与Vector执行路径 | 每个AIV是一个通信producer分片 |
| SIMT | 单指令多线程执行模式 | 并行处理top-k、去重、slot和WQE字段 |
| SIMT VF | 以SIMT方式执行的vector function | 承载Combine中的metadata读取和WQE构造 |
| lane | warp内线程编号 | 描述WQE字段在warp内的并行分工 |
| top-k | Router为token选择的expert集合 | 决定目标rank和slot |
| slot | 接收buffer中的token记录索引 | 定位远端写入位置 |
| payload | 被传输的token数据 | 包括hidden、metadata和scale factors |
| Jetty | HCOMM/URMA endpoint与队列资源 | 每个AIV一个，服务多个peer |
| Channel | 建立在Jetty上的connection context | 提供远端endpoint、远端内存和队列context |
| SQ / CQ | 提交队列/完成队列 | 承载WQE提交与CQE完成 |
| WQE / SQE | Work Queue Entry / Submission Queue Entry | WQE表示一次提交；SQE指WQE的header/context部分 |
| SGE | Scatter-Gather Element | 描述本地payload的地址和长度 |
| doorbell | 通知硬件有新WQE的机制 | 由AIV的scalar侧发布 |
| URMA | Unified Remote Memory Access | 执行远端写入 |
| MTE | Memory Transfer Engine | 执行bulk数据传输 |
| CQE | Completion Queue Entry | 记录WQE完成 |

# DeepEP-Ascend SIMT 通信软文大纲（v0.1）

日期：2026-10-10
状态：只做大纲，正文待确认后再写

## 1. 文章定位

这篇软文的主线不是介绍一个通信库，而是回答一个问题：在 MoE 这种“每个 token 的目标 rank、专家、槽位都在变化”的场景里，为什么 SIMT 是合适的通信执行模型？

文章以 DeepEP-Ascend 的实现为主，拆解它的三个关键动作：

1. Host 侧为每个 AIV 建立独立 Jetty；
2. Dispatch/Combine 的 SIMT worker 在 Device 侧直接构造 SQE/WQE；
3. WQE 里的 SGE 直接指向本地 payload，URMA 负责把数据写到远端接收窗口。

写作时保持一个边界：DeepEP-Ascend 路径不是“每个 lane 都各自 ring doorbell”。准确模型是每个 AIV 拥有一个私有 Jetty/SQ；SIMT VF 负责路由、槽位、SGE 和 WQE 字段的并行构造；WQE 写入 SQ 后，由该 AIV 的 scalar 侧更新 head 并 ring doorbell。这个区分很重要，否则容易把“per-AIV 独立提交”误写成“per-lane 独立提交”。

## 2. 核心主张

DeepEP-Ascend 展示了 SIMT 在通信领域的三个契合点：

1. 动态路由天然适合 SIMT。top-k 扫描、rank 去重、expert histogram、槽位分配、前缀和都是典型的不规则分支和归约。DeepEP-Ascend 把这些逻辑放在 512-thread 的 SIMT persistent worker 里，而不是交回 Host 或集中 service。
2. WQE 字段天然适合 lane 映射。SQE header、peer metadata、SGE length/address 都是按 word/字段组织的。DeepEP-Ascend 代码把这些字段分配到不同 lane，一条 WQE 可以由一个 warp 并行拼出来。
3. per-AIV Jetty 避免 producer 串行化。Host 为每个 AIV 创建一个 Jetty；每个 AIV 处理一段连续 token，并拥有自己的 SQ；64 个 AIV 可以同时构造 WQE，不需要一个集中 transport service 逐条解析 command。

一句话概括：SIMT 负责“把不规则的路由结果变成规则的 WQE 字段”，URMA 负责“把 SGE 指向的 payload 写到远端”。

## 3. 目标读者

- 做 MoE、EP、分布式通信或推理框架的工程师；
- 关心 Ascend C、SIMT、AIV 执行模型的开发者；
- 想判断 SIMT 是否适合自己业务的技术决策者。

默认读者知道 MoE 和 EP，但不假设熟悉 HCOMM、Jetty、SQE/WQE。

## 4. 推荐标题

1. DeepEP-Ascend：让 SIMT 直接拼通信 WQE
2. MoE 路由到 URMA：DeepEP-Ascend 的 SIMT 通信路径
3. 每个 AIV 一个 Jetty：DeepEP-Ascend 如何用 SIMT 消掉通信中间层
4. 从 top-k 到 WQE：DeepEP-Ascend 的设备侧通信执行模型

公众号钩子式标题可用：为什么 MoE 通信适合 SIMT？看 DeepEP-Ascend 怎么做。

## 5. 正文结构

### 0. 导语：通信的难点不是“搬数据”，而是“知道搬给谁”

建议从一个具体动作切入：router 输出 top-k；同一个 token 可能去多个 rank；每个 token 的目标 expert、rank、slot、payload 长度都不同；如果这段逻辑放回 Host，会带来同步、launch 和状态搬运开销。

引出主张：DeepEP-Ascend 把“路由结果到 WQE 字段”的转换留在 Device 侧，并用 per-AIV Jetty 保证多个 AIV 可以并行提交。

### 1. MoE Dispatch/Combine 的通信形状

解释两个操作：

- Dispatch：按 top-k 把 token 发给 expert 所在 rank；
- Combine：把 expert 输出送回原 token 所在 rank，并做聚合。

强调三个特点：动态、稀疏、不均衡。可用一张图表示 token 到 top-k、rank/expert/slot 的映射。

### 2. DeepEP-Ascend 的总体架构

建议用一张主架构图，主干是：

Host 侧：HCCL communicator，UB_MEM 地址发现，per-AIV Jetty endpoints，peer channel table。

Device 侧：dispatch_impl / combine_impl，SIMT persistent worker，route scan / histogram / dedup / prefix / slot，WQE field construction，SQ write，scalar head update + doorbell，URMA payload write。

需要明确：Host 只做资源准备；Device 侧不经过通用 command queue；SIMT 与 scalar/MTE 在同一个 kernel 内协作。

### 3. Host 侧：为什么是“每个 AIV 一个 Jetty”

对应代码 csrc/kernels/comm/hccl.hpp 的 acquire_jetties。

关键事实：

- 为每个 AIV 创建一个 Jetty endpoint；
- 每个 Jetty 可以服务多个 peer；
- Host 要求所有 peer 使用同一个 local UBC_CTP endpoint；
- Device 侧通过 jetty_ptrs 表按 AIV 和 peer 索引。

写作重点：这不是“每个 peer 一个 Jetty”，而是“每个 AIV 一个 Jetty”。这让每个 AIV 拥有独立 SQ，避免多个 AIV 争用同一个 producer。这是 DeepEP-Ascend 多 AIV 并发模型的基础。

配图可以是：AIV0 到 Jetty0/SQ0，AIV1 到 Jetty1/SQ1，直到 AIV63 到 Jetty63/SQ63。

### 4. Dispatch：从 top-k 到 SQE/WQE

对应 deep_ep/include/deep_ep/impls/ep/dispatch.hpp。建议拆成四步。

#### 4.1 每个 AIV 拿一段连续 token

- kNumMaxTokensPerVecCore 决定每个 AIV 负责的 token 上限；
- token range 连续，减少跨 AIV 同步；
- 每个 AIV 内部再交给 512 个 SIMT thread。

#### 4.2 SIMT 做路由统计和去重

DeepEP-Ascend worker 里的动作：

- 读取 top-k；
- expert histogram；
- rank histogram；
- 同一 token 的重复 rank 去重；
- 计算逻辑 SQE/SGE 区间；
- 计算目标 slot；
- 生成 sgep_to_entry 映射。

可强调 SIMT 的价值：top-k 天然映射到 lane；asc_ballot、popc、asc_shfl、reduce_add 这类 warp primitive 正适合做去重和归约。这些操作如果交给集中 service，会退化成串行或需要额外 command 协议。

#### 4.3 WQE 字段的 lane 映射

DeepEP-Ascend 代码中的典型映射：

- SQE header 使用前 6 个 lane；
- peer metadata 从 lane 1 开始；
- 每个 SGE pair 使用 4 个 lane；
- 每个 WQEBB 8 个 u64；
- asc_stcg 直接把 sqe_word 写到 SQ。

建议配一张“lane 到 WQE 字段”的表：

| lane | 字段 |
| --- | --- |
| 0 | SQE header / opcode / owner / index |
| 1–5 | peer info / token id / remote address |
| 6–7 | SGE 0：hidden length / source address |
| 8–9 | SGE 1：metadata length / source address |
| 10–13 | 后续 SGE pair |
| 其余 | padding / NOP |

这张图是全文最能体现“SIMT 契合通信”的部分。

#### 4.4 Payload 与完成语义

DeepEP-Ascend Dispatch 的 payload 通过 SGE 描述：

- hidden payload 指向 x；
- metadata payload 指向 metadata_send_buffer；
- URMA 根据 WQE 里的 remote address 写到对端接收窗口。

完成语义：

- 最后一个物理 SQE 设置 strong order、fence、CQE；
- scalar 侧在 VF 返回后统一更新 SQ head 并 ring doorbell；
- benchmark 场景再用 barrier 等待全部 URMA traffic。

### 5. Combine：把通信提交嵌进任务流水

对应 deep_ep/include/deep_ep/impls/ep/combine.hpp。

DeepEP-Ascend Combine 的关键点：

1. load_meta_issue_wqe 是 SIMT VF；
2. 每个 task 处理 8 个 token；
3. 每个 token 构造一个 WQE，两个 SGE：hidden payload 和可选 top-k weights；
4. lane 0–9 直接填 WQE 字段；
5. 每 8 个 task 批量 ring 一次 doorbell；
6. 结尾追加 strong-order NOP 请求 CQE，用于完成检查。

建议重点写：Combine 不是“先算完再通信”，而是把 metadata load、WQE 构造和 reduce 流水化。SIMT 负责把每个 token 的来源 rank 和目标 slot 变成 WQE 字段；scalar 只在批量边界更新 head 和 doorbell。

### 6. 为什么这条路径能体现 SIMT 的优势

建议用一张对比表，但不贬低其他实现：

| 工作类型 | 更适合的执行路径 | 原因 |
| --- | --- | --- |
| top-k 扫描 | SIMT | 每个 token 的目标不同 |
| rank 去重 | SIMT warp primitive | ballot/popc/shuffle 天然匹配 |
| expert/rank histogram | SIMT | 分支多、需要归约 |
| slot/SGE 映射 | SIMT | 每个 token 一份不规则映射 |
| WQE 字段构造 | SIMT | 字段可按 lane 展开 |
| SQ head/doorbell | scalar | 需要明确发布边界 |
| 大块 payload 搬运 | URMA/MTE | 硬件路径更高效 |

核心结论：SIMT 的优势不是替代 URMA，而是把“不规则的路由结果”变成“规则的硬件描述符”。

### 7. 性能证据

README 的公开数据：

| EP size | Dispatch GB/s | Combine GB/s |
| ---: | ---: | ---: |
| EP8 | 373–375 | 345–347 |
| EP16 | 348–352 | 338–341 |
| EP32 | 335–340 | 320–324 |
| EP64 | 323–327 | 294–298 |
| EP128 | 313–320 | 272–278 |

README 说明：

- 测量口径包含 issue 和 drain，排除最终 epilogue；
- Dispatch 在 EP32 以内可达到物理 payload 带宽的约 90–95%；
- Combine 仍有本地 reduce 和 HBM/URMA contention。

GitCode 技术报告中的 EP16 案例可作为补充：

| 操作 | 前处理 | URMA | 后处理 | 带宽 |
| --- | ---: | ---: | ---: | ---: |
| Dispatch BF16 | 58.021 μs | 649.602 μs | 298.877 μs | 508.467 GB/s |
| Dispatch FP8 | 71.031 μs | 318.665 μs | 239.513 μs | 518.258 GB/s |
| Combine BF16 | 37.861 μs | 769.748 μs | 151.046 μs | 429.103 GB/s |

写作时注意：这是公开报告数据，不写成我们自测数据；不做跨环境外推；不把 ASC-COMM 报告的数据与 GitHub 仓代码逐行等同。

### 8. 与 ASC-COMM 技术报告的关系

需要单独说明，避免读者混淆：

- GitHub DeepEP-Ascend 仓：直接操作 HcommUrmaSqeCtx / HcommUrmaSgeCtx，手工组 WQE；
- GitCode ASC-COMM 报告：描述更高层的 Hcomm::WriteNbi、MakeBatchHandle、BatchCommit、Drain API。

二者不是同一层实现。正文可以以 GitHub 仓作为代码级证据，以 ASC-COMM 报告补充“通信 API 分层”和性能案例，但不把报告里的 API 名直接套到仓源码上。

### 9. 另一个扩展方向：专职 transport service kernel

DeepEP-Ascend 的 direct WQE 要求资源模型配合：每个 AIV 拥有自己的 Jetty/SQ，避免多个 producer 争用队列所有权。

另一种工程路线是让 producer 与队列所有权解耦：SIMT producer 生成 put、signal、flush 等语义 command；专职 transport service kernel 独占 channel/Jetty，把 command 解析成 WQE/SQE，并统一维护 head/tail、completion 和错误诊断状态。

这一段作为第二篇的引子：不谈来源、不谈性能，只解释 direct WQE 与 service 分层的取舍。后续展开点包括 command 粒度、channel 独占方式、completion 跨代语义，以及 producer 与 service 的流水线深度。

### 10. 小结

回扣三个主张：

1. 动态路由适合 SIMT；
2. WQE 字段适合 lane 映射；
3. per-AIV Jetty 让多 producer 并发自然成立。

结尾句建议：DeepEP-Ascend 证明了一件事，通信不只是搬运 payload，更是把不规则业务意图编译成硬件描述符。SIMT 恰好擅长这件事。

## 6. 配图清单

1. 主架构图：Host 资源准备、per-AIV Jetty、Device SIMT worker、URMA；
2. per-AIV Jetty 图：AIV0–AIV63 各自拥有 Jetty/SQ；
3. Dispatch 数据流图：top-k、histogram/dedup/prefix/slot、SGE mapping、WQE、SQ、URMA；
4. lane-to-WQE 映射图：每个 lane 负责哪个 WQE 字段；
5. Combine 任务流水图：metadata load、WQE 构造、reduce、批量 doorbell；
6. 性能图：EP8–EP128 Dispatch/Combine 带宽；
7. 可选架构图：higher-level Hcomm API 与 transport service kernel，分别作为第二篇的引子。

## 7. 写作红线

1. 不写“每个 lane 独立 ring doorbell”；
2. 不写“DeepEP-Ascend 路径完全不用 scalar”；
3. 不写“SIMT 替代 URMA”；
4. 不写“所有场景性能提升”；
5. 不把 ASC-COMM 报告和 GitHub 仓混成同一实现；
6. 不引用内部任务 ID、日志路径或私有环境细节。

## 8. 已有素材复用清单

### 8.1 可以直接复用

来自 epv2-ascend-simt-urma-article-draft-zh.md：

- “阅读说明”的三条前置知识：Host/Device 执行模型、EP、对称窗口；
- 第 1 节对 MoE 通信“动态、稀疏、不均衡”的解释；
- 第 2 节术语表中的 AICore/AIV、SIMT、URMA、WQE/CQE、SQ/CQ、doorbell；
- 第 5 节控制面与数据面的论述，尤其“payload 不先进 UB，由 URMA 直接写远端窗口”；
- 第 9 节性能口径：timings include issue and drain but exclude final epilogues；
- 写作风格：一分钟版、术语表、代码路径追踪、边界说明。

来自 ascend-hcomm-simt-communication-overlap.html：

- Collective 与 one-sided communication 的对比；
- communicator/team/window/channel/SQ/CQ 的心智模型；
- URMA SQE 48B、SGE 16B、CQE 64B 的结构解释；
- “通信不是一次拷贝”的开场心智模型。

来自 Netlayer 文档：

- netlayer0-vs-netlayer1-analysis-zh.md 的物理拓扑解释：layer 1 是 SuperPoD 外部 Clos/PEER2NET，layer 0 是单机 PEER2PEER 直连。正文可用一小段说明 DeepEP-Ascend 路径依赖 netlayer 1 的外部 Clos 通路，但不展开内部 NPU8P 适配细节；
- 该文档的 per-AIV Jetty 模型图：AIV0–AIV63 分别拥有 Jetty0/SQ0–Jetty63/SQ63。这个图是解释 DeepEP-Ascend 多 producer 模型最直接的基础；
- 该文档对 “shared queue” 的澄清：一个 AIV 的 Jetty 可服务多个 peer，但不同 AIV 不共享 Jetty。这个句子应保留为正文关键句；
- 该文档的性能统计口径：bench_msprof 采样 dispatch_impl / combine_impl，使用 kernel dur_ns，50 个采样平均，字节公式为 per-rank 逻辑 URMA 字节，最终 epilogue 不计入；
- 该文档的“性能预期”段落：多 producer 同时提交、WQE 编码和 CQE drain 分布在多个 AIV、与 token 分片天然匹配、低延迟路径受 host launch 和单队列串行化影响更小。这些是解释 SIMT 结构优势的好素材。

来自其他 docs 材料：

- docs/deepep-teach/MISSION.md 的三条写作原则：性能数字标明实测/公开参考/估算，不混用 workload，不宣称未验收能力；
- docs/deepep-teach/reference/deepep-v2-ascend-950-guide.html 可用于核对 workload 和 EPBuffer 使用方式；
- docs/ascend-reference/ascend-hcomm-team-channel-model-zh.md 的 Communicator / Team / Window / Channel / Endpoint / MR / SQ / CQ 分层解释，适合压缩成新文术语节。

### 8.2 改写后复用

- 现有文章第 6 节 Team/Window/Channel 资源模型：DeepEP-Ascend 仓使用 per-AIV Jetty 和 peer channel table，需要按 DeepEP-Ascend 资源模型重写；
- 现有文章第 4 节 staged transport 的动机：可改成“为什么 DeepEP-Ascend 路径能把 WQE 构造放回 SIMT”；
- 现有文章第 9 节 Netlayer 0/1 对比：可压缩成背景，不作为新文主线；
- simt-and-aicore-main-scalar-division-zh.md 中的“SIMT/AICore 分工”结论：可用于解释 SIMT 拼 WQE、scalar 管 doorbell、URMA/MTE 管 payload。
- dispatch-simt-parallelization.md 和 ascend-72-aiv-pipeline-design.md：可借鉴“一个 AIV 负责一段 token、AIV 内 512 SIMT threads 分工”的叙事，但要按 DeepEP-Ascend 仓源码重写，不使用本仓 staged 阶段名；
- epv2-ascend-multi-channel-design-spec-zh.md：可借鉴 channel / Jetty / SQ 深度 / WQE 数量的概念区分，用于解释“增加队列不等于增加物理链路”；
- netlayer0-adaptation-diagnosis-zh.md：可借鉴每 AIV 独占 Jetty 与共享 Jetty 冲突的原因，但只作为技术背景，不进入对外软文细节。

### 8.3 只做背景对照

- asc-comm-official-simt-comparison-zh.md：说明 asc-comm SIMT API 与当前 staged transport 的差异，但不要把它混同为 DeepEP-Ascend GitHub 仓；
- deep-ep-ascend-communication-implementation-review-zh.md：当前仓库实现细节，只用于一句话对照；
- official-simt-backend-abba-zh.md：内部实验数据，不用于对外软文。
- dispatch-multichannel-jetty-validation-zh.md：内部验证数据，适合用来提醒写作时不要把 channel 数、Jetty 数和物理链路数混为一谈，但不直接引用数据；
- dispatch-small-transfer-and-cq-analysis-zh.md、dispatch-tail-bottleneck-analysis-zh.md、epv2-ascend-rank-tail-analysis-zh.md：内部性能诊断，不进入对外软文；可借鉴其“stage 归因不能直接外推端到端”的方法论。

### 8.4 不建议复用

- 当前实现的 TransportCommand ABI 细节；
- 内部 benchmark 任务、日志、环境路径；
- 我们自己的 0.73x / 0.83x 近似对齐数据；
- 容易让读者误解为“DeepEP-Ascend 路径就是每个 lane ring doorbell”的表述。

### 8.5 配图复用

可直接复用或改版：

- MoE routing 图：token 到不同 expert/rank 的直观解释；
- control/data plane 图：控制描述符与 payload 分离；
- dispatch/combine call chains 图：改成 top-k 到 WQE 的调用链；
- netlayer0-vs-netlayer1 图：仅作为内部理解材料，不进入正文配图；
- put lifecycle 图：改成“top-k → histogram/dedup → SGE → WQE → SQ → doorbell → URMA”的路径生命周期。

新增配图建议：

- direct WQE 与 transport service 的分层对比图：一边是 producer 内完成路由到 WQE，另一边是 producer 生成语义 command、service 独占队列并完成 WQE/SQE；
- Hcomm API 层次图：应用侧 WriteNbi / MakeBatchHandle / BatchCommit / Drain 在上，底层 WQE/SGE/SQ/CQ 在下，用虚线表示封装边界。

Netlayer 文档新增可改版图：

- Official netlayer1 ownership 图：直接改绘为 DeepEP-Ascend per-AIV Jetty 架构图；
- Repository netlayer0 ownership 图：仅作为内部理解材料，不进入正文配图；
- WQE 提交时序图：可改成“SIMT 写 SQE → scalar 更新 head/doorbell → URMA payload → CQE/drain”的时序。

## 9. 参考材料

### DeepEP-Ascend 代码

- Repository: https://github.com/deepseek-ai/DeepEP-Ascend
- Commit: 3b25377d04b24fc6154698ded78a2bcb2c59afff
- Key files: csrc/kernels/comm/hccl.hpp；deep_ep/include/deep_ep/comm/handle.hpp；deep_ep/include/deep_ep/impls/ep/dispatch.hpp；deep_ep/include/deep_ep/impls/ep/combine.hpp；tests/ep/test_ep.py

### ASC-COMM 技术报告

- https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek_v4_1/deepseek_v4.1_asc_comm_tech_report.md

### 仓库内已有参考

- docs/ascend-reference/asc-comm-official-simt-comparison-zh.md
- docs/ascend-reference/simt-and-aicore-main-scalar-division-zh.md
- docs/ascend-reference/deep-ep-ascend-communication-implementation-review-zh.md

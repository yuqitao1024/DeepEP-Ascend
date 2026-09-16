# DeepEP Ascend SIMT-URMA 通信软文策划

## 1. 文档定位

本文是 DeepEP Ascend SIMT 通信方案的中文软文策划稿，目标是为后续技术文章、项目介绍和性能发布材料提供统一叙事。

文章应突出真实的工程价值：让 SIMT 参与动态通信控制，让 UB_MEM/AICore 承接 URMA 队列提交，让 DeepEP 的 dispatch/combine 算子通过统一 facade 使用 HCCL/HCOMM 资源。

当前实现采用 **SIMT-fronted staged transport**：SIMT VF 生成通信命令，AICore transport service 在服务边界使用 UB、SQ/CQ 和 `st_dev/ld_dev` 完成底层提交。因此，除非后续版本真正实现 SIMT 直接 doorbell，否则标题和正文不应把当前方案描述为“SIMT 直接执行完整 URMA doorbell”。

## 2. 推荐文章主线

```text
MoE 路由为什么带来不规则通信
    -> 传统 collective/host 控制的控制面成本
    -> SIMT 为什么适合生成动态通信意图
    -> UB_MEM 和 AICore 如何承接 URMA 提交
    -> Team/Window/Channel 如何组织通信资源
    -> dispatch/combine 如何复用统一 transport facade
    -> 如何验证端到端性能和长尾收益
    -> 当前边界与下一步演进
```

核心叙事不是“用一种新硬件指令替代旧通信库”，而是：

> 把 token 路由、目标 peer、窗口 offset 和通信命令生成放回设备侧，使动态 MoE 通信控制更接近数据产生的位置；再用专用 service 将这些意图高效落到底层 URMA 队列。

## 3. 可以主打的技术亮点

### 3.1 路由即通信

DeepEP 的 dispatch/combine 不是规则的大块 collective。router 产生 top-k 结果后，每个 token 可能对应不同的 expert、peer、目标 offset 和 payload 大小。

SIMT 线程可以在设备侧完成：

- 根据路由结果选择目标 peer；
- 将 team-local peer 翻译为 world peer；
- 计算 symmetric window offset；
- 判断边界、空 payload 和完成条件；
- 生成 put、inline value、signal、flush 等通信命令。

这使通信控制能够和 token 元数据处理保持在同一执行域内，减少不必要的 host-side metadata 转换。

### 3.2 SIMT 与 AICore 的分工

不要把 SIMT 和 SIMD 写成简单的替代关系。更准确的表述是：

| 工作类型 | 更适合的执行路径 |
| --- | --- |
| 规则、大块、同构的数据搬运 | SIMD/AICore |
| 动态 peer 选择和分支 | SIMT |
| 每个 token 目标不同 | SIMT |
| 批量组织 SQE、门铃和完成轮询 | AICore service |
| 端到端通信与路由融合 | SIMT 生成 + AICore 提交 |

SIMT 负责动态性，AICore service 负责硬件提交吞吐。两者结合，比“SIMT 全部替代 SIMD”更符合当前实现和可验证的性能假设。

### 3.3 UB_MEM 作为通信控制面的高速工作区

UB_MEM 的重点不是承载全部 payload，而是承担通信提交阶段的控制面工作：

- 聚合 SIMT 产生的固定格式 command；
- 组织 SQE、SGE 和临时 descriptor；
- 保存服务阶段需要快速访问的状态；
- 配合 `st_dev/ld_dev` 完成 SQ/CQ 和 doorbell 操作；
- 让复杂的 GM descriptor 访问集中在受控的 service 路径。

真正的 payload 仍然通过注册的 symmetric window 和 URMA 机制完成远端访问。文章应明确区分“UB_MEM 控制面”和“URMA payload 数据面”，避免让读者误解为所有 token 都要先复制到 UB。

### 3.4 Team、Window、Channel 的统一资源模型

可以用下面的层次解释通信资源：

```text
communicator
    -> team：参与者集合、rank 和拓扑域
        -> window：各 rank 对称布局的可远端访问内存
            -> channel：Team 内的并行通信 lane
                -> SQ/CQ、doorbell 和底层通信队列
```

- **Team** 定义通信域。当前抽象包含 `world`、`scale-up` 和 `scale-out` 逻辑 team。
- **Window** 提供统一的远端寻址基础。发送端保存 offset，目标地址由 peer window base 加 offset 得到。
- **Channel** 提供 Team 内的并行执行资源。一个 peer 可以有多个 channel，设备侧按 team 的 channel table 和 `channel_counts` 做索引。
- **Jetty** 如果在底层 HCOMM/URMA 实现中出现，属于 channel 下面的 endpoint/队列实现细节，不是当前 DeepEP operator facade 的一级抽象。

上层算子只需要描述“向哪个 peer 写哪段数据”，不需要自己拼装 HCOMM/URMA 的底层队列对象。

### 3.5 算子侧易用性

统一 `DeviceTransportFacade` 屏蔽了以下细节：

- HCCL communicator 的复用和校验；
- team/window/channel 的创建和销毁；
- world rank 与 team-local peer 的翻译；
- symmetric window 地址解析；
- channel table 的索引和有效性检查；
- signal、generation、flush 和 barrier 的语义；
- CANN device-visible ABI 的结构体布局。

算子开发者面对的是 `put`、`put_value`、`signal`、`flush` 等通信语义，而不是 URMA SQE 字段和 doorbell 细节。这种分层也让 CUDA/NCCL 路径和 Ascend backend 保持相对独立。

### 3.6 工程落地速度

“实现快”应当作为工程方法的结果，而不是未经量化的宣传承诺。可强调以下做法：

- 复用已有 ProcessGroupHCCL communicator；
- 只定义需要的 device-visible POD ABI，不引入 HCOMM 内部 C++ 类体系；
- 用 ABI probe 检查尺寸、对齐和字段 offset；
- 用统一 facade 将新通信 primitive 接入 dispatch/combine；
- 先通过 stub transport 和 host probe 验证接口，再切换真实 CANN 资源；
- 新增通信操作主要扩展 command opcode 和 service handler，而不是重写资源生命周期。

这种方式的亮点是依赖边界清晰、可测试、可替换，而不是单纯减少代码行数。

## 4. 性能表述和验证方法

### 4.1 不建议直接宣称

以下表述需要避免，除非有同版本、同 workload 的数据支撑：

- “SIMT 通信性能全面优于 SIMD”；
- “所有场景都能获得更高带宽”；
- “已经实现 SIMT 直接 doorbell”；
- “完全消除了 AICore service 开销”；
- “无需任何同步即可保证 payload 可见”。

### 4.2 建议验证的指标

性能章节应优先关注端到端 MoE 价值：

- dispatch/combine 的端到端 latency；
- p50、p95、p99 长尾 latency；
- router 结束到第一批通信命令提交的启动时间；
- CPU launch 和 host synchronization 次数；
- command generation 和 service submission 时间；
- 不同 top-k、peer 数和 payload size 下的有效带宽；
- 通信与排序、GEMM 等计算的重叠收益；
- channel 数从 1 到 2/4 时的收益和资源成本；
- expert 负载不均衡时的尾部 rank 表现。

至少应覆盖三类 workload：

1. 大块、规则、均匀通信，用于比较基础搬运能力；
2. 小包、多 peer、动态路由，用于体现 SIMT 控制优势；
3. 长尾明显的专家负载，用于观察调度和完成语义的稳定性。

更稳健的结论形式是：

> 在动态稀疏通信中，SIMT-fronted 路径减少了控制面转换，并为通信和路由计算融合提供了更自然的执行粒度；最终收益取决于消息规模、peer 分布、channel 数和 service 提交效率。

## 5. 建议文章结构

### 标题候选

- 《让 SIMT 参与通信：DeepEP Ascend 的 URMA/UB_MEM 一体化路径》
- 《从 token 路由到 URMA 提交：DeepEP Ascend 的 SIMT 通信执行层》
- 《SIMT 生成、UB_MEM 聚合、URMA 提交：DeepEP Ascend 通信架构实践》
- 《面向稀疏 MoE 的 Ascend SIMT 通信：动态路由与 URMA 的协同设计》

### 正文建议章节

1. **MoE 通信的难点**：动态、稀疏、不均衡，不能只用规则 collective 的视角理解。
2. **为什么引入 SIMT**：把 peer 选择、offset 计算和命令生成放回设备侧。
3. **为什么不是 SIMT 单独完成一切**：CANN 9.2.0 下底层 doorbell 和 SQ/CQ 提交由 AICore service 承接。
4. **UB_MEM/URMA 分层**：控制面组织与 payload 数据面分离。
5. **Team/Window/Channel 资源模型**：从 communicator 到实际 peer/channel 的映射。
6. **dispatch/combine 接入方式**：统一 facade、signal/generation、flush 和 barrier。
7. **性能验证**：按动态稀疏 workload 报告端到端结果和长尾，而不是只报裸带宽。
8. **边界与路线图**：跨机 RoCE、device get、真实异步 request、多 service worker 等后续能力。

## 6. 当前边界和后续演进

当前方案的对外描述应保持以下边界：

- 主要验证目标是 Ascend 950、CANN 9.2.0、单机多 NPU 的通信语义；
- SIMT 路径当前是 staged transport，命令在 AICore service 边界完成 URMA 提交；
- operator 侧遵循统一 facade，生产 capability 需要以端到端验证结果为准；
- 跨主机 scale-out/RoCE、device-side get、完整异步 request 状态机和多 service worker 需要单独验证；
- 任何性能领先结论都应附带硬件、CANN、channel 配置和 workload 条件。

后续可以围绕以下方向形成第二篇文章或性能专题：

1. SIMT command batching 和多 channel 调度；
2. 通信计算重叠及 pipeline 边界；
3. 长尾 expert 场景的 channel/peer 负载均衡；
4. device-side completion 和 request 状态机；
5. 跨机 URMA/RoCE 的 topology 与资源扩展；
6. 在编译器支持成熟后评估更直接的 SIMT doorbell 路径。

## 7. 参考代码和设计文档

- `csrc/backends/ascend/transport/device_transport_facade.hpp`
- `csrc/backends/ascend/transport/cann_transport.cpp`
- `csrc/backends/ascend/transport/device_transport_commands.hpp`
- `docs/ascend-design/epv2-ascend-simt-urma-transport.md`
- `docs/ascend-design/epv2-ascend-transport-contract.md`
- `docs/ascend-design/epv2-ascend-multi-channel-design-spec-zh.md`


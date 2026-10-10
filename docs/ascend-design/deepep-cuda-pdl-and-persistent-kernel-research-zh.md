# DeepEP CUDA PDL 与 Hybrid-EP Persistent Kernel 调研记录

日期：2026-10-10
状态：调研记录，未排期实施

## 0. 调研对象

| 分支 | commit | 关注点 |
| --- | --- | --- |
| DeepEP CUDA main | 93eb6eb238127e96c6d7a4a625a6dad158348509 | 主线 PDL，以及 Dispatch/Combine 与 epilogue 的衔接 |
| DeepEP CUDA hybrid-ep | 10d4dd7377d5bce900fbb4b80cce863764892b95 | warp-specialized persistent kernel 的执行模型 |

本文只记录公开源码可见的事实和可借鉴方向，不把这些方向等同于当前 Ascend staged transport 的来源，也不预设性能收益。

## 1. 结论摘要

DeepEP CUDA 里需要区分两类优化：

1. 主线 PDL（Programmatic Dependent Launch）：用于 Dispatch/Combine 主 kernel 与 epilogue kernel 的低开销串联，减少 CPU 参与 kernel chaining 的次数。
2. Hybrid-EP persistent kernel：一种 warp-specialized persistent kernel 形态。一个 CUDA block 独占 SM，kernel 内部按 warp 角色组成流水线，用 shared-memory FIFO、ready/tail 标志和 coordinator 推进数据流。

二者都不是常驻 transport service kernel 等待外部 signal 执行通信。不要把它们与当前 Ascend staged transport service 混同。

## 2. 主线 PDL

### 2.1 证据

DeepEP CUDA main 分支的 Dispatch kernel 末尾附近调用：

cudaTriggerProgrammaticLaunchCompletion()

Dispatch 与 Combine 的 epilogue launch 配置中显式设置 enable_pdl = true，对应底层 launch attribute 为 programmatic stream serialization。当前源码中 Trigger 调用可见于 Dispatch / Hybrid Dispatch，Combine 主 kernel 末尾则通过 barrier 等待发送到达，再进入已启用 PDL 的 reduce epilogue。关键文件：

- deep_ep/include/deep_ep/impls/ep/dispatch.cuh
- deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh
- csrc/kernels/ep/dispatch.hpp
- csrc/kernels/ep/combine.hpp
- deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh
- deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh

### 2.2 作用

传统方式需要：

main kernel finish -> CPU/runtime observes completion -> launch epilogue

PDL 允许 epilogue 的 launch 提前准备，主 kernel 通过 programmatic completion 事件触发后续执行，减少 CPU round-trip 和 kernel chain 空隙。

PDL 解决的是依赖 kernel 的启动衔接，不改变通信数据面本身，也不等价于把通信提交交给一个常驻 service。

### 2.3 对 Ascend 的启发

当前 Ascend 阶段已经观察到 kernel/queue 空洞是主要瓶颈之一。可研究：

1. epilogue 提前 launch；
2. 主通信 kernel 与 epilogue 的设备侧事件衔接；
3. 减少 host 每阶段重复提交；
4. 保留 correctness、generation 和 completion 语义。

Ascend 是否存在等价的 programmatic dependent launch 语义，需要单独验证，不能直接把 CUDA PDL API 映射为已有能力。

## 3. Hybrid-EP Persistent Kernel

### 3.1 证据

Hybrid-EP 分支文档明确说明：

The dispatch and combine kernels in Hybrid-EP are warp-specialized persistent kernels.

并解释：

- Persistent：每个 CUDA block 独占一个 SM，覆盖整个 kernel 生命周期；
- Warp-specialized：block 内不同 warp group 组成流水线；
- Independent blocks：数据切成 chunk，分布到多个 block。

相关文件：

- csrc/hybrid_ep/backend/hybrid_ep_backend.cuh
- csrc/hybrid_ep/executor/
- docs/Hybrid-EP_Implementation.md

dispatch_kernel 与 combine_kernel 定义在 hybrid_ep_backend.cuh；launch 侧按 warp group 组织角色，并配置动态 shared memory。

### 3.2 执行模型

Hybrid-EP Dispatch 中 warp 角色大致包括：

- G2S warp：global memory 到 shared memory；
- S2G warp：shared memory 到 global/remote buffer；
- RDMA sender / receiver warp；
- 融合模式下另有 permute G2S / S2G warp group。

Combine 中还包括 reduce warp group：

- G2S warp：global memory 到 shared memory；
- reduce warp：跨 rank 或节点内预聚合；
- S2G warp：写回本地或远端 buffer；
- RDMA warp：跨节点发送/接收；
- 融合模式下另有 unpermute G2S / reduce warp group。

核心结构：

persistent block -> role-specialized warps -> shared-memory cyclic FIFO -> ready/tail/head flags -> coordinator controls flow control -> kernel exits after complete workload

### 3.3 解决什么

它主要解决 kernel 内部流水化和角色分工：

1. 数据搬运、通信和 reduction 在同一个 kernel 内重叠；
2. coordinator 控制 buffer release 和发送进度；
3. 减少 kernel 间切换和中间全局缓冲；
4. 适合 chunked pipeline 和通信计算重叠。

它不是让 kernel 无限驻留等待外部 signal；kernel 仍然处理完本轮 workload 后退出。

## 4. 与当前 Ascend staged transport 的关系

当前 Ascend 默认路径：

SIMT producer -> TransportCommand -> AICore transport service -> URMA WQE/SQ/CQ

Hybrid-EP persistent kernel 不是这个结构的直接来源，也没有证据表明 staged transport 参考了它。当前 staged transport 更应被描述为一种工程上的 command/service 分层设计：producer 编码语义 command，专职 service 统一处理资源所有权、错误诊断和 completion 协议。

Hybrid-EP 启发的是外层 kernel 调度与流水组织，而不是通信 service 的所有权模型。

可借鉴方向：

1. 把 Dispatch/Combine 组织成 persistent/warp-specialized kernel；
2. 用设备侧 flag/head/tail 控制 chunk 流水；
3. 在同一 kernel 内划分 producer、service、receiver、reduce 角色；
4. 减少 host 阶段 launch 和 kernel 间空洞。

需要避免：

1. 把 persistent kernel 等同于常驻 transport service；
2. 直接照搬 CUDA warp/block 语义；
3. 弱化 completion、generation、错误诊断；
4. 未验证前宣称能降低 launch 开销。

## 5. 后续候选工作

### 5.1 Stage chaining / launch-ahead

先分析当前阶段间 kernel/queue 空洞，再研究是否能把稳定参数预提交，让 epilogue 在通信完成边界自动衔接。

设计要点：

1. 明确主 kernel 到 epilogue 的数据依赖和可见性边界；
2. 保持 generation / request / completion 语义；
3. 不把 launch-ahead 误做成跨代 buffer 复用；
4. 优先覆盖 Dispatch copy epilogue 与 Combine reduce epilogue 这类稳定链路。

验收：

- 不改变 correctness；
- 不引入跨代复用 race；
- ABBA 显示 mean/p95 稳定改善；
- 不能只让 profile Event 变短而端到端不变。

### 5.2 Persistent operator kernel

将 Dispatch/Combine 的 producer、release、epilogue 角色放进更少的 persistent kernel 中，使用设备侧状态推进。

设计要点：

1. 先做角色划分：producer、receiver、reduce、release、epilogue；
2. 用有限深度的 cyclic FIFO 或 buffer ring 控制飞行数据；
3. 用明确的 ready/tail/completion 标志替代隐式时序假设；
4. 保留错误退出和 stream 语义，避免为了 persistent 而牺牲可诊断性。

验收：

- stream 和 shutdown 语义安全；
- 不因 persistent kernel 导致死锁；
- 错误路径可退出；
- 对典型 8-rank case 有稳定收益。

### 5.3 与 direct HCOMM SIMT facade 组合

如果未来 direct facade 对接 official ASC-COMM SIMT API，可研究：

persistent/warp-specialized operator kernel -> per-peer dedicated lane -> HCOMM SIMT WriteNbi / Drain

这不是当前优先级，需在 direct facade probe 通过后再评估。

### 5.4 排序建议

1. 先做 stage gap 量化，确认 launch/queue 空洞占比；
2. 若空洞占比高，优先做 stage chaining / launch-ahead，改动面小于完整 persistent kernel；
3. 若主要瓶颈是阶段间数据搬运和 reduce 无法重叠，再评估 persistent operator kernel；
4. direct HCOMM SIMT facade 独立推进，避免与 persistent kernel 同时重构 hot path。

## 6. 参考

- DeepEP CUDA main：https://github.com/deepseek-ai/DeepEP/tree/93eb6eb238127e96c6d7a4a625a6dad158348509
- DeepEP CUDA hybrid-ep：https://github.com/deepseek-ai/DeepEP/tree/10d4dd7377d5bce900fbb4b80cce863764892b95
- Hybrid-EP 实现文档：https://github.com/deepseek-ai/DeepEP/blob/10d4dd7377d5bce900fbb4b80cce863764892b95/docs/Hybrid-EP_Implementation.md
- docs/ascend-design/official-asc-comm-facade-integration-plan-zh.md

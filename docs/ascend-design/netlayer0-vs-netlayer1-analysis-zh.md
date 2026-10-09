# netlayer=0 与 netlayer=1 全面对比分析

日期：2026-10-09。本文面向 Ascend 950PR / CANN 9.3 / HCOMM 的自定义
AIV-URMA 数据面，比较 `HcclRankGraphGetLayers` 返回的 layer 0 和 layer 1。
结论只适用于本文列出的接口和实测环境，不能直接外推到其他 CANN 版本。

## 1. 一句话结论

layer 1 是官方 DeepEP-Ascend 数据面假设的 SuperPoD 外部 Clos 通信层：官方
实现固定选择 `layers[1]`，先为每个 peer 获取 `UB_MEM` channel，用于发现远端
buffer 地址；再为每个 AIV 创建共享 Jetty，并通过 `UBC_CTP` 建立共享队列
channel。layer 0 是 NPU8P 可用的机内直连层，当前独立 channel 路径选择
`UB_CTP`。因此二者不是简单的配置差异，而是物理路径、协议组合和队列模型
都不同的通信路径。

## 2. 硬件与拓扑能力

### 2.0 物理拓扑事实

NPU8P 的驱动拓扑文件 `atlas_950_1.json` 显示：0–7 卡在 layer 0 有完整 28 条
`PEER2PEER` 直连，每个 rank 都有 7 个 peer，8 卡是全互联。layer 1 在该拓扑
文件中表现为每卡到 CLOS 网络的 `PEER2NET` 连接，不是 layer 0 那种成对的
`PEER2PEER` 边。二者都能表达 8 rank all-to-all，但物理路径不同：layer 0 是
机内 1DMESH 直连；layer 1 是 SuperPoD 外部 CLOS/RDMA 网络。

这修正了早期诊断中的一个错误：当时 `HcclRankGraphGetLinks` 只返回 peer 1，
根因是 `HcclRankGraphGetLayers` 的库内缓冲被后续查询复用。必须在进入循环前
复制 `layers[...]`，之后所有 rank 均能枚举 7 个 peer。

### 2.1 layer 选择

HCCL rank graph 通过以下接口暴露可用网络层：

```text
HcclRankGraphGetLayers(comm, &layers, &layer_count)
HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &link_count)
```

`CommLink` 给出本端 endpoint、远端 endpoint 和协议。官方 DeepEP-Ascend
固定选择 `layers[1]`；NPU8P 诊断显示当前环境选择 layer 1 会让
`HcclChannelAcquire` 等待 120 秒超时，而 layer 0 能返回
UBC_CTP/UB_MEM 等链路。

### 2.2 链路协议

当前 NPU8P 的 layer 0 rank pair 可同时返回 `UB_CTP` 和 `UB_MEM` 链路。协议
数值为 `UB_CTP=4`、`UBC_TP=5`、`UB_MEM=6`。当前独立 channel URMA WRITE 路径
在 layer 0 选择 `UB_CTP`，因为 NPU8P 实测该协议可以完成 8 rank all-to-all
建链、CQE drain 和 payload 校验。`UB_MEM` 是另一类内存语义路径，官方
DeepEP-Ascend 在 layer 1 用它做 peer buffer 地址发现和数据面访问，不能默认
与 layer 0 的 `UB_CTP` URMA WRITE 队列模型等价。

### 2.3 物理带宽

本仓库尚未在同一 8 卡环境实测 layer 0/1 的饱和带宽。已有 DeepEP transport
基准证明 layer 0 的 UBC_CTP URMA WRITE 可以达到可用吞吐，但这不是 layer 1
的对照结果。合理的性能结论必须等待 `benchmarks/netlayer_ab` 在同一硬件、
同一 CANN、同一 payload/请求规模下完成 A/B 测试。

## 3. 软件实现差异

官方 DeepEP-Ascend 的 layer 1 初始化流程可以概括为：

1. `HcclRankGraphGetLayers` 后立即复制 `layers[1]`，避免库内缓冲被后续查询
   复用；
2. 对每个 peer 选择 `UB_MEM` link，批量 `HcclChannelAcquire`，再通过
   `HcclChannelGetRemoteMems` 找到远端 workspace 地址；
3. 对每个 peer 选择 `UBC_CTP` link，要求所有 peer 使用同一 local endpoint；
4. 每个 AIV 创建一个 Jetty endpoint，并在其上注册 workspace；
5. 通过 `HcommEndpointGetListenPort`、HCCL all-gather 和端口交换完成监听端口
   发现；
6. 使用 `HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE=1` 创建共享队列 channel，
   并用 `HcommChannelGetStatus` 轮询到 ready；
7. 将 channel handle 表复制到 device，kernel 按 AIV/peer 索引解析。

这套实现的重点是：`UB_MEM` 负责 peer 地址空间，`UBC_CTP` 负责 URMA 发送
队列；共享 Jetty 由官方 kernel 的调度逻辑保证使用顺序。官方性能数据基于
Ascend 950DT、CANN 9.2.0 和特定 PoC HDK，README 标明 EP8 dispatch 约
373–375 GB/s、combine 约 345–347 GB/s，且均使用 netlayer 1。

| 维度 | layer 1 / 官方路径 | layer 0 / 当前 NPU8P 路径 |
| --- | --- | --- |
| layer 选择 | `layers[1]` | `layers[0]` |
| 资源模型 | 每个 AIV 一个 Jetty/SQ | 共享 Jetty 或少量独立 Jetty |
| 并发假设 | 64 个 AIV 同时提交 | 共享队列要求调用者串行化 |
| SQE 位置 | 每 AIV 从自己的 Jetty head 计算 | 多 AIV 使用同一 head 会写同一 SQE |
| CQ/drain | 每 AIV/队列独立推进 | 共享队列的 CQ 所有权需要明确 |
| DeepEP 官方代码 | 直接匹配 | 需要重构资源表和提交协议 |

上表中的“当前 NPU8P 路径”指本仓库独立 AB 基准，而不是 DeepEP 生产实现。
它通过编译宏隔离两种协议：`NETLAYER_AB_USE_UB_CTP` 对应 layer 0，
`NETLAYER_AB_USE_UB_MEM` 对应 layer 1。NPU8P 已验证 layer 0 能完整运行；
layer 1 编译通过，但 `HcclChannelAcquire` 在该环境返回 status 9，不能作为
性能结论。

官方 kernel 中的关键模型是：

```text
HcommJettyInfo::load_jetty_info(..., vec_core_idx, lane_idx)
HcommJetty jetty(..., vec_core_idx)
```

即 AIV 索引决定 Jetty/SQ。每个 AIV 使用自己的 SQE slot 和 head 推进，最后
独立 ring doorbell。这使 dispatch/combine 可以把不同 token/expert 的发送
工作分发到多个 AIV，通信提交和计算重叠。

layer 0 的诊断适配曾把 64 个 AIV 全部指向共享 Jetty index 0。此时每个 AIV
的局部 `psqe_idx` 都从 0 开始，它们读取同一个 packed head 后写同一个物理
SQE 区间。第一个 SQE 被写 64 次，但 doorbell 只按某一个 AIV 的局部计数推
进，硬件看到不完整的 WQE 序列，最终返回 CQE status=6。

CANN 9.3 的 `hcomm_channel.h` 对共享队列的说明是：不同 channel 不支持并发
使用，需要调用者按业务顺序串行调用。因此共享 Jetty 不是只改 doorbell 就
能修复的锁竞争问题，而是接口契约层面的串行队列。

## 4. 使用方式差异

### 4.0 AB 基准的一键用法

`benchmarks/netlayer_ab/build_ab.sh` 一次构建两个场景：

```bash
source /path/to/cann/set_env.sh
./build_ab.sh
```

产物分别是 `build-layer0/netlayer_ab` 和 `build-layer1/netlayer_ab`。二者
使用相同 host/device 代码，只通过编译宏选择 `UB_CTP` 或 `UB_MEM`。

`run_ab.sh` 默认以 8 rank、每 peer 256 MiB payload、16 MiB WQE、5 次 warmup、
20 次测量运行两个二进制，并输出 `results-ab/summary.txt`：

```bash
PAYLOAD_BYTES=$((256*1024*1024)) \
CHUNK_BYTES=$((16*1024*1024)) \
ITERATIONS=20 WARMUP=5 WORLD_SIZE=8 ./run_ab.sh
```

如果目标环境内存或队列深度不足，优先降低 payload，不要降低 rank 数；8 rank
是本次对比的最小有效规模。若 16 MiB WQE 触发队列限制，可按 8/4/2 MiB 阶梯
下降，并记录实际配置。

### 4.1 推荐的独立 AB 测试方式

为了比较通信栈本身，`benchmarks/netlayer_ab` 做了如下约束：

- 不使用 DeepEP、PyTorch、NumPy 或 pip 依赖；
- host 侧直接调用 ACL/HCCL/HCOMM；
- 每个 rank 注册同一个 payload MR；
- 按 `--layer-index 0/1` 选择 rank graph layer；
- 每个 peer 创建独立 AIV channel，避免共享 Jetty；
- device 侧直接编码 64B URMA WRITE WQE/SGE；
- 每 peer 发送 256 MiB，默认 16 MiB 一个 WQE；
- host 前后 barrier，使用 steady_clock 计时；
- 校验 CQE status 和远端 rank marker。

### 4.2 性能预期

layer 1 如果具备每 AIV 独立 SQ，其优势主要在高并发小/中请求场景：

- 多 producer 同时提交，不需要共享队列调度；
- WQE 编码和 CQE drain 可以分布在多个 AIV；
- 与 dispatch/combine 的 token 分片天然匹配；
- 低延迟路径受 host launch 和单队列串行化影响更小。

layer 0 的独立 channel 也能达到一定吞吐，但当前单 producer 基准只回答
“链路和 WQE 数据面是否可用、大请求带宽是多少”。它不能证明 layer 0 可以
复刻官方 64 AIV 并发模型。若共享 Jetty 是唯一资源，则高并发场景需要全局
producer 或严格串行化，软件提交开销和长尾都会上升。

大 payload 单流测试可能显示两者差距不大，因为瓶颈在链路带宽而不是队列并
发；小请求和高并发测试更能暴露 netlayer 1 的结构优势。因此 A/B 报告应固定
多个 payload/chunk 组合，而不是只看一个 64 MiB case。

## 5. AB 测试建议矩阵

| 场景 | 目的 |
| --- | --- |
| 64 MiB/peer, 4 MiB/WQE | 基线大请求吞吐 |
| 256 MiB/peer, 16 MiB/WQE | 默认饱和带宽 case，8 rank all-to-all |
| 128 MiB/peer, 8 MiB/WQE | 饱和链路，观察是否达到平台上限 |
| 16 MiB/peer, 1 MiB/WQE | 请求频率和 CQ 压力 |
| 8 MiB/peer, 256 KiB/WQE | 小请求开销和队列深度影响 |
| 多 channel / 多 AIV 扩展版 | 验证并发提交模型，后续在基线稳定后增加 |

初始版本先使用一个 AIV 和每 peer 一个 channel，是为了保证 netlayer 0/1
代码路径一致并先跑通功能。若该 case 没有达到链路上限，应优先增加请求大小
和 payload，再扩展 producer 数量。

## 6. 风险和解释规则

1. 不能把 layer 0 共享 Jetty correctness 失败解释为链路带宽不足；它是队列
   所有权和并发契约问题。
2. 不能用不同 CANN/HCOMM 版本的结果直接比较；ABI 和资源策略可能变化。
3. 不能跨 NPU 比较绝对 `GetSystemCycle()` 值；cycle 原点不同，host wall
   clock 才是跨 rank 主指标。
4. `send_sum_gibps` 是发送侧逻辑带宽求和，可能高于物理链路；评估 all-to-all
   关键路径时还要看最慢 rank envelope。
5. 正确性验证通过是性能数据有效的前置条件。

## 7. 当前结论

layer 0 和 layer 1 的核心差异不是 endpoint 数组下标，而是“队列资源是否支
持每 AIV 独占”。官方 layer 1 路径依赖这一能力实现多 producer 并发。当前
NPU8P 的 layer 0 适配如果只提供共享 Jetty，就与官方模型结构性冲突。独立
channel AB 基准可以回答两条 layer 的基础带宽和单 producer 延迟差异；若要
评估 DeepEP 真实 workload，还需要后续多 AIV/多 channel 版本。

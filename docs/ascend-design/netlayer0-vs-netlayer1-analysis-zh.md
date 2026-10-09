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

## 3. 两套软件实现的差异

本节比较的是两类已经存在或已经验证的软件形态：

1. **官方 DeepEP-Ascend netlayer1 实现**：EP kernel 直接拥有 Jetty/SQ，多个
   AIV 同时生成并提交 WQE。
2. **本仓库 netlayer0 通信实现**：业务 kernel 只生成 transport command，由
   单个 transport service 统一解析 channel、提交 WQE 和 drain CQ。

二者不是“同一个 kernel 换一个 layer 参数”的关系，而是 producer/consumer
所有权模型完全不同。

### 3.0 官方 DeepEP-Ascend netlayer1：EP kernel 直接提交

官方实现的初始化流程可以概括为：

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

这里的 "shared queue" 是 Jetty 维度上的共享：一个 AIV 的 Jetty 可服务多个
peer，但 **不同 AIV 不共享同一个 Jetty**。因此每个 AIV 仍然有独立 SQ/CQ。

```text
Official netlayer1 ownership

             EP dispatch/combine kernel
             +--------------------------------------+
             | AIV0   AIV1   AIV2 ... AIV63          |
             +--|------|------|---------|-----------+
                |      |      |         |
                v      v      v         v
             Jetty0 Jetty1 Jetty2 ... Jetty63
                |      |      |         |
                | SQ0  | SQ1  | SQ2 ...| SQ63
                v      v      v         v
             UBC_CTP/URMA network
                |      |      |         |
                v      v      v         v
             peer0 peer1 peer2 ... peerN
```

官方 kernel 中的关键模型是：

```text
HcommJettyInfo::load_jetty_info(..., vec_core_idx, lane_idx)
HcommJetty jetty(..., vec_core_idx)
```

即 AIV index 决定 Jetty/SQ。每个 AIV 使用自己的 SQE slot 和 head 推进，最后
独立 ring doorbell。这使 dispatch/combine 可以把不同 token/expert 的发送
工作分发到多个 AIV，通信提交和计算重叠。

### 3.1 本仓库 netlayer0：单 transport service 提交

本仓库生产实现没有把 EP kernel 和 WQE 提交耦合在一起。它分成三层：

1. **业务 AICore kernel**：只写 `TransportCommand`，例如 `Put`、
   `Signal`、`RemoteAdd64`、`Flush`、`Barrier`；
2. **transport service**：一个 executor 顺序消费命令；
3. **HCOMM/URMA channel**：按 `peer * channel_count + channel` 选择独立
   channel，然后提交 WQE 并 drain。

```text
Repository netlayer0 ownership

             dispatch/combine/business kernels
             +--------------------------------------+
             | AIV0   AIV1   AIV2 ... AIV63          |
             |  |      |      |         |            |
             |  +------v------+---------+---+        |
             |        TransportCommandQueue          |
             +------------------|-------------------+
                                |
                                v
                     single transport executor
                     (exactly one lane posts/drains)
                                |
                     +----------+-----------+
                     |          |           |
                     v          v           v
                  peer0/ch0  peer1/ch0 ... peer7/ch0
                     |          |           |
                     v          v           v
                  UB_CTP independent channels
```

在 AICore 原生路径中，service 内部直接解析 channel 的 SQ/CQ context，按
队列深度计算 slot/owner bit，用 MTE3 copy 发布 WQE，再更新 head 和 doorbell。
在 official SIMT 路径中，service 调 HCOMM 的 `WriteNbi`、`AtomicFAA`、
`Drain`。两种后端都保持同一个契约：**一个 service invocation 内只有一个
producer 拥有所有 channel**。

这个设计与共享 Jetty 的串行契约兼容，同时保留了对多 peer 并发通信的抽象。
它并发的是多个 peer/channel，而不是多个 WQE producer。

### 3.2 Device channel 表布局差异

```text
Official netlayer1 table

 index      0 ... 63                64 ... 64+P-1
        +--------------+          +----------------+
        | Jetty/AIV 0  | ...      | peer channels  |
        +--------------+          +----------------+
        kernel uses vec_core_idx to select producer queue
        kernel uses peer offset to load peer metadata


Repository netlayer0 table

 peer             0       1       2       3   ...   7
 channel       +-------+-------+-------+-------+-----+
        0      | p0,c0 | p1,c0 | p2,c0 | p3,c0 | ... |
               +-------+-------+-------+-------+-----+
        1      | p0,c1 | p1,c1 | p2,c1 | p3,c1 | ... |
               +-------+-------+-------+-------+-----+

 address = peer * channel_count + channel
```

官方表按“producer/Jetty”和“peer metadata”分块；本仓库表是二维
peer/channel 矩阵。若把官方 netlayer1 的表布局直接改造成
`[shared jetty, peer0, peer1, ...]`，只是改了索引，并没有解决 SQ 所有权问题。

### 3.3 WQE 提交时序差异

```text
Official netlayer1 multi-producer

 AIV0: load own Jetty head -> build SQE -> write SQ -> ring DB
 AIV1: load own Jetty head -> build SQE -> write SQ -> ring DB
 ...
 AIV63: load own Jetty head -> build SQE -> write SQ -> ring DB
 barrier: every AIV drains its own CQ


Repository netlayer0 single service

 producers: append TransportCommand (no WQE, no doorbell)
 service: acquire queue ownership
 service: for each command:
            resolve peer/channel
            resolve local/remote MR
            write WQE
            update head + doorbell
          flush/barrier:
            drain CQ
            update tails
 service: publish completion
```

官方模型中 WQE 构造和 doorbell 分布在 64 个 AIV 上；本仓库模型中这些动作被
集中到一个 executor。业务 kernel 与通信提交的 overlap 依赖 command queue 的
异步性，而不是多个 WQE producer 同时写 SQ。

### 3.4 为什么共享 Jetty 不能靠全局 SQE offset 修复

layer 0 诊断曾尝试把 64 个 AIV 全部指向共享 Jetty index 0，并给每个 AIV 分配
全局唯一 SQE offset。该方案能编译，但会触发 vector core exception 341
（VEC 访问 UB 越界）。

这说明官方 dispatch 的多 AIV 路径除了物理 SQE slot 之外，还隐含依赖
“每个 AIV 独立加载自己的 Jetty info 和 head”。当所有 AIV 共享一个 Jetty 时，
WQE 构造、head snapshot、owner bit、CQ tail 和 drain 语义都会交织。CANN
接口契约也明确要求共享 queue 的调用者串行化。

因此对 netlayer0 更稳的适配方向不是继续调共享 SQ offset，而是参考本仓库：
把 WQE 提交收敛到一个 service producer。

### 3.5 已验证证据

本仓库的多 channel 验证已经证明 layer0 的独立 channel 模型可用：

| 每 peer channel 数 | 每 rank SQ/CQ 套数 | 实际使用队列数 | 结论 |
| ---: | ---: | ---: | --- |
| 1 | 7 | 7 | 默认稳定路径 |
| 2 | 14 | 14 | 可用，端到端收益不稳定 |
| 4 | 28 | 28 | 可用，典型 case 性能退化 |

诊断同时验证了每条队列的 SQ 提交数与 CQ 完成数相等。这说明“单 service
producer + 独立 per-peer channel”在 NPU8P layer0 上是 correctness 已验证的
通信模型。

官方 netlayer1 性能数据基于 Ascend 950DT、CANN 9.2.0 和特定 PoC HDK，README
标明 EP8 dispatch 约 373–375 GB/s、combine 约 345–347 GB/s。这些数据不能
直接换算到当前 NPU8P/CANN 9.3 环境，只能作为官方目标环境的参考。

#### 官方 EP8 性能统计口径

官方 README 对上述数值的说明是：**Bandwidth ranges cover all ranks, using
timings that include issue and drain but exclude final epilogues**。对应实现
不是完整 API 端到端计时，而是 `tests/ep/test_ep.py` 中的 profiler 采样：

1. 先设置 `set_barrier_in_prologue(True)`，用 `bench_msprof` 对
   `dispatch_impl` / `combine_impl` 采通信 kernel，对
   `dispatch_copy_epilogue_impl` / `combine_reduce_epilogue_impl` 采最终
   epilogue；
2. 再设置 `set_barrier_in_prologue(False)`，单独采前一个 kernel，用于报告
   barrier 被移到 epilogue 时的 prologue 侧时间；它不是输出 URMA 带宽的
   那次采样；
3. 输出的 URMA GB/s 使用通信 kernel 的 `dur_ns`，而不是 API wall time 或
   NPU Event 端到端时间；
4. `dur_ns` 来自 Torch-NPU/FFTS profiler 的 kernel 开始/结束时间，并在
   50 个采样上按 kernel 求平均；
5. Dispatch 的字节公式是
   `num_recv_tokens * count_bytes(x, topk_idx, topk_weights) / num_tokens`；
   Combine 是 `num_recv_tokens * count_bytes(input_for_combine) /
   input_for_combine.size(0)`。它们是逻辑 URMA 字节，不是完整 API 逻辑字节，
   也不是物理链路字节。

因此官方 EP8 350–375 GB/s 只覆盖“通信 issue + drain + 通信 kernel 排队/执行”
的边界，并明确排除最终 epilogue。与本仓库正式 benchmark 的 max-rank NPU
Event 端到端口径不能直接互比；与本仓库仅统计 producer/release kernel 的
诊断口径也不能直接互比。

### 3.6 差异总表

| 维度 | 官方 netlayer1 | 本仓库 netlayer0 |
| --- | --- | --- |
| layer | `layers[1]` | `layers[0]` |
| 地址发现 | peer `UB_MEM` channel | `HcclChannelGetRemoteMems` |
| 发送协议 | `UBC_CTP` Jetty/URMA | `UB_CTP` independent channel |
| 队列所有权 | 每 AIV 一个 Jetty/SQ | 单 service producer |
| EP kernel 职责 | 直接构造和提交 WQE | 只生成 TransportCommand |
| channel 表 | AIV Jetty 区 + peer 区 | `peer * channel_count + channel` |
| 并发位置 | 多 producer | 多 peer/channel，单 producer |
| CQ drain | 每 AIV/队列独立推进 | service 统一 drain |
| 共享 queue 语义 | Jetty 内共享，跨 AIV独立 | service 串行拥有所有 channel |
| 已验证规模 | 官方 950DT 环境 | NPU8P 8 rank，1/2/4 channel |

### 3.7 对 netlayer0 打通的架构建议

1. 保留官方 EP kernel 的 dispatch/combine 语义；
2. 把直接 WQE 写入替换成 TransportCommand 生成；
3. 引入单 producer transport service；
4. host 初始化采用本仓库模型：每 peer 独立 `UB_CTP` channel，默认 1 channel；
5. device 表采用 `peer * channel_count + channel`；
6. 先跑通 2 rank smoke，再扩展 8 rank；
7. correctness 通过后再评估 2 channel；默认不启用 4 channel，因为本仓库实测
   其在典型 case 中退化。

这个方案牺牲官方 netlayer1 的多 WQE producer 并行性，但符合 netlayer0 的队列
所有权约束，并且有本仓库生产实现的 correctness 证据。

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

# DeepEP-Ascend：SIMT 生成，AICore 提交的 URMA 通信路径

日期：2026-09-25
状态：公众号正文初稿
配套大纲：`docs/ascend-design/epv2-ascend-simt-urma-article-draft-outline-zh.md`

## 阅读说明

本文面向正在做 MoE、分布式通信或 Ascend 后端适配的工程师，介绍 DeepEP-Ascend 的设备侧通信执行层：SIMT 在算子内部生成通信命令，AICore service 在服务边界构造 URMA WQE 并驱动 SQ/CQ。

阅读前建议先了解三个概念：

- Host / Device 执行模型：Host 负责 Python/C++ 控制流、资源初始化和 kernel launch，Device 上的 AICore/AIV 执行算子；
- 专家并行（Expert Parallelism，EP）：不同 expert 分布在不同 rank 上，token 需要在 rank 之间流动；
- 对称窗口（symmetric window）：各 rank 按同一规则注册一段可远端访问的内存，发送端可以用“本端地址减本端 base 得到的 offset”定位远端地址。

需要先划清一个边界：本文讲的不是官方 SHMEM 或 AIV 直驱 URMA 路径，而是 DeepEP-Ascend 当前采用的 **SIMT-fronted staged transport**。SIMT 不直接敲 SQ/CQ doorbell，它生成固定格式的通信命令；AICore service 在 VF 返回后统一解释命令、构造 WQE、发布 SQE 并处理完成队列。这个区别会贯穿全文。

> 一句话概括：SIMT 负责“我要发给谁、写到哪、发多少”；AICore service 负责“怎么把一批通信意图高效提交给 URMA 队列”。

```text
【图 1：主架构图】
dispatch / combine operator
        |
DeviceTransportFacade
        |
SIMT command encoder
        |
GM command buffer
        |
AICore transport service
        |
URMA WQE / SQ / CQ / doorbell
        |
HCCL / HCOMM team, window, channel
```

## 1. 要解决的问题

MoE 的通信看起来是 all-to-all，但它和规则 collective 不是一类问题。Router 每一步都会产生新的 top-k 结果，同一个 batch 里的 token 可能被送往不同 expert；这些 expert 又分布在不同 rank 上。于是每个 token 的目标 peer、目标 expert、目标 offset 和 payload 长度都可能不同。

这带来三个直接后果：

1. **动态**：通信意图由路由结果决定，无法在运行前完全展开成固定通信计划；
2. **稀疏**：每个 rank 只和一部分 peer 有真实数据交互，通信矩阵不是满密度；
3. **不均衡**：热门 expert 会集中在少数 rank 上，尾部 rank 的完成时间会拉长整个操作。

传统 Host 调度通信的方式会遇到控制面成本。Device 算子先算出路由元数据，回到 Host 由 Host 发起通信，再等通信完成后继续 Device 后处理。每一轮都有 launch、同步和状态搬运开销。更麻烦的是，路由信息本来就在 Device 上产生，来回转换只会增加延迟和出错面。

DeepEP-Ascend 的思路是把这段“路由结果到通信意图”的转换留在 Device 侧。SIMT producer 在算子内部读路由结果，完成 peer 选择、rank 翻译、offset 计算和边界判断，然后把结果写成 `put`、`put_value`、`remote_add`、`flush` 这类固定格式命令。AICore service 随后把这批命令翻译成 URMA 队列操作。

```text
【图 2：MoE 路由示意图】
token 0 ──> expert 19 / rank 2
token 1 ──> expert 88 / rank 0
token 2 ──> expert 7  / rank 3
...
同一个 batch 内，每个 token 的目标都由 router 结果决定
```

这个设计和“AIV 直驱 URMA”有什么区别？后者让 AIV 在调用点内直接构造并提交 WQE；本文的 staged transport 则是 SIMT 先写命令，AICore service 统一提交。两者都是 Device 侧通信，但数据面调度模型不同。staged transport 的收益不在“少一层就一定更快”，而在把 DeepEP 需要的动态路由控制、完成协议和错误处理放在一个可控的执行边界上。

## 2. 术语定义

| 术语 | 含义 | 本文中的作用 |
| --- | --- | --- |
| Host | CPU 侧控制程序 | 初始化 communicator、注册内存、启动 kernel |
| Device | NPU 侧执行环境 | 执行算子、生成和提交通信 |
| AICore / AIV | 昇腾 AI Core 与 Vector 执行路径 | 承接数据并行搬运、SIMT producer 和 AICore service |
| SIMT | 单指令多线程执行模式 | 靠近 token 路由元数据，生成通信命令 |
| TransportCommand | DeepEP 自定义的固定格式命令 | 在 SIMT 和 AICore service 之间传递通信意图 |
| GM | Global Memory | 存放 command buffer、payload 和控制状态 |
| UB / UB_MEM | Unified Buffer 及相关快速存储 | service 阶段组织 SQE、SGE 和临时 descriptor |
| HCCL / HCOMM | 昇腾通信库及其内部通信实现 | 提供 communicator、链路、内存注册和队列资源 |
| Team | 通信参与者集合 | 描述 world、scale-up、scale-out 等逻辑域 |
| Window | 对称可远端访问内存 | 提供 offset 到远端地址的映射 |
| Channel | Team 内的通信 lane | 绑定 SQ/CQ、token 和链路上下文 |
| URMA | Unified Remote Memory Access | 将远端读写表示为硬件可执行操作 |
| WQE / CQE | 工作队列项 / 完成队列项 | 描述一次远端操作及其完成状态 |
| SQ / CQ | 提交队列 / 完成队列 | 承载 WQE 提交与 CQE 完成 |
| doorbell | 通知硬件队列有新工作的机制 | 由 AICore service 用 `st_dev` 写入 |

## 3. 总体架构：从 operator 到 URMA

整个通信路径分成六层：

1. **dispatch / combine operator**：只关心业务语义，“这个 token 应该去哪个 expert，这个 expert 输出应该回到哪个 token”；
2. **DeviceTransportFacade**：向算子暴露 `put`、`put_value`、`remote_add`、`signal`、`flush` 等通信原语；
3. **SIMT command encoder**：把 facade 参数翻译成固定大小、trivially-copyable 的 `TransportCommand`；
4. **GM command buffer**：backend 拥有的命令队列，SIMT 发布命令，AICore service 消费；
5. **AICore transport service**：校验命令、解析 team/window/channel、构造 WQE/SGE、写 SQ、ring doorbell、poll CQ；
6. **HCCL/HCOMM 资源层**：继续拥有连接建立、内存注册、token、队列分配和 teardown。

用一个 `put` 的生命周期串起来：

```text
【图 3：put 生命周期】
operator 调用 facade.put(...)
    -> SIMT 校验 team、peer、channel 和地址
    -> 写 TransportCommand 到 GM command buffer
    -> 发布 count，VF 返回
    -> AICore service 读取 command
    -> 解析远端 window 地址和 channel
    -> 构造 SQE/SGE
    -> 写 SQ 并用 st_dev ring SQ doorbell
    -> 等待 CQE 完成
    -> 更新 CQ consumer 和 doorbell
```

这个分层的价值是让职责清楚。算子不感知 SQE 字段，SIMT 不直接操作 doorbell，AICore service 不理解 MoE 路由，CANN 继续负责连接和注册资源。任何一层出问题，都能在明确边界上定位。

## 4. 为什么是 staged transport

最直接的回答：当前路径下，SIMT 不能安全地直接执行完整 URMA doorbell 提交。

当前实现以 CANN 9.3.0 为基线。在这个执行模型里，DeepEP 的通信 producer 是 `__simt_vf__`，而 URMA SQ/CQ 提交所需的 `st_dev` / `ld_dev` 属于 AICore 执行域，不能从 SIMT VF 安全调用；`__stg` 虽然能在 SIMT 中编译，但它和 `st_dev` 的设备语义不同，不能当作 doorbell 的替代。CANN 的 HCOMM、PTO URMA 和 MoE 通信参考实现也都使用 `st_dev` 处理 SQ/CQ doorbell，没有参考实现支持用普通 GM store 替代。

因此 DeepEP-Ascend 选择了 staged transport：

```text
SIMT producer
  -> 追加 TransportCommand 到 GM command queue
  -> VF 返回
  -> AICore service 逐条解释命令
  -> 构造 URMA WQE / SQ / CQ 操作
  -> 提交、drain、发布完成
```

它的代价是明确的：多了一层命令编码、一次 GM 命令发布和一次 service 解析。它的好处同样是明确的：

- SIMT 能在离路由元数据最近的位置生成通信意图；
- WQE 构造、队列提交、doorbell 和 CQ 处理集中在 AICore service；
- `flush`、`barrier`、generation、错误诊断可以按 DeepEP 的协议实现；
- 未来如果换成更直接的 SIMT 提交路径，替换边界也在这里。

### 4.1 看一眼命令 ABI

`TransportCommand` 是 128 字节、64 字节对齐的 POD：

```cpp
enum class TransportCommandOpcode : std::uint32_t {
    kNone,
    kPut,
    kPutValue64,
    kRemoteAdd64,
    kSignal,
    kFlush,
    kBarrier,
};

struct alignas(64) TransportCommand {
    TransportCommandOpcode opcode;
    TransportTeam team;
    CooperationScope scope;
    MemorySegment segment;
    RemoteActionKind action_kind;

    int32_t peer;
    uint32_t channel;
    DeviceOptions options;
    uint32_t value_bytes;
    uint32_t signal_index;
    int32_t world_peer;

    DeviceAddress source;
    DeviceAddress destination;
    uint64_t bytes;
    uint64_t value;
    uint64_t symmetric_offset;
    uint64_t timeout_cycles;
    uint64_t reserved1[6];
};
```

它刻意不包含 HCOMM 内部类型，也不包含 SQE 字段。SIMT 只写“语义”：目标 team、目标 peer、channel、源地址、目标 offset、字节数和可选动作。service 再把这些语义翻译成队列操作。

这种 ABI 带来两个工程收益：

1. **边界稳定**：CANN 内部类结构变化不会直接穿透到 DeepEP 算子；
2. **可测试**：host probe 可以检查结构尺寸、对齐、字段 offset 和 trivially-copyable 属性，不匹配就失败。

不要把它理解成“SIMT 全部替代 AICore”。两者是分工：SIMT 处理动态分支，AICore service 处理批量提交吞吐。

| 工作类型 | 更适合的执行路径 |
| --- | --- |
| 规则、大块、同构的数据搬运 | SIMD/AICore |
| 动态 peer 选择和分支 | SIMT |
| 每个 token 目标不同 | SIMT |
| 批量组织 SQE、doorbell、CQ 轮询 | AICore service |
| 端到端通信与路由融合 | SIMT 生成 + AICore 提交 |

```text
【图 4：直接 SIMT 提交 vs staged transport】
直接提交路径：
SIMT -> 构造 WQE -> 写 SQ -> ring doorbell -> poll CQ

staged transport 路径：
SIMT -> 写 TransportCommand
       -> AICore service
       -> 构造 WQE -> 写 SQ -> ring doorbell -> poll CQ
```

## 5. 控制面与数据面

一个容易产生的误解是：“既然用了 UB_MEM，是不是所有 token 都要先复制到 UB？”

不是。DeepEP-Ascend 把控制面和数据面分开了。

**UB_MEM 主要服务通信控制面**：

- 聚合 SIMT 生成的固定格式命令；
- 组织 SQE、SGE 和临时 descriptor；
- 保存 service 阶段快速访问的状态；
- 配合 `st_dev` / `ld_dev` 完成 SQ/CQ 和 doorbell 操作。

**真正的 payload 仍走 symmetric window 和 URMA**：

```text
【图 5：控制面 / 数据面分离】
控制面：TransportCommand / SQE / SGE / CQE / signal
    主要经过 GM command buffer 和 UB service 工作区

数据面：token payload
    从本地注册内存出发
    通过 URMA 写入 peer symmetric window
```

远端地址的生成规则也很简单：

```text
offset = local_operand - local_window_base
remote_address = peer_window_base + offset
```

发送端不需要知道远端物理地址，只需要维护本端 operand 在 symmetric window 里的 offset。service 会读取 peer window 表，把 offset 转成合法 URMA 目标地址，并校验边界。

这种设计的意义在于：控制面的动态性由 SIMT 承接，数据面的大块搬运交给 URMA。两者不互相拖累。

## 6. 资源模型：Team、Window、Channel

DeepEP-Ascend 用三层资源模型组织通信：

```text
communicator
    -> team：参与者集合、rank 和拓扑域
        -> window：各 rank 对称布局的可远端访问内存
            -> channel：Team 内的并行通信 lane
                -> SQ/CQ、doorbell 和底层通信队列
```

**Team** 定义通信域。DeepEP 的 facade 中有 `world`、`scale_up`、`scale_out` 三个逻辑 team，算子可以按拓扑选择通信域，而不需要理解 HCCS、PCIe 或 RoCE 的具体差异。

**Window** 提供远端寻址基础。所有 rank 按同一规则注册对称内存后，发送端只需要 offset，不需要在 kernel 里维护每张卡的远端物理地址表。

**Channel** 提供并行执行资源。一个 peer 可以有多个 channel，每个 channel 绑定自己的队列和 token。service 会按 team 的 channel table 和 per-member channel counts 解析具体队列。

Jetty 如果在底层实现中出现，它属于 HCOMM/URMA 的 endpoint/队列实现细节，不是 DeepEP operator facade 的一级抽象。算子开发者不需要理解它。

上层算子最终只描述一件事：“向哪个 peer 写哪段数据”。team 翻译、window 解析、channel 索引、token 校验和队列生命周期都留在 transport 层。

## 7. dispatch / combine 如何接入

`DeviceTransportFacade` 屏蔽了这些细节：

- HCCL communicator 的复用和校验；
- team / window / channel 的创建与销毁；
- world rank 与 team-local peer 的翻译；
- symmetric window 地址解析；
- channel table 索引和有效性检查；
- signal、generation、flush 和 barrier 语义；
- CANN device-visible ABI 的结构体布局。

对 dispatch 来说，producer 侧根据路由结果分组 token，生成 record 和远端 payload 命令；release 阶段先保证 payload 可见，再发布 count、generation 和 signal。consumer 侧等待所有 source ready，再 validate record、统计 expert 数量、分配输出并 copy。

对 combine 来说，producer 侧把各 expert 输出写回目标 rank；本 rank 目标可以直接落位，远端 rank 仍通过 staging 和 HCOMM put 发送。consumer 侧 acquire 并 validate contributor slot，再按 source token reduce、加权重和完成输出。

```text
【图 6：Dispatch/Combine 调用链】
Dispatch:
  routing / grouping
    -> TransportCommand (put, signal, flush)
    -> AICore service
    -> URMA payload + control publication
    -> consumer acquire / validate / copy

Combine:
  expert outputs
    -> TransportCommand (put, signal, flush)
    -> AICore service
    -> URMA payload + control publication
    -> acquire / validate / reduce / weights
```

capability bit 在这里起关键作用。代码里存在某个接口，不等于生产可用；编译通过也不等于端到端语义正确。DeepEP-Ascend 只有在对应的多 rank 语义测试通过后，才开启对应 capability。这是防止“看起来支持，实际边界不清”的机制。

## 8. 工程落地路径

DeepEP-Ascend 的适配没有从零重写通信栈，而是按一条可验证的路径推进：

1. **复用已有 HCCL communicator**：进程启动后校验 rank 和 world size，不额外建立一套进程组；
2. **定义最小 device-visible POD ABI**：只定义 SQE、SGE、CQE、team、window、channel 等必要布局，不引入 HCOMM 内部 C++ 类体系；
3. **用 ABI probe 校验结构**：检查尺寸、对齐和字段 offset，不匹配就构建或初始化失败，不做 best-effort 继续；
4. **先 stub，后真实资源**：先在 stub transport 上验证 facade 语义，再切换到真实 HCCL/HCOMM 资源；
5. **扩展命令而不是重写生命周期**：新增通信操作主要扩展 command opcode 和 service handler；
6. **保持 CUDA/NCCL 路径独立**：Ascend 后端不反向污染 CUDA 实现。

```text
【图 7：工程推进路径】
facade contract
    -> stub transport
    -> ABI probe
    -> host resource probe
    -> staged AICore service
    -> dispatch / combine integration
    -> multi-rank acceptance
```

这套方法的重点不是“代码行数少”，而是依赖边界清楚、可测试、可替换。CANN 9.3.0 的 communication-domain 路径负责 rank graph、内存注册、AIV channel 和远端 MR 查询，DeepEP 在 host transport 层把这些资源整理成设备可见的 Team / Window / Channel 表，再让 device facade 保持稳定。

## 9. 性能验证口径

性能部分是这篇文章最需要克制的地方。没有同版本、同 workload、同配置的数据，就不要宣称自己优于某个基线。

建议至少覆盖三类 workload：

1. **大块、规则、均匀通信**：验证基础搬运能力；
2. **小包、多 peer、动态路由**：观察控制面收益；
3. **长尾 expert 负载**：观察完成语义和尾部 rank 表现。

建议报告的指标包括：

- dispatch / combine 端到端 latency；
- p50 / p95 / p99；
- router 结束到第一批通信命令提交的启动时间；
- CPU launch 和 host synchronization 次数；
- command generation 和 service submission 时间；
- 不同 top-k、peer 数、payload size 下的有效带宽；
- 通信与排序、GEMM 等计算的重叠收益；
- channel 数从 1 到 2/4 的收益和资源成本。

每张性能图都必须附带条件：

| 条件 | 示例 |
| --- | --- |
| 硬件 | Ascend 950，单机 8 NPU |
| CANN / HCOMM | 9.3.0，并注明 HCOMM 包来源 |
| workload | tokens、hidden、top-k、experts、dtype |
| 路由 | balanced / unbalanced / manifest |
| channel | 1 / 2 / 4 |
| 协议 | warmup、sample 数、seed |
| 路径 | direct / hybrid、sync / async |

一个更稳健的结论形式是：

> 在动态稀疏通信中，SIMT-fronted 路径减少了控制面转换，并为通信和路由计算融合提供了更自然的执行粒度；最终收益取决于消息规模、peer 分布、channel 数和 service 提交效率。

```text
【图 8：性能验证占位图】
横轴：workload / payload size / channel count
纵轴：end-to-end latency 与 P99
必须标注硬件、CANN、rank、top-k、warmup/sample 协议
```

## 10. 当前边界与后续演进

当前实现的主要验证边界：

- **硬件与软件**：Ascend 950，单机多 NPU，CANN/HCOMM 9.3.0；
- **执行模型**：SIMT-fronted staged transport，不是 SIMT 直接 doorbell；
- **生产能力**：以端到端验证和 capability bit 为准，不能只看代码存在；
- **跨机能力**：物理 RoCE scale-out 仍需单独验证，不能由逻辑多主机结果外推。

后续演进方向：

1. **SIMT command batching**：减少每条命令的固定发布开销；
2. **多 channel 调度**：按 payload 字节和 peer 分布自适应选择 channel 数；
3. **通信计算重叠**：把 release、acquire 和计算阶段进一步流水化；
4. **device-side completion**：完善 request 状态机和细粒度完成等待；
5. **跨机 URMA / RoCE**：扩展 topology、注册和完成模型；
6. **多 service worker**：提高命令消费和队列提交并发；
7. **评估官方 SIMT Hcomm 路径**：在单 lane、单 channel 的最小 probe 上做 A/B，再评估 producer 改造。

官方 asc-comm 已经提供了 SIMT 直接提交 Hcomm 的能力，历史阻塞点正在消失。但它要求一个 channel 由单个 lane 驱动，而 DeepEP 当前 producer 是多 block、多 lane 形态，不能简单替换。合理的路线是先做最小 probe，再评估是否为每个 channel 设计唯一提交 lane。

## 11. 小结

DeepEP-Ascend 这条通信路径的核心不是“用一条新指令替代旧通信库”，而是把 MoE 动态路由的通信控制放回 Device 侧：

- SIMT 在算子内部根据路由结果生成通信命令；
- AICore service 在明确边界上构造 URMA WQE 并驱动 SQ/CQ；
- symmetric window 承载 payload 数据面，UB_MEM 服务控制面；
- Team / Window / Channel 统一组织通信资源；
- dispatch / combine 通过同一个 facade 接入，不直接感知 HCOMM/URMA 细节。

这套设计当前的价值是可控和可验证：动态路由控制靠近数据产生位置，硬件提交集中在 service 边界，完成协议和错误处理有明确归属。它的性能收益需要按 workload 条件验证，而不是用一句“更快”概括。

后续我们会继续补三件事：多 channel 的功能和性能 A/B、通信计算重叠的 pipeline 边界，以及跨机 RoCE 路径。性能数据齐了以后，再单独写一篇验证专题。

## 参考入口

- `csrc/backends/ascend/transport/device_transport_facade.hpp`
- `csrc/backends/ascend/transport/device_transport_commands.hpp`
- `csrc/backends/ascend/transport/aicore_transport_service.hpp`
- `csrc/backends/ascend/transport/cann_transport.cpp`
- `docs/ascend-design/epv2-ascend-simt-urma-transport.md`
- `docs/ascend-design/deep-ep-ascend-communication-implementation-review-zh.md`
- `docs/ascend-design/asc-comm-official-simt-comparison-zh.md`

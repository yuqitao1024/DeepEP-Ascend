# SIMT 与 AICore Main Scalar 执行分工

日期：2026-10-10。本文记录当前 DeepEP-Ascend 默认 staged transport 中
SIMT VF、AICore scalar 控制面和 MTE 数据面的分工。讨论范围只覆盖本仓库默认构建；
`DEEP_EP_ASCEND_OFFICIAL_SIMT` 是实验分支，本文单独说明，不把它混入默认结论。

## 1. 一句话结论

默认实现不是“纯 SIMT”，也不是“纯 AICore/Main Scalar”。它是一个分层模型：

- **SIMT producer**：负责路由、计数、记录构造、局部数据处理，并把通信意图编码为
  `TransportCommand`，追加到 device global memory 的 command queue；
- **AICore transport service**：作为唯一 consumer，顺序解释 command，解析
  peer/channel/MR，构造 URMA WQE；
- **AICore scalar + MTE 混合提交**：WQE 字段由 scalar 逻辑组装，WQE body 通过
  MTE3 从 UB 拷贝到 SQ；SQ head、doorbell、CQ tail 和完成状态由 scalar
  `ld_dev/st_dev` 类设备访问发布/观察；
- **SIMT/AICore epilogue**：接收侧 metadata、copy、reduce、validation 和 complete
  阶段继续大量使用 SIMT VF。

因此，“默认 AICore/URMA service” 只表示**通信命令的执行后端**不走 HCOMM SIMT
API；它并不表示 Dispatch/Combine 整体没有 SIMT。

## 2. 默认 Dispatch/Combine 的阶段分工

### 2.1 Dispatch

Dispatch 的典型路径可以概括为：

```text
SIMT producer plan / record / control
    -> SIMT producer release
       -> 追加 TransportCommand 到 GM queue
AICore service execute
    -> 校验 command / channel / MR
    -> scalar 组 WQE
    -> MTE3 写 SQ
    -> scalar 更新 head + doorbell
    -> scalar bypass-DCache 轮询 CQE
SIMT epilogue
    -> 计数 / metadata / prefix / copy / validate / complete
```

其中 producer release 本身是 `__simt_vf__` 函数，由外层
`__global__ __vector__` kernel 通过 `asc_vf_call` 调用。它内部构造
`DeviceTransportFacade`，再通过 facade 把 `Put`、`Signal`、`Flush` 等
通信意图写入 `TransportCommandQueue`。

Dispatch 中涉及 SIMT 的典型阶段包括：

- route plan / acquire route plan；
- producer plan；
- producer control；
- producer record；
- producer release；
- persistent producer / persistent release 系列；
- publish counts / publish route plan；
- epilogue acquire；
- epilogue assign destinations；
- epilogue count experts；
- epilogue metadata；
- epilogue prefix / parallel prefix；
- epilogue copy outputs；
- epilogue validate records；
- epilogue complete；
- reduce errors。

### 2.2 Combine

Combine 的结构类似：

```text
SIMT producer plan / record / control
    -> SIMT producer release / local copy
       -> 追加 TransportCommand
AICore service execute
    -> 提交 URMA WQE 并 drain
SIMT epilogue
    -> acquire / prepare / reduce / tail / weights / validate / complete
```

Combine 中涉及 SIMT 的典型阶段包括：

- producer plan；
- producer plan prefix；
- producer control；
- producer record；
- producer release；
- producer local copy；
- epilogue acquire；
- epilogue prepare vector slots；
- epilogue reduce；
- epilogue vector tail；
- epilogue weights；
- epilogue validate；
- epilogue clear index；
- epilogue complete；
- epilogue reduce errors。

`direct_combine_producer_release_vf` 与 Dispatch 的 release 类似，也是
`__simt_vf__`，外层通过 `asc_vf_call` 进入，再由 `DeviceTransportFacade`
追加通信命令。

### 2.3 Barrier

`barrier_producer_vf` 同样是 SIMT VF。外层 kernel 先执行 SIMT producer，
然后进入 `transport::service::execute()`。因此 barrier 也是
“SIMT command producer + AICore service”结构。

## 3. TransportCommandQueue 如何衔接 SIMT 与 service

核心数据结构：

- `TransportCommand`：128 字节，一条通信意图；
- `TransportCommandQueue`：64 字节队列头，包含 `capacity/count/generation`
  和 commands、service state、diagnostic 的地址；
- `TransportServiceState`：记录 `consumed_count`、`active`、
  `consumed_generation`；
- `StagedTransportContext`：挂在 `DeviceTransportContext.backend_context`，
  指向 command queue。

SIMT producer 的追加流程：

```text
读 queue.count
    -> 写 TransportCommand 到 commands + count
    -> 清空 service.consumed_generation
    -> system fence
    -> 发布 queue.count = count + 1
    -> system fence
```

AICore service 的消费流程：

```text
读取 queue.count
    -> 从 service.consumed_count 开始顺序遍历 command
    -> 校验 ABI / topology / peer / channel / MR / address
    -> 构造并提交 WQE
    -> 更新 service.consumed_count
    -> drain 完成后发布 consumed_generation = queue.generation
```

这个 queue 是顺序批次队列，不是通用 MPMC queue。当前实现依赖 staged producer
与 service boundary 的顺序约定，service 是唯一 consumer；如果未来允许多个
consumer/lane 并发消费，需要原子 claim、per-lane 进度、错误聚合和 barrier
coordinator 等新协议。

## 4. AICore/URMA service 内部的 Main Scalar 与 MTE 分工

AICore/URMA 原生路径不是纯 Main Scalar。它是 scalar 控制面加 MTE 数据面：

### 4.1 WQE 构造：scalar 逻辑

`urma::make_write`、`urma::make_inline_write64`、`urma::make_faa64` 等
helper 先在 AICore scalar 代码中构造 request/WQE 结构，并修补 SQ slot、
owner bit 等字段。随后 `wqe_scratch.SetValue(...)` 把 request 字段写入
UB scratch。

### 4.2 SQ 写入：MTE3 DataCopy

关键路径：

```text
S -> MTE3 event
for each 64B WQE basic block:
    AscendC::DataCopy(SQ global, UB scratch, 64B)
MTE3 -> S event
```

也就是说，WQE payload 不是逐字 scalar store 到 SQ，而是从 UB 通过 MTE3
批量拷贝。

### 4.3 SQ head / doorbell：scalar store

WQE 写完后：

```text
st_dev(peer.sq->head, packed position/request count)
st_dev(peer.sq->doorbell, position)
```

head 与 doorbell 的发布是 scalar device store。代码中的
`aicore::store_device` 最终映射到 `st_dev`。

### 4.4 CQ drain：scalar bypass-DCache 访问

CQ drain 的主要动作：

- 读取 SQ head / tail，计算 expected completion；
- 读取 CQ tail；
- 使用 bypass-DCache load 观察 CQE owner/status；
- 校验 status/substatus；
- 推进 CQ tail；
- 更新 CQ doorbell 和 SQ tail。

这些控制字访问由 scalar 完成。CQE 状态读取使用
`ReadGmByPassDCache`/`load_published`，避免 AICore DCache 观察到旧值。

### 4.5 准确描述

因此 AICore/URMA service 的准确描述是：

```text
scalar 控制面驱动 URMA 队列
+ MTE3 批量写 WQE/SQ
+ scalar 发布 head/doorbell/tail
+ scalar bypass-DCache 观察 CQE
```

不能把它称为纯 Main Scalar 实现。

## 5. Experimental Official SIMT 后端

`DEEP_EP_ASCEND_OFFICIAL_SIMT=ON` 时，service 后端切换到实验性的 HCOMM SIMT
executor。它并不是整条路径都变成 SIMT。

执行结构：

```text
AICore service shell
    -> 校验 staged context / queue ABI / command / channel
    -> asc_vf_call(official_simt::execute_vf, dim3(32))
       -> 只有 threadIdx.x == 0 的 lane 运行 Executor
       -> Hcomm::Init / WriteNbi / WriteValueNbi / AtomicFAA / Drain
    -> asc_sync_vec
    -> AICore system fence
```

所以该分支是：

- command contract 校验：AICore/scalar；
- 实际通信执行：单 lane SIMT；
- WQE/SQ/CQ 细节：不手工管理，交给 HCOMM SIMT API；
- profiling：没有 AICore/URMA 路径的细粒度 service/WQE cycle 拆分。

单 lane 的原因是协议和所有权约束，不是随意保守：

1. 当前 `TransportCommandQueue` 只有一个 `consumed_count` 与
   `active`，没有 per-lane claim/progress/error 协议；
2. executor 在一次 invocation 内独占所有 channel，保证 command 顺序、
   flush/barrier 语义和 diagnostic 首错归属；
3. HCOMM SIMT channel 的 SQ head/tail 和内部状态需要唯一 owner；
4. 多 lane 同时驱动同一 channel 可能造成 head 丢失或 doorbell 发布未完成
   WQE。

该实验分支与官方 DeepEP-Ascend direct-SIMT 数据面也不同。官方 EP kernel 会在
编译期把 SQE 构造分散到多个 lane/warp，并为每个 AIV 建立独立 Jetty/SQ；
本仓库实验分支只是把已有 command queue 顺序翻译成 HCOMM SIMT API。

## 6. 与官方 direct-SIMT 路径的边界

官方 DeepEP-Ascend 数据面是 producer 与通信提交耦合更紧的模型：

```text
官方：EP SIMT kernel
    -> 多 lane/warp 构造 SQE
    -> 每个 AIV 拥有 Jetty/SQ
    -> 直接写 SQ / ring doorbell / drain
```

本仓库默认模型是：

```text
本仓库：SIMT producer
    -> 只追加 TransportCommand
AICore service
    -> 统一组 WQE / 写 SQ / doorbell / drain CQ
```

因此两者差异不是“是否使用 SIMT”，而是：

- SIMT 是否直接提交通信；
- SQ/WQE 所有权是 per-AIV 还是集中 service；
- command queue 与 AICore service boundary 是否存在；
- profiling 的边界如何定义。

## 7. 结论表

| 层次 | 默认路径 | Experimental Official SIMT |
| --- | --- | --- |
| Dispatch route/plan/record/control | SIMT VF | SIMT VF |
| Dispatch/Combine producer release | SIMT VF | SIMT VF |
| Command 追加 | SIMT VF 写 GM command queue | SIMT VF 写 GM command queue |
| Command 校验 | AICore service | AICore service shell |
| WQE 构造 | AICore scalar | HCOMM SIMT API 内部 |
| SQ 写入 | AICore MTE3 | HCOMM SIMT API 内部 |
| head/doorbell/CQ tail | AICore scalar `st_dev` | HCOMM SIMT API 内部 |
| CQE 观察 | AICore scalar bypass-DCache | HCOMM `Drain` |
| Dispatch epilogue | SIMT VF 为主 | SIMT VF 为主 |
| Combine epilogue | SIMT VF 为主 | SIMT VF 为主 |
| HCOMM SIMT `WriteNbi` 等 API | 不使用 | 使用，单 lane executor |

最终结论：默认 DeepEP-Ascend 是 **SIMT command producer + AICore scalar/MTE
URMA service + SIMT epilogue** 的分层实现；AICore service 不是纯 Main Scalar，
而是 scalar 控制面与 MTE 数据面的混合。

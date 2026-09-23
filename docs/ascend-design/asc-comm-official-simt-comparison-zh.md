# asc-comm 官方 SIMT 数据面与当前实现对比

日期：2026-09-21

## 结论摘要

官方仓 [asc-comm](https://gitcode.com/cann/asc-comm) 已经提供了真正可从
`__simt_vf__` 直接调用的 Hcomm SIMT URMA 数据面。当前 DeepEP-Ascend
实现没有调用这层公开接口，而是保留了一条显式的 staged transport 路径：

```text
DeepEP SIMT producer
  -> 追加 TransportCommand 到 GM command queue
  -> VF 返回
  -> AICore service 逐条解释命令
  -> 自研 URMA WQE/SQ/CQ 代码提交和 drain
```

官方 SIMT 路径则是：

```text
DeepEP SIMT producer
  -> AscendC::simt::Hcomm::WriteNbi / AtomicFAA / ...
  -> lane-private WQE staging
  -> SQ 写入
  -> 128B DWQE doorbell
  -> Drain 消费 CQ
```

因此两者差异不是“同一实现的接口包装差异”，而是数据面调度模型不同。官方
版有机会减少 command queue 编码、VF/AICore service 边界、命令逐条解析和
AICore MTE 往返开销；尤其对小粒度、高 WQE 数的场景，理论上有明显收益。

但当前不能直接把 DeepEP 全量切换到官方 SIMT Hcomm：

1. 官方普通 SIMT 接口要求一个 channel 由单个 lane 驱动；延迟提交尤其不能
   由多个 lane 混写同一 channel，否则 doorbell PI 可能发布其他 lane 尚未
   写完的 WQE。
2. DeepEP 当前 producer 是多 block/多 lane 形态，不能简单把每个 transport
   facade 调用点替换成官方 Hcomm 调用。
3. 官方 SIMT 普通接口没有独立的 Commit 接口，只能通过后续 `commit=true`
   的任务发布整批。
4. `Drain` 是 channel 级完成，不是每个 request 的细粒度完成。
5. DeepEP 当前 `flush_async/wait`、barrier、signal 等语义建立在 staged
   command ABI 和 service state 上，需要重新映射。

所以合理的路线不是“直接替换”，而是先做一个官方 SIMT Hcomm backend 的
最小 probe/分支：单 lane producer、单 channel、只覆盖 put/put_value/
remote_add/flush 四个核心语义，与当前 staged backend 在同一 2 卡环境做
A/B。通过后再评估是否改造 producer 调度，让每个 channel 拥有唯一提交
lane，或者使用多 channel 分摊单 lane 提交瓶颈。

## 官方仓版本和核心文件

本次分析对象：

```text
repository  https://gitcode.com/cann/asc-comm
commit      9280cf6ea12a0cd7d00acc0709fe8834c669c3e1
version.cmake ASCCOMM_VERSION=9.2.0
target      Ascend 950PR/950DT, dav-3510
protocol    COMM_PROTOCOL_UB_CTP / URMA
```

关键文件：

| 职责 | 文件 |
| --- | --- |
| SIMT 公开接口 | `include/aicore/hcomm/hcomm_simt.h` |
| SIMT 模板转发 | `src/aicore/hcomm/impl/hcomm_simt_impl.h` |
| V310 SIMT URMA 实现 | `src/aicore/hcomm/impl/platform_v310/hcomm_simt_urma.h` |
| SQ 预留和 head/tail 编码 | `src/aicore/hcomm/common/hcomm_simt_utils.h` |
| 功能样例 | `examples/aicore/hcomm/04_simt_urma` |
| 性能样例 | `examples/aicore/hcomm/05_simt_urma_perftest` |

公开 SIMT 操作包括：

- `WriteNbi`
- `WriteValueNbi`
- `ReadNbi`
- `WriteWithNotifyNbi`
- `AtomicFAA`
- `AtomicCAS`
- `Drain`
- `LocalBufferAddr`
- `RemoteBufferAddr`

`Init(buff, len)` 只保留签名，当前实现不需要 workspace；WQE 在 lane 私有
staging 中构造。

## 官方实现的关键语义

### 1. 单 lane 驱动 channel

官方头文件明确要求每个接口由单个 lane 调用，且 Hcomm 对象是 lane-private
状态，不能在 lane 间共享。一个 channel 不能同时由多个 lane 驱动。

原因在于 SQ 预留不是原子操作，而是单 lane 的普通 read-modify-write：

```text
headAddr 低 32 位: curHead
headAddr 高 32 位: wqeCnt
提交时一次写入 head + delta
```

如果多 lane 驱动，head 更新会丢失，doorbell 也可能发布其他 lane 未完成的
WQE。

### 2. WQE 先进入 lane-private staging

官方实现先把 64B/128B WQE 写到 `HcommSimtWqeStage`，再复制到 SQ。立即
提交时，doorbell 从同一份 staged bytes 中复制，保证 SQ 和 DWQE 内容一致。

这对性能很重要：

- 避免在 GM 上构造半成品 WQE；
- 避免不同 lane 交叉写相邻 basic block；
- WQE 布局可完全展开和 `#pragma unroll`。

### 3. 立即提交必须占满 128B DWQE

官方规则：

- deferred Write/Read 占 1 个 64B basic block；
- immediate Write/Read 拆成两个 SGE，占 2 个 basic block；
- notify 和 atomic 固定占 2 个 basic block。

SIMT 没有独立 `Commit`。延迟任务停留在 SQ，直到后续 `commit=true` 的
任务通过 128B DWQE 发布覆盖整批的 producer index。

### 4. doorbell 写法更保守

官方 SIMT doorbell 使用：

```text
asc_threadfence()
asc_stwt(..., ulonglong2)  // 128-bit store
asc_dcci_single(dwqe)
asc_dcci_single(dwqe + 64)
```

注释明确说明：普通 64-bit store 曾被观察到 NIC 会消费 header/fetch SGE，
但丢失后面的 atomic operand words。

当前 DeepEP AICore service 用 `st_dev` 写 doorbell，同样是 CANN reference
认可的路径；两者硬件语义上等价，但编译域不同。

### 5. PollCq 直接读 CQE 原始字

官方 SIMT `PollCq` 直接读 32-bit CQE word，而不是通过 MTE2 把 CQE 拷到
UB。它还会根据 CQE 中的 completed slot 推进 `sqTail`。

当前 DeepEP `drain_channel` 在 AICore 中通过 MTE2 将 64B CQE 拷到 UB 再
检查 owner/status。这个路径更通用，但每次 CQE poll 都有同步和搬运开销。

## 当前 DeepEP-Ascend 实现模型

相关文件：

| 职责 | 文件 |
| --- | --- |
| SIMT facade 与命令编码 | `csrc/backends/ascend/transport/device_transport_commands.hpp` |
| AICore service | `csrc/backends/ascend/transport/aicore_transport_service.hpp` |
| 自研 URMA WQE builder | `csrc/backends/ascend/transport/urma_wqe.hpp` |
| CANN ABI 兼容层 | `csrc/backends/ascend/transport/cann_compat.hpp` |
| Host 资源和通道 | `csrc/backends/ascend/transport/cann_transport.cpp` |
| 设计文档 | `docs/ascend-design/epv2-ascend-simt-urma-transport.md` |

当前模型的历史原因是明确的：CANN 9.2.0 时公开 Hcomm/AIN 只有 `__aicore__`
实现，`__simt_vf__` 不能直接调用；SIMT 里直接使用 `st_dev` doorbell 会
触发 Bisheng 编译器 crash。因此当时选择了 staged command ABI。

现在官方 asc-comm 已经提供 SIMT-qualified 实现，历史阻塞点消失。

## 逐项差异

| 维度 | 当前 DeepEP | 官方 asc-comm SIMT |
| --- | --- | --- |
| 调用域 | SIMT 只追加 command | SIMT 直接提交 WQE |
| producer 数 | 多 lane 可追加 command queue | 一个 channel 只能单 lane 驱动 |
| command 开销 | 每 64B TransportCommand 需 GM 写入、fence、count 发布 | 无 command 层 |
| service 开销 | AICore 逐条解析、地址解析、通道校验 | 调用点内直接解析和提交 |
| WQE staging | AICore UB scratch + MTE3 | SIMT lane-private stack staging |
| SQ 写入 | AICore MTE3 block copy | SIMT store，按 BB 处理 ring wrap |
| doorbell | AICore `st_dev` | SIMT `asc_stwt` 128-bit store + fence + dcci |
| CQE poll | AICore MTE2 拷 UB | SIMT 直接读 CQE 原始字 |
| 批量提交 | command queue 后统一 service | 延迟任务由最后一个 commit 发布 |
| 失败诊断 | 结构化 DeviceTransportDiagnostic | 接口返回 -1 / CQE status |
| 语义面 | put、put_value、remote_add、signal、flush、barrier、request | Write/Read/Notify/FAA/CAS/Drain |

## 同机实测：官方 SIMT perftest

环境：

```text
host        NPU8P-ALT
devices     1,2
CANN        /data/disk2/cann_version/0916/use_cann/cann-9.3.0
asc-comm    9280cf6ea12a0cd7d00acc0709fe8834c669c3e1
example     examples/aicore/hcomm/05_simt_urma_perftest
iterations  1024
warmup      100
mode        single-lane rank0 -> rank1
```

结果：

| payload | 提交方式 | AverageIssue | CompletionTime | 带宽 |
| ---: | --- | ---: | ---: | ---: |
| 1KB | immediate | 2132 ns/WQE | 2.624 ms | 0.400 GB/s |
| 1KB | last | 1226 ns/WQE | 1.706 ms | 0.615 GB/s |
| 16KB | immediate | 2138 ns/WQE | 2.628 ms | 6.383 GB/s |
| 16KB | last | 1227 ns/WQE | 1.784 ms | 9.402 GB/s |
| 64KB | immediate | 2191 ns/WQE | 2.684 ms | 24.999 GB/s |
| 64KB | last | 1280 ns/WQE | 3.402 ms | 19.725 GB/s |

观察：

1. official SIMT 单 lane 的 WQE 下发开销约 1.2µs 到 2.2µs。
2. 延迟提交能显著降低下发开销，但对大 payload 的总完成时间不一定更好。
3. 这些数字只代表单 lane、单 channel 点对点路径，不能直接外推到 DeepEP
   多 block、多 expert、多 channel 的端到端性能。

同一环境、同一 2-rank representative case 下，当前 DeepEP staged transport
的结果为：

```text
case      ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0
tokens    8192/rank
hidden    7168
top-k     8
experts   256
device    1,2
samples   20

dispatch mean 8.223 ms, p50 8.034 ms, p95 10.716 ms
logical dispatch bandwidth 89.245 GB/s
```

注意：DeepEP case 的 payload 聚合、route 分布、copy 阶段和通信阶段与
官方 perftest 完全不同。该数据只说明当前路径在真实 case 上可用且有较高
聚合带宽，不构成与官方 perftest 的直接 A/B 结论。

## 是否应该切换

### 不建议立即全量切换

原因：

1. 当前 DeepEP 已经在 CANN 9.3.0 上通过典型 case 功能验收，直接替换引入
   的调度改造风险大。
2. 官方 SIMT channel 的单 lane 约束与当前多 lane producer 不匹配。
3. DeepEP 的 request/barrier/signal 语义需要重新设计。
4. 当前没有官方 backend 与 staged backend 在同一 DeepEP case 下的 A/B。

### 建议分三步验证

1. **最小官方 backend probe**
   - 单 producer lane；
   - 单 peer、单 channel；
   - 实现 put、put_value、remote_add、flush；
   - 复用当前 host 侧 HcclChannel 资源；
   - 与 staged backend 在同一 case 中切换。

2. **同负载 A/B**
   - 使用同一 2-rank representative case；
   - 保持 route、tokens、hidden、experts、channel 数一致；
   - 对比 dispatch mean/p95 和 logical bandwidth；
   - 使用 stage profile 区分下发开销、payload 传输和 drain。

3. **多 producer 调度设计**
   - 方案 A：每个 channel 一个专职提交 lane，其他 lane 先写本地 request，
     再由提交 lane 统一发布；
   - 方案 B：按 peer/channel 拆分 producer，天然保证单 lane 驱动；
   - 方案 C：如果官方后续提供 multi-producer/lock 语义，再评估公开锁接口。

### 性能收益的预期边界

官方版最可能改善的是：

- 小 payload、高命令数场景的每 WQE 固定开销；
- command queue 写入和 AICore service 解释开销；
- CQE poll 的 MTE2/UB 往返；
- producer 与通信提交之间的阶段边界。

官方版不一定改善的是：

- 大 payload 的 HCCS/URMA 链路带宽；
- 多 peer 竞争和 rank tail；
- route 计划、expert count、copy 等 DeepEP 计算阶段；
- 单 lane 提交吞吐不足时，反而可能成为新瓶颈。

## 初步判断

官方 asc-comm SIMT 数据面是我们当前 staged transport 的一个重要替代方向，
尤其适合作为“去掉中间 command/service 层”的长期方案。它解决了 CANN 9.2
时代 SIMT 无法直接提交 URMA 的历史阻塞。

但在 DeepEP 的多 lane producer 形态没有改造前，直接切换风险大于收益。建议
先做最小 probe，并用同一 DeepEP representative case 做 A/B。只有当 probe
证明官方路径在相同语义和相同负载下稳定优于 staged 路径，再进入全量适配。

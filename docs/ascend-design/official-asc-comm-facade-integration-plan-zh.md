# Official ASC-COMM SIMT Facade 对接设计草案

日期：2026-10-10
状态：设计记录，未排期实施

## 1. 目标

评估把 official ASC-COMM / HCOMM SIMT API 从当前 transport service executor 上移到 DeviceTransportFacade 对应的 device implementation 层，使通信语义在 producer 侧直接落地，减少 command queue 与 service 边界开销。

本文只记录方案，不作为已实现能力或性能结论。

## 2. 当前状态

当前默认路径：

operator / release protocol -> DeviceTransportFacade::put / put_value / signal / flush -> TransportCommandQueue -> AICore transport service -> 自研 URMA WQE / SQ / CQ / doorbell。

实验路径：

operator / release protocol -> DeviceTransportFacade -> TransportCommandQueue -> AICore service shell -> official_simt executor（单 lane） -> HCOMM SIMT WriteNbi / WriteValueNbi / AtomicFAA / Drain。

该实验路径的价值是保留现有 command ABI、generation、request、barrier 和 diagnostic 协议，用最小改动验证 official SIMT API 的可用性，并与自研 URMA service 做 A/B。

## 3. 目标架构

长期更合理的 direct-facade 路径：

operator / release protocol -> DeviceTransportFacade::put / put_value / remote_add_release / signal / flush -> official HCOMM SIMT backend implementation -> WriteNbi / WriteValueNbi / AtomicFAA / Drain。

分层原则：

- facade 保持稳定；
- official SIMT backend 是 device implementation 的一种，不是 transport service 的内部实现；
- command queue 和 service shell 不参与 hot path；
- request / generation / barrier 语义在 backend boundary 内重新映射。

## 4. 接口映射

| 当前 facade 语义 | Official HCOMM SIMT API | 说明 |
| --- | --- | --- |
| put | WriteNbi | 远端 payload 写 |
| put_value | WriteValueNbi<uint64_t> | 64-bit 控制值写 |
| put_control_slot | WriteNbi | 16B 控制槽写，可后续评估 inline |
| remote_add_release | AtomicFAA<uint64_t> | 带发布顺序的远端累加 |
| signal add / increment | AtomicFAA | 远端 signal 累加 |
| signal set | WriteValueNbi | 远端 signal 置位 |
| flush | Drain | channel 完成等待 |
| flush_async / wait | request + Drain | 需重新设计 request 状态 |
| device_barrier | FAA + signal polling | 保留 scale-out -> scale-up 协议 |

## 5. 关键设计问题

### 5.1 Channel ownership

Official HCOMM SIMT API 要求一个 channel 在一次执行期内由唯一 lane 驱动。direct-facade 路径必须明确 producer 到 channel 的映射。

候选方案：

1. lane 0 统一提交：最保守，兼容现有调用形态，但不能体现多 producer 优势；
2. per-peer 专职 lane：每个 peer / channel 有唯一提交 lane，适合当前 Dispatch/Combine 的 peer 分区；
3. producer 分片 + channel 独占：更接近 DeepEP-Ascend 的 per-AIV Jetty 模型，但需要重排 producer。

第一版建议从 per-peer 专职 lane 开始，避免直接引入多 producer SQ 协议。

### 5.2 Request / generation 状态

当前 flush_async / wait 依赖：

- TransportCommandQueue.generation；
- TransportServiceState.consumed_generation；
- DeviceRequest.command_begin / command_end；
- diagnostic first-error。

direct-facade 后没有 service state，需要重新定义：

- request 保存 peer / channel 集合，Drain 完成后发布 completion；
- 或维护 backend-local request ring；
- barrier 仍需明确的 generation 与 timeout。

### 5.3 Ordering 和 release 语义

当前 release protocol 依赖：

1. payload put；
2. flush；
3. count / generation put_value；
4. signal。

direct-facade 必须保持这个顺序。需要验证：

- WriteNbi 与后续 WriteValueNbi / AtomicFAA 的顺序；
- Drain 后远端可见性；
- signal 写是否需要额外 fence；
- barrier fetch-result buffer 的发布顺序。

### 5.4 Diagnostics

service 层现在能统一记录 command index、opcode、peer、channel、backend status。direct-facade 后，错误要移到每个 API 调用点记录，或维护 backend-local diagnostic ring。

### 5.5 Build 和资源模型

需要新增 backend 选择，而不是继续放在 DEEP_EP_ASCEND_OFFICIAL_SIMT service executor 下：

staged_native：command queue + AICore URMA service。

official_direct：facade -> HCOMM SIMT API。

Host 资源仍复用当前 channel table，但 backend 需要拿到：

- channel handle；
- remote base；
- sync base；
- window bounds；
- fetch-result buffer。

## 6. 分阶段计划

### Phase A：单 lane direct probe

目标：验证 API 层直接调用是否成立。

- 一个 producer lane；
- 一个 peer / 一个 channel；
- 覆盖 put、put_value、AtomicFAA、Drain；
- 与 staged backend 同负载 A/B；
- 不改生产 operator。

### Phase B：facade backend 化

目标：把 official SIMT API 变成 device transport implementation。

- 新增 official_direct backend；
- facade API 直接映射 HCOMM SIMT API；
- 保留 request / generation 最小语义；
- 单 lane 负责全部 channel。

### Phase C：per-peer lane

目标：恢复多 producer 并发。

- producer 按 peer / channel 分片；
- 每个 channel 唯一 lane；
- 验证 Dispatch / Combine 的 producer 调度；
- 保持 diagnostic 聚合和 first-error 语义。

### Phase D：性能验收

- same-binary ABBA；
- 覆盖典型 8-rank case；
- 关闭 stage profile；
- 比较 mean / p95 和 logical bandwidth；
- 不以单 lane API 微测外推端到端收益。

## 7. 风险

1. direct-facade 改动会触碰 operator hot path，风险高于 service executor 实验；
2. HCOMM SIMT channel 单 lane 约束可能与当前多 lane producer 冲突；
3. request / barrier 语义重写后容易引入 completion race；
4. 如果 immediate commit 固定开销高，需要研究 deferred commit / batching；
5. 该路径不代表 DeepEP-Ascend per-AIV Jetty 模型已经复现，仍需资源模型配合。

## 8. 当前结论

当前 service 层对接是低风险实验，适合验证 API 可用性；direct-facade 是更合理的长期架构方向，能减少 command encoding、GM queue 和 service 解析开销。实施前必须先解决 channel ownership、request / generation、ordering 和 diagnostic 语义。

# Official ASC-COMM SIMT Facade 对接设计草案

日期：2026-10-10，更新于 2026-10-13
状态：Phase B correctness/performance 已验证；默认路径不变，official_direct 仍为实验开关

## 1. 目标

评估把 official ASC-COMM / HCOMM SIMT API 从当前 transport service executor 上移到 DeviceTransportFacade 对应的 device implementation 层，使通信语义在 producer 侧直接落地，减少 command queue 与 service 边界开销。

本文记录方案与 Phase B 第一版实现边界。除非显式打开实验开关，本文不改变
默认 AICore/URMA 路径，也不给出任何性能结论。

## 2. 当前状态

当前默认路径：

operator / release protocol -> DeviceTransportFacade::put / put_value / signal / flush -> TransportCommandQueue -> AICore transport service -> 自研 URMA WQE / SQ / CQ / doorbell。

实验路径：

operator / release protocol -> DeviceTransportFacade -> TransportCommandQueue -> AICore service shell -> official_simt executor（单 lane） -> HCOMM SIMT WriteNbi / WriteValueNbi / AtomicFAA / Drain。

该实验路径的价值是保留现有 command ABI、generation、request、barrier 和 diagnostic 协议，用最小改动验证 official SIMT API 的可用性，并与自研 URMA service 做 A/B。

2026-10-11 新增 Phase B 实验分支：`DEEP_EP_ASCEND_OFFICIAL_DIRECT=1` 时，
DeviceTransportFacade 直接包含 direct HCOMM SIMT 实现；默认 0 时仍走
staged AICore/URMA service。该分支复用 host 侧 channel table、对称窗口、
sync window、request、generation 与 diagnostic ABI，避免为验证 API 改动
public host 协议。

历史备注：2026-10-12 曾做过 experimental official Ain executor
(`DEEP_EP_ASCEND_OFFICIAL_AIN=1`)，让 staged command executor 调用
`AscendC::Ain`。2026-10-13 direct HCOMM correctness/performance 收敛后，
该路线已删除：它仍保留 producer/command queue 和 service 边界，而 direct
HCOMM 已经在 facade 侧直接执行通信。当前只保留
`DEEP_EP_ASCEND_OFFICIAL_DIRECT=ON / OFF` 两态。

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

Phase B 第一版实际采用更保守的 lane 0 统一提交：所有 facade API 仅由
threadIdx.x == 0 驱动，并遍历该 transport context 的全部 peer/channel。
这能先验证 direct API 与 release protocol，但串行提交会使通信下限受单个
lane 的 WQE issue 能力限制，不能把它当作最终性能形态。

Phase C 再评估 per-peer 专职 lane 或 producer 分片。这个顺序 intentionally
把“API 正确性”和“并发资源模型”拆开，避免第一轮 correctness 失败时无法
区分是 HCOMM API 映射错误还是多 lane channel ownership 错误。

### 5.2 Request / generation 状态

当前 flush_async / wait 依赖：

- TransportCommandQueue.generation；
- TransportServiceState.consumed_generation；
- DeviceRequest.command_begin / command_end；
- diagnostic first-error。

direct-facade 不再产生 TransportCommand，但 Phase B 仍复用
TransportServiceState 的少量字段作为 completion ABI：

- `queue->generation`：operator 每代 reset 时发布的 generation；
- `state->consumed_generation`：request 完成后的发布字段；
- `barrier_generation`：device barrier 继续使用的单调 phase；
- `consumed_count`：保持不变，不代表 direct 路径消费命令数。

第一版语义如下：

1. flush_async() 只做校验并发布 request；
2. request 保存当前 queue->generation，command_begin/end 与
   consumed_target 均为 0；
3. wait() 由 lane 0 调 HCOMM Drain()；
4. Drain 成功后写 state->consumed_generation = request.queue_generation，
   并把 request 标记 completed；
5. Drain 或 diagnostic 失败时保留 first-error 并把 request 标记 failed；
6. direct 路径不增加 queue->count / state->consumed_count。

这样 Dispatch/Combine epilogue 中已有的 transport.consumed_generation()
generation 检查可以继续工作，而不需要修改上层协议。

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

需要新增 backend 选择，而不是继续放在 transport service executor 下：

staged_native：command queue + AICore URMA service。

official_direct：facade -> HCOMM SIMT API。

Host 资源仍复用当前 channel table，但 backend 需要拿到：

- channel handle；
- remote base；
- sync base；
- window bounds；
- fetch-result buffer；无独立 fetch buffer 时，fallback 到已注册 sync
  window 的 peer * 8 偏移，与 native executor 的规则一致。

Phase B 保持该资源模型。AtomicFAA 的 fetch 落点统一通过
fetch_result_target(table, staged, peer) 计算：standalone barrier context
若发布专用 fetch buffer 则优先使用；普通 transport context 使用 sync window
第一行。该规则与 default_fetch_result() 和官方 SIMT staged executor 对齐。

2026-10-13 修正：direct HCOMM 的 AtomicFAA fetch-result 不能 fallback 到
sync window 第一行。dispatch release signal 恰好占用 logical signal row 0；
device_barrier 的 fetch-result 写回会覆盖尚未被 epilogue 消费的 release
signal，导致 kBarrier/kSignal completion timeout。direct transport context
必须在 command memory 尾部追加独立的、已注册 fetch-result scratch：

- 分配大小为 world_size * sizeof(uint64_t)；
- command queue 的 base、capacity 与 registration cookie 保持不变；
- StagedTransportContext::fetch_results 指向 command queue 末尾；
- fetch_result_bytes 覆盖全部 world member；
- sync window 只承载协议 signal/barrier，不承载 API 内部返回值。

该修正已在 NPU8P 4-rank 同步 direct 用例验证：修复前稳定复现
kBarrier/kSignal completion timeout；修复后同 case 完整通过。

2026-10-13 Phase B 8-rank formal performance 结果：

- Workload：EP8 / FP8 / align 128 / 16384 tokens / hidden 7168 / top-k 6 /
  256 experts / num-sms 64；
- 采集协议：official bench_msprof，10 warmup、50 iteration，external
  barrier，flush L2，cold L2 epilogue，skip correctness check；
- 计时对象：direct producer-release kernel（SIMT HCOMM issue/drain）；
- debug 中间结果任务：`task_20261013_130923_238615525390`，exit=0；
- debug 中间结果输出：
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/official-direct-benchmark-fetch-scratch-formal-r1/fp8-8rank-official-direct.json`；
- 最终 clean-copy 复测任务：`task_20261013_191359_40100608280`，exit=0；
- 最终 clean-copy 输出：
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/local-repo-r6-formal-benchmark-r1/fp8-8rank-official-direct.json`。
- clean-copy CMake cache：
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/local-repo-r6/build-r6/CMakeCache.txt`
  确认 `DEEP_EP_ASCEND_OFFICIAL_DIRECT=ON`。

| Operation | Mean time | Median time | Per-rank bandwidth | Staged service 对齐结果 | Official 参考区间 |
|---|---:|---:|---:|---:|---:|
| Dispatch | 1635.607 us | 1636.511 us | mean 331.867 GB/s，range 328.596–335.575 GB/s | 274.49 GB/s | 373–375 GB/s |
| Combine | 3021.620 us | 3021.394 us | mean 345.021 GB/s，range 343.495–346.738 GB/s | 285.82 GB/s | 345–347 GB/s |

三种实现的同 workload 对比（staged 数据来自 2026-10-09
`fp8-8rank-aligned-workload-service-issue-drain`；direct 数据来自上表
clean-copy；official 数据来自官方 README 的 EP8 参考区间）：

| Operation | Staged transport | Direct HCOMM | Official DeepEP-Ascend | Direct vs staged | Direct vs official |
|---|---:|---:|---:|---:|---:|
| Dispatch | 1977.41 us / 274.49 GB/s | 1635.61 us / 331.87 GB/s | 373–375 GB/s | time 0.83x，带宽 1.21x | 带宽 0.89x |
| Combine | 3647.37 us / 285.82 GB/s | 3021.62 us / 345.02 GB/s | 345–347 GB/s | time 0.83x，带宽 1.21x | 带宽 0.99–1.00x |

上表为最终 clean-copy 结果，取代同日 debug 副本中 Dispatch 2349.612 us /
237.115 GB/s、Combine 3344.049 us / 311.981 GB/s 且存在明显 per-rank skew
的中间结果。该结果说明 fetch-result 与 clean-copy 修正后 direct 路径可以
稳定完成 8-rank formal run，且没有再出现 completion timeout。Clean-copy
各 rank Dispatch time 集中在 1616.5–1653.6 us，Combine time 集中在
3004.9–3043.2 us，此前 debug run 的 rank skew 未复现。Combine 已达到
official 参考区间；Dispatch 优于 staged service，但仍低于 official 参考值
约 10%。

边界说明：上述数据来自 experimental Phase B direct HCOMM 实现。其
producer release 仍是 1 block × 512 lane 的 control launch，并由 lane 0
驱动 HCOMM issue/drain；它不是官方 per-AIV Jetty 的 64-AIV 并发模型。因
此这组数据只能用于验证 Phase B 的 correctness 与当前性能形态，不能作为
official parity 或最终架构的性能结论。

### 5.6 Release 与 transport service 的执行域

旧 staged 路径中，release lane 0 和 transport service 不是同一个 block，
也不是同一个执行域：

- release lane 0 位于 `direct_*_producer_release_kernel` 的 SIMT VF；
- transport service 位于后续 outer `dispatch_kernel` / `combine_kernel`
  的 AICore scalar 逻辑；
- 两者都使用 control launch（1 block × 512 lane），但代码不保证它们落在
  同一个物理 AIV，也不依赖这一点。

当前 direct HCOMM 模式下，service 已退化为空 shim：

- HCOMM issue 在 release kernel 的 lane 0；
- Drain 在后续 `direct_*_pipeline_wait_kernel` 的 lane 0；
- outer kernel 仍会被 launch，但其中的 service 调用是 no-op。

因此 direct 模式还需要调整 launch 编排：跳过空 service 外层 kernel，或把
release 与 wait 合并，减少一次无意义的 kernel launch。Phase B 先保留该
形态以最小化协议改动，后续再优化。

### 5.7 当前 launch shape 与 AIV 分工

本节按 `--num-sms=64` 的 8-rank direct scale-up 场景描述当前实现。代码中
的 "AIV 数" 指 kernel launch 的 block 数；每个 block 内 VF 宽度为 512
lane。不要把它理解成 64 个独立 kernel。

#### 5.7.1 通用 launch shape

| Launch 类别 | Shape | 典型 stage | 有效执行域 |
|---|---:|---|---|
| data launch | 64 block × 512 lane | Dispatch Group/Record/Copy、Combine Plan/Record/Reduce/Weights 等 | 64 个 AIV 并行做分片数据处理 |
| control launch | 1 block × 512 lane | Control、Prefix、Release、Acquire、Complete 等 | 1 个 AIV；release/control 相关逻辑通常只由 lane 0 执行 |

Dispatch 的非 profile pipeline 有两种形态：

- CPU-sync / split dispatch：9 个 stage，停在
  `EpilogueExpertPrefix`，后续 copy 由 split epilogue 路径处理；
- 非 CPU-sync：12 个 stage，包含 `EpilogueMetadata`、`EpilogueCopy`、
  `EpilogueComplete`。

这里的 CPU-sync 指 kernel mode 中的 `kCpuSync`。Python API 传入
`do_cpu_sync=True` 且 async/stream overlap 关闭时，默认 device-prefix 配置
会生效，此时反而走非 `kCpuSync` 的 12-stage 路径；async/stream overlap
打开时 device prefix 被禁用，才落入 9-stage split dispatch 路径。

Profile pipeline 会在 `ProducerRelease` 后额外拆出
`ProducerReleaseControl` 与 `ProducerReleaseBarrier`；Dispatch profile
另有 CPU-sync 11 stage / 非 CPU-sync 14 stage 两种形态。

#### 5.7.2 Dispatch stage 的 kernel 与职责

| Stage | Launch | 当前实际 launch 顺序 | 职责 |
|---|---:|---|---|
| ProducerControl | 1 AIV | outer `dispatch_kernel` → specialized control/plan VFs | outer 先 reset transport generation；specialized VF 清理状态、校验 layout、准备 route plan |
| ProducerGroup | 64 AIV | specialized group kernel，随后直接 return | 按 token/top-k 分组，生成 owner、tile count 和 destination 计数 |
| ProducerPrefix | 1 AIV | specialized prefix VF → outer `dispatch_kernel` → 可选 error/route-plan VFs | 生成 rank/expert prefix；early route plan 时发布并等待 route plan |
| ProducerRecord | 64 AIV | specialized record VF → outer `dispatch_kernel` | specialized VF 处理 record 元数据、scale factor 和 vector 后缀；outer AICore 处理 bulk payload 的 UB/MTE copy |
| ProducerRelease | 1 AIV | specialized release VF → outer `dispatch_kernel` | staged：VF 写 command queue，outer 执行 AICore transport service；direct：VF 直接 HCOMM issue + Drain，outer service 为 no-op |
| EpilogueAcquire | 1 AIV | specialized acquire VF → outer | 等待远端 control/count 发布 |
| EpilogueValidate | 1 AIV | specialized validate VF → outer | 校验 receive record 与控制信息 |
| EpilogueExpertCount | 1 AIV | specialized count VF → outer | 统计本 rank 收到的 expert/token 数 |
| EpilogueExpertPrefix | 1 AIV | specialized prefix VF → outer，可选 publish counts | 生成输出 prefix；CPU-sync split 路径在此结束 |
| EpilogueMetadata | 64 AIV | specialized metadata/assign/reduce/clear VFs → outer | 生成 source metadata、destination slot、错误归约和 padding 清理 |
| EpilogueCopy | 64 AIV | 可 specialized copy VF → outer | 从 symmetric window / staging 拷贝到用户输出 |
| EpilogueComplete | 1 AIV | outer → specialized complete kernel | 完成状态与 generation 收尾 |

#### 5.7.3 Combine stage 的 kernel 与职责

| Stage | Launch | 当前实际 launch 顺序 | 职责 |
|---|---:|---|---|
| ProducerControl | 1 AIV | outer `combine_kernel` → specialized control VF | outer reset generation；specialized VF 清理/初始化控制状态 |
| ProducerPlan | 64 AIV | outer → specialized plan VF | 生成 combine record 计划与 payload 分片 |
| ProducerPlanPrefix | 1 AIV | outer → specialized plan-prefix VF | 生成 rank/record prefix |
| ProducerRecord | 64 AIV | specialized record VF → outer | specialized VF 处理 record/元数据；outer AICore 处理 bulk payload 或 expanded vector reduce |
| ProducerRelease | 1 AIV | specialized release VF → outer | staged：VF 写 command，outer 执行 service；direct：VF 直接 HCOMM issue + Drain；若关闭 direct local placement，outer 还承担 local copy |
| ProducerLocalCopy（仅 profile 或关闭 direct placement） | 1 AIV | specialized local-copy VF | staging 到 receive window 的本地拷贝 |
| EpilogueAcquire | 1 AIV | outer → acquire/clear-index VFs | 等待 control/count，并清理 index |
| EpilogueValidate | 64 AIV | outer → specialized validate VF | 校验 combine record |
| EpilogueValidateReduce | 1 AIV | outer → reduce-error/prepare-slot VFs | 错误归约，准备 vector slot |
| EpilogueReduce | 64 AIV | outer；vector 条件满足时不追加专用 reduce kernel | 多来源 record reduce；top-k/hidden 满足 vector 条件时由 outer 直接做 vector reduce，否则追加 specialized reduce kernel |
| EpilogueWeights | 64 AIV | outer → 可选 tail kernel → weights kernel | top-k weight 汇总 |
| EpilogueComplete | 1 AIV | outer → specialized complete kernel | 完成状态与 generation 收尾 |

#### 5.7.4 staged 与 direct 的关键差异

1. staged 的通信执行域是后续 outer kernel 内嵌的 AICore transport service；
   direct 的通信执行域前移到 specialized release VF 的 lane 0。
2. direct 非 pipeline 模式下，`flush_payload` 在 release VF 内完成 Drain；
   因此 outer release kernel 在默认 direct local placement 下只剩 no-op
   service 调用。
3. direct pipeline 模式下，release VF 使用 `flush_async` 发布 request，
   后续 `direct_dispatch_pipeline_wait_kernel` 的 lane 0 调 `wait()` 并
   Drain。
4. direct 模式的 ProducerControl outer kernel 不能简单删除，因为它仍通过
   shim reset generation/count/diagnostic；若要删除，需要把 reset 移入
   specialized control VF。
5. Combine release outer kernel 也不能无条件删除：当
   `direct_local_placement=0` 时它承担 local copy。默认推荐开启 direct
   local placement 时，它才近似为空。

#### 5.7.5 对后续 launch 优化的约束

当前 direct release 是 "1 block × 512 lane，只有 lane 0 有效"，不是
"64 AIV 中一个 AIV 空闲"。因此不能把该 release block 直接看作可加入
63 个数据 AIV 的第 64 个成员；它是一个独立 control launch。

目标形态应是把 release 语义合并进 64-AIV producer kernel：

- 64 个 AIV 继续并行做 record/payload 预处理；
- 每个 peer/channel 由唯一 owner lane 调 HCOMM API，满足单 lane channel
  ownership；
- 未拥有 channel 的 lane 不直接操作同一 channel，避免 HCOMM handle 竞争；
- 计时对象改为该合并 kernel，而不是 transport service 或单独 release
  kernel。

在 channel 数为 1 时，8-rank 只有 7 个远端 peer，可并发的 HCOMM owner
lane 最多约 7 个；其余 lane 仍应承担数据预处理。若后续启用多 channel，
可并发 owner 数近似为 `7 × channel_count`，但仍受 channel 表资源限制。

## 6. 分阶段计划与 16K/top-6 复筛边界

### Phase A：单 lane direct probe

目标：验证 API 层直接调用是否成立。

- 一个 producer lane；
- 一个 peer / 一个 channel；
- 覆盖 put、put_value、AtomicFAA、Drain；
- 与 staged backend 同负载 A/B；
- 不改生产 operator。

### Phase B：facade backend 化（当前阶段）

目标：把 official SIMT API 变成 device transport implementation。

- 新增 official_direct backend；
- facade API 直接映射 HCOMM SIMT API；
- 保留 request / generation 最小语义；
- 单 lane 负责全部 channel。

实现细节：

- 所有写操作使用 immediate commit；deferred commit、batch handle 和多 lane
  ownership 放到后续阶段；
- put / put_control_slot 使用 WriteNbi；
- put_value / signal set 使用 WriteValueNbi<uint64_t>；
- remote_add_release / signal add-increment 使用 AtomicFAA<uint64_t>；
- flush / wait 使用逐 channel Drain()；
- 每次 API 调用构造 lane-private Hcomm 对象并 Init()；当前实现中 Init()
  是 no-op，功能可用，但这是后续 profile 前需要确认的固定开销；
- direct 路径尚未接入 staged service 的 per-stage profile；official
  reporting 的 stage profile 在该后端上暂缺，不能用它推断 direct 性能。
- 2026-10-12 补充 direct 性能采集口径：direct 后端不应使用
  `stage_service_issue_drain`。它的 HCOMM SIMT issue 与 Drain 发生在
  `direct_dispatch_producer_release_kernel` /
  `direct_combine_producer_release_kernel` 内，因此 msprof 直接统计这两个
  kernel。单 case 入口和 matrix 入口均新增
  `--transport-mode staged|official_direct`；默认 `staged` 完整保留原
  service 统计方式，`official_direct` 才切换到 direct kernel 统计。
  该开关只影响报告层选择的 kernel，不改变已编译后端的 launch 代码。
  但这不等于 direct 性能形态已经完成：Phase B 当前仍把 producer record
  与 producer release 拆成两个 stage；release kernel 使用 control launch，
  并由 lane 0 统一驱动全部 channel。因此它还没有实现“64 个 AIV 在同一个
  kernel 内完成通信数据预处理与真实通信”的目标。Phase C 需要把 release
  producer 与 data producer 合并，按 peer/channel 分配唯一 owner lane，
  才能让 64 AIV 分别驱动独立通信路径，并继续用该合并 kernel 的 msprof
  时间作为 direct 性能口径。

2026-10-11 已在 NPU8P 完成该实验分支的 no-device 编译验证：

- 直连 NPU8P，设备 0–7 当前由其他任务占用，本验证未占卡；
- CANN/HCOMM：/usr/local/Ascend/cann-9.3.0；
- Python：/data/disk2/pyptouser/yuqitao/deepep-venv-py311；
- task：task_20261011_102930_99922226581（首轮通过）、
  task_20261011_103735_107215124273（代码清理后复验）；
- 结果：exit=0，_C target 100% built；
- extension SHA256：
  81f06d0a7252d2994605442eba799dfd3ba9afbce37c60efdceeceba3f7981c9；
- 验证副本：/data/disk2/pyptouser/yuqitao/official-direct-20261011/repo。

该结果只证明 direct 后端在该 CANN/HCOMM 树上可编译，尚未证明 2-rank 或
8-rank correctness，也未做性能结论。

2026-10-12 direct r4 构建与 2-rank correctness 验证结果：

- 构建通过：task
  `task_20261012_154522_168863128814`，exit=0；日志为
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/build-official-direct-r4.log`。
- 2-rank correctness 尚未通过：task
  `task_20261012_155000_209878217491`，exit=1；日志为
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/direct-correctness-r4.log`。
- 失败点发生在 Buffer 初始化，尚未进入 device kernel：
  `HcommMemAlloc` 成功后，`HcclCommSymWinRegister` 在两个 rank 均返回
  backend error 9。

2026-10-13 direct r6 correctness 定位与复验结果：

- 前 11 个正常 FP8 case 已可运行；首个失败点收敛到 cached dispatch 的
  route digest：8-rank `cached-representation-change` 中 per-rank counts
  正确，但 `cached_route_digest_entry()` 结果恒为 0。
- 用固定常量替换 digest 证明 route-plan VF launch、GM 读入、block 内
  32-bit atomic 归约、64-bit 重组和 host readback 都正确。
- 逐步二分确认：直接 `expert + encoded`、helper 返回简单加法、64-bit
  乘法、32-bit 移位/或、以及不含 lambda 的完整 32-bit mix 均能得到
  非零结果。只有原 helper 中 lambda 捕获的 64-bit mix 序列会被当前
  CANN 9.3.0 SIMT VF lowering 折叠为 0。
- 修正方式是移除 lambda，把 32-bit mix 展开成普通局部变量语句。该
  digest 保持 commutative、非零，并与原实现一样只作为 cached handle
  的附加身份校验；digest mismatch 仍会回退 full planner。
- 修正后扩展 SHA256：
  `12f69af346c8d76624422bf5cf34fdcc9e3b25ac0ee4b0264c611e49764fa51d`。
- 2-rank task `task_20261013_072332_40206557159` 通过 12/12 FP8 cases；
  8-rank task `task_20261013_072446_40267177556` 通过 12/12 FP8 cases。
  运行时环境为 `/usr/local/Ascend/cann-9.3.0`、设备 0–7，且显式设置
  `DEEP_EP_ASCEND_PREFLIGHT=full`。
- `malformed-inputs` 在 stable local-only preflight 下会进入 runtime 并
  超时，这是 2026-09-24 hot-path preflight 优化的既有语义，不是 direct
  后端新问题；该用例需要 full preflight 拦截单 rank 非法输入。
- 本轮 debug 副本仍含 `[DEBUG-route-c57a]`、`[DEBUG-8c4f]` 和
  `[TEARDOWN-DIAG]` 临时探针，不能直接合入；最终迁移时只保留 digest
  helper 的最小实现修正。

2026-10-13 clean copy 复验与 Phase B final 状态：

- clean copy 不再包含 debug 探针。除上述 lambda digest 修正外，还同步了
  direct 路径必需的三处修正：`direct_dispatch_cached_route_plan.asc`
  的 UB scratch 布局为 `2 + 2 * kTopkSubgroupWidth`；direct barrier 使用
  immediate `ChannelHandle` 加 `Drain()`，不再落回 batch workspace；
  direct 模式的空 command generation 由
  `official_direct_service_shim.hpp` 消费，保证 barrier continuation
  不会误等待。
- 最终构建配置必须保持 `DEEP_EP_ASCEND_OFFICIAL_DIRECT=ON`。历史上
  强制或联动打开 AIN 时，8 rank 会在 `register_symmetric_window` 阶段
  返回 backend error 9；该路线已删除。
- 8-rank clean-copy correctness task
  `task_20261013_191032_394628225846` 使用设备 0–7、
  `/usr/local/Ascend/cann-9.3.0`，exit=0，日志末尾为
  `Phase 3F 8-rank FP8 runtime matrix passed (12 cases)` 和
  `LOCAL_REPO_R6_FP8_8RANK_OK`。
- 该结果证明最终 local repo 副本已复现 debug 副本的 12/12 FP8 correctness。
  下文官方 basic ring 的 ret=9 记录对应当时强制 AIN/SIMT window 路径的
  探测，不应再被解读为 direct HCOMM 路径本身不可用。

为区分实现问题与运行时能力问题，使用 asc-comm 官方
`examples/aicore/ain/01_basic_ring` 做最小 sanity check。该示例只执行：

```text
HcclCommInitRootInfo
  -> HcommMemAlloc(2 MiB send/recv window)
  -> HcclCommSymWinRegister
  -> HcclTeamCreate
  -> Ain kernel
```

结果：

| CANN/HCOMM | task | 结果 | 第一失败点 |
| --- | --- | --- | --- |
| `/usr/local/Ascend/cann-9.3.0` | `task_20261012_160355_326698212875` | exit=1 | `basic_ring.asc:139`，`HcclCommSymWinRegister`，ret=9 |
| `/data/disk2/cann_version/0916/use_cann/cann-9.3.0` | `task_20261012_160827_396924418039` | exit=1 | `basic_ring.asc:139`，`HcclCommSymWinRegister`，ret=9 |

日志已归档：

- `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/basic-ring-usr-local-cann.log`
- `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/basic-ring-disk-cann.log`

两套安装树的 `libhccl.so` / `libhcomm.so` SHA256 均不同，但最小官方用例
均失败在同一 host API，因此当前证据优先指向运行时 symmetric window
注册能力或包兼容性，而不是 DeepEP direct 集成代码、window 大小、
`HcclCommInitRootInfoConfig` 差异或 SIMT channel ownership。下一步 blocker
处理顺序：

1. 使用官方 basic ring 继续探测可用 CANN/HCOMM 组合，要求它先通过后再
   回到 DeepEP direct correctness；
2. 若获得可用包，先复跑 direct 2-rank empty-input / asymmetric-routing，
   再跑 regular cases；
3. direct 8-rank 性能必须在 correctness 通过后进行，当前不能把 Phase B
   编译成功解释为性能已验证。

补充说明：HCCL/HCOMM 错误码 9 对应 `HCCL_E_TIMEOUT`。由于失败发生在两个
rank 并行的 `HcclCommSymWinRegister`，且在 180 秒外层 timeout 内稳定复现，
这里更合理的解读是注册内部等待某个尚未就绪的资源/状态直到超时，而不是
调用方传入了明显非法参数。官方 basic ring README 标称支持
Ascend 950PR/DT 且 CANN >= 9.1.0；当前两套 9.3.0 均未满足其实际运行时
依赖，说明版本标称与当前安装包能力之间可能存在包内组件差异。

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

Phase B 的验证顺序固定为：CPU-only build、2-rank correctness、8-rank
correctness，最后才做 official reporting performance。staged 后端使用
官方 bench_msprof + stage issue-and-drain 对齐；direct 后端使用同一官方
bench_msprof 采集协议，但计时对象改为实际执行 HCOMM SIMT issue/drain 的
producer-release kernel，不使用 raw bench_ep.py。

2026-10-13 的 Phase B 验收只覆盖一个代表性 case：
`ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`。它不能替代：

- staged transport 的 144-case 全量基线；
- BF16 dispatch / FP8 dispatch 的差异；
- alignment 1、bias、handle copy、previous event、sync/async 和 comm-stream
  allocation 组合；
- direct HCOMM 后端的 144-case correctness 和性能边界。

在 direct HCOMM 后端完成自己的 144-case 矩阵前，不应把单 case 的
“Combine 达到官方参考区间、Dispatch 低约 10%”外推为 direct 后端的全矩阵
结论。

## 7. 风险

1. direct-facade 改动会触碰 operator hot path，风险高于 service executor 实验；
2. HCOMM SIMT channel 单 lane 约束可能与当前多 lane producer 冲突；
3. request / barrier 语义重写后容易引入 completion race；
4. 如果 immediate commit 固定开销高，需要研究 deferred commit / batching；
5. 该路径不代表 DeepEP-Ascend per-AIV Jetty 模型已经复现，仍需资源模型配合。

## 8. 当前结论

当前 service 层对接是低风险实验，适合验证 API 可用性；direct-facade 是更合理的长期架构方向，能减少 command encoding、GM queue 和 service 解析开销。实施前必须先解决 channel ownership、request / generation、ordering 和 diagnostic 语义。

## 9. 16K/top-6 workload 与优化项适配性

### 9.1 workload 变化

历史代表性优化 workload 是 8192 tokens/rank、top-k 8。当前官方
144-case 全量基线是 16384 tokens/rank、top-k 6，并且 Dispatch 实际使用
expanded_dispatch，Combine 实际使用 reduced_combine。

因此，不能把历史 8K/top-8 的优化结论直接外推到当前 16K/top-6：

- token 数增加会改变 producer/consumer 划分与长尾；
- top-k 从 8 降到 6 会降低每个 source token 的 fan-out 与 metadata 压力；
- expanded Dispatch 的控制流和普通 direct Dispatch 不同；
- reduced Combine 的输入布局和普通 Combine 不同；
- 144 个 case 还叠加了 mode、dtype、bias、hcopy、prev、async、alloc 等
  参数差异，不能把一个 case 的趋势当作全矩阵结论。

### 9.2 selector 与当前 workload 的关系

下面按当前官方 workload 的实际调用模式归组。判断依据是 host selector 的
eligibility 条件，不是 selector 名称：例如 token fanout / early route plan
都明确排除 expanded 模式，而 device prefix 组的 selector 不排除 expanded。

### 9.3 逐项审计：哪些默认优化带 8K/top-8 形态假设

| 优化 / 常量 | 代码中的形态假设 | 16K/top-6 下的实际状态 | 处理建议 |
|---|---|---|---|
| Dispatch token fanout，`kDispatchTokenFanoutBufferBytes=7168` | FP8 hidden row 必须是 7168 byte；top-k ≤ 8；world size ≤ 8；明确排除 expanded/cached/stream/hybrid | 官方口径 Dispatch 使用 expanded_dispatch，因此当前不会命中；top-k 6 本身满足上限 | 不需要改代码；expanded 若未来重开 fanout，已有 C20.2 ABBA 证明收益不成立 |
| Dispatch consumer tile 8192 byte | 针对一条 7168-byte hidden row 的单次 MTE 迭代优化 | hidden=7168 不变，tile 与 top-k/token 数无关；主要受 stream/cached/cpu-sync eligibility 影响 | 保留默认；按 case 记录是否命中，不做 16K/top-6 特改 |
| Device prefix / parallel prefix | 依赖 uncached、CPU-sync、non-stream 等模式条件 | 官方 expanded case 的 async/stream 组合会改变命中情况；这不是 token/top-k 数值特化 | 不改默认；复筛时先记录每个 case 的 selector 实际状态 |
| Cached route plan | 历史验收在 8192 tokens/top-k 8 下做，但算法是 capacity/world/top-k 动态校验 | 16K/top-6 correctness 已通过；当前官方性能报告主要使用 expanded/reduced，cached 路径不是主统计对象 | 保留；如后续统计 cached case，需单独按 16K/top-6 复测 |
| Combine producer payload tile 3584 elements | 3584 = 7168/2，来自 hidden=7168 的 full-record tile；与 token 数/top-k 无关 | hidden 仍为 7168，top-k 6 不破坏该假设；reduced Combine 走 expanded vector reduce 侧 | 保留默认；如 direct 后端瓶颈在 producer record，可做 3584 vs 1024 AB |
| Combine local copy tile 32768 byte | 针对 representative hidden=7168 的本地 staging DataCopy 调优 | hidden 不变，token 数增加通常增强大 tile 方向；无 top-k 8 硬编码 | 保留默认；direct local placement 已减少该路径重要性 |
| Combine vector reduce tile 512 / prefetch 4 | 512-element tile 与 hidden=7168 的 14 次迭代匹配；历史验收为 top-k 8 | 当前 top-k 6 仍走通用 vector reduce 分支；`kCombineCommonTopk=8` 的专用 AOT 实例不会命中，top-k 6 会落入动态模板 | 这是最明确的复筛对象：需要在 16K/top-6 下比较 512 与 1024，以及 prefetch 4 vs 2 |
| Expanded vector reduce | 面向 expanded/reduced Combine；top-k ∈ [1,32] | top-k 6 命中，且 reduced Combine 是官方主统计对象 | 保留；它是 reduced Combine 的关键优化，不应关闭 |

结论：真正带有“top-k 8 专属”硬编码的是 Combine epilogue 的
`kCombineCommonTopk=8 / kCombineCommonHidden=7168` AOT 专用实例。当前
top-k 6 会自然退回通用实现，不会错误启用该实例，但它也意味着历史上的
top-k 8 专用优化在当前主口径中没有贡献。Dispatch token fanout 的 7168
hidden 特化虽存在，但 expanded dispatch 的 selector 条件使其不会命中。

因此，当前不应直接删除历史优化，也不应凭 8K/top-8 的收益继续扩大它们；
优先复筛对象是 Combine vector reduce tile/prefetch，其次才是 producer
payload tile。Dispatch 侧的 token fanout / early route plan 对当前
expanded_dispatch 主路径没有影响。

以下优化项只在 non-expanded Dispatch 中生效，当前官方 expanded_dispatch
不会命中：

- `DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT`；
- `DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN`。

以下 Dispatch 优化项在 expanded 模式下仍可继续生效：

- `DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX`；
- `DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX`；
- `DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES=8192`。

补充修正：仅从 selector 名称推导“expanded 模式下仍生效”并不充分。当前
官方 benchmark 的 expanded_dispatch 在 async/stream overlap 打开时，
`do_cpu_sync` 路径与 stream-mode eligibility 会影响 device prefix 与
parallel prefix 的实际选择；144-case 中也存在同步、异步、previous-event
和 comm-stream allocation 的组合。因此本节应按下述方式理解：

- expanded/async case 只说明 token fanout 与 early route plan 不适用；
- device prefix、parallel prefix、consumer tile 和 Combine vector-reduce
  系列是否真正命中，必须以每个 case 的 runtime selector 记录或 stage
  trace 为准；
- 不能把某一个同步 case 的命中结果外推到全部 144 case。

当前 reduced Combine 相关的 selector 包括：

- `DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE`；
- `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE=512`；
- `DEEP_EP_ASCEND_COMBINE_PRODUCER_PAYLOAD_TILE=3584`；
- `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_PREFETCH=4`；
- `DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768`；
- `DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT=1`。

这些仍是优化候选，但 direct HCOMM Phase B 尚未做逐 selector 的
same-binary AB。当前更明确的处理规则是：

1. 保留代码和默认 selector，不做实现删除或默认值调整；
2. 先采集 direct HCOMM 的 144-case 基线；
3. 从报告中筛出 Dispatch 或 Combine 明显低于 staged 全量基线的 case；
4. 对这些 case 检查实际 runtime selector 命中情况；
5. 只对已命中且收益方向可疑的 selector 做 same-binary AB；
6. AB 的默认值修改必须另过 correctness 边界，不能由一轮性能 run 直接决定。

这个顺序把“16K/top-6 适配性复筛”限定为 direct HCOMM 的后续工作，而不是
把历史 8K/top-8 结论直接迁移到当前默认实现。

### 9.4 raw bench_ep.py 数据不能作为判定依据

此前有一轮 raw bench_ep.py AB 曾被用来观察 expanded vector reduce，但它
不能作为正式结论：

- case 是 hcopy0，不是历史 representative 的 hcopy1；
- 未使用官方 bench_msprof、stage profile 与 align_official_reporting；
- device event 只覆盖外层 kernel，没有覆盖 staged service completion；
- 任务只锁 4 张卡，但 torchrun 实际使用 8 rank；
- prefetch=2 是非法值，后来改为 prefetch=1，但输出文件名仍保留 prefetch2。

这轮数据只能说明关闭 expanded vector reduce 后 Combine 明显变慢，不能
用于量化收益。后续所有 selector 复筛都必须使用 144-case 官方口径，至少
覆盖代表性 case，并在修改默认值前给出 correctness 边界。

### 9.5 2026-10-13 selector 复筛结果

本轮复筛使用同一个 `DEEP_EP_ASCEND_OFFICIAL_DIRECT=ON` 构建，同一 case
`ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`，EP8、16384 tokens、
hidden 7168、top-6、256 experts、64 AIV。raw 入口使用
`bench_ep.py`，10 warmup / 30 iteration；official 入口使用
`bench_ep_msprof.py --transport-mode official_direct`，10 warmup / 50
iteration。

结论按收益排序如下：

| 调整 | raw `reduced_combine` mean | raw 相对默认 | msprof 受影响 kernel | 结论 |
|---|---:|---:|---:|---|
| 关闭 expanded vector reduce | 109683 us | +780% | `direct_combine_producer_record` 54.6 us -> 98373 us | 必须保留；关闭后灾难性退化 |
| 关闭 direct local placement | 16224 us | +30.2% | `combine_kernel` 473 us -> 825 us | 必须保留；明显收益 |
| vector reduce prefetch 4 -> 2 | 16052 us | +27.8% | `combine_kernel` 473 us -> 801 us | 必须保留；这是最大可控收益点 |
| producer payload tile 3584 -> 1024 | 13980 us | +11.3% | `combine_kernel` 473 us -> 591 us | 保留 3584 |
| vector tile 1024 vs 512（均在 depth 2） | 15942-16275 us | +26.9%-29.5% | `combine_kernel` 约 800 us | 1024 与 512 在 depth 2 下接近；相对默认的退化主要来自失去 prefetch 4，而不是 tile 从 512 变为 1024 |

两点解释：

1. `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_PREFETCH=4` 只在 512-element
   tile 下有效。设置 `VECTOR_REDUCE_TILE=1024` 时，host selector 会把
   prefetch 强制降回 depth 2，因此 `1024 + prefetch=4` 实际是
   `1024 + depth2`。在该约束下，1024 与 512 在 depth 2 时差异不大；
   真正显著的是 depth 4 相对 depth 2 的收益。
2. official msprof 的 `urma` 字段只统计
   `direct_combine_producer_release_kernel`，本轮 selector 主要影响
   `combine_kernel` 或 `direct_combine_producer_record`。因此只看 release
   kernel 会得到约 3.03 ms 的近似不变结果，掩盖上述差异。评估这些 selector
   时必须同时读取 `all_kernels` 里的受影响 kernel，或保留 raw event 作为
   辅助证据。

raw event 覆盖用户可见的 async operation 边界，包含 epilogue 与
current-stream wait，因此只用于同 binary 相对比较；带宽结论仍以 official
msprof 的通信 kernel 边界为准。

原始任务与输出：

- raw selector 第一轮：`task_20261013_223938_241685015569`，
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/optim-ab-r1/`；
- official selector 第一轮：`task_20261013_225118_250798323503`，
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/optim-msprof-r1/`；
- raw direct-local / expanded-reduce 对照：
  `task_20261013_230106_283819016957`，
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/optim-ab-r2/`；
- official direct-local / expanded-reduce 对照：
  `task_20261013_230519_286533518629`，
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/optim-msprof-r2/`；
- top-8 对照：`task_20261013_230940_30572337908`，
  `/data/disk2/pyptouser/yuqitao/official-direct-20261012-r1/optim-msprof-topk8-r1/`。

补充对照：同一 workload 把 top-k 从 6 改为 8 后，official release 为
3594 us / 345.9 GB/s，top-6 为 3024 us / 341.3 GB/s；top-8 的
`combine_kernel` 为 380 us，top-6 为 473 us。由于 top-8 的数据量比 top-6
高 1/3，这不能直接证明 top-6 专用 AOT 会带来等比例收益。当前证据只说明
top-8 特化路径效率较高；在 release kernel 占主导的 official 口径下，
补 top-6 特化的优先级低于保留现有 prefetch / direct local placement /
expanded vector reduce。

Dispatch 侧 selector 在当前 official async/expanded case 下大多不生效：
`async_with_compute_stream=true` 会置位 stream mode，而 device prefix、
parallel prefix、consumer tile 均要求 non-stream；token fanout 又明确排除
expanded。因此这轮不需要把 Dispatch selector 纳入同一优先级排序。

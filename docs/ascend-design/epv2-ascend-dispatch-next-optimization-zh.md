# Dispatch 下一阶段性能优化设计

## 目标

本文基于 2026-09-24 NPU8P 的三次 8-rank stable profile，规划 Normal Dispatch
下一阶段优化。后续开发以本文为基准，避免回到已经失效的历史结论。

固定 workload：

```text
world_size=8
num_tokens=8192
hidden=7168
num_topk=8
num_experts=256
data_blocks=64
FP8 Dispatch
warmups=30
iterations=30
DEEP_EP_ASCEND_PREFLIGHT=stable
DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=1
```

数据来源：

- `combine-gating-stable-8rank-topk8-8192tokens-30iter-rerun1.json`
- `combine-gating-stable-8rank-topk8-8192tokens-30iter-rerun2.json`
- `combine-gating-stable-8rank-topk8-8192tokens-30iter-rerun3.json`
- NPU8P task `task_20260924_191032_253197525980`

## 当前结论

三次 run 的 Normal Dispatch mean 为 7.247-7.479 ms，logical bandwidth 为
1041.065-1074.459 GB/s。历史 P7A 中的 `producer_record` 与
`release_barrier` 已经不再是主要瓶颈。

三次 run 的稳定阶段耗时如下：

| Stage | mean range | 说明 |
| --- | ---: | --- |
| release_payload | 0.905-0.914 ms | 最大单一设备 stage |
| epilogue_copy | 0.521-0.522 ms | 稳定 D8 copy |
| release_control | 0.256-0.268 ms | 控制发布 |
| producer_record | 0.166-0.168 ms | 已从历史 multi-ms 降下来 |
| release_barrier | 0.041-0.043 ms | signal-only 后已很小 |

更重要的是，per-rank device stage activity 总量只有约 2 ms，而端到端约
7.3 ms。critical path report 显示最大 unattributed gap 为 2.3-4.0 ms。
因此当前的主要矛盾是 stage 间调度、等待和 host 边界，而不是单个 kernel
内部的计算量。

## 优化项

### D1. Dispatch host count bridge 优化

优先级：P0。

当前非 cached 路径在 kernel 完成后仍执行：

1. count bridge D2H；
2. host 侧 rank/expert prefix 构造与校验；
3. public count bridge H2D。

三次 profile 的 host 阶段：

| 阶段 | mean range | max range | rank spread range |
| --- | ---: | ---: | ---: |
| counts_to_host | 0.153-0.215 ms | 0.286-0.447 ms | 0.160-0.434 ms |
| prefix_to_device | 0.193-0.270 ms | 0.294-0.548 ms | 0.160-0.517 ms |
| host_prefix | 约 0.002 ms | 约 0.003 ms | <0.001 ms |

两项跨设备边界合计约 0.35-0.49 ms，并且会放大 rank skew。

设计方向：

1. 为 stable/full preflight 增加模式：
   - stable：保留最小 count readback 与公开 API 必需字段；
   - full：保留当前完整 host 校验与诊断。
2. 优先做设备侧 public count publication：
   - 在 D6/D7 内生成 public expert prefix 与 unaligned count；
   - host 只读取 rank tail 与错误状态；
   - 不再 D2H 全量 count bridge 后再 H2D public bridge。
3. 若设备侧 publication 风险过高，先做保守合并：
   - 一次 D2H 只读必要元素；
   - 将 host prefix 构造合并进 count bridge；
   - public bridge 使用异步 H2D，并保留 stream ordering。

验收标准：

- stable 模式下 `dispatch_counts_to_host + dispatch_prefix_to_device`
  降至 0.10 ms 以下，或证明剩余部分为公开 API 必需；
- 8-rank submit/end spread 降低；
- Normal Dispatch mean/p95 改善或保持 neutral；
- cached/hybrid/stream/expanded 路径不回归。

### D2. Stage 间固定 gap 优化

优先级：P0。

rerun1 中每个 rank 都出现稳定 gap：

| Gap | 典型值 |
| --- | ---: |
| producer_group -> producer_prefix | 约 0.44 ms |
| epilogue_validate -> epilogue_validate_reduce | 约 0.51 ms |
| epilogue_metadata -> epilogue_copy | 约 0.39 ms |
| epilogue_expert_count -> epilogue_expert_prefix | 约 0.33 ms |

这些 gap 的总量超过多数 stage 本身。当前怀疑点是 one-stage-per-launch、
AICore/VF 分离提交和 transport service 顺序执行带来的固定调度成本，而不是
数据量。

设计方向：

1. 新增 launch-boundary profile：
   - host submit start/end；
   - AICore launch start/end；
   - VF service start/end；
   - device stage start/end；
   - no-op generic stage 与真实专用 VF stage 分开。
2. 合并小控制 stage：
   - producer control/group/prefix 可作为一个 control launch；
   - epilogue validate/validate_reduce/expert count/expert prefix/metadata
     可评估合并；
   - 保留 profile 模式下的语义化 stage ID。
3. 允许无依赖 stage 并发：
   - transport service 与 epilogue metadata 准备并发；
   - control publication 与 payload flush 的重叠仅在协议顺序证明安全后启用。

验收标准：

- 每个固定 gap 的主因可归类为 host submit、AICore launch、VF service 或
  真实数据依赖；
- 合并后端到端 mean 至少改善 0.3 ms；
- 不改变输出布局、generation、错误协议和 buffer 复用语义。

### D3. Release payload/control 拆分

优先级：P1。

当前 release 聚合阶段耗时：

| Stage | mean range |
| --- | ---: |
| release_payload | 0.905-0.914 ms |
| release_control | 0.256-0.268 ms |

command metrics 显示每 rank 约 7 个 put、约 286 MB payload，SQ/CQ
high watermark 均为 3，说明当前没有证据表明队列深度或 channel 数量是瓶颈。

设计方向：

1. 将 release_payload 拆分为：
   - put_staged_records_striped 构造与提交；
   - transport service submit；
   - payload flush/CQ wait；
   - completion 观察与错误处理。
2. 将 release_control 拆分为：
   - count/generation put_value；
   - release signal；
   - 本地 store_release；
   - peer publish observation。
3. 在得到拆分数据后，再评估：
   - payload 分批发布；
   - control signal 与 payload completion 的安全重叠；
   - WQE 数量是否需要降低。

验收标准：

- release_payload 内部每段有独立计时；
- 能明确 0.9 ms 中 transport 执行、提交和等待占比；
- 任何重排都必须通过 two-generation buffer reuse 和错误注入测试。

### D4. D8 epilogue copy 优化

优先级：P1。

当前 `epilogue_copy` 稳定在约 0.52 ms。8192-byte consumer tile 已经是
retained 默认，因此继续单纯调大 tile 的收益可能有限。

设计方向：

1. 增加 local/remote record 维度：
   - local copy 时间；
   - remote copy 时间；
   - hidden/scale/metadata 各自 copy 时间。
2. 评估双缓冲：
   - 一块 GM record 正在向 public output copy；
   - 下一块 record 已开始 GM-to-UB load。
3. 评估 arrival-driven copy：
   - 按 source rank 或 fixed source-prefix interval 观察到 payload ready 后
     提前 copy；
   - 不能绕过 validate、completion 和 reuse barrier。
4. 重新筛选 tile 大小时只做 same-binary ABBA，不与 D1/D2 改动混合。

验收标准：

- `epilogue_copy` mean 降低至少 15%，或证明其受 GM 带宽下限约束；
- Normal Dispatch mean/p95 不回退；
- Expanded/Cached Dispatch correctness 不回归。

### D5. Cached Dispatch 专项

优先级：P1。

当前五操作中 Cached Dispatch 约 70.3-71.4 ms，远高于 Normal Dispatch 的
约 7.3 ms。Cached path 不使用 device prefix、parallel prefix、token fanout
等 normal 路径 selector，并且仍有大量 host count bridge 与校验逻辑。

设计方向：

1. 单独采集 cached stage/host profile，不与 Normal Dispatch 混比。
2. 拆分 cached path 的：
   - host D2H/H2D；
   - cached control validation；
   - destination slot/metadata 复用；
   - producer release；
   - consumer epilogue。
3. 评估将已验证的 normal-path 机制迁移到 cached path：
   - device-side public prefix；
   - consumer tile specialization；
   - release split；
   - launch 合并。
4. 若 public handle/cache ABI 需要变化，必须单独设计，不直接修改 normal path。

验收标准：

- Cached Dispatch mean 从约 70 ms 至少降低 20%；
- cached handle reuse、two-generation 和 expanded cached correctness 通过；
- Normal/Expanded Dispatch 不因共享代码改动回归。

## 明确不做

1. 继续优化 `producer_record` 作为第一优先级：当前约 0.17 ms，已不是瓶颈。
2. 继续优化最终 `release_barrier` 作为第一优先级：signal-only 后约 0.04 ms。
3. 默认增加 HCOMM channel：当前 SQ/CQ watermark 只有 3，且 `CHANNELS=2`
   历史无收益。
4. 在 launch-boundary profile 完成前直接重写为大规模 pipeline。
5. 修改 logical bandwidth 公式或 workload 来制造优化结果。

## 开发顺序

1. D1 host count bridge 保守优化；
2. D2 launch-boundary profile 与小 stage 合并；
3. D3 release 拆分 profile；
4. D4 epilogue copy 双缓冲/arrival-driven；
5. D5 Cached Dispatch 专项。

每一步都必须：

1. 先提交 host contract/source contract；
2. 本地构建通过；
3. NPU8P 8-rank correctness 通过；
4. same-binary ABBA 或三次 30-iteration profile；
5. 只有端到端 mean/p95 改善或明确 neutral 且有诊断价值时才 retain。

## NOP 指令现状

代码中仍有两类容易混淆的写法。

### 1. 真实 NOP 指令

真实 NOP 存在于 polling loop 中：

- AICore path: `csrc/backends/ascend/transport/aicore_intrinsics.hpp`
  - `kPollingNopCycles = 800`
  - `AscendC::Nop<kPollingNopCycles>()`
- SIMT path: `csrc/backends/ascend/transport/simt_intrinsics.hpp`
  - `kPollingNopCycles = 55`
  - 循环调用 55 次 `asc_nop()`

调用位置包括 transport service、Dispatch/Combine acquire 和 persistent
pipeline 的 wait loop。这些 NOP 是有意保留的 polling cadence，用于降低
远程观察的访存压力，避免紧密轮询；不是残留的占位代码。

### 2. 源码占位符

`dispatch.asc` 与 `combine.asc` 中大量 `(void)0` 不是机器 NOP 指令。
它们主要来自历史实现被移除或由专用 VF stage 替代后的源码占位，用来保留
分支结构、stage 顺序和可读性。编译器通常会消除这些表达式；它们不等价于
`AscendC::Nop` 或 `asc_nop()`。

后续整理建议：不要把 `(void)0` 当成性能问题直接删除。应结合 stage
合并设计，在保证 source contract 和生成代码可读性的情况下分批清理或
重构分支结构。

## Driver 25.6 NOP-removal experiment

2026-09-24 在 NPU8P 上做了只改远端临时代码的 A/B，没有修改本地仓库。
Driver 为 `25.6.rc2.b023`，CANN/HCOMM 为 9.3.0。实验将两个底层
polling NOP helper 改为空实现：

- AICore path：`AscendC::Nop<800>()`；
- SIMT path：55 次 `asc_nop()`。

所有调用点和 polling loop 语义保持不变，`(void)0` 源码占位符没有删除。

结果：

| Run | Result |
| --- | --- |
| 5 warmup / 5 iteration smoke | exit 0，1 case passed |
| 30 warmup / 30 iteration | exit 0，1 case passed |

30-iteration 结果与当前 baseline 三次 run 对比如下：

| Configuration | Dispatch mean | Dispatch p95 | Logical bandwidth |
| --- | ---: | ---: | ---: |
| NOP retained rerun1 | 7.266 ms | 7.969 ms | 1071.584 GB/s |
| NOP retained rerun2 | 7.479 ms | 7.870 ms | 1041.065 GB/s |
| NOP retained rerun3 | 7.247 ms | 7.715 ms | 1074.459 GB/s |
| NOP removed | 6.958 ms | 7.372 ms | 1119.035 GB/s |

NOP removed 的关键 stage 与 baseline 基本相同：

| Metric | NOP removed |
| --- | ---: |
| producer_record | 0.168 ms |
| release_payload | 0.913 ms |
| release_control | 0.256 ms |
| release_barrier | 0.043 ms |
| epilogue_copy | 0.522 ms |
| cq_wait | 0.793 ms |
| service_submit | 0.521 ms |

结论：

1. Driver 25.6.rc2.b023 下，移除 polling NOP 的 correctness 通过；
2. 单次 30-iteration 显示 Dispatch mean 约 6.96 ms，p95 约 7.37 ms，
   相对三次 NOP retained baseline 的 mean 范围改善约 0.29-0.52 ms；
3. stage 内部耗时基本不变，说明收益大概率来自等待 loop 调度/轮询节奏，
   而不是 payload 或 copy 工作量变化；
4. 该实验尚未做 same-binary ABBA 和多轮重复，不足以直接 retain；
5. 已将真实 polling NOP 做成 build-time selector：
   `DEEP_EP_ASCEND_POLLING_NOP`，默认 `0`，即 AICore/SIMT helper 不再
   发出真实 NOP；设为 `1` 时恢复原调用，用于旧 driver 或 A/B 对照。
   `dispatch.asc`/`combine.asc` 中的源码占位 `(void)0` 不受该开关影响。

### Polling NOP selector 验收矩阵

1. 构建对照：
   - `DEEP_EP_ASCEND_POLLING_NOP=0`（默认）；
   - `DEEP_EP_ASCEND_POLLING_NOP=1`。
2. Correctness：
   - 至少 8-rank；
   - 覆盖 FP8/BF16、alignment 1/128、handle copy、cached、expanded、
     previous event、async、allocate on comm stream；
   - 观察卡死、超时、SIGSEGV 或 protocol error。
3. Performance：
   - 同 workload、同 commit、同 CANN/HCOMM；
   - 0/1 按 ABBA 或 AABB BBAA 交替，各至少三次 30-iteration；
   - 记录 Dispatch/Combine mean、p95 和 logical bandwidth。
4. Soak：
   - 典型 8-rank case 至少 100-300 iterations；
   - `DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=1` 必须保留。

### 2026-09-25 macro A/B 结果

环境：

- NPU8P-ALT，8 rank，device 0-7；
- Driver `25.6.rc2.b023`；
- CANN/HCOMM 9.3.0（`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`）；
- workload：8192 tokens、hidden 7168、top-k 8、256 experts、64 AIV；
- `DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=1`；
- 两个独立 binary 分别以 `DEEP_EP_ASCEND_POLLING_NOP=0/1` 构建。

Correctness：

| Test | NOP=0 | NOP=1 |
| --- | --- | --- |
| 5-case 代表子集 | 5 cases passed | 5 cases passed，run 结束后偶发 rank SIGSEGV |
| 12-case 矩阵 | 12 cases passed | 未重复完整矩阵 |
| 单 case 复跑 | 通过 | 失败 case 单独复跑通过 |

12-case 子集覆盖 FP8/BF16、alignment 1/128、bias 0/1/2、handle copy、
previous event、async、allocate on comm stream。两次 run 都是在 benchmark
报告 "cases passed" 之后、进程退出阶段出现 SIGSEGV；没有出现卡死、
timeout 或 protocol error。NOP=1 也在 correctness 和多轮 soak/ABBA 中复现
同样的退出阶段 SIGSEGV，因此该现象与 NOP=0 无关，更像当前环境或测试
shutdown 路径的既有偶发问题。

Soak（典型 case，30 warmup / 300 iterations，各 2 次）：

| Configuration | Dispatch mean | Dispatch p95 | Combine mean | Combine p95 |
| --- | ---: | ---: | ---: | ---: |
| NOP=0 | 7.068 ms | 7.531 ms | 14.576 ms | 15.158 ms |
| NOP=1 | 7.105 ms | 7.733 ms | 14.666 ms | 15.195 ms |

ABBA（30 warmup / 30 iterations，0/1 各 3 次）：

| Configuration | Dispatch mean | Dispatch p95 | Combine mean | Combine p95 |
| --- | ---: | ---: | ---: | ---: |
| NOP=0 | 7.014 ms | 7.471 ms | 14.392 ms | 14.824 ms |
| NOP=1 | 7.020 ms | 7.363 ms | 14.579 ms | 15.066 ms |

结论：

1. Driver 25.6.rc2.b023 下，默认不调用真实 polling NOP 未出现卡死；
2. NOP=0 的 300-iteration soak 两轮通过，功能层面可以保留默认关闭；
3. Dispatch mean/p95 和 Combine 指标在 0/1 之间方向不一致，差异在正常
   run-to-run 波动范围内，不能证明稳定性能收益；
4. 早期手动实验的约 0.3-0.5 ms mean 收益未被 controlled ABBA 复现；
5. 保留 `DEEP_EP_ASCEND_POLLING_NOP=1` 作为旧 driver 或归因对照，不作为
   性能优化推荐。

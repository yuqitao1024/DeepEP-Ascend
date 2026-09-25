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

2026-09-25 实现结论：

- 已实现独立 `direct_dispatch_publish_counts` VF kernel，在 D6 expert
  prefix 后直接生成 public expert prefix 与 unaligned count。
- 不把 public 指针追加到既有 prefix kernel 参数链，避免复现 argument list
  330 类问题；full preflight 自动关闭该优化，保留原完整 host 校验。
- stable 路径 host 侧只做一次 kernel count bridge D2H，host 内存内拆出
  rank tail 与本地 expert counts；不再执行 public count bridge H2D。
- NPU8P 8-rank、8192 tokens、hidden 7168、top-k 8、256 experts、
  30 warmups/30 iterations，D1 与 baseline 各 3 次：

| Operation | Baseline mean | D1 mean | 变化 |
| --- | ---: | ---: | ---: |
| dispatch | 7.306 ms | 6.969 ms | -4.6% |
| expanded_dispatch | 20.620 ms | 20.291 ms | -1.6% |
| cached_dispatch | 70.527 ms | 67.804 ms | -3.9% |
| combine | 14.707 ms | 14.878 ms | +1.2% |
| reduced_combine | 15.039 ms | 15.226 ms | +1.2% |

- D1 三次 run 的 `counts_to_host + prefix_to_device` 为 0.288、0.338、
  0.646 ms；baseline 三次为 0.868、0.664、0.923 ms。收益方向稳定，但
  host readback 本身仍有较大 run-to-run 抖动。
- full preflight 与 stable correctness case 均通过。30 iteration 长跑中，
  D1 与 baseline 都出现过 rank teardown 阶段 SIGSEGV；case 本身
  显示 `1 cases passed`，且崩溃也在未启用 D1 的 baseline 出现，因此不是
  D1 新引入问题。该 teardown 问题需单独跟踪，不能算作功能失败。

原始结果归档在 NPU8P 工作区：

- D1: `/home/pyptouser/yuqitao/deepep-d1-publish/results/d1-perf-run{1,2,3}.json`
- Baseline: `/home/pyptouser/yuqitao/deepep-baseline/results/d1-baseline-run{1,2,3}.json`
- Stable/full smoke:
  `deepep-d1-publish/results/d1-smoke-r5.json` 与
  `deepep-d1-publish/results/d1-full-smoke-r5.json`
- 语义修复后的最终 stable smoke:
  `deepep-d1-publish/results/d1-final-smoke.json`，结果 `1 cases passed`。
- Stream/previous-event 代表 case（previous-event、async、async+allocate、
  async+bias2）合跑结果 `3 cases passed`，归档为
  `deepep-d1-publish/results/d1-stream-regress.json`。该进程同样在 case
  全部通过后的 teardown 阶段出现 SIGSEGV，与 D1 的 correctness 结果无关。

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

2026-09-25 no-op 跳过项复核结论：

- 在包含 D1 的当前代码上重测
  `DEEP_EP_ASCEND_SKIP_EPILOGUE_NOOP=1`，非 profile direct Dispatch 的
  acquire/validate redundant AICore kernel 被跳过；profile 模式仍保留该
  stage 计时。
- 初次三次测试使用了 `--profile-stages`。该开关在 profile 模式下会主动
  关闭跳过逻辑，因此初次数据不是 no-op 跳过的有效 A/B 证据。
- 补做三次无 `--profile-stages` 的 ABBA 对照。二进制归属复核为：
  `deepep-d1-publish` 是 OFF，`deepep-d2-noop` 是 ON。Dispatch OFF
  均值 6.969 ms，ON 均值 6.810 ms；Expanded OFF 20.291 ms，ON
  20.180 ms；Cached OFF 67.805 ms，ON 67.507 ms。Normal Dispatch 的
  差异在 run-to-run 波动范围内；ON 对 Expanded/Cached 有稳定小幅
  正向，但幅度远小于 stage gap 本身。
- 代表 correctness 集（normal、previous-event、async、async+bias2）通过，
  `4 cases passed`。结果归档于 NPU8P
  `/home/pyptouser/yuqitao/deepep-d2-noop/results/d2-noop-{run1,run2,run3,regress}.json`。
- 结论：撤回默认开启，恢复 `OFF`。该开关仍可用于诊断，但不作为生产
  性能优化推荐。
- 随后把非 profile 跳过范围从 acquire/validate 扩展到
  validate_reduce/expert_count/expert_prefix/metadata 四个 redundant
  AICore launch。三次 30-iteration 结果：Dispatch mean 6.919 ms，对比
  仅跳过 acquire/validate 的 6.810 ms 无稳定收益；但 Expanded Dispatch
  20.083 ms、Cached Dispatch 67.604 ms 均有稳定小幅改善。代表
  correctness 集 `4 cases passed`。
- 按逐项独立保留规则，本次扩展暂不保留，恢复只跳过 acquire/validate；
  Expanded/Cached 的正向结果记录为后续专门验证线索，不能作为单独保留
  Normal Dispatch 回退项的理由。

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

#### D3 计时口径修正（2026-09-26）

本轮先校正 profile，再决定是否修改发布协议。此前 D3 smoke 不能作为
优化收益证据：Dispatch 的 release-ablation mask 只包含 stage 13/14，
漏掉 stage 1–12；phase 推导仍使用旧的 14/15 编号。现修正为完整的
1–14，并将 expert_count（stage 8）归入 consumer compute。

计时定义：

- `service.start/end` 保留首尾时间戳，二者差值是跨 launch 的 envelope。
- `service_active_cycles` 累加每次 service 执行区间，计时器位于独立
  noinline wrapper，避免在大型 executor 中保留长生命周期计时状态。
- payload/control/flush/barrier 四类 command cycles 互不重叠；flush
  包括未显式入队的最终 drain，control 包括 Signal。
- `cq_drain_cycles` 是 CQ drain 函数的耗时，包含轮询、CQ 处理和状态更新，
  不能等同于纯网络等待；它与 barrier poll 都是上述 command 时间的子集。
- `other_active_cycles` 只从 active 时间扣除四类 command，不再次扣除
  drain/poll。`launch_gap_cycles` 单独记录 envelope 与 active 的差值。
- 缺失或不合法的计时标为 unavailable，不用零值冒充实测。跨 rank 的
  max 仅用于观察各项上界，不可相加或用于占比；占比需使用同一 rank。

新增 profile 字段后 ABI 从 3 升至 4，host/device 必须一起重新编译。
非 profile 路径中的新增时钟读取通过 `if constexpr` 编译移除。逻辑
带宽公式及 benchmark event 计时边界保持原样。历史表中的
`service_submit` 是旧 envelope 口径，不能直接作为当前 active 提交耗时。

上一轮任务 `task_20260925_234411_233712123803` 在结果输出前超时，
不是已知的 teardown SIGSEGV；旧 `d3-smoke.json` 是此前成功任务遗留。
本轮使用唯一输出文件名，并分别验证退出状态、correctness 和全部 rank
的 profile，避免误用旧结果。尚未根据这些诊断保留任何协议重排优化。

修正后的三次 profile：NPU8P、devices 0–7，CANN
`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`，同树 HCOMM，
8192 tokens / hidden 7168 / top-k 8 / experts 256 / FP8 / num_sms 64。
`d3-attribution-v4-r1/r2/r3.json` 全部五个操作、八个 rank 的 profile
一致性检查通过，任务正常退出，未复现先前的输出前超时。

以下每列取该次 **service active 最慢的同一 rank**，单位为 cycles，
不能将 drain 与 flush 再相加：

| Dispatch 指标 | r1 (rank 4) | r2 (rank 2) | r3 (rank 2) |
| --- | ---: | ---: | ---: |
| service active | 1254978 | 1253398 | 1242317 |
| payload command | 73571 | 74556 | 69646 |
| control command | 203209 | 200100 | 183394 |
| flush command | 901788 | 907680 | 922091 |
| CQ drain（嵌套） | 800354 | 807532 | 830587 |
| other active | 76410 | 71062 | 67186 |
| launch gap（active 外） | 108805 | 110284 | 107686 |

CQ drain 占 active 约 64%–67%。signal-only 配置下没有 Barrier opcode，
因此 barrier command/poll 为零是有效结果，不代表 release_barrier stage
不存在。每 rank 仍为 7 个 payload put、总计 30 个 commands、SQ/CQ
high watermark 3，采样结束时 SQ/CQ depth 都为零。

这些计时包含 profile 自身开销，仅用于定位。当前 command counters
覆盖整个 operation，尚未独立拆开 release VF 的命令构造及逐 peer
control/signal 发布；不能将 `release_payload` 的 AICore stage span
当作包含该 VF 的端到端 release 时间。

本轮独立实验：将 CQ owner/status 轮询从 64B MTE2/UB 往返改为
cache-bypassing 32-bit scalar read，保持完成检查、tail/doorbell 更新和
payload→control 发布顺序不变。

#### D3 CQ scalar poll：保留（2026-09-26）

三组无 profile 的 ABBA，A 为上述计时修正后的 MTE2 版本，B 仅替换
CQ poll。每次 warmups=2、iterations=30，五个操作均使用同一典型
case `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`。测量边界、
逻辑带宽公式和性能 selector 完全一致；没有重排 payload/control。

Dispatch device event mean（ms）：

| Batch | A1 | B1 | B2 | A2 | A mean | B mean | 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abba2 | 6.612239 | 6.661284 | 6.234582 | 6.577938 | 6.595088 | 6.447933 | 2.231% |
| abba3 | 6.653921 | 6.507129 | 6.414394 | 6.570464 | 6.612193 | 6.460761 | 2.290% |
| abba4 | 6.594972 | 6.588084 | 6.414740 | 6.620831 | 6.607901 | 6.501412 | 1.612% |

每组分别以两次 A、两次 B 的 run mean 求平均；三组总体 Dispatch
均值约从 6.605 ms 降至 6.470 ms，改善约 2.04%。保留依据是三组
独立 ABBA 的均值改善复现，不设置百分比门槛。单次 run 和 p95 仍有
波动，不声称 tail latency 有稳定收益。

其他操作的 batch mean 改善（正数为更快）：

| Operation | abba2 | abba3 | abba4 |
| --- | ---: | ---: | ---: |
| Expanded Dispatch | +0.516% | -0.284% | +0.183% |
| Cached Dispatch | +0.315% | -0.205% | -0.165% |
| Combine | +0.615% | +0.811% | -0.713% |
| Reduced Combine | +1.330% | +0.313% | +0.028% |

三组共 12 次运行、每次五操作 correctness 均通过；候选单独一次
10-iteration profile 的全部 rank 也通过归因一致性检查。未观察到
输出前卡死。已知 teardown SIGSEGV 仅在 JSON 完整写出且五操作全部
通过后接受，不视作正常退出；相关日志保留。未修改既有完成错误码、
owner/status 校验、重试计数、SQ/CQ tail 或 doorbell 更新。

复现与证据均在 NPU8P `/home/pyptouser/yuqitao/deepep-d3/`：

- `results/d3-attribution-v4-r{1,2,3}.json`：归因基线；
- `results/d3-cq-scalar-profile.json`：候选 profile；
- `results/d3-cq-abba{2,3,4}-{A1,B1,B2,A2}.json` 及同名 `.log`；
- `d3-run-verified.sh`、`.scratch/d3-abba.sh`：环境与运行命令；
- A 二进制 SHA256：`f8cd209b52e9b1ce0abb3b2b9cef40907b61ff87275db885dfa82cf3eaf19124`；
- B 二进制 SHA256：`0eb9dc732d1e5fe9fffa9ec83993810c557802146dfa8737fd49ae08975db8c7`；
- 三组任务：`task_20260926_032650_304226011631`、
  `task_20260926_032943_305511715446`、`task_20260926_033306_30687755755`。

`abba1` 在 A1 已写结果后的已知 teardown SIGSEGV 处中断，不纳入
完整 ABBA 统计。以上结论覆盖 NPU8P 的指定环境和典型 case，不扩展
声称其他 CANN/设备或所有 workload 均有收益。

最终源码额外补充了 active interval 超出 envelope 时的拒绝检查，重新
编译并运行 `task_20260926_033841_309434719540`：
`d3-final-profile.json` 的五操作、全部 rank 归因检查通过；
`d3-final-performance.json` 的五操作 correctness 通过，30-iteration
mean 分别为 Dispatch **6.522 ms**、Expanded Dispatch 19.987 ms、
Cached Dispatch 67.293 ms、Combine 14.669 ms、Reduced Combine
14.948 ms。这一单次收尾运行不替代上述三组 ABBA。performance 运行
在写出完整结果后出现已知 teardown SIGSEGV，未出现计算期间错误。
最终二进制 SHA256 为
`c6d9af28b527d360c6bfce57136d41db414db9c20f9c7e3608db668be187735d`。
本地 benchmark/transport/SIMT-URMA 相关测试 **136 passed, 3 skipped**；
另通过 profile codegen boundary 的两个 source contract 检查。

D3 本轮完成 service 归因及一个有重复收益的优化；release VF 内部的
命令构造、逐 peer 发布细分仍待补齐。分批 payload 和 control/completion
重叠尚未实施，后续仍须按 two-generation reuse 和错误注入要求验证。

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

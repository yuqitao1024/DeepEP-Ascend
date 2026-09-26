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

这些 gap 的总量超过多数 stage 本身。最初怀疑点是 one-stage-per-launch、
AICore/VF 分离提交和 transport service 顺序执行带来的固定调度成本，而不是
数据量。2026-09-26 的完整 kernel trace 已修正这一归因：大部分固定 gap
是阶段计时未覆盖的专用 VF kernel，见下文“D4 后普通 Dispatch 关键路径复核”。

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
- 合并后端到端 mean 的改善可通过重复对照稳定复现，不设置绝对耗时门槛；
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

计时边界核查：该 stage clock 主要覆盖 AICore hidden payload copy。
FP8 scale、top-k weight 和 hidden scalar tail 由前置的
`direct_dispatch_epilogue_copy_outputs_vf` 处理，metadata 也有独立 stage；
因此不能把 0.52 ms 当成整个输出拷贝路径的时间。local/remote record 在此
阶段均从本卡 receive shard 读取，remote 表示数据来源 rank，不能将这段
GM 拷贝时间解释为 P2P 传输时间。

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

- 收益必须通过重复对照稳定复现，不设置百分比门槛；stage 缩短需同时
  对照未开启 profiling 的端到端 Dispatch，不能仅凭单次 stage 数值保留；
- Normal Dispatch mean/p95 不回退；
- Expanded/Cached Dispatch correctness 不回归。

#### D4 双缓冲实测（2026-09-26）

实现仅调整 AICore hidden payload copy：使用两个 `TileBytes` 大小的 UB
buffer，按实际提交的 tile 交替使用 event 0/1。MTE2 在复用 buffer 前等待
该 buffer 的 MTE3 完成，下一条 record 的 load 可以与上一条 record 的
store 重叠；退出时消费两个完成事件，包括无工作和单 tile 的情况。
所有 consumer launcher 的 UB 分配同步增加到 `2 * TileBytes`。
不改变 VF 参数列表、输出 layout、producer 或跨 rank 协议。

典型 hidden 为 7168 bytes，8192-byte tile 一次即可覆盖一条 record；因此
buffer 轮换必须跨 record，而不能在每条 record 开始时重新初始化。
Expanded 中跳过的 lane 不消耗 buffer/event。

环境与对照方法：

- NPU8P，队列独占 device 0–7，8 rank；CANN/HCOMM 使用
  `/data/disk2/cann_version/0916/use_cann/cann-9.3.0` 及其 `aarch64-linux`。
- A 为 `1037d33` 的 D3 最终二进制；B 为仅加入上述双缓冲的二进制。
  两者均为 Release，NOP/diagnostics/testing/skip-noop 关闭，signal-only 开启。
- 固定 8192 tokens/rank、hidden 7168、top-k 8、256 experts、64 AIV，
  `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`。
  stable preflight、device/parallel prefix、token fanout 开启；consumer tile
  8192；Combine 使用既有 `32768/512`、direct local placement 和 expanded
  vector reduce 配置。
- 三组独立 A1/B1/B2/A2，每次进程重启，2 warmups、30 iterations，
  不开启 stage profiling，保留完整功能校验。每个版本共 6 次 run。
  此处 warmups 为 2，不与文首历史 30-warmup 数据混算。

下表正值表示耗时降低；p95 列先计算每次 run 的 p95，再对 A/B 各两次取均值。

| 指标 | ABBA 1 | ABBA 2 | ABBA 3 |
| --- | ---: | ---: | ---: |
| Dispatch mean | +3.969% | +7.309% | +6.472% |
| Dispatch p95 | +3.791% | +11.709% | +6.369% |
| Expanded Dispatch mean | +13.146% | +14.866% | +13.892% |
| Cached Dispatch mean | +3.497% | +3.528% | +3.066% |
| Combine mean | -1.303% | +1.553% | -0.355% |
| Reduced Combine mean | -1.118% | +1.726% | -0.651% |

各版本 6 次 run 的平均耗时，以及用同一逻辑字节数除以该平均耗时得到的带宽：

| 操作 | A mean ms | B mean ms | A logical GB/s | B logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | 6.536389 | 6.149285 | 1191.191 | 1266.178 |
| Expanded Dispatch | 19.914144 | 17.131861 | 464.569 | 540.017 |
| Cached Dispatch | 67.392924 | 65.125878 | 115.533 | 119.554 |
| Combine | 14.604170 | 14.608250 | 746.445 | 746.236 |
| Reduced Combine | 15.015298 | 15.015563 | 726.006 | 725.994 |

逻辑带宽仍为所有 rank 的逻辑通信/拷贝/归并字节数口径，不是单链路 P2P
带宽。Dispatch 三种模式的收益均稳定复现；Combine 两项方向不一致，汇总
差异接近零，不宣称其有性能收益。

独立 profile pair（每次 2 iterations）中的 `epilogue_copy` stage span：

| 操作 | A cycles | B cycles |
| --- | ---: | ---: |
| Dispatch | 523730 | 317830 |
| Expanded Dispatch | 6660934 | 3904499 |
| Cached Dispatch | 5262707 | 2952823 |

按现有报告的 1 GHz 时钟口径，Normal hidden copy 为 0.524 → 0.318 ms。
该单次 profile 仅用于归因，保留依据是上述未开启 profiling 的重复对照。
本次没有新增 local/remote 独立计时或 scale/weight VF 独立计时，不能用表中
数字推导这些部分的耗时；arrival-driven copy 和新 tile sweep 未纳入本次。

验证与已知限制：

- 典型用例的 profile pair 和三组 ABBA 共 14 次 run 均完成五操作功能校验。
  部分原版 run 在结果成功写出后出现已知 teardown SIGSEGV；仅在确认全部
  结果通过、成功标记早于 SIGSEGV、无其他退出错误后接受。
- 本地 layout/tiling、production dispatch state/layout、parallel layout
  host-callable 三项检查通过；NPU8P 完整 CANN 编译通过。
- 新增 `tests/ascend/production/run_dispatch_copy_boundaries.py` 独立验证
  hidden、FP8 scale、路由 metadata 和 padding，不传 top-k weights。
  八个场景已在 8 rank 全部通过，覆盖五种 tile、奇偶 tile 数、scalar tail、
  空输入/路由、Expanded 无效 lane、Cached/Expanded Cached 及
  previous-event async；典型用例覆盖大量 record 间的 buffer 轮换。
  首次运行因测试脚本未将列主序 scale 转成 contiguous 后再作字节视图而
  失败；修正后完整重跑通过，最终退出阶段的已知 SIGSEGV 按上述规则核验。
- 扩展的五操作边界用例发现既有 weight 问题：8 rank、17 tokens、hidden
  4865、BF16、top-k 2、16 experts、masked ratio 0.5、tile 512 时，原版和
  双缓冲均复现某一接收 rank 的有效 Dispatch weights 变成 0，并传播至
  Combine；出错 rank 会变化。原版一次无诊断复跑还在 180 秒限制下超时。
  此问题未修复，不把该五操作边界用例记为通过，也不声称全矩阵无问题。

结论：保留 D4 双缓冲；其典型性能和独立 payload 边界验证已完成。
weight 问题仍是独立待办。复跑 payload 回归时，在相同 CANN/HCOMM/Python
环境和队列分配下执行：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module tests.ascend.production.run_dispatch_copy_boundaries \
  --num-sms 64 --output d4-copy-boundaries.json
```

原始数据保存在 NPU8P `/home/pyptouser/yuqitao/deepep-d4/results/`：

- `d4-control-profile.json`、`d4-buffer-profile.json`；
- `d4-abba{1,2,3}-{A1,B1,B2,A2}.json` 及同名 `.log`；
- `d4-copy-boundaries-v2.json` 和同名 `.log` 为最终 payload 回归结果，
  task `task_20260926_043424_13073638856`；
- `d4-tail-tile512.log`、`d4-tail-control512.log`、
  `d4-weight-{control,double-buffer}.log` 保留失败及对照证据。
- A SHA256：`c6d9af28b527d360c6bfce57136d41db414db9c20f9c7e3608db668be187735d`；
  B SHA256：`75cd9b35f5dbcd29a832f7cabde77b4ff625f5da011a309666e1ab5a7a2ab4f2`。
- 编译 task `task_20260926_040157_31757501652`；profile pair task
  `task_20260926_040415_319115028530`；ABBA tasks
  `task_20260926_041249_1229072165`、`task_20260926_041545_12418695524`、
  `task_20260926_041822_125384114887`；weight 对照 task
  `task_20260926_042722_128910326848`（任务脚本成功表示两次诊断已执行，
  两个 workload 本身均退出 1，并非功能通过）。

### D4 后普通 Dispatch 关键路径复核（2026-09-26）

按当前开发范围，暂停 D5 Cached Dispatch，集中优化普通 Dispatch。
基线为 `645fa84`，NPU8P device 0-7、8 rank、8192 tokens/rank、hidden
7168、top-k 8、256 experts、FP8、64 AIV。CANN/HCOMM 仍使用
`/data/disk2/cann_version/0916/use_cann/cann-9.3.0`，沿用 D4 的优化开关。
基线二进制 SHA256：
`75cd9b35f5dbcd29a832f7cabde77b4ff625f5da011a309666e1ab5a7a2ab4f2`。

#### 阶段采样

三次独立 `--profile-stages --warmups 30 --iterations 30` 均通过五操作功能
校验。以下 mean/p95 属于开启阶段 profiling 的计时路径，不与 D4 无 profiling
的 6.149 ms 直接比较，也不据此判断回退。

| Run | Dispatch mean / ms | p95 / ms | Host completion anchor spread / ms |
| --- | ---: | ---: | ---: |
| 1 | 6.559796 | 7.037377 | 0.31439 |
| 2 | 6.573279 | 6.914325 | 0.26603 |
| 3 | 6.583718 | 6.873264 | 0.37044 |

各 rank 已计时 stage 活动约 1.745-1.809 ms；阶段之间尚未归属的时间约
2.25-3.20 ms。这部分不能标成 NPU idle 或 host launch 开销。
Release attribution 的 max-per-rank CQ drain 为 810504/811306/812556
cycles，flush command 为 903840/915266/910554 cycles；两者存在包含关系，
不能相加。不同 rank 的最大值也不能合成为一条实际关键路径。

同三次采样的 host count bridge（max-per-rank）为 0.373/0.315/0.325 ms，
prelaunch setup 为 0.089/0.149/0.091 ms，prefix H2D 均为 0。
这与 D1 的 device-side public prefix 路径一致；约 0.5 ms 的固定 VF 工作
应与 host preflight 分开归因。

#### 完整 kernel trace

另外使用 Torch-NPU CPU/NPU profiler，在普通 Dispatch 的 30 次预热后
捕获三次调用；HCCL 入口同步的 host 提交放在调用标记外，但没有额外等待
设备完成，因此其异步执行可能延续到标记内。五操作 benchmark 校验通过，
随后八个 rank 都导出了 trace，任务正常退出 0。第 0 次捕获有明显 profiler
启动扰动，固定 kernel 耗时统计只使用第 1/2 次，共 16 个 rank/capture 样本。

| 专用 kernel | min / μs | median / μs | max / μs |
| --- | ---: | ---: | ---: |
| epilogue_validate_records | 521.460 | 522.217 | 523.135 |
| producer_prefix | 435.098 | 435.791 | 436.204 |
| epilogue_count_experts | 327.694 | 329.170 | 329.777 |
| epilogue_parallel_prefix | 169.314 | 175.490 | 176.181 |
| epilogue_metadata | 169.892 | 170.616 | 171.621 |
| producer_release | 160.963 | 166.369 | 171.057 |
| epilogue_copy_outputs | 164.742 | 165.776 | 167.026 |
| producer_record | 117.869 | 119.043 | 122.056 |
| epilogue_acquire | 66.228 | 68.658 | 1687.572 |

例如 rank 0 第 2 次捕获，从首个 Dispatch kernel 到最后的 complete kernel
跨度为 3.68189 ms，kernel 时长之和 3.68129 ms，间隙合计仅约 0.59 μs。
原来约 0.44/0.52/0.33/0.18 ms 的阶段间 gap，分别对应 producer prefix、
validate records、count experts、parallel prefix 的真实执行。
metadata→copy 的约 0.39 ms 区间还包含 metadata、assign destinations、
reduce errors、clear padding 和 copy outputs 专用 kernel。

该 rank 的完整 host 调用约 5.975 ms，首个 kernel 前约 0.558 ms，最后一个
kernel 后约 1.735 ms。首个 kernel 前的区间可能包含入口 HCCL 等待，不能
视为纯 host 开销。host 尾部可见 narrow 和 CPU tensor copy；当前 trace
没有覆盖所有自定义 runtime API，不能把这些剩余时间全部归因于 preflight
或 pybind。另一些 rank/capture 确有较长间隙（最高约 1.345 ms）和 acquire
等待（最高约 1.688 ms）。两次捕获不足以判断其发生频率或证明同步缺陷。

后续按单变量推进：

1. 专家计数的线程内 histogram 已完成独立对照，未取得稳定收益，撤回实现
   （见下文）。后续计数优化需先验证存储布局与并行粒度，不能假定普通局部
   数组一定比 GM 计数快。
2. validate records 的并行粒度实验见下文；其后评估 producer/epilogue prefix
   的串行扫描。保留全部校验和确定性的错误汇总，不通过删检查获得收益。
3. 入口 HCCL 显式等待后的 20 次 capture 已完成（见下文）；下一步细分
   kernel 完成后的结果回读、runtime completion 和 handle 收尾时间，同时
   保留 host 提交边界测量，区分晚提交与跨 rank 等待。
4. 所有候选以无 profiling 的重复 ABBA 验收，普通 Dispatch 收益稳定才保留。

原始结果位于 NPU8P `/home/pyptouser/yuqitao/deepep-d4/results/`：
`dispatch-after-d4-profile-{1,2,3}.json`、
`dispatch-after-d4-kernel-trace.json`、`dispatch-after-d4-traces/rank{0..7}.json`。
阶段采样 task 为 `task_20260926_044701_133165922041`，完整 trace task 为
`task_20260926_044925_134136317897`。

#### 等待入口 HCCL 完成后的补充 trace

在 D4 基线上，调用标记前显式 `torch.npu.synchronize()`，然后捕获 20 次
普通 Dispatch。剔除每 rank 第 0 次，剩余 8 × 19 = 152 个 rank/capture
样本。五操作校验通过，trace 全部导出，任务退出 0。

| 区间 | median / μs | p95 / μs | max / μs |
| --- | ---: | ---: | ---: |
| 调用标记开始 → 首个 Dispatch kernel | 265.928 | 438.356 | 582.259 |
| 首个 → 最后一个 Dispatch kernel | 3884.224 | 4123.776 | 4241.832 |
| 上述 kernel 之间的间隙总和 | 13.196 | 23.428 | 412.121 |
| epilogue acquire kernel | 240.877 | 493.392 | 613.832 |
| 最后一个 Dispatch kernel → 调用标记结束 | 1557.563 | 2520.315 | 3125.087 |

这是 rank/capture 样本的分布，不能相加各行 p95/max，也不是 benchmark 的
逐 iteration max-rank latency。标记结束包含显式设备同步和返回值处理。
最后一行还可能包含数据回读/其他 runtime 操作，不能标成纯 CPU 或设备空闲。

固定 kernel 再次得到确认：validate median 522.055 μs，producer prefix
435.728 μs，count experts 329.165 μs，epilogue prefix 175.530 μs。
因此可以优先优化实际 VF 工作；另需细分较大的调用尾部，避免把它与阶段
gap 混在一起。显式同步改变了各 rank 的进入时序，且所有数据仍在 profiler
下，不能根据本次 acquire 最大值较小就宣称长尾已修复。

结果在 D4 目录的 `results/dispatch-entry-synced-trace.json` 和
`results/dispatch-entry-synced-traces/rank{0..7}.json`，task 为
`task_20260926_050616_141016430903`。

#### 专家计数私有 histogram 对照：不保留

单变量候选只改 `direct_dispatch_epilogue_count_experts.asc`：每线程仍拥有
一个 128-record tile，在 local experts ≤ 32 时用 32 个 `uint64_t` 局部
计数器累加，结束后写回 GM；更大 expert 数保留旧分支。未修改 ABI、workspace
布局、错误传播或同步协议。独立目录编译完成，源码目录对照只有这个文件不同。

候选完整 trace 中 count experts 的 median 从 329.170 μs 增至 392.195 μs，
其他固定 VF kernel 基本不变。仅凭源码减少 GM 更新次数，不能推断实际访存
或指令成本下降；本次没有编译器 lowering 证据，不将变慢进一步归因于某种
特定 spill 或 UB 行为。

三组无 profiling 的 ABBA，每次 2 warmups / 30 iterations，与 D4 验收口径
一致。A 为 D4，B 为私有 histogram。每组表值是两个 A/B run 各自均值：

| 组 | A Dispatch mean / ms | B mean / ms | mean 收益 | A p95 / ms | B p95 / ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.157462 | 6.149411 | +0.131% | 6.699107 | 6.896977 |
| 2 | 6.074498 | 6.362223 | -4.737% | 6.583800 | 6.736078 |
| 3 | 6.243174 | 6.604998 | -5.796% | 6.792412 | 7.191284 |

六个等长 run 合并：A **6.158378 ms / 1264.309 GB/s**，B **6.372211 ms /
1221.882 GB/s**。带宽沿用八 rank 聚合 logical bytes，不能解释为单链路 P2P
带宽。A 与 D4 先前 6.149285 ms / 1266.178 GB/s 的结果一致。

12 次 run 都完成五操作功能校验，没有卡死或协议错误。B 第一组第一次在
完整成功结果后出现已知 teardown SIGSEGV，按既定严格检查接受；其余正常
退出。Expanded Dispatch 合并均值为 17.274230 → 17.406393 ms，亦无收益。
结论：撤回候选，生产 kernel 维持 `645fa84`；不新增性能开关或诊断代码。

原始 A 结果在 `/home/pyptouser/yuqitao/deepep-d4/results/`，B 在
`/home/pyptouser/yuqitao/deepep-dispatch-hist/results/`，文件名均为
`hist-abba{1,2,3}-{A1,B1,B2,A2}.json` 及对应 `.log`（各目录只有对应 A 或 B）。
B 的 `hist-kernel-trace.json`、`hist-traces/rank{0..7}.json` 和实验源码保留在
独立目录中。候选二进制 SHA256 为
`1a1f4e6f9d4017bad8798f9007d8ba69cd3375690505c8ad8bfb058f89eb271e`。
编译 task `task_20260926_045604_135826716442`，候选 trace task
`task_20260926_045806_136990814002`，ABBA task
`task_20260926_045900_137634923157`，均已结束。

本轮临时采样/分析脚本与下载结果归档于本地
`/tmp/deepep-dispatch-after-d4-20260926.3mBLpT/scratch/`；远端新增脚本归档于
`/home/pyptouser/yuqitao/dispatch-after-d4-20260926.R8hw7E/`。主工作树只保留
本文的测量与决策记录，不保留实验 kernel 或临时诊断代码。

### 普通 Dispatch 接收记录校验并行化（2026-09-26）

基于上一轮已确认的约 522 μs 校验耗时，本轮只改
`direct_dispatch_epilogue_validate_records.asc` 的 uncached tile 分工。
对照 A 为 D4 二进制 `75cd9b35...`，候选 B 为
`38a0b3f211d8d43c25aea814ba40a28ea4487bbe96570e87693a2b9387cfd25d`。
仍使用 NPU8P device 0-7、8192 tokens/rank、hidden 7168、top-k 8、256
experts、FP8、64 AIV 和同一 CANN/HCOMM、优化开关。

实现与协议边界：

- 原来每线程串行检查一个 128-record tile；普通路径改为每个 32-thread
  subgroup 拥有一个 tile，每 lane 检查连续 4 条记录。
- 每 lane 仍按记录顺序取第一个错误，`asc_ballot` 选最早出错的 lane，
  `asc_shfl` 传递该 lane 的错误与 diagnostic peer，只有 lane 0 写 tile
  结果。连续分段保证最小出错 lane 对应最早出错记录，不能改成仅按错误码
  取最小值，也不能直接把记录交错分配后仍按 lane 顺序取错误。
- 跨 tile 的 atomic-min、block 内完成汇总、最后一个 block 发布错误的
  顺序均保持原样。kernel 参数、workspace、generation 和发布协议未变。
- cached 模式和不满完整 subgroup 的 launch 保留逐线程扫描。D5 继续暂停。

#### 功能回归

新增 `tests/ascend/production/run_dispatch_validation.py` 和 host adapter
`tests/ascend/core_ops/dispatch_validate_adapter.cpp`。adapter 使用生产
launcher 声明与真实 C++ POD，Python 只传指针和标量，不复制复杂参数 ABI。
测试从已加载的 `_C` 取得生产 launcher 地址，构造本地 device receive records，
与独立串行 oracle 比较 status、每 tile 错误、全局候选、完成计数和 tile
输出边界。

原版与候选均在 device 0-7 通过，每设备 33 个场景，包括：

- 1/8 个 source 的布局；capacity 0/1/17/129/137；空路由、部分 source count；
- 32/128/512 threads、跨多次 grid-stride 的 8193 capacity、最后一个部分 tile；
- 同 lane 多个错误、不同 lane 中错误码相反的先后顺序、跨 tile 多个错误；
- 已有 status 不被覆盖、cached identity 和 expanded cached 非本地 slot；
- top-k 1/2/8/32。

该测试直接覆盖生产校验 kernel，但不提交跨 rank transport 操作；真正的
8-rank 协议和输出验收另由五操作 benchmark 完成。复跑时，在匹配的
CANN/Python 环境中，通过主机规定的 CPU/NPU 队列分别执行：

```bash
c++ -std=c++17 -shared -fPIC -I. \
  tests/ascend/core_ops/dispatch_validate_adapter.cpp \
  -o build/dispatch-validate-adapter.so

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/ascend/production/run_dispatch_validation.py \
  --adapter "$PWD/build/dispatch-validate-adapter.so" \
  --output "$PWD/results/dispatch-validation"
```

#### Kernel 归因

候选同样在入口 HCCL 完成后捕获 20 次普通 Dispatch，剔除首轮，8 × 19 个
样本中 validate kernel median **46.121 μs**，p95 **47.511 μs**，范围
45.003-48.310 μs。此前对照 median 为 522.055 μs，减少约 476 μs。
producer prefix 435.750 μs、count experts 329.833 μs、epilogue prefix
175.596 μs，未随该修改明显改变。该结果支持减少每 lane 串行记录数的假设；
最终保留依据仍是无 profiling 的端到端重复对照。

#### 端到端 ABBA：保留

三组 ABBA，每次 2 warmups / 30 iterations，不开 stage profiling。全部
报告的 workload fingerprint、64 AIV、30 个采样和五操作结果已核对。
每组的 A/B mean 和 p95 分别是两个对应 run 的指标均值：

| 组 | A Dispatch mean / ms | B mean / ms | mean 收益 | A p95 / ms | B p95 / ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.352133 | 5.852685 | 7.863% | 7.008137 | 6.290833 |
| 2 | 6.219534 | 5.833915 | 6.200% | 6.652391 | 6.439599 |
| 3 | 6.311999 | 6.002317 | 4.906% | 6.841331 | 6.475157 |

六个等长 run 合并：

| 操作 | A mean / ms | B mean / ms | B logical GB/s |
| --- | ---: | ---: | ---: |
| Dispatch | 6.294555 | 5.896305 | 1320.503 |
| Expanded Dispatch | 17.281871 | 16.703687 | 553.859 |
| Cached Dispatch | 65.133734 | 65.161330 | 119.489 |
| Combine | 14.587793 | 14.514459 | 751.058 |
| Reduced Combine | 15.072036 | 15.036815 | 724.968 |

Normal Dispatch mean 改善约 **6.327%**，logical bandwidth 从
**1236.956 → 1320.503 GB/s**；仍是八 rank 聚合逻辑带宽，并非单链路 P2P
吞吐。这里的百分比只与本轮配对 A 对比，不与上一轮 6.158 ms 的基线混算。
Expanded Dispatch 三组 mean 改善 4.355%/2.973%/2.696%，合并约 3.346%。
Cached/Combine/Reduced Combine 各组方向不一致，未观察到稳定变化。

验证完成情况：

- 原版和候选各自 8 × 33 个 kernel 回归全部通过。
- 完整 CANN 编译通过；本地 receive-validation 契约、production dispatch
  state/layout、C++ layout/tiling 三项检查通过；host adapter 也通过
  `-Wall -Wextra -Werror` 编译。
- 12 次 ABBA、一次候选 trace，以及 BF16 同步、FP8 previous-event +
  async + allocate-on-comm-stream 两个补充 case，均完成五操作功能校验，
  本轮没有 SIGSEGV、卡死、超时或协议错误。
- 这不是完整 144-case 矩阵验收；此前 D4 记录的小规模 weight 边界问题
  不在本次修改范围内，仍未修复。

结论：保留接收记录 subgroup 并行校验，无需新增性能开关。后续固定 kernel
优先看约 436 μs 的 producer prefix，再评估约 329 μs 的专家计数及约
176 μs 的 epilogue prefix；调用尾部的结果回读/handle 收尾继续单独归因。

原始数据：

- A：`/home/pyptouser/yuqitao/deepep-d4/results/validate-abba{1,2,3}-{A1,A2}.json`；
- B：`/home/pyptouser/yuqitao/deepep-dispatch-validate/results/` 下的
  `validate-abba{1,2,3}-{B1,B2}.json`、`validate-bf16-sync.json`、
  `validate-fp8-async.json`、`validate-parallel-trace.json`，以及对应 `.log`；
- 相同 B 目录内的 `validate-{serial,parallel}-errors.rank{0..7}.json`
  保存两版 33-case 结果，`validate-parallel-traces/rank{0..7}.json` 保存 trace；
- adapter 编译 task `task_20260926_060207_152818122900`；原版回归 task
  `task_20260926_060229_152895320037`；候选编译 task
  `task_20260926_060356_15338624653`；候选回归及 trace task
  `task_20260926_060608_154577515958`；补充 case 与 ABBA task
  `task_20260926_060732_155451626370`。所有任务均退出 0。

临时脚本和下载数据归档于本地
`/tmp/deepep-dispatch-validate-20260926.SJ6VvQ/scratch/`，远端新增采样/执行
脚本归档于 `/home/pyptouser/yuqitao/dispatch-validate-20260926.MEgemf/`。
生产源码不含本轮临时诊断，新增的 adapter/runner 仅作为可复跑设备回归保留。

### 普通 Dispatch producer prefix 错误扫描并行化（2026-09-26）

本轮在接收校验优化 `ce54e17` 上继续，只修改 producer prefix 的首个错误
扫描。典型 case 不变：NPU8P device 0-7、8192 tokens/rank、hidden 7168、
top-k 8、256 experts、FP8、64 AIV，使用
`/data/disk2/cann_version/0916/use_cann/cann-9.3.0` 与树内配套 HCOMM。
基线 A 二进制为 `38a0b3f211d8d43c25aea814ba40a28ea4487bbe96570e87693a2b9387cfd25d`，
候选 B 为 `673bc59e40bfa8ad8f45da19cb7c2b42d4c4414ee9c2aa9093c335f5d83769f8`。

#### 假设与实现

按优先级检查：thread 0 串行扫描全部 tile errors；tile counts 的前缀读写；
每 rank 汇总 chunk partial sums。此次仅验证第一项。

每个 grouping tile 包含 4 个 token，因此典型输入有 2048 个 tile。
原实现先由 thread 0 串行读取 2048 个错误槽位，再进入已有的两级前缀计算。
改为 block 内线程按 grid-stride 分摊扫描，每线程遇到自己的首个错误后，
对一个 UB uint32 槽位执行 atomic-min，选择最小的**出错 tile 下标**。
无错误时不执行 atomic-min；thread 0 最终读取选中 tile 的完整 packed error。

以下语义保持不变：

- 最早出错 tile 优先，不能用最小 packed error 代替；只有 thread 0
  将该错误带入 rank-0 发布，其他 rank 不因此额外写错误。
- rank count 超出 capacity 时，该 rank 的 capacity overflow 仍覆盖当前错误。
- UB 初始化后、错误下标归并后分别同步；thread 0 读取完整错误后保留
  原有同步，之后才允许将 tile_errors 前部复用为 chunk partial sums。
- 前缀算法、launch、参数 ABI、workspace 布局、generation、route staging
  与后续协议均未变化，无新增性能开关。

#### 功能回归

新增 `tests/ascend/core_ops/dispatch_prefix_adapter.cpp` 和
`tests/ascend/production/run_dispatch_prefix.py`。host adapter 构造真实
C++ POD，调用已加载生产 `_C` 中的 prefix launcher；runner 用合成 grouping
输出与独立串行 CPU oracle 比较 counts、exclusive prefixes、错误、status、
early-route staging 和输出保护区。仅允许 parallel path 的 tile_errors
前部作为临时 scratch 改写，其余 workspace 字节必须符合预期。

原版与候选各在八张卡上通过每卡 94 个场景（各 752 项）：虚拟 world
1/3/8/32，32/128/512 threads，0/1/threads-1/threads/2049 tiles；首尾、
跨 lane、同 lane 和多个错误；全 tile 错误；容量溢出优先级；已有 status
及已有 rank error；不整除 world 的串行回退；early-route 输出。
该测试不执行 transport，跨 rank 输出及同步仍由完整五操作用例验证。

复跑命令（按主机规定分别提交 CPU/NPU 队列）：

```bash
c++ -std=c++17 -shared -fPIC -I. \
  tests/ascend/core_ops/dispatch_prefix_adapter.cpp \
  -o build/dispatch-prefix-adapter.so

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/ascend/production/run_dispatch_prefix.py \
  --adapter "$PWD/build/dispatch-prefix-adapter.so" \
  --output "$PWD/results/dispatch-prefix"
```

#### Kernel 归因

与上一轮相同，在入口 HCCL 完成后采集 20 次普通 Dispatch，去掉首轮，
两版各 152 个样本。每次仍包含 28 个 DeepEP kernel。

| Kernel | A median / μs | B median / μs |
| --- | ---: | ---: |
| producer prefix | 435.750 | 132.507 |
| receive validate | 46.121 | 46.094 |
| count experts | 329.833 | 329.976 |
| epilogue parallel prefix | 175.596 | 175.491 |
| metadata | 170.434 | 170.513 |

producer prefix 减少 303.243 μs（约 69.6%）；B p95 为 133.051 μs，
范围 131.766-134.149 μs。数据支持串行错误扫描是主要固定开销的假设。
acquire 等待和调用尾部耗时随 rank 到达时间波动，不把其差值全部归因于
这次修改；是否保留仍以无 profiling 的端到端重复对照为准。

#### 端到端 ABBA：保留

先补测两个非典型调用场景，均为五操作通过：BF16 同步
`prefix-bf16.json`，以及 FP8 previous-event + async +
allocate-on-comm-stream `prefix-async.json`。随后执行三组无 profiling 的
A1/B1/B2/A2，每组 2 warmups / 30 iterations，并核对 workload fingerprint、
64 AIV、30 个采样、五操作结果和二进制哈希。

| 组 | A Dispatch mean / ms | B mean / ms | mean 收益 | A p95 / ms | B p95 / ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.919067 | 5.361466 | 9.420% | 6.379939 | 5.712232 |
| 2 | 5.809218 | 5.458504 | 6.037% | 6.247656 | 5.874960 |
| 3 | 5.887448 | 5.588990 | 5.069% | 6.405610 | 6.127460 |

六个等长 run 合并：

| 操作 | A mean / ms | B mean / ms | B logical GB/s |
| --- | ---: | ---: | ---: |
| Dispatch | 5.871911 | 5.469653 | 1423.507 |
| Expanded Dispatch | 16.681919 | 16.400436 | 564.100 |
| Cached Dispatch | 65.171538 | 65.198233 | 119.422 |
| Combine | 14.598520 | 14.521632 | 750.687 |
| Reduced Combine | 14.965438 | 14.970332 | 728.187 |

Normal Dispatch 合并 mean 改善 **6.851%**，logical bandwidth 从
**1325.989 → 1423.507 GB/s**；仍是八 rank 聚合逻辑带宽。Expanded Dispatch
三组 mean 改善 2.275%/1.619%/1.168%，与该 kernel 同时服务 expanded 输出
相符。Cached Dispatch、Combine 和 Reduced Combine 方向不一致，认为无稳定
变化。

验证完成情况：

- 原版与候选的 8 × 94 个 prefix 边界场景全部通过；
- 候选完整 CANN 编译通过；本地 production dispatch state/layout 与
  C++ layout/tiling 检查通过；host adapter 通过
  `-Wall -Wextra -Werror` 编译；
- BF16 同步与 FP8 异步补充 case 均五操作通过；
- 三组 ABBA 共 12 次 run，每次五操作通过；
- 一次 B run 在所有结果已写出且五操作全部通过后出现既知的 teardown
  SIGSEGV，严格检查脚本确认 `KNOWN_TEARDOWN_SIGSEGV_AFTER_VALID_RESULT`，
  整个队列任务仍退出 0，未出现结果前失败、卡死或协议错误；
- 不是完整 144-case 矩阵验收，此前 D4 记录的小规模 weight 边界问题仍
  未修复。

结论：保留 producer prefix 错误扫描并行化，无需新增性能开关。下一个固定
kernel 优先级是约 330 μs 的专家计数；producer prefix 仍余约 132 μs，
其中 tile counts 的两级前缀读写和 rank chunk 汇总是后续候选。

原始数据：

- A：`/home/pyptouser/yuqitao/deepep-dispatch-validate/results/prefix-abba{1,2,3}-{A1,A2}.json`；
- B：`/home/pyptouser/yuqitao/deepep-dispatch-prefix/results/` 下的
  `prefix-abba{1,2,3}-{B1,B2}.json`、`prefix-bf16.json`、
  `prefix-async.json`、`prefix-parallel-trace.json` 与对应 `.log`；
- 相同 B 目录内的 `prefix-{serial,parallel}-errors.rank{0..7}.json`
  保存两版 94-case 结果，`prefix-parallel-traces/rank{0..7}.json` 保存
  trace；
- adapter 编译 task `task_20260926_063045_16195003386`；原版边界 task
  `task_20260926_063100_16200348206`；候选编译 task
  `task_20260926_063150_16234383643`；候选边界与 trace task
  `task_20260926_063340_16348406207`；补充 case 与 ABBA task
  `task_20260926_063455_16433221367`。所有任务均退出 0。

临时脚本与原始采样已归档到本地
`/tmp/deepep-dispatch-prefix-20260926.URXmHM/scratch/`；生产源码不含
临时诊断，adapter/runner 仅作为可复跑设备回归保留。

### D5. Cached Dispatch 专项（暂停）

优先级：暂停；2026-09-26 按用户要求，先集中优化普通 Dispatch。

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

- Cached Dispatch mean 的改善可通过重复对照稳定复现，不设置百分比门槛；
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
5. D4 后普通 Dispatch 的专用 VF kernel 与长尾归因；D5 Cached Dispatch 暂停。

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

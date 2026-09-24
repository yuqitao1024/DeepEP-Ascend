# Ascend 性能宏与配置速查

本文是 DeepEP Ascend 性能相关开关的集中索引，目标是快速回答两个问题：

1. 当前生产路径默认已经启用了哪些优化？
2. 哪些开关还能用于对照、诊断或实验？

本文不替代各阶段设计文档。开关的完整实验数据、任务号和验收结论仍在对应
P3-P7 文档中；本文记录当前代码的实际默认值、生效条件和推荐用法。

## 快速结论

当前 8-rank 典型 case 的生产路径中，已通过验收的 Dispatch/Combine runtime
selector 在满足条件时默认启用。显式设置成下表推荐值，通常只是强制当前默认路径，
不应期待新的性能变化；显式设置成 `0` 才是关闭优化、用于 A/B 对照的 control。

| 操作 | Selector | 未设置时的默认值 |
| --- | --- | --- |
| Dispatch | `DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX` | eligible 时开启 |
| Dispatch | `DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX` | eligible 时开启 |
| Dispatch | `DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES` | eligible 时 8192 |
| Dispatch | `DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT` | eligible 时开启 |
| Combine | `DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY` | eligible 时 32768 |
| Combine | `DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT` | eligible 时开启 |
| Combine | `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE` | eligible 时 512 |
| Combine | `DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE` | eligible 时开启 |

“eligible 时”表示调用模式满足该优化的边界条件。selector 的显式 `1` 不会强行
打开不满足条件的路径，而是返回 disabled；非法值在 host 边界 fail-fast。

## 运行时 Dispatch selector

### `DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX`

- 取值：`0`、`1`，或未设置。
- 未设置：direct、uncached、CPU-sync、non-hybrid、non-stream Dispatch 满足
  条件时默认开启。
- 作用：把 D5-D8/F0 组织成连续 device pipeline，移除 D6 到 D7 之间的 host
  count bridge。它仍保留最终 count readback 和公开 tensor 的 H2D publication。
- 对照：`0` 强制回到保守路径；`1` 显式选择优化路径。
- 注意：只影响普通 direct Dispatch。cached、hybrid、stream、expanded 等路径
  不适用。

### `DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX`

- 取值：`0`、`1`，或未设置。
- 未设置：依赖 device-prefix 路径且调用 eligible 时默认开启。
- 作用：把 D6 local-expert tile 前缀从单线程串行扫描改为每个 active SIMT
  thread 负责一个 local expert 列，再做短串行 rank/global prefix。
- 对照：`0` 保留串行 D6 control。
- 注意：不改变 token routing 或 prefix 语义，只改变计算任务划分。

### `DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES`

- 取值：`512`、`1024`、`2048`、`4096`、`8192`，或未设置。
- 未设置：device-prefix eligible 时默认 `8192`；不 eligible 时关闭。
- 作用：选择 D8 consumer hidden/scale copy 的 AICore tile specialization。
  aligned body 走 `DataCopy`，不足 32-byte 的 scalar tail 仍由 SIMT 处理。
- 对照：`512` 是保守 control；其他值用于形状筛选。
- 注意：该 selector 本身不是越大越好。历史筛选中 2048/4096 与 8192 在不同
  阶段各有表现，当前生产默认为 8192。

### `DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT`

- 取值：`0`、`1`，或未设置。
- 未设置：FP8、direct、uncached、non-expanded、non-hybrid、non-stream、
  grouping-eligible 且无 pipeline 的调用默认开启。
- 作用：P7.0 hidden vector fan-out：每个 source token 一次加载 aligned body，
  再 fan out 到 top-k destination records，减少重复 GM 读。
- 对照：`0` 回到 retained P6 path。
- 约束：top-k `<= 8`，world size `<= 8`，hidden aligned body 必须是当前支持的
  7168-byte plan；目的 slot pipeline 或 source pipeline 开启时不适用。

## 运行时 Combine selector

### `DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT`

- 取值：`0`、`1`，或未设置。
- 未设置：direct、non-hybrid Combine eligible 时默认开启。
- 作用：保留 direct local placement 优化，减少本地 contributor 的间接转换。
- 对照：`0` 回到保守 local placement。

### `DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY`

- 取值：`0`、`1`、`512`、`1024`、`2048`、`4096`、`8192`、`16384`、
  `32768`，或未设置。
- 未设置：direct、non-hybrid eligible 时默认 `32768`。
- 作用：选择 local payload copy 的 `DataCopy` tile；`1` 等价于 `32768`。
- 对照：`0` 关闭 DataCopy specialization，回到 SIMT copy control。
- 注意：当前生产默认是 `32768`，不是历史文档中较早阶段的 2048-byte tile。

### `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE`

- 取值：`0`、`1`、`512`，或未设置。
- 未设置：direct、non-hybrid eligible 时默认 `512`。
- 作用：选择 vector reduction tile。`1` 与 `512` 等价。
- 对照：`0` 关闭 vector tile specialization。

### `DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE`

- 取值：`0`、`1`，或未设置。
- 未设置：direct、expanded、non-hybrid、`allow_multiple_reduction=true` 且
  top-k 在 `[1, 32]` 时默认开启。
- 作用：P5.4 expanded/reduced Combine 的向量归约，使用
  `DataCopy -> Cast -> Add -> Cast -> DataCopy`，并保持 lane 顺序的浮点累加。
- 对照：`0` 回到 scalar control。
- 注意：Normal Combine 不是它的适用路径；它主要影响 expanded/reduced Combine。

## 实验或非默认 runtime selector

### `DEEP_EP_ASCEND_CHANNELS`

- 取值：`1` 到 `4`，默认 `1`。
- 作用：为每个 scale-up peer 请求多条 transport channel/QP/SQ。
- 状态：实现存在，但当前不是生产默认。8-rank 的完整功能和性能验收数据尚未
  记录，因此不能默认推荐。
- 建议验证顺序：先小规模 correctness，再 8-rank 30 warmup/30 iterations
  performance，并与单 channel 同 commit、同 CANN/HCOMM、同 workload 对照。
- 注意：开启多 channel 不等于提高 `num_sms`。两者影响不同资源层，不能混在
  同一个 A/B 里归因。

## 编译宏

Ascend build 的编译宏在构建时固化，修改后必须重新构建 extension。

| 编译宏 | 默认 | 建议 |
| --- | --- | --- |
| `DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY` | `ON` | 生产构建保持开启。设为 `0` 可恢复最终 direct producer release barrier，用于协议对照 |
| `DEEP_EP_ASCEND_TESTING` | `OFF` | 性能和生产构建不要开启。开启后包含 validation-only 诊断和 fault hooks |
| `DEEP_EP_ASCEND_ACQUIRE_DIAGNOSTICS` | `OFF` | 只在需要 acquire 长尾归因时构建。性能测试不要开启 |
| `DEEP_EP_ASCEND_SKIP_EPILOGUE_NOOP` | `OFF` | 历史候选已做过 correctness A/B，但端到端无收益；不作为生产优化推荐 |

`DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=ON` 只跳过最终 direct Dispatch/Combine
producer release barrier；payload/control put、CQ drain、release signal、
route-plan barrier 和 service completion check 仍保留。它是通过 correctness、
consecutive-generation 和 buffer-reuse 验收后保留的生产默认。

## 不建议再打开的历史候选

这些开关曾用于实验，当前结论是不应作为生产性能优化打开：

- `DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_SLOTS`
  - destination-slot pipeline。早期证据有收益，但后续受组合约束和 retained
    路径变化影响，不再作为默认推荐；只在明确要复现历史实验时使用。
- `DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES`
  - source-token pipeline。2026-09-20 快照中多 chunk 配置在 prefix/count
    correctness 上失败，不能作为可用性能 knob。
- `DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN=1`
  - isolated candidate 已被拒绝，不应与 token fan-out 或其他 pipeline 混用。
- `DEEP_EP_ASCEND_COMBINE_RELEASE_FLUSH_BARRIER=1`
  - 三轮 8-rank ABBA 方向不一致，实现已回退。
- `DEEP_EP_ASCEND_COMBINE_METADATA_CACHE`、
  `DEEP_EP_ASCEND_COMBINE_METADATA_BUCKETS`、
  `DEEP_EP_ASCEND_COMBINE_PACKED_CONTROL_PUT`
  - 相关实现和 selector 已被移除或回退，不应在文档或脚本中继续当作可用开关。
- `DEEP_EP_ASCEND_COMBINE_PIPELINE_CHUNK_ROWS`
  - 历史 phase-A 候选，不作为当前生产推荐。

若要重新评估其中某一项，应从新的 profile 出发，先做小规模 correctness，再做
same-binary ABBA；不要把多个历史候选叠加在一次 run 里。

## Profile 与 benchmark 辅助变量

这些变量不改变核心通信协议，但会影响统计口径或执行开销：

- `DEEP_EP_ASCEND_PREFLIGHT=stable|full`
  - 默认 `stable`。stable 模式把逐调用 host 校验移到稳定边界，full 模式保留
    旧诊断能力。
- `DEEP_EP_ASCEND_PROFILE_STAGES=1`
  - 开启 stage profile。会保留或增加 profile 相关 kernel/字段，适合归因，
    不应和无 profile 结果混比。
- `DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS`
  - benchmark 进程级 timeout，默认 300 秒。

## 推荐的 8-rank 对照脚本口径

与当前 stable preflight 基线可比的 run，建议固定：

```bash
export DEEP_EP_PLATFORM=ascend
export DEEP_EP_DISABLE_TORCH_COMPILE=1
export DEEP_EP_ASCEND_PREFLIGHT=stable

# 以下显式值等价于当前 eligible 默认值；用于防止外部环境意外覆盖。
export DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX=1
export DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX=1
export DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES=8192
export DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT=1
export DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768
export DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT=1
export DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE=512
export DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE=1

# 如果要测试真正非默认项，例如多 channel，必须单独改 DEEP_EP_ASCEND_CHANNELS，
# 不要同时叠加其他变量。
```

当前典型 case 的 workload：

```text
case: ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0
world_size=8, num_tokens=4096, hidden=7168
num_topk=6, num_experts=256, num_sms=56
warmups=30, iterations=30
```

口径说明：历史 P5-P7 优化验收使用的“representative workload”是 8192
tokens、top-k 8、72 data blocks。2026-09-24 的 host-preflight 分析、selector
复核和跨机器对照使用的是本文记录的 4096 tokens、top-k 6、56 data blocks
口径；两者不是同一 workload，不能直接混比。

2026-09-24 所有 selector 复核和跨机器对照 JSON 中的 workload 元数据均已
确认：8-rank 三次 selector 复核、NPU8P 4-rank、NPU4Px2 4-rank 均为
`num_tokens=4096`、`num_topk=6`。其中 8-rank workload fingerprint 为
`dd523e34557a25bbda218f7346e772c147e1dc2a0035241ec1f9712b565dbbc3`；
两个 4-rank 对照的 fingerprint 相同，为
`b8f607ea337761f2c581c4221dd5dfb54e5fc49d6d7e8a03f11896fc8a9cfdb5`。

如果目标是与既有 8-rank stable preflight 数据直接比较，保持 `--profile-stages`。
如果目标是正式无 profile 性能，则去掉 `--profile-stages`，并重新建立基线，
不要跨口径混合比较。

## 2026-09-24 selector 复核

以下三次 run 使用同一 NPU8P、CANN 9.3.0/HCOMM、commit `9476200`、
30 warmup/30 iterations、`--profile-stages` 和同一个典型 case。差异只在
selector 环境；correctness 均通过。

| Run | Dispatch mean | Dispatch GB/s | Combine mean | Combine GB/s |
| --- | ---: | ---: | ---: | ---: |
| 默认 selector | 5.993 ms | 542.324 | 11.530 ms | 402.290 |
| 显式保留 selector | 5.920 ms | 548.990 | 11.223 ms | 413.293 |
| 显式保留 selector + `CHANNELS=2` | 6.053 ms | 536.879 | 11.715 ms | 395.943 |

结论：

1. 显式设置已保留 selector 与默认结果同量级，个别差异在 run-to-run 范围内，
   说明当前默认优化已经生效，不是宏漏开。
2. `DEEP_EP_ASCEND_CHANNELS=2` 在该 case 上没有收益，且五项操作均略慢；
   因此不改变默认值，仍保持 `1`。

结果文件与任务：

- 默认 selector：`preflight-stable-8rank-30iter.json`，
  task `task_20260924_140149_96065026468`。
- 显式保留 selector：`preflight-stable-8rank-30iter-optimized.json`，
  task `task_20260924_141730_112040026649`，
  SHA-256 `170cb2469c8663dfd13009505b00c16854cf2a42e2da324014b7fd7d1182080b`。
- `CHANNELS=2`：`preflight-stable-8rank-channels2-30iter.json`，
  task `task_20260924_141958_11324995765`，
  SHA-256 `7f1063b5ed3ddf370fc553a35ffbb0426c7d2fde39a1942cff9c257bc4e55526`。

## 2026-09-24 NPU4Px2 / NPU8P 4-rank 对照

同一 commit `37d8276`、stable preflight、4-rank、hidden 7168、top-k 6、
256 experts、56 data blocks、30 warmup/30 iterations，并且都使用
`--profile-stages`。每 rank 4096 tokens。

| Host | Device | CANN/HCOMM | Allocation |
| --- | --- | --- | --- |
| NPU4Px2 | Ascend950PR，设备 4-7 | CANN 9.3.0，`/data/y00621698/pkg-9.3/cann-9.3.0` | 直接运行；运行前 `npu-smi` 确认 4-7 组内无其他进程 |
| NPU8P | Ascend950DT，设备 0-3 | CANN 9.3.0，`/data/disk2/cann_version/0916/use_cann/cann-9.3.0` | task-submit |

NPU4Px2 的 4-7 卡属于同一个 UB group；NPU8P 的 0-3 卡由 task-submit 锁定。
两台机器都是 4-rank 全 UB 互联。NPU4Px2 直接运行时需要显式设置
`HCCL_NPU_SOCKET_PORT_RANGE=60000-60099`，否则会遇到默认 16666 端口已被
绑定的问题。

| Operation | NPU4Px2 mean | NPU8P mean | Time ratio | NPU4Px2 GB/s | NPU8P GB/s | Bandwidth ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 3.218 ms | 5.609 ms | 0.574 | 375.122 | 215.216 | 1.743 |
| Expanded Dispatch | 9.979 ms | 12.494 ms | 0.799 | 165.825 | 132.434 | 1.252 |
| Cached Dispatch | 27.083 ms | 33.001 ms | 0.821 | 44.568 | 36.576 | 1.218 |
| Combine | 5.965 ms | 9.665 ms | 0.617 | 298.954 | 184.517 | 1.620 |
| Reduced Combine | 6.523 ms | 10.217 ms | 0.639 | 273.395 | 174.542 | 1.565 |

结论：在该 4-rank representative case 上，NPU4Px2/Ascend950PR 五项操作都
快于 NPU8P/Ascend950DT。普通 Dispatch 和 Combine 的差距最大，logical
bandwidth 分别约为 1.74x 和 1.62x；expanded/cached 路径差距较小，约为
1.22x 到 1.25x。这个跨机器结论只能用于描述两台机器在上述固定环境下的表现，
不能单独归因给某一个硬件或驱动差异。

结果文件与任务：

- NPU4Px2：`/data/y00621698/deepep-37d8276/results/npu4px2-4rank-stable-30iter.json`，
  SHA-256
  `429cab942bca89b46f1abd235dc2b3da277b75295c830bbb272074e9bc211006`。
- NPU8P：`/home/pyptouser/yuqitao/deepep-preflight-9476200/results/preflight-stable-4rank-30iter.json`，
  task `task_20260924_140434_96976420629`，SHA-256
  `cb99f66314468908779064bb56a7d5cde0ff91422a8ad8259dc44b0661836a95`。

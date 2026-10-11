# 官方口径 144-case EP8 性能报告

日期：2026-10-10 至 2026-10-11。

## 1. 测试目标与结论

本报告按 DeepSeek 官方 DeepEP-Ascend bench_msprof 性能采集口径，覆盖当前支持的完整 144 个 EP8 case。全部 144 个 case 均通过，无失败项。

| 指标 | 数值 |
|---|---:|
| Case 总数 | 144 |
| 通过 / 失败 | 144 / 0 |
| Dispatch p50 | 258.67 GB/s |
| Combine p50 | 271.89 GB/s |
| Dispatch min / max | 226.91 / 300.49 GB/s |
| Combine min / max | 190.51 / 320.31 GB/s |

整体看，dispatch 与 combine 的结果均落在同一量级，未出现显著超过物理链路能力的离群值。BF16 dispatch 高于 FP8 dispatch，主要来自两者每 token 逻辑字节数不同；combine 的 FP8 与 BF16 结果基本一致。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 机器 | NPU8P 直连环境 |
| 设备 | Ascend 950DT，8 卡，设备 ID 0–7 |
| CPU 架构 | aarch64，384 逻辑 CPU |
| CANN | /usr/local/Ascend/cann-9.3.0 |
| Python | /data/disk2/pyptouser/yuqitao/deepep-venv-py311，Python 3.11 |
| Torch / Torch-NPU | 2.13.0+cpu / 2.13.0.rc1 |
| 代码仓 | 本仓库 DeepEP-Ascend |
| 远端验证目录 | /data/disk2/pyptouser/yuqitao/deepep-ascend-41691e3 |
| 结果目录 | /data/disk2/pyptouser/yuqitao/results/official-reporting-matrix-20261010-r3 |

## 3. 测试方式

| 项目 | 值 |
|---|---:|
| World size | 8 |
| num_tokens | 16,384 |
| hidden | 7,168 |
| num_topk | 6 |
| num_experts | 256 |
| seed | 0 |
| unbalanced_ratio | 1.0 |
| masked_ratio | 0.0 |
| AIV 数 | 64 |
| msprof warmup / iterations | 10 / 50 |
| stage warmup / iterations | 1 / 1 |
| Workload fingerprint | 20bce2a7990c4ddc8fd4ff7697e2c2b673bfb44e81dc44d56255821afcf841e0 |

每个 case 依次执行三步：

1. bench_ep_msprof.py：按官方 bench_msprof 语义采集 10 warmup、50 sample 的 kernel profile。
2. bench_ep.py --profile-stages：采集同一 workload 的 stage profile。
3. align_official_reporting.py：把 msprof 与 stage 数据映射到统一报告口径。

报告的 dispatch / combine 带宽均使用 stage_service_issue_drain，即 service_submit + cq_wait 作为当前 staged transport 与官方 URMA issue-and-drain 语义的映射。表格中的数值为该 case 的 p50 带宽，单位 GB/s。

执行时按 3 个 case 一批提交 8 卡任务，避免长时间独占整台机器；中断后使用同一 workload fingerprint 恢复，已通过的 case 不重复执行。

### Case ID 与模式差异

Case ID 的格式为：

```text
ep-<dtype>-align<expert_alignment>-bias<num_bias>-hcopy<do_handle_copy>-prev<with_previous_event>-async<async_with_compute_stream>-alloc<allocate_on_comm_stream>
```

这些字段不改变 token 数、hidden、top-k、expert 数和路由 manifest；它们改变的是 Dispatch/Combine 的数据布局、API 参数和执行流模式。

| 字段 | 取值 | 含义 |
|---|---|---|
| `dtype` | `fp8` / `bf16` | Dispatch payload 类型。`fp8` 会将 hidden payload 量化为 FP8，并携带 per-token-group scale factor；`bf16` 直接传输 BF16 hidden payload。本报告中 dispatch 的 FP8/BF16 字节数不同，combine 输出统一为 BF16。 |
| `align` | `1` / `128` | Dispatch 的 expert alignment。`128` 表示按 128 个 token 粒度对齐 expert 分组边界，可能引入 padding；`1` 表示不启用这种 expert 粒度对齐。 |
| `bias` | `0` / `1` / `2` | Combine 侧 bias 输入形式。`0` 表示不传 bias；`1` 表示传单个 BF16 bias tensor；`2` 表示传两个 BF16 bias tensor 组成的 tuple。 |
| `hcopy` | `0` / `1` | Dispatch `do_handle_copy` 开关。`1` 表示返回的 handle 元数据需要拷贝，不与输入 `topk_idx` 复用存储；`0` 表示关闭该拷贝路径。 |
| `prev` | `0` / `1` | 是否向通信操作传入 `previous_event`，用于让通信流等待前序事件。`1` 时必须启用通信流分配，因此矩阵中没有 `prev1-alloc0` 组合。 |
| `async` | `0` / `1` | `async_with_compute_stream` 开关。`1` 表示通信返回 event handle，由当前 compute stream 等待完成；`0` 表示同步完成等待。 |
| `alloc` | `0` / `1` | `allocate_on_comm_stream` 开关。`1` 表示输出和中间资源在专用通信流上分配，配合 event/previous-event 路径；`0` 表示使用当前执行流。 |

组合展开规则为：`hcopy × align × dtype × bias × prev × async × alloc`，其中 `prev=1` 时 `alloc` 只取 `1`。因此每个 `hcopy/align/dtype/bias` 基础组合有 6 种 stream/event 模式，最终得到：

```text
2 (hcopy) × 2 (align) × 2 (dtype) × 3 (bias) × 6 (有效 stream/event 组合) = 144
```

## 4. 维度汇总

| 数据类型 | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| 数据类型 | bf16 | 72 | 254.76 / 288.97 / 288.45 / 300.49 | 240.68 / 271.91 / 270.34 / 320.31 |
| 数据类型 | fp8 | 72 | 226.91 / 250.91 / 250.19 / 265.71 | 190.51 / 271.63 / 268.55 / 284.32 |

| expert alignment | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| expert alignment | 1 | 72 | 226.91 / 256.47 / 268.70 / 298.35 | 190.51 / 271.89 / 268.58 / 284.32 |
| expert alignment | 128 | 72 | 229.74 / 259.92 / 269.94 / 300.49 | 238.75 / 271.83 / 270.31 / 320.31 |

| bias 数量 | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| bias 数量 | 0 | 48 | 226.91 / 258.67 / 269.32 / 296.57 | 190.51 / 271.27 / 267.96 / 320.31 |
| bias 数量 | 1 | 48 | 243.59 / 258.44 / 269.91 / 300.49 | 241.76 / 272.69 / 270.19 / 278.22 |
| bias 数量 | 2 | 48 | 229.74 / 254.33 / 268.73 / 297.08 | 225.04 / 271.91 / 270.19 / 281.79 |

| handle copy | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| handle copy | 0 | 72 | 239.96 / 265.71 / 270.90 / 300.49 | 238.75 / 272.36 / 270.33 / 279.59 |
| handle copy | 1 | 72 | 226.91 / 257.11 / 267.73 / 297.08 | 190.51 / 271.42 / 268.56 / 320.31 |

| previous event | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| previous event | 0 | 96 | 243.09 / 258.67 / 269.78 / 300.49 | 190.51 / 271.42 / 268.39 / 320.31 |
| previous event | 1 | 48 | 226.91 / 256.53 / 268.40 / 298.35 | 241.76 / 272.68 / 271.55 / 281.79 |

| async with compute stream | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| async with compute stream | 0 | 72 | 226.91 / 258.67 / 268.79 / 298.35 | 190.51 / 271.91 / 269.28 / 320.31 |
| async with compute stream | 1 | 72 | 239.96 / 258.44 / 269.85 / 300.49 | 238.75 / 271.63 / 269.61 / 284.32 |

| comm-stream allocation | 分组 | Case 数 | Dispatch min/p50/mean/max（GB/s） | Combine min/p50/mean/max（GB/s） |
|---|---|---:|---:|---:|
| comm-stream allocation | 0 | 48 | 243.09 / 258.44 / 268.33 / 296.76 | 225.04 / 271.99 / 269.09 / 320.31 |
| comm-stream allocation | 1 | 96 | 226.91 / 259.92 / 269.81 / 300.49 | 190.51 / 271.52 / 269.62 / 284.32 |

## 5. 全部 144 case 结果

| # | Case | 状态 | Dispatch GB/s | Combine GB/s |
|---:|---|---|---:|---:|
| 1 | ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0 | passed | 245.47 | 270.45 |
| 2 | ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc1 | passed | 248.67 | 258.51 |
| 3 | ep-fp8-align128-bias0-hcopy1-prev0-async1-alloc0 | passed | 254.41 | 250.82 |
| 4 | ep-fp8-align128-bias0-hcopy1-prev0-async1-alloc1 | passed | 243.59 | 271.27 |
| 5 | ep-fp8-align128-bias0-hcopy1-prev1-async0-alloc1 | passed | 256.53 | 264.62 |
| 6 | ep-fp8-align128-bias0-hcopy1-prev1-async1-alloc1 | passed | 255.27 | 281.29 |
| 7 | ep-fp8-align128-bias1-hcopy1-prev0-async0-alloc0 | passed | 251.47 | 276.84 |
| 8 | ep-fp8-align128-bias1-hcopy1-prev0-async0-alloc1 | passed | 243.59 | 265.36 |
| 9 | ep-fp8-align128-bias1-hcopy1-prev0-async1-alloc0 | passed | 257.11 | 276.13 |
| 10 | ep-fp8-align128-bias1-hcopy1-prev0-async1-alloc1 | passed | 259.92 | 276.79 |
| 11 | ep-fp8-align128-bias1-hcopy1-prev1-async0-alloc1 | passed | 255.45 | 276.51 |
| 12 | ep-fp8-align128-bias1-hcopy1-prev1-async1-alloc1 | passed | 254.10 | 270.53 |
| 13 | ep-fp8-align128-bias2-hcopy1-prev0-async0-alloc0 | passed | 246.64 | 273.96 |
| 14 | ep-fp8-align128-bias2-hcopy1-prev0-async0-alloc1 | passed | 252.51 | 271.83 |
| 15 | ep-fp8-align128-bias2-hcopy1-prev0-async1-alloc0 | passed | 250.72 | 275.99 |
| 16 | ep-fp8-align128-bias2-hcopy1-prev0-async1-alloc1 | passed | 243.53 | 274.35 |
| 17 | ep-fp8-align128-bias2-hcopy1-prev1-async0-alloc1 | passed | 229.74 | 278.49 |
| 18 | ep-fp8-align128-bias2-hcopy1-prev1-async1-alloc1 | passed | 249.92 | 281.79 |
| 19 | ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc0 | passed | 277.39 | 320.31 |
| 20 | ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc1 | passed | 295.28 | 262.54 |
| 21 | ep-bf16-align128-bias0-hcopy1-prev0-async1-alloc0 | passed | 286.44 | 257.59 |
| 22 | ep-bf16-align128-bias0-hcopy1-prev0-async1-alloc1 | passed | 290.88 | 260.04 |
| 23 | ep-bf16-align128-bias0-hcopy1-prev1-async0-alloc1 | passed | 288.05 | 267.48 |
| 24 | ep-bf16-align128-bias0-hcopy1-prev1-async1-alloc1 | passed | 287.44 | 271.52 |
| 25 | ep-bf16-align128-bias1-hcopy1-prev0-async0-alloc0 | passed | 288.40 | 273.22 |
| 26 | ep-bf16-align128-bias1-hcopy1-prev0-async0-alloc1 | passed | 287.47 | 267.91 |
| 27 | ep-bf16-align128-bias1-hcopy1-prev0-async1-alloc0 | passed | 258.44 | 274.89 |
| 28 | ep-bf16-align128-bias1-hcopy1-prev0-async1-alloc1 | passed | 283.18 | 273.54 |
| 29 | ep-bf16-align128-bias1-hcopy1-prev1-async0-alloc1 | passed | 288.91 | 273.49 |
| 30 | ep-bf16-align128-bias1-hcopy1-prev1-async1-alloc1 | passed | 293.15 | 269.81 |
| 31 | ep-bf16-align128-bias2-hcopy1-prev0-async0-alloc0 | passed | 286.95 | 264.79 |
| 32 | ep-bf16-align128-bias2-hcopy1-prev0-async0-alloc1 | passed | 288.97 | 271.91 |
| 33 | ep-bf16-align128-bias2-hcopy1-prev0-async1-alloc0 | passed | 287.76 | 270.28 |
| 34 | ep-bf16-align128-bias2-hcopy1-prev0-async1-alloc1 | passed | 290.26 | 259.95 |
| 35 | ep-bf16-align128-bias2-hcopy1-prev1-async0-alloc1 | passed | 294.40 | 272.68 |
| 36 | ep-bf16-align128-bias2-hcopy1-prev1-async1-alloc1 | passed | 292.90 | 274.80 |
| 37 | ep-fp8-align1-bias0-hcopy1-prev0-async0-alloc0 | passed | 250.91 | 248.13 |
| 38 | ep-fp8-align1-bias0-hcopy1-prev0-async0-alloc1 | passed | 253.87 | 190.51 |
| 39 | ep-fp8-align1-bias0-hcopy1-prev0-async1-alloc0 | passed | 256.47 | 271.63 |
| 40 | ep-fp8-align1-bias0-hcopy1-prev0-async1-alloc1 | passed | 249.27 | 284.32 |
| 41 | ep-fp8-align1-bias0-hcopy1-prev1-async0-alloc1 | passed | 226.91 | 275.46 |
| 42 | ep-fp8-align1-bias0-hcopy1-prev1-async1-alloc1 | passed | 254.74 | 272.38 |
| 43 | ep-fp8-align1-bias1-hcopy1-prev0-async0-alloc0 | passed | 249.57 | 274.40 |
| 44 | ep-fp8-align1-bias1-hcopy1-prev0-async0-alloc1 | passed | 251.35 | 260.98 |
| 45 | ep-fp8-align1-bias1-hcopy1-prev0-async1-alloc0 | passed | 247.22 | 265.70 |
| 46 | ep-fp8-align1-bias1-hcopy1-prev0-async1-alloc1 | passed | 256.10 | 272.06 |
| 47 | ep-fp8-align1-bias1-hcopy1-prev1-async0-alloc1 | passed | 243.69 | 258.23 |
| 48 | ep-fp8-align1-bias1-hcopy1-prev1-async1-alloc1 | passed | 254.04 | 270.47 |
| 49 | ep-fp8-align1-bias2-hcopy1-prev0-async0-alloc0 | passed | 243.09 | 225.04 |
| 50 | ep-fp8-align1-bias2-hcopy1-prev0-async0-alloc1 | passed | 253.39 | 273.36 |
| 51 | ep-fp8-align1-bias2-hcopy1-prev0-async1-alloc0 | passed | 248.00 | 271.09 |
| 52 | ep-fp8-align1-bias2-hcopy1-prev0-async1-alloc1 | passed | 249.62 | 271.39 |
| 53 | ep-fp8-align1-bias2-hcopy1-prev1-async0-alloc1 | passed | 241.68 | 269.03 |
| 54 | ep-fp8-align1-bias2-hcopy1-prev1-async1-alloc1 | passed | 250.60 | 265.79 |
| 55 | ep-bf16-align1-bias0-hcopy1-prev0-async0-alloc0 | passed | 254.76 | 255.34 |
| 56 | ep-bf16-align1-bias0-hcopy1-prev0-async0-alloc1 | passed | 293.95 | 277.35 |
| 57 | ep-bf16-align1-bias0-hcopy1-prev0-async1-alloc0 | passed | 285.35 | 263.31 |
| 58 | ep-bf16-align1-bias0-hcopy1-prev0-async1-alloc1 | passed | 284.54 | 264.54 |
| 59 | ep-bf16-align1-bias0-hcopy1-prev1-async0-alloc1 | passed | 283.98 | 278.66 |
| 60 | ep-bf16-align1-bias0-hcopy1-prev1-async1-alloc1 | passed | 278.01 | 275.32 |
| 61 | ep-bf16-align1-bias1-hcopy1-prev0-async0-alloc0 | passed | 287.66 | 271.83 |
| 62 | ep-bf16-align1-bias1-hcopy1-prev0-async0-alloc1 | passed | 288.87 | 258.85 |
| 63 | ep-bf16-align1-bias1-hcopy1-prev0-async1-alloc0 | passed | 282.32 | 257.21 |
| 64 | ep-bf16-align1-bias1-hcopy1-prev0-async1-alloc1 | passed | 295.99 | 266.05 |
| 65 | ep-bf16-align1-bias1-hcopy1-prev1-async0-alloc1 | passed | 280.48 | 274.70 |
| 66 | ep-bf16-align1-bias1-hcopy1-prev1-async1-alloc1 | passed | 278.86 | 241.76 |
| 67 | ep-bf16-align1-bias2-hcopy1-prev0-async0-alloc0 | passed | 284.01 | 276.27 |
| 68 | ep-bf16-align1-bias2-hcopy1-prev0-async0-alloc1 | passed | 297.08 | 281.40 |
| 69 | ep-bf16-align1-bias2-hcopy1-prev0-async1-alloc0 | passed | 292.25 | 274.44 |
| 70 | ep-bf16-align1-bias2-hcopy1-prev0-async1-alloc1 | passed | 290.99 | 271.42 |
| 71 | ep-bf16-align1-bias2-hcopy1-prev1-async0-alloc1 | passed | 287.85 | 268.97 |
| 72 | ep-bf16-align1-bias2-hcopy1-prev1-async1-alloc1 | passed | 284.54 | 276.93 |
| 73 | ep-fp8-align128-bias0-hcopy0-prev0-async0-alloc0 | passed | 265.71 | 265.21 |
| 74 | ep-fp8-align128-bias0-hcopy0-prev0-async0-alloc1 | passed | 248.23 | 250.58 |
| 75 | ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0 | passed | 252.98 | 272.40 |
| 76 | ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc1 | passed | 247.83 | 270.22 |
| 77 | ep-fp8-align128-bias0-hcopy0-prev1-async0-alloc1 | passed | 253.57 | 276.13 |
| 78 | ep-fp8-align128-bias0-hcopy0-prev1-async1-alloc1 | passed | 241.04 | 270.84 |
| 79 | ep-fp8-align128-bias1-hcopy0-prev0-async0-alloc0 | passed | 248.61 | 276.58 |
| 80 | ep-fp8-align128-bias1-hcopy0-prev0-async0-alloc1 | passed | 256.70 | 269.55 |
| 81 | ep-fp8-align128-bias1-hcopy0-prev0-async1-alloc0 | passed | 250.56 | 266.88 |
| 82 | ep-fp8-align128-bias1-hcopy0-prev0-async1-alloc1 | passed | 257.40 | 266.95 |
| 83 | ep-fp8-align128-bias1-hcopy0-prev1-async0-alloc1 | passed | 247.57 | 254.42 |
| 84 | ep-fp8-align128-bias1-hcopy0-prev1-async1-alloc1 | passed | 254.42 | 274.77 |
| 85 | ep-fp8-align128-bias2-hcopy0-prev0-async0-alloc0 | passed | 253.83 | 278.42 |
| 86 | ep-fp8-align128-bias2-hcopy0-prev0-async0-alloc1 | passed | 254.33 | 276.89 |
| 87 | ep-fp8-align128-bias2-hcopy0-prev0-async1-alloc0 | passed | 251.24 | 238.75 |
| 88 | ep-fp8-align128-bias2-hcopy0-prev0-async1-alloc1 | passed | 252.62 | 265.90 |
| 89 | ep-fp8-align128-bias2-hcopy0-prev1-async0-alloc1 | passed | 245.93 | 278.49 |
| 90 | ep-fp8-align128-bias2-hcopy0-prev1-async1-alloc1 | passed | 251.19 | 274.39 |
| 91 | ep-bf16-align128-bias0-hcopy0-prev0-async0-alloc0 | passed | 286.56 | 272.75 |
| 92 | ep-bf16-align128-bias0-hcopy0-prev0-async0-alloc1 | passed | 294.58 | 276.35 |
| 93 | ep-bf16-align128-bias0-hcopy0-prev0-async1-alloc0 | passed | 290.66 | 267.57 |
| 94 | ep-bf16-align128-bias0-hcopy0-prev0-async1-alloc1 | passed | 290.02 | 240.68 |
| 95 | ep-bf16-align128-bias0-hcopy0-prev1-async0-alloc1 | passed | 295.24 | 277.67 |
| 96 | ep-bf16-align128-bias0-hcopy0-prev1-async1-alloc1 | passed | 289.71 | 271.01 |
| 97 | ep-bf16-align128-bias1-hcopy0-prev0-async0-alloc0 | passed | 288.63 | 275.33 |
| 98 | ep-bf16-align128-bias1-hcopy0-prev0-async0-alloc1 | passed | 295.70 | 268.61 |
| 99 | ep-bf16-align128-bias1-hcopy0-prev0-async1-alloc0 | passed | 287.19 | 274.90 |
| 100 | ep-bf16-align128-bias1-hcopy0-prev0-async1-alloc1 | passed | 300.49 | 272.69 |
| 101 | ep-bf16-align128-bias1-hcopy0-prev1-async0-alloc1 | passed | 292.40 | 269.05 |
| 102 | ep-bf16-align128-bias1-hcopy0-prev1-async1-alloc1 | passed | 289.42 | 274.46 |
| 103 | ep-bf16-align128-bias2-hcopy0-prev0-async0-alloc0 | passed | 280.90 | 260.68 |
| 104 | ep-bf16-align128-bias2-hcopy0-prev0-async0-alloc1 | passed | 292.94 | 277.08 |
| 105 | ep-bf16-align128-bias2-hcopy0-prev0-async1-alloc0 | passed | 294.10 | 272.77 |
| 106 | ep-bf16-align128-bias2-hcopy0-prev0-async1-alloc1 | passed | 290.60 | 273.36 |
| 107 | ep-bf16-align128-bias2-hcopy0-prev1-async0-alloc1 | passed | 286.96 | 261.82 |
| 108 | ep-bf16-align128-bias2-hcopy0-prev1-async1-alloc1 | passed | 291.03 | 265.89 |
| 109 | ep-fp8-align1-bias0-hcopy0-prev0-async0-alloc0 | passed | 258.67 | 274.25 |
| 110 | ep-fp8-align1-bias0-hcopy0-prev0-async0-alloc1 | passed | 252.19 | 268.40 |
| 111 | ep-fp8-align1-bias0-hcopy0-prev0-async1-alloc0 | passed | 256.39 | 275.19 |
| 112 | ep-fp8-align1-bias0-hcopy0-prev0-async1-alloc1 | passed | 251.39 | 273.06 |
| 113 | ep-fp8-align1-bias0-hcopy0-prev1-async0-alloc1 | passed | 243.20 | 272.13 |
| 114 | ep-fp8-align1-bias0-hcopy0-prev1-async1-alloc1 | passed | 253.33 | 272.36 |
| 115 | ep-fp8-align1-bias1-hcopy0-prev0-async0-alloc0 | passed | 250.70 | 276.92 |
| 116 | ep-fp8-align1-bias1-hcopy0-prev0-async0-alloc1 | passed | 247.56 | 269.15 |
| 117 | ep-fp8-align1-bias1-hcopy0-prev0-async1-alloc0 | passed | 252.32 | 273.58 |
| 118 | ep-fp8-align1-bias1-hcopy0-prev0-async1-alloc1 | passed | 251.89 | 269.83 |
| 119 | ep-fp8-align1-bias1-hcopy0-prev1-async0-alloc1 | passed | 249.26 | 277.32 |
| 120 | ep-fp8-align1-bias1-hcopy0-prev1-async1-alloc1 | passed | 248.96 | 275.95 |
| 121 | ep-fp8-align1-bias2-hcopy0-prev0-async0-alloc0 | passed | 247.15 | 262.49 |
| 122 | ep-fp8-align1-bias2-hcopy0-prev0-async0-alloc1 | passed | 252.15 | 279.59 |
| 123 | ep-fp8-align1-bias2-hcopy0-prev0-async1-alloc0 | passed | 251.56 | 271.99 |
| 124 | ep-fp8-align1-bias2-hcopy0-prev0-async1-alloc1 | passed | 247.30 | 269.21 |
| 125 | ep-fp8-align1-bias2-hcopy0-prev1-async0-alloc1 | passed | 246.95 | 266.87 |
| 126 | ep-fp8-align1-bias2-hcopy0-prev1-async1-alloc1 | passed | 239.96 | 264.23 |
| 127 | ep-bf16-align1-bias0-hcopy0-prev0-async0-alloc0 | passed | 286.09 | 272.35 |
| 128 | ep-bf16-align1-bias0-hcopy0-prev0-async0-alloc1 | passed | 293.43 | 261.14 |
| 129 | ep-bf16-align1-bias0-hcopy0-prev0-async1-alloc0 | passed | 292.61 | 271.89 |
| 130 | ep-bf16-align1-bias0-hcopy0-prev0-async1-alloc1 | passed | 290.82 | 268.20 |
| 131 | ep-bf16-align1-bias0-hcopy0-prev1-async0-alloc1 | passed | 296.57 | 272.86 |
| 132 | ep-bf16-align1-bias0-hcopy0-prev1-async1-alloc1 | passed | 294.23 | 275.60 |
| 133 | ep-bf16-align1-bias1-hcopy0-prev0-async0-alloc0 | passed | 290.38 | 276.76 |
| 134 | ep-bf16-align1-bias1-hcopy0-prev0-async0-alloc1 | passed | 292.83 | 276.06 |
| 135 | ep-bf16-align1-bias1-hcopy0-prev0-async1-alloc0 | passed | 292.77 | 255.00 |
| 136 | ep-bf16-align1-bias1-hcopy0-prev0-async1-alloc1 | passed | 292.32 | 273.32 |
| 137 | ep-bf16-align1-bias1-hcopy0-prev1-async0-alloc1 | passed | 298.35 | 275.39 |
| 138 | ep-bf16-align1-bias1-hcopy0-prev1-async1-alloc1 | passed | 281.69 | 278.22 |
| 139 | ep-bf16-align1-bias2-hcopy0-prev0-async0-alloc0 | passed | 280.46 | 270.24 |
| 140 | ep-bf16-align1-bias2-hcopy0-prev0-async0-alloc1 | passed | 286.96 | 271.17 |
| 141 | ep-bf16-align1-bias2-hcopy0-prev0-async1-alloc0 | passed | 296.76 | 273.96 |
| 142 | ep-bf16-align1-bias2-hcopy0-prev0-async1-alloc1 | passed | 285.85 | 275.06 |
| 143 | ep-bf16-align1-bias2-hcopy0-prev1-async0-alloc1 | passed | 292.41 | 269.81 |
| 144 | ep-bf16-align1-bias2-hcopy0-prev1-async1-alloc1 | passed | 288.63 | 273.94 |

## 6. 结果文件

- 权威汇总：远端 official-reporting-matrix-20261010-r3/matrix-summary.json。
- 可读汇总：远端 official-reporting-matrix-20261010-r3/matrix-summary.md。
- 每个 case 的原始 msprof、stage、对齐 JSON/Markdown 与日志分别位于远端结果目录的 msprof/、stage/、alignment/、logs/ 子目录。
- 本地保留了一份最终摘要快照：.scratch/official-reporting-matrix-20261010-r3/。


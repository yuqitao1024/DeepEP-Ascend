# DeepEP-Ascend 官方性能统计口径归档

本文归档 2026-09-30 之后完成的 DeepSeek 官方 DeepEP-Ascend 性能统计口径对齐工作。结论聚焦在“如何测、如何算、如何解释”，不把包版本差异作为实现差异。

## 归档内容

| 文件 | 说明 |
|---|---|
| [`deep_ep/utils/testing_msprof.py`](../../deep_ep/utils/testing_msprof.py) | 官方 `bench_msprof` 采集器副本，来源 commit `3b25377 Initial public release`；内容与官方文件逐行一致，归档时仅追加了一个文件末尾空行，源文件 SHA256 为 `8ade9aa62c7cdef8d3cf061145bbc641cbd966a508a6d2be4c2dd6c9e3bc0c75`。 |
| [`bench_ep_msprof.py`](../bench_ep_msprof.py) | 本仓库的官方语义 msprof 基准脚本。 |
| [`align_official_reporting.py`](../align_official_reporting.py) | 报告层对齐脚本，将当前 kernel 边界、stage 计数与官方参考口径映射到同一带宽公式。 |
| [`results/official_reporting/fp8-8rank-official-workload.json`](results/official_reporting/fp8-8rank-official-workload.json) | 历史正式对齐结果：10 warmup、50 sample。 |
| [`results/official_reporting/fp8-8rank-official-workload.md`](results/official_reporting/fp8-8rank-official-workload.md) | 历史正式结果的可读报告。 |
| [`results/official_reporting/fp8-8rank-smoke-summary.json`](results/official_reporting/fp8-8rank-smoke-summary.json) | NPU8P 冒烟验证摘要，保留原始文件 SHA256。 |
| [`results/official_reporting/fp8-8rank-formal-summary.json`](results/official_reporting/fp8-8rank-formal-summary.json) | 2026-10-09 修正口径后的 NPU8P 正式结果摘要。 |
| [`results/official_reporting/fp8-8rank-formal-service-issue-drain.json`](results/official_reporting/fp8-8rank-formal-service-issue-drain.json) | 2026-10-09 正式对齐 JSON。 |
| [`results/official_reporting/fp8-8rank-formal-service-issue-drain.md`](results/official_reporting/fp8-8rank-formal-service-issue-drain.md) | 2026-10-09 正式对齐报告。 |
| [`results/official_reporting/fp8-8rank-aligned-workload-service-issue-drain.json`](results/official_reporting/fp8-8rank-aligned-workload-service-issue-drain.json) | 2026-10-09 负载/字节口径对齐后的正式结果摘要。 |
| [`results/official_reporting/fp8-8rank-aligned-workload-service-issue-drain.md`](results/official_reporting/fp8-8rank-aligned-workload-service-issue-drain.md) | 2026-10-09 负载/字节口径对齐后的可读报告。 |

## 负载与字节口径对齐后的复测

2026-10-09 晚间按官方 `tests/ep/test_ep.py` 的负载语义补齐三项测试层差异后重新采集：

- expanded dispatch 开启 `do_zero_padding=True`；
- expanded dispatch 使用 row-major scale factor（`use_tma_aligned_col_major_sf=False`）；
- combine 的 URMA 字节只计 BF16 hidden payload，不计 top-k weights。
- 官方参考 workload 与本轮 benchmark 的 data path 均启动 64 个 AIV：当前固定 `num_sms=64`，官方 direct scale-up 路径同样为 64 AIV。

本轮仍在直连 NPU8P 设备 0–7，CANN /usr/local/Ascend/cann-9.3.0。msprof task 为 task_20261009_211108_88754524853，stage task 为 task_20261009_211205_93635418509，均为 10 warmup、50 sample 且 exit=0。

| 操作 | service_submit + cq_wait | 折算带宽 | 官方均值 | 相对值 |
|---|---:|---:|---:|---:|
| dispatch | 1977.41µs | 274.49 GB/s | 374 GB/s | 0.73x |
| combine | 3647.37µs | 285.82 GB/s | 346 GB/s | 0.83x |

与上一轮相比，dispatch 从 263.52 GB/s 提高到 274.49 GB/s；combine 的结果同时受时间变化与字节口径修正影响，从 291.07 GB/s 变为 285.82 GB/s。本轮两个结果都低于 EP8 物理带宽上限，也低于官方参考值。AIV 启动数不是剩余差距的来源：双方均为 64 AIV。

完整结果摘要见 tests/ascend/benchmark/results/official_reporting/fp8-8rank-aligned-workload-service-issue-drain.json 与同名 Markdown。

## 使用方法

1. 构建 Ascend 版本并进入 8 rank 环境。
2. 以官方语义运行：

   ```bash
   torchrun --standalone --nproc_per_node=8 \
     tests/ascend/benchmark/bench_ep_msprof.py \
     --case ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0 \
     --num-tokens 16384 --hidden 7168 --num-topk 6 --num-experts 256 \
     --warmups 10 --iterations 50 --skip-check \
     --output results/official-msprof/fp8-8rank.json
   ```

3. 若需要把 msprof 结果与 stage profile 对齐，运行：

   ```bash
   python3 tests/ascend/benchmark/align_official_reporting.py \
     --msprof results/official-msprof/fp8-8rank.json \
     --stage results/stage/fp8-8rank.json \
     --output results/official-reporting/fp8-8rank-aligned.json
   ```

历史正式结果中的 stage JSON 已经因为旧远端目录清理丢失；本归档保留当时已经生成的对齐 JSON 与 Markdown。2026-10-09 已重新完成 50-sample msprof 与同 workload 的 stage profile 正式采集，并归档修正后的摘要。

## 官方统计口径

官方 DeepEP-Ascend 的 `bench_msprof` 做了以下事情：

- 每次测量前 flush 8GB L2；
- barrier 前让 device 保持 busy 的 matmul；
- 跨 rank barrier；
- 通信 launch 后清零 256MB dirty buffer，强制 cold-L2 epilogue；
- 使用 FFTS kernel time；
- 带宽使用 per-rank received records 对应的 scaleup logical bytes；
- 默认 10 warmup、50 sample。

本仓库脚本复刻了上述协议，但由于当前实现缺少官方 `set_barrier_in_prologue()` 与 `defer_epilogue` API，只能把 barrier 放在通信外部，且 epilogue 边界无法与官方完全一致。

## 物理带宽与 kernel 边界检查

Netlayer0 的单链物理带宽上限约 50GB/s，不能把 675 GB/s 或 854 GB/s 解释为物理带宽效率；官方口径的 GB/s 也不是网卡式单向链路速率，而是“per-rank 逻辑字节数 / 官方 URMA kernel 耗时”的归一化指标。

尽管如此，若一个归一化结果显著高于官方参考，仍然必须检查 kernel 边界。这个检查推翻了最初按 `dispatch_kernel` / `combine_kernel` 计算 msprof 带宽的口径：

- 当前路径把 producer、release/service、外层控制和 epilogue 拆成多个 kernel；
- `dispatch_kernel` / `combine_kernel` 是外层通信 kernel，并不等价于官方 `dispatch_impl` / `combine_impl`；
- `direct_dispatch_producer_release_kernel` / `direct_combine_producer_release_kernel` 只把 `TransportCommand` 追加到 device command queue，不执行真实 URMA service；
- service 在后续 `dispatch_kernel` / `combine_kernel` 与 barrier kernel 中的 `transport::service::execute()` 里提交和 drain；
- 2026-10-09 正式 msprof 中，dispatch producer-release 平均约 143µs、外层 kernel 约 805µs；combine producer-release 平均约 163µs、外层 kernel 约 1057µs；
- 同 workload stage profile 中，dispatch `service_submit + cq_wait` 约 2.060ms；combine 约 3.588ms。

因此，旧的 `msprof_current_kernel_time` 只会得到一个过小的时间分母，进而得到超过物理链路带宽的表观带宽。这不是实现性能更好，而是统计边界错误。

归档脚本的修正口径如下：

- msprof 侧保留外层 `dispatch_kernel` / `combine_kernel` 作为原始 kernel 诊断，不把它们或 producer-release kernel 当作官方等价 service kernel；
- stage 侧以 `service_submit + cq_wait` 作为官方 URMA issue-and-drain 语义的代理；
`stage_network` 在该结果中为 `publication + service_submit + cq_wait + barrier_wait`，比 issue-and-drain 更宽，不能作为首选对标口径；
consumer/epilogue 不属于官方 URMA kernel 计时。
stage 聚合对每个 rank 的各 phase 取最大值，因此它是跨 rank 的墙钟近似，不是把所有 block 的 cycles 相加；用 `service_submit + cq_wait` 对齐 URMA issue-and-drain 时，仍要注意它是 staged transport 的服务边界代理，不等于官方直驱 kernel 的逐指令边界。

当前 staged transport 没有单一 msprof kernel 等价官方 URMA issue-and-drain。2026-10-09 的 50-sample msprof 原始 JSON 和 5.5MB stage JSON 保留在远端验证目录，未直接入库；本地归档摘要记录其 SHA256、关键 kernel、stage phase 与结果。

## 历史 EP8 结果

Case 为 `ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`，参数为 16384 tokens、hidden 7168、top-k 6、256 experts、8 rank。

| 操作 | 当前 msprof kernel | stage network | stage cq_wait | stage producer | stage producer+network | 官方参考 |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 675.28 GB/s | 238.30 | 359.57 | 313.23 | 135.34 | 373–375 |
| combine | 854.47 GB/s | 292.59 | 342.70 | 298.35 | 147.72 | 345–347 |

### 关键结论

- 旧 `msprof_current_kernel_time` 不能和官方参考比较：它选择了外层 `dispatch_kernel` / `combine_kernel`，遗漏了真正执行 URMA service 的 release kernel。
- 当前最接近官方 URMA issue-and-drain 语义的代理是 `service_submit + cq_wait`。
- 按该口径计算：
  - dispatch 为官方均值的约 `0.96x`；
  - combine 为官方均值的约 `0.99x`。
- 因此，之前“我们比官方快 1.8x–2.5x”的表述不成立，那是 kernel 边界错配带来的表观结果。
- 用户指出的物理带宽检查是有效的红灯：Netlayer0 单链约 50GB/s，675–854GB/s 的表观带宽说明时间分母缺了 transport service。修正后的口径应使用 transport service kernel，或使用 stage 的 `service_submit + cq_wait`。

## 2026-10-09 正式 EP8 结果

运行环境为直连 NPU8P，设备 0–7，CANN `/usr/local/Ascend/cann-9.3.0`，Python 为既有 deepep-venv-py311。代码是 main HEAD `3497395` 的远端验证副本；为了让 16K token 用例跑通，远端副本额外包含 workspace 对齐修改，未进入本地仓库。

- msprof task：`task_20261009_173805_2798064565`，10 warmup、50 sample，exit=0；
- stage task：`task_20261009_175931_305367014262`，同 case，exit=0；
- case：`ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`，16384 tokens、hidden 7168、top-k 6、256 experts、EP8；
- 字节口径：per-rank scaleup logical bytes 的平均值。

| 操作 | service_submit | cq_wait | issue-and-drain | 折算带宽 | 官方均值 | 相对值 |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 557.86µs | 1501.83µs | 2059.69µs | 263.52 GB/s | 374 GB/s | 0.70x |
| combine | 504.97µs | 3082.61µs | 3587.58µs | 291.07 GB/s | 346 GB/s | 0.84x |

同一轮 msprof 的原始 kernel 证据也否定了“producer-release 就是 transport service kernel”的判断：

| 操作 | producer-release 均值 | 外层 kernel 均值 | issue-and-drain |
|---|---:|---:|---:|
| dispatch | 143.49µs | 804.59µs | 2059.69µs |
| combine | 163.01µs | 1057.29µs | 3587.58µs |

因此，本轮正式结论是：

- dispatch 约为官方均值的 `0.70x`；
- combine 约为官方均值的 `0.84x`；
- 之前仅用 `cq_wait` 得到的 `0.96x` / `0.99x` 低估了 service submit 时间，不能作为对齐结论；
- 之前任何 3.6–6.5 TB/s 级别的表观带宽都是 kernel 边界错误，不是可实现性能。

## FP8 scale factor 结论

hidden=7168 时，官方与当前路径每个 token 都传输 224 个 scale 字节：

- 官方：56 个 float32，以 112 个 int16 视图参与 pack；
- 当前：56 个 float32。

dispatch 的每 token 总逻辑字节数为 7464：

- 7168 字节 FP8 payload；
- 224 字节 scale factor；
- 48 字节 top-k index；
- 24 字节 top-k weight。

因此，官方 pack 为 2 字节、当前 pack 为 4 字节的差异，主要是 API/layout 兼容性问题，不是这个 workload 上的带宽差异来源。

本轮 benchmark 已把 scale factor 的 layout 对齐为 row-major，但未把 host ABI 从 float32 改成官方 int16 视图；这是因为当前实现只接受 float32/int32 scale pack，直接传 int16 会在 preflight 报 invalid_sf_tensor。对该 workload 来说，两种 ABI 每 token 都是 224 字节，带宽口径一致。

## NPU8P 冒烟验证

验证时间：2026-10-09。

- 机器：直接连接 NPU8P；
- CANN：`/usr/local/Ascend/cann-9.3.0`；
- Python：`/data/disk2/pyptouser/yuqitao/deepep-venv-py311/bin/python`；
- 代码：远端验证副本，等价于本仓库 main HEAD `4293698`；
- task：`task_20261009_165709_216548018488`；
- 设备：`0,1,2,3,4,5,6,7`；
- 结果：`exit=0`；
- 参数：1 warmup、2 iteration，仅用于脚本可运行性验证。

| 操作 | kernel | GB/s 范围 | 均值 | kernel 时间 |
|---|---|---:|---:|---:|
| dispatch | `dispatch_kernel` | 668.63–681.83 | 675.88 | 795.60–810.00 µs |
| combine | `combine_kernel` | 980.37–992.00 | 986.00 | 1052.13–1064.45 µs |

这个冒烟结果只证明归档脚本在当前代码和 NPU8P 环境上可以正常运行，不能作为正式性能数据。

完整原始 JSON 约 240KiB，主要是每个 rank 重复的 `all_kernels` 明细；运行日志也主要是 warning 和路径信息。二者均未直接提交，归档只在 `fp8-8rank-smoke-summary.json` 中保留每个 rank 的 dispatch/combine 选定 kernel、耗时、字节数与带宽，并记录原始 JSON 的 SHA256。如需逐 kernel 复查，应在下次正式 50-sample 运行中保留原始输出。

远端验证副本中有一处一次性 workspace 修改，仅用于 16K token 用例跑通，未进入本仓库：

`csrc/backends/ascend/elastic/tiling.hpp` 中 `2 * kPublicElasticBufferAlignment` 改为 `16 * kPublicElasticBufferAlignment`，runtime workspace 从 4MiB 提升到 32MiB。

## 遗留注意事项

- 当前实现缺少官方 `set_barrier_in_prologue()`。
- 当前实现缺少官方 `defer_epilogue`。
- 当前 FP8 scale factor 为 column-major，官方参考实现是 row-major。
- stage profile 是一次聚合观测，不是 50 个独立 sample；其 min/mean/max 反映 rank/mapping 差异，不是迭代方差。
- stage cycle counter 没有显式频率，脚本用 `device_timeline_cycles.envelope_cycles / device_seconds.mean` 校准，属于近似。
- 历史正式结果使用 CANN 9.3.0，官方 README 使用 CANN 9.2.0；按既定策略，包版本差异不作为实现差异。
- 历史原始 msprof JSON 和原始 stage JSON 已随旧远端目录删除；2026-10-09 原始 JSON 仍在远端验证目录中，本归档只保留摘要和对齐结果。

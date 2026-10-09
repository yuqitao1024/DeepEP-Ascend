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

历史正式结果中的 stage JSON 已经因为旧远端目录清理丢失；本归档保留当时已经生成的对齐 JSON 与 Markdown。

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
- URMA service 在 `direct_dispatch_producer_release_kernel` / `direct_combine_producer_release_kernel` 中执行；
- 历史冒烟 profile 中，dispatch 外层 kernel 约 799µs，而 stage 的 `service_submit + cq_wait` 约 2.09ms；combine 外层 kernel 约 1224µs，stage 的 `service_submit + cq_wait` 约 3.57ms。

因此，旧的 `msprof_current_kernel_time` 只会得到一个过小的时间分母，进而得到超过物理链路带宽的表观带宽。这不是实现性能更好，而是统计边界错误。

归档脚本的修正口径如下：

- msprof 侧选择 transport service kernel：`direct_dispatch_producer_release_kernel` / `direct_combine_producer_release_kernel`；
- stage 侧以 `service_submit + cq_wait` 作为官方 URMA issue-and-drain 语义的代理；
`stage_network` 在该结果中为 `publication + service_submit + cq_wait + barrier_wait`，比 issue-and-drain 更宽，不能作为首选对标口径；
consumer/epilogue 不属于官方 URMA kernel 计时。

当前遗留限制是：历史 50-sample msprof 原始 JSON 已丢失，无法重新提取 service kernel 的 50-sample 统计；归档的 msprof 数值仍保留当时外层 kernel 的结果，只作为错误口径的历史证据。下一次 NPU8P 正式运行应使用修正后的脚本并保留原始输出。

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
- 历史原始 msprof JSON 和原始 stage JSON 已随旧远端目录删除，本归档只保留生成的对齐摘要。

# DeepEP-Ascend official reporting alignment

Case: `ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`

## Bandwidth mappings

### dispatch (`expanded_dispatch`)

Bytes per rank: 542,779,281

| Mapping | Min GB/s | Mean GB/s | Max GB/s | vs official mean |
|---|---:|---:|---:|---:|
| `msprof_current_kernel_time` | 668.62 | 675.28 | 678.39 | 1.81x |
| `stage_network` | 238.30 | 238.30 | 238.30 | 0.64x |
| `stage_cq_wait` | 359.57 | 359.57 | 359.57 | 0.96x |
| `stage_producer` | 313.23 | 313.23 | 313.23 | 0.84x |
| `stage_producer_plus_network` | 135.34 | 135.34 | 135.34 | 0.36x |

### combine (`reduced_combine`)

Bytes per rank: 1,044,253,815

| Mapping | Min GB/s | Mean GB/s | Max GB/s | vs official mean |
|---|---:|---:|---:|---:|
| `msprof_current_kernel_time` | 851.40 | 854.47 | 860.50 | 2.47x |
| `stage_network` | 292.59 | 292.59 | 292.59 | 0.85x |
| `stage_cq_wait` | 342.70 | 342.70 | 342.70 | 0.99x |
| `stage_producer` | 298.35 | 298.35 | 298.35 | 0.86x |
| `stage_producer_plus_network` | 147.72 | 147.72 | 147.72 | 0.43x |

## Comparability caveats

- **kernel_boundary:** Current dispatch_kernel/combine_kernel cover a wider boundary than official dispatch_impl/combine_impl; msprof numbers are therefore lower bounds on official-equivalent bandwidth.
- **set_barrier_in_prologue:** Current implementation lacks the official deep_ep._C.set_barrier_in_prologue API; the external barrier could not be moved into the communication kernel prologue.
- **defer_epilogue:** Current implementation lacks the official defer_epilogue API; epilogue work cannot be split into the official separate kernels.
- **fp8_scale_factor_layout:** Current expanded FP8 path uses column-major scale factors while the official public reference uses row-major. Byte count is the same for hidden=7168, but access locality is not identical.
- **stage_profile_sample_count:** Stage profile has one aggregated observation, unlike msprof's 50 sampled iterations; its min/mean/max represent rank/mapping variation, not iteration variance.
- **stage_clock_calibration:** Stage JSON has cycle counters but no explicit frequency. Seconds are calibrated as device_timeline_cycles.envelope_cycles divided by device_seconds.mean; this is an approximation, not an independent hardware timer measurement.
- **package_version:** Ignored per comparison policy; CANN/Torch package versions are not treated as implementation differences.

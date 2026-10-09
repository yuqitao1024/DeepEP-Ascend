# DeepEP-Ascend official reporting alignment

Case: `ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`

## Bandwidth mappings

### dispatch (`expanded_dispatch`)

Bytes per rank: 542,779,281

| Mapping | Min GB/s | Mean GB/s | Max GB/s | vs official mean |
|---|---:|---:|---:|---:|
| `msprof_raw_outer_kernel_time` | 3598.66 | 3784.63 | 3865.84 | 10.12x |
| `stage_service_issue_drain` | 263.52 | 263.52 | 263.52 | 0.70x |
| `stage_cq_wait` | 361.41 | 361.41 | 361.41 | 0.97x |
| `stage_network` | 237.91 | 237.91 | 237.91 | 0.64x |
| `stage_producer` | 315.81 | 315.81 | 315.81 | 0.84x |
| `stage_producer_plus_network` | 135.69 | 135.69 | 135.69 | 0.36x |

### combine (`reduced_combine`)

Bytes per rank: 1,044,253,815

| Mapping | Min GB/s | Mean GB/s | Max GB/s | vs official mean |
|---|---:|---:|---:|---:|
| `msprof_raw_outer_kernel_time` | 6057.19 | 6409.74 | 6567.90 | 18.53x |
| `stage_service_issue_drain` | 291.07 | 291.07 | 291.07 | 0.84x |
| `stage_cq_wait` | 338.76 | 338.76 | 338.76 | 0.98x |
| `stage_network` | 290.48 | 290.48 | 290.48 | 0.84x |
| `stage_producer` | 720.46 | 720.46 | 720.46 | 2.08x |
| `stage_producer_plus_network` | 207.01 | 207.01 | 207.01 | 0.60x |

## Comparability caveats

- **kernel_boundary:** The current implementation has no single msprof kernel that matches official URMA issue-and-drain. producer_release only appends transport commands; use stage_service_issue_drain for the preferred proxy.
- **set_barrier_in_prologue:** Current implementation lacks the official deep_ep._C.set_barrier_in_prologue API; the external barrier could not be moved into the communication kernel prologue.
- **defer_epilogue:** Current implementation lacks the official defer_epilogue API; epilogue work cannot be split into the official separate kernels.
- **fp8_scale_factor_layout:** Current expanded FP8 path uses column-major scale factors while the official public reference uses row-major. Byte count is the same for hidden=7168, but access locality is not identical.
- **stage_profile_sample_count:** Stage profile has one aggregated observation, unlike msprof's 50 sampled iterations; its min/mean/max represent rank/mapping variation, not iteration variance.
- **stage_clock_calibration:** Stage JSON has cycle counters but no explicit frequency. Seconds are calibrated as device_timeline_cycles.envelope_cycles divided by device_seconds.mean; this is an approximation, not an independent hardware timer measurement.
- **package_version:** Ignored per comparison policy; CANN/Torch package versions are not treated as implementation differences.

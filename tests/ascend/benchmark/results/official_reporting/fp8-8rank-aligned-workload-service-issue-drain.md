# DeepEP-Ascend official reporting alignment

Case: `ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`

This run aligns the benchmark workload with the official EP8 test:

- expanded dispatch uses `do_expand=True`, `do_zero_padding=True`, and row-major scale factors;
- combine bandwidth counts only the BF16 hidden payload, excluding top-k weights from the official URMA byte formula;
- dispatch uses FP8 payload plus scale bytes and combine uses BF16 payload, with official top-k index and weight dtypes.
- the official reference workload and this benchmark both launch the data path with 64 AIVs.

## Bandwidth mappings

### dispatch (`expanded_dispatch`)

Bytes per rank: 542,779,281

| Mapping | Mean GB/s | vs official mean |
|---|---:|---:|
| `stage_service_issue_drain` | 274.49 | 0.73x |

### combine (`reduced_combine`)

Bytes per rank: 1,042,508,544

| Mapping | Mean GB/s | vs official mean |
|---|---:|---:|
| `stage_service_issue_drain` | 285.82 | 0.83x |

## Interpretation

The service issue-and-drain proxy remains below the physical EP8 ceiling. AIV
launch count is not a source of the remaining gap: both sides use 64 AIVs.
Dispatch improved from 263.52 to 274.49 GB/s after row-major scale factors and
zero padding were enabled. Combine changed from 291.07 to 285.82 GB/s after
its byte formula was corrected to exclude top-k weights; the time also moved
from 3.588ms to 3.647ms in this run.

Full min/mean/max mappings, per-rank msprof records, and stage counters are in
the remote source artifacts listed in the JSON summary. The original full JSON
files are intentionally not committed.


## official_reporting

This directory archives the official reporting-alignment scripts' historical
output and the 2026-10-09 NPU8P smoke validation output.

- `fp8-8rank-official-workload.json` and `.md`: the aligned historical result
  for 10 warmups and 50 samples.
- `fp8-8rank-formal-summary.json`: compact summary of the corrected formal
  NPU8P run on 2026-10-09, including stage service issue-and-drain and the
  raw kernel observations used to reject producer-release as a service proxy.
- `fp8-8rank-formal-msprof-summary.json`: the per-rank 50-sample msprof
  record with the repeated `all_kernels` map removed.
- `fp8-8rank-formal-stage-profile-summary.json`: the corrected formal stage
  profile with per-block raw counters removed.
- `fp8-8rank-formal-service-issue-drain.json` and `.md`: the corrected
  formal report generated from the same msprof and stage-profile runs.
- `fp8-8rank-smoke-summary.json`: a compact per-rank summary of the NPU8P
  smoke validation.

The original full smoke JSON, formal msprof JSON, formal stage-profile JSON,
and runtime logs were intentionally not archived. The compact summaries
retain the selected kernels, timing, bytes, bandwidth, stage mappings, and
source-file SHA256 values. The smoke source SHA256 was
`9b5be60369d9fd7039c239a89cc8f1c7f9118de7ceca8dc4d18d8b1ce1c7224b`.

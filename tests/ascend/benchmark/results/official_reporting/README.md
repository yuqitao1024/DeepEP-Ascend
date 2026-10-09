## official_reporting

This directory archives the official reporting-alignment scripts' historical
output and the 2026-10-09 NPU8P smoke validation output.

- `fp8-8rank-official-workload.json` and `.md`: the aligned historical result
  for 10 warmups and 50 samples.
- `fp8-8rank-smoke-summary.json`: a compact per-rank summary of the NPU8P
  smoke validation.

The original full smoke JSON and runtime log were intentionally not archived.
The JSON was about 240 KiB and primarily repeated detailed `all_kernels`
records for every rank; the summary retains the selected dispatch/combine
kernels, timing, bytes, and bandwidth for every rank. Its SHA256 was
`9b5be60369d9fd7039c239a89cc8f1c7f9118de7ceca8dc4d18d8b1ce1c7224b`.

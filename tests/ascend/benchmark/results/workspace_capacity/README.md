# Runtime workspace capacity A/B

Date: 2026-10-10.

The default Ascend elastic runtime workspace reservation is increased from 4MiB
(`2 * kPublicElasticBufferAlignment`) to 32MiB
(`16 * kPublicElasticBufferAlignment`). The larger reservation removes the
`dispatch capacity exceeds runtime storage` failure for the official-aligned
16384-token EP8 workload while leaving the actual producer, transport service,
and epilogue data paths unchanged.

## A/B protocol

- Host: direct NPU8P, devices 0-7;
- CANN: /usr/local/Ascend/cann-9.3.0;
- Python: /data/disk2/pyptouser/yuqitao/deepep-venv-py311/bin/python;
- Case: `ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0`;
- Tokens: 8192 per rank, hidden 7168, top-k 6, 256 experts;
- `num_sms=64`;
- 10 warmup, 30 iterations;
- Order: W32-1, W4-1, W4-2, W32-2;
- Each side combines two runs and 60 timed samples;
- Timing uses the maximum device time across eight ranks.

Both binaries use the same main `d78a57e` source and the same default
AICore/URMA transport backend. The only source difference is the workspace
reservation constant.

## Results

The table reports W32 relative to W4. Positive change means W32 is faster.

| Operation | 4MiB ms | 32MiB ms | Delta | Relative |
|---|---:|---:|---:|---:|
| Dispatch | 13.062718 | 12.992807 | -0.000070 | +0.54% |
| Expanded Dispatch | 54.729856 | 54.577851 | -0.000152 | +0.28% |
| Cached Dispatch | 11.538074 | 11.380869 | -0.000157 | +1.36% |
| Combine | 8.117452 | 8.038813 | -0.000079 | +0.97% |
| Reduced Combine | 7.823616 | 7.713198 | -0.000110 | +1.41% |

The 32MiB side is slightly faster in this run, but the differences are within
normal short-run variation. The important conclusion is that there is no
performance regression. The change reserves more device memory but does not add
kernel work or change the data layout.

## Tasks and artifacts

| Run | Task |
|---|---|
| W32-1 | `task_20261010_082950_2790302772` |
| W4-1 | `task_20261010_083215_283959118440` |
| W4-2 | `task_20261010_083256_284345514878` |
| W32-2 | `task_20261010_083336_28483654194` |

Raw JSON files and `summary.json` are archived in this directory.


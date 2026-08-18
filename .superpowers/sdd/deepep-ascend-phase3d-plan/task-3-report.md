# Task 3 Report: Device-Only Two-Stage Dispatch And Reverse Combine

## Implementation Summary

- Enabled Ascend hybrid construction and buffer sizing only for an exact 2x2
  topology while preserving the zero-byte CPU communicator/buffer contract.
- Added explicit diagonal dispatch stages: origin-to-ingress over scale-out,
  then ingress-to-destination over scale-up. Both stages flush payload before
  publishing generation/count controls, signal, and participate when empty.
- Added complete device-produced route records after the ABI-v4 descriptor in
  `token_metadata_at_forward`. Direct handles remain descriptor-only.
- Reused cached route geometry while rebinding the descriptor and every route
  record to the current dispatch generation.
- Added reverse diagonal combine: destination-to-ingress over scale-up ordered
  by recorded `forwarded_slot`, then ingress-to-origin over scale-out ordered by
  recorded `ingress_slot`. The compact ordering supports both expanded
  reduction settings without changing the direct combine-record layout.
- Added logical reference coverage, source/runtime contracts, and AICore runner
  cases for local, scale-up, scale-out, diagonal, empty, cached, single-reduce,
  and multiple-reduce behavior.

## Two-Stage Schedules

For world rank `scale_out_rank * 2 + scale_up_rank`, dispatch `0 -> 3` uses
ingress rank `2`:

```text
dispatch: 0 --scale-out--> 2 --scale-up--> 3
combine:  3 --scale-up----> 2 --scale-out-> 0
```

Direct local, same-domain scale-up, and same-rail scale-out rows keep their
single-team paths. Diagonal commands never select `TransportTeam::kWorld`.

The hybrid handle byte layout is:

```text
DispatchHandleDescriptor
HybridRouteRecord[descriptor.route_record_count]
```

Each diagonal combine record carries its two recorded slots in the existing
aligned record tail. Route-slot bounds and duplicates are rejected before any
payload or control publication. The first reverse stage compacts records in
`forwarded_slot` order; the ingress stably reorders them by `ingress_slot`
before returning them to the origin's contributor shard.

## TDD Evidence

Initial RED before implementation:

```bash
python3 -m pytest tests/ascend/test_hybrid_smoke.py \
  tests/ascend/test_core_operator_contract.py -q
```

```text
2 failed, 44 passed
```

An intermediate runner-contract RED produced `9 failed, 37 passed`; after the
logical model was complete, the remaining production runner RED was
`1 failed, 45 passed`.

The first remote Bisheng compile, task `task_20260818_152419_1155253449`,
failed because device code called host-only `mode_bit`/`has_mode` helpers and
assigned a whole local `HybridRouteRecord` into `__gm__`. Focused source
contracts were added first, then the kernels switched to inline device-safe bit
tests and field-by-field global-memory stores.

The second compile, task `task_20260818_153016_1478323542`, exposed the same
annotation class for `is_complete_hybrid_route_stage_flags`. A focused contract
was watched fail and then pass after applying the established
`DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE` annotation.

The third compile, task `task_20260818_154437_2801831870`, confirmed Bisheng
does not allow host validators in the same Ascend translation unit to call that
device-only helper. The regression now locks the domain split: the combine
device path calls the annotated helper, while both host validators compare
directly with `kHybridRouteCompleteStageFlags`.

The fourth compile, task `task_20260818_154952_29962120340`, built all
production Task 3 sources and the compile probe successfully through 71%. Its
only failures were three runner-only host declarations using an unqualified
transport namespace outside `deep_ep::ascend`. Exact source contracts were
watched fail, then the declarations were changed to
`deep_ep::ascend::transport::DeviceTransportContext`.

Self-review then exposed contributor-order batching in reverse combine. A RED
case reordered route-table rows relative to recorded slots; the final
implementation now uses `forwarded_slot` and `ingress_slot` for the two reverse
stages.

## Verification

Focused local GREEN:

```text
python3 -m pytest tests/ascend/test_hybrid_smoke.py \
  tests/ascend/test_core_operator_contract.py -q
48 passed
```

Full local regression:

```text
python3 -m pytest tests/ascend/test_core_operator_contract.py \
  tests/ascend/test_python_api.py tests/platform -q
74 passed, 3 skipped, 20 subtests passed
```

`git diff --check` and `python3 -m py_compile` for both new Python files also
completed successfully.

Final serialized incremental AOT, task
`task_20260818_155853_33334029760`, ran on permitted device 6 and exited `0`.
The production probe was already built at 71%; the corrected runner compiled,
linked, and built its target at 100%. The device lock was released.

## Device-Safety Audit

- Audited every newly added call in `__simt_vf__` and global AICore paths in
  `dispatch.asc` and `combine.asc`. Project helpers are device annotated;
  kernel-only mode tests are literal bit operations. Device-only helpers have
  no host callers in the same Ascend translation units.
- Audited new global-memory writes. Route and hybrid combine metadata are stored
  field-by-field, and payload relocation uses byte copies between `__gm__`
  addresses; no whole local aggregate is assigned to global memory.
- Added explicit system fences after local forwarding-buffer writes and before
  transport puts in the scale-up dispatch and scale-out combine return stages.
- Empty participants publish zero counts in both dispatch and combine stages.
- Dispatch route records are emitted in deterministic origin-rank/slot order.
- Reverse combine validates route identity, completion flags, slot sentinels,
  global slot bounds, and duplicates before staging or publishing commands.
- The implementation retains `num_cpu_buffer_bytes == 0`; no mapped-host-memory
  dependency was introduced.

## Files Changed

- `csrc/backends/ascend/elastic/dispatch.asc`
- `csrc/backends/ascend/elastic/combine.asc`
- `csrc/backends/ascend/elastic/dispatch_state.hpp`
- `csrc/backends/ascend/elastic/kernels.hpp`
- `csrc/backends/ascend/elastic/runtime.cpp`
- `csrc/backends/ascend/elastic_buffer.hpp`
- `tests/ascend/core_ops/core_operator_runner.asc`
- `tests/ascend/production/run_hybrid_smoke.py`
- `tests/ascend/test_hybrid_smoke.py`
- `tests/ascend/test_core_operator_contract.py`

## Concerns

- The available TaskQueue policy permits at most two NPUs per task, while the
  production hybrid smoke requires four ranks. Bisheng AOT compilation covers
  the device source, and the exact 2x2 behavior is covered by the independent
  model and AICore probe cases, but a four-rank hardware runtime was not run in
  this environment.
- Reverse-stage slot ordering is a bounded allocation-free O(n^2) device scan.
  It is deterministic and preserves fixed storage, but large-capacity hybrid
  performance has not yet been benchmarked.

# Ascend direct EP 72-AIV pipeline design

**Status:** Implemented and qualified on NPU8P

## Goal

Allow the Ascend direct, single-host dispatch and combine paths to run their
large data stages on 72 AI Vector blocks while retaining the current one-block
behavior as a supported fallback. The change must preserve output ordering,
generation handling, transport release ordering, cached handles, and the
five-second operation watchdog.

This work is the first implementation slice of P0.1 in
`docs/ascend-design/epv2-ascend-performance-optimization.md`. The user referred
to it as P0.0; this specification uses the repository's existing P0.1 name in
code and test artifacts.

## Scope

Included:

- Direct pure-scale-up dispatch and combine.
- BF16 and FP8 dispatch.
- Compact, expanded, cached, synchronous, CPU-sync split, and asynchronous
  modes already supported by the direct path.
- A configurable data block count from 1 through 72.
- A canonical maximum-performance setting of 72 blocks.
- Separate one-block control kernels and multi-block data kernels submitted to
  the same stream.
- Host contract tests, ASC compile probes, two-device correctness tests, and a
  representative performance comparison on NPU8P.

Excluded:

- Hybrid and physical scale-out execution. These continue to require one
  block.
- P0.2 subgroup grouping, P0.3 vector/DataCopy movement, metadata histogram
  rewrites, and HCOMM publication batching.
- Changes to benchmark shapes, logical-byte formulas, timeout values, or
  public output ordering.
- Automatic tuning of the block count. This slice accepts an explicit count;
  tuning across 1, 8, 16, 32, 64, and 72 follows after the 72-block path is
  correct and measurable.

## Why the current kernel cannot simply launch 72 blocks

`dispatch_kernel` and `combine_kernel` currently execute a sequence of
`asc_vf_call` stages inside one outer `__global__ __vector__` kernel. Several
stages clear shared status, publish transport commands, execute the AICore
transport service, write completion generations, or assume a block-local
barrier. Launching 72 copies of the outer kernel would therefore:

- clear or update the same workspace from multiple blocks;
- construct and execute duplicate transport commands;
- publish control before every payload writer has completed;
- run later stages without a device-wide barrier; and
- write the same completion generation multiple times.

A cross-block spin barrier is not acceptable. It assumes all 72 blocks are
resident together and can deadlock when the scheduler cannot make that
guarantee.

## Selected architecture

The host launcher submits a sequence of global kernels to one stream. Kernel
completion on that stream is the device-wide stage boundary.

Two launch shapes are carried in `CoreTiling`:

```text
control_launch = 1 block, 512 SIMT threads
data_launch    = num_sms blocks, 512 SIMT threads per block
```

The public Ascend `num_sms` argument means `data_launch.num_blocks`. Valid
values are 1 through 72 for direct pure-scale-up mode. Hybrid mode accepts
only 1. `num_qps` remains 0 because this change does not alter HCOMM channel
selection.

Value 0 keeps the existing Python default behavior:

- dispatch resolves `num_sms=0` to 72 on the direct Ascend path;
- combine resolves `num_sms=0` from `handle.num_sms`;
- explicit `num_sms=1` selects the compatibility fallback; and
- a dispatch handle records the resolved value so combine uses the same
  resource policy by default.

The tiling ABI is incremented. Runtime descriptor validation reconstructs both
launch shapes from the input block count rather than accepting a host-mutated
descriptor.

## Stage split

### Dispatch producer

The direct producer is submitted in this order:

1. `dispatch_producer_control`: one block; reset transport state, validate
   fixed layout, and clear workspace status and counters.
2. `dispatch_producer_plan`: one block in this slice; preserve current
   deterministic destination-slot assignment.
3. `dispatch_producer_reduce_errors`: one block.
4. `dispatch_producer_record`: data grid; copy token records into disjoint
   staging slots.
5. `dispatch_producer_release`: one block; copy or publish completed staging
   shards only after the record kernel has finished.
6. `dispatch_transport_service`: one block; execute the staged HCOMM commands.

The data record stage uses:

```text
global_thread = blockIdx.x * blockDim.x + threadIdx.x
global_stride = gridDim.x * blockDim.x
```

Each destination slot already has one deterministic owner. The 72-block stage
uses that slot as its disjoint write unit and does not use an atomic result as
public ordering.

### Dispatch epilogue

Control, acquire, prefix, expert counting, validation, and completion remain
one-block stages in this slice. The hidden payload and scale-factor output
copy becomes a separate data-grid kernel. Expanded padding writes are included
only where ownership is disjoint and already encoded by the current slot
layout. Any stage whose present algorithm assigns one owner per rank or expert
remains one block until the later metadata work.

The CPU-sync split path submits its epilogue stages after host-visible counts
have been read, using the same data block count from the original tiling.

### Combine producer

The direct producer is submitted in this order:

1. producer control: one block;
2. deterministic plan and error reduction: one block each;
3. record construction: data grid over disjoint record slots;
4. local copy and transport release: one block each;
5. transport service execution: one block.

### Combine epilogue

Acquire, contributor planning, validation, deterministic first-error
selection, and completion publication remain one block. BF16 output reduction
is a data-grid kernel over disjoint `(token, hidden)` elements. Routing-weight
output uses a data grid when each output element has one owner. Completion is
submitted only after both data kernels finish.

The current `allow_multiple_reduction` and expanded behavior is preserved. The
data stage changes iteration ownership, not which records contribute to a
token. Expanded combine with `allow_multiple_reduction=false` still uses the
legacy general algorithm and is submitted with the one-block control launch;
the host launcher never submits that algorithm on the 72-block data grid.

## Kernel interface rules

- Control kernels reject or ignore `blockIdx.x != 0` defensively but are
  launched with one block.
- Data kernels derive work from global thread and stride. They never publish
  transport control or final generation values.
- Every global stage wrapper takes the existing `CoreTiling` descriptor to
  avoid a second argument ABI for topology and layout.
- The existing one-block outer kernels remain available only as the
  compatibility implementation selected by `data_num_blocks == 1` until the
  split pipeline has passed device qualification. The 72-block path uses the
  split launcher exclusively.
- Launch errors identify the stage name rather than reporting only
  `dispatch kernel launch failed` or `combine kernel launch failed`.

## Python and runtime behavior

Ascend preflight accepts `num_sms` in `[0, 72]` for direct pure-scale-up
dispatch and combine, rejects negative values and values above 72, and keeps
`num_qps=0`. Hybrid preflight rejects explicit values other than 0 or 1.

The C++ backend repeats these checks before allocation or launch. It passes the
resolved data block count into `build_dispatch_tiling` and
`build_combine_tiling`. Capacity checks no longer require
`tiling.launch.num_blocks == 1`; they validate the explicit control and data
launch shapes instead.

Benchmark runtime configuration changes from fixed `num_sms=1` to an explicit
profile or CLI value. Ascend profiles and the direct CLI default use 72;
`--num-sms 1` selects the comparison baseline. The selected value is recorded
in `benchmark.json` without changing the shared workload manifest or its
fingerprint. Existing tests that specifically exercise fallback or hybrid
behavior continue to pass 1.

## Correctness invariants

- One destination slot has one record writer.
- One `(token, hidden)` output element has one combine reduction writer.
- Payload kernels finish before release publication.
- Release acquisition finishes before receive-side data kernels start.
- Receive-side data kernels finish before completion generation publication.
- The same stream orders every stage in one operation.
- Async events are recorded after the final completion kernel, not after an
  intermediate data stage.
- Previous-event dependencies are waited before the first control stage.
- Status and deterministic first-error selection preserve the existing rank,
  record, and lane precedence.
- Cached dispatch slot indices and handle metadata remain byte compatible.

## Testing strategy

Testing follows red-green cycles.

Host and source contract tests first prove:

- direct tiling accepts 1 and 72 data blocks and rejects 0 after resolution,
  73, and hybrid 72;
- runtime descriptor reconstruction detects a forged launch shape;
- Python preflight resolves default direct dispatch to 72 and combine reuses
  the handle count;
- explicit 1 remains supported;
- launch recording observes ordered control/data/control stages with shapes
  `1/72/1`;
- global work partition helpers cover every item exactly once for block counts
  1 and 72, including empty and non-divisible workloads; and
- no data-stage interface can publish release or completion state.

ASC compile probes instantiate direct BF16 and FP8 dispatch plus compact and
expanded combine at 72 blocks. Existing hybrid probes remain one block.

NPU8P qualification uses only TaskQueue and the permitted device pair `0,1`
or `6,7`, with at most one active task. It proceeds in this order:

1. one-device stage-launch smoke for 1 and 72 blocks;
2. two-device direct BF16 dispatch/combine correctness at 1 and 72;
3. two-device FP8 dispatch plus BF16 combine at 72;
4. cached and expanded representative cases at 72;
5. one canonical representative case, one warmup and at least three measured
   iterations, comparing 1 against 72; and
6. the existing allowed two-device smoke matrix.

The five-second operator watchdog remains unchanged. A timeout, output
mismatch, stale generation, missing event, or malformed artifact fails the
qualification even if latency improves.

## Acceptance criteria

- Profiler or launch trace proves that the selected data stages launch 72
  blocks and execute on more than one AI Vector.
- Direct BF16, FP8, cached, expanded, synchronous, and supported asynchronous
  cases match the one-block reference output.
- Hybrid and logical scale-out paths still reject or select the one-block
  fallback.
- The representative canonical dispatch and combine operations complete
  within the existing watchdog.
- Repeated measurements report the targeted stage and every unaffected public
  operation without imposing a fixed percentage threshold.
- Benchmark schema, workload fingerprint, case IDs, and logical-byte formulas
  are unchanged.

## NPU8P qualification evidence

The qualification used CANN 9.2.0, the pinned
`hcomm-deepep-current` package, the `deepep-ascend-py310` environment, and
devices `6,7`. The extension build completed in
`task_20260821_131712_269628026508`. Production ASC compilation also completed
in `task_20260821_124045_259050332208` and the final combine pipeline probe in
`task_20260821_130428_267183818000`.

Correctness coverage:

- `task_20260821_132046_27085297174`: two-rank BF16 compact, expanded,
  cached, combine, and reduced-combine paths with one data block;
- `task_20260821_132148_27114842719`: the same workload and case with 72 data
  blocks; and
- `task_20260821_132615_272404714981`: FP8 dispatch with previous-event and
  asynchronous compute-stream mode at 72 data blocks.

All three tasks completed with one case passed after reference checks. The
small `16 x 128` workload showed that seven stage submissions can dominate
latency; it is a correctness smoke, not performance evidence.

The archived representative measurement used two ranks and BF16 with one
warmup and three measured iterations. Baseline task
`task_20260821_132309_271487832248` and 72-block task
`task_20260821_132456_271989713619` used the same workload fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`.

| Operation | 1 block mean | 72 blocks mean | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | 1417.60 ms | 150.35 ms | 9.43x | 89.39% |
| Expanded dispatch | 2556.65 ms | 152.06 ms | 16.81x | 94.05% |
| Cached dispatch | 1409.17 ms | 143.09 ms | 9.85x | 89.85% |
| Combine | 1002.76 ms | 97.82 ms | 10.25x | 90.25% |
| Reduced combine | 1278.79 ms | 121.04 ms | 10.56x | 90.53% |

The benchmark reports record `device.num_sms` as 1 and 72 respectively. Host
tiling contracts and the ASC launcher compile prove the data stages consume
the 72-block launch shape. A profiler-level active-core trace was not captured
in this pass, so the report field is configuration evidence rather than a
claim that all 72 AI Vector cores were simultaneously resident.

## Rollback and diagnosis

Explicit `num_sms=1` is the operational rollback. A failed 72-block run must
record the operation, stage, generation, rank, data block count, task ID, and
diagnostic status. Increasing timeouts or silently selecting one block is not
an accepted recovery path.

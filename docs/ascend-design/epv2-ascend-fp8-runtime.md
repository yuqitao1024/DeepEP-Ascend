# Ascend EPv2 Phase 3F FP8 Runtime Design

## Status

Synchronous single-host execution is implemented and qualified at 2, 4, and
8 ranks. The pure-scale-up async/stream-overlap extension described here is
the next implementation slice. Performance qualification, physical
multi-host execution, FP8 combine, the Phase 3E.3 pre-epilogue dependency
boundary, and Phase 3E.4 same-buffer multiflight remain outside this scope.

## Goal

Implement Ascend EPv2 FP8 dispatch with exact scale-factor transport and
pure-scale-up async overlap while preserving the existing BF16 combine
contract, BF16 behavior, public API signatures, transport ordering, generation
safety, and teardown.

## Scope

Phase 3F supports the existing `ElasticBuffer.dispatch` tuple input:

- payload: contiguous NPU `torch.float8_e4m3fn`, shape `[tokens, hidden]`;
- scale factors: NPU `torch.float32` or packed UE8M0x4 `torch.int32`, shape
  `[tokens, sf_packs]`, with arbitrary positive two-dimensional strides; and
- output scale factors: the same dtype and logical shape, either row-major or
  the existing `use_tma_aligned_col_major_sf` strided layout.

The selected combine path remains the existing synchronous BF16 input and
BF16 output path. FP8 is a dispatch wire and expert-input format, not a new
combine reduction format. Normal, expanded, aligned zero-padded, cached,
weighted, empty, asymmetric, and `-1` lane routing are in scope. Direct
scale-up and already-supported logical hybrid routes use the same record
contract.

The async extension covers cached and CPU-synchronized non-cached dispatch,
normal and expanded layouts, communication-stream allocation, completion
events, and `previous_event` ordering. It does not enable
`previous_event_before_epilogue`, same-buffer multiflight, logical hybrid or
physical scale-out async, mapped CPU memory, graph capture, FP8 combine
accumulation, new quantization kernels, or performance tuning.

## Selected Approach

Extend the existing versioned dispatch record with exact FP8 payload and SF
bytes. Keep one transport schedule and one generation/control protocol for
BF16 and FP8. The record layout is derived from element size and SF pack
geometry; every route, including hybrid forwarding, copies the complete record
as opaque bytes. The dispatch epilogue writes payload and SF into their public
output tensors using explicit strides.

This approach is preferred because the current producer, forwarding stages,
and epilogue already carry SF pointers and token-layout offsets. It preserves
the accepted ordering and failure model while limiting new behavior to record
geometry, validation, and strided SF copies.

Rejected alternatives:

1. Support only contiguous FP32 factors. This is smaller but omits the upstream
   packed UE8M0 and column-major output contracts.
2. Dequantize FP8 to BF16 on the host-facing path before communication. This
   changes the dispatch bandwidth and exact payload contract and would make
   Phase 3F a BF16 fallback rather than FP8 runtime support.

## Public Contract

Python collective preflight accepts exactly two payload forms:

- BF16 tensor with no SF tensor; or
- `(float8_e4m3fn tensor, SF tensor)` where SF is float32 or int32, has two
  dimensions, matches the token count, has at least one pack, resides on the
  same NPU, and has positive strides.

Preflight records payload dtype, hidden width, SF dtype, SF pack count, SF
layout class, output SF layout selection, routing mode, and existing topology
fields. Token count remains wildcarded so asymmetric ranks are valid. A dtype,
pack-count, or layout disagreement fails collectively before output allocation
or kernel launch.

`use_tma_aligned_col_major_sf=False` returns SF strides `[sf_packs, 1]`.
`True` returns strides `[1, align(output_tokens, 4)]`, matching the upstream
16-byte SF alignment because each pack is four bytes. The option is meaningful
only for FP8; BF16 calls may leave it false and reject true.

Cached handles continue to cache routing, not payload representation. A valid
handle may route a later BF16 or FP8 payload with identical token, hidden,
expert, top-k, alignment, topology, and generation-family geometry. The
current call's FP8/SF contract is still collectively validated.

## Layout And Capacity

`SymmetricWindowInput.element_bytes` describes dispatch payload bytes and gains
an explicit `scale_factor_bytes`. Dispatch record order remains:

1. payload bytes;
2. SF bytes;
3. top-k indices;
4. top-k weights; and
5. source metadata.

Combine records always reserve `hidden * 2` bytes for BF16, independent of the
dispatch element size. Existing BF16 geometry stays byte-for-byte unchanged.

The public FP8 size hint uses one-byte payloads and the upstream conservative
bound `ceil(hidden / 32) * 4` SF bytes per token. An actual dispatch uses its
exact `sf_packs * 4` geometry and rejects a request larger than that bound or
the allocated communication buffer. All size arithmetic remains checked and
aligned through the existing layout builders.

## Host Runtime

The Ascend C++ entry point validates tensor device, dtype, shape, stride, and
payload/SF pairing before reserving an operation or allocating outputs. It
builds dispatch tiling with `ElementKind::kFloat8E4M3`, exact SF pack count, and
four-byte packs. BF16 tiling remains unchanged.

The output payload uses `x.options()`. The output SF uses `sf.options()` and
`torch::empty_strided` with the selected public stride. Dispatch arguments
carry input and output SF token/pack strides in element units. Empty input is
valid and still returns correctly typed empty payload and SF tensors.

The core runtime accepts FP8 only for dispatch. It requires input SF for a
non-empty FP8 producer and output SF for a non-empty FP8 copy stage, while
rejecting either representation for BF16. Combine continues to reject an FP8
element kind.

## Async And Stream Overlap

FP8 uses the Phase 3E.2 two-stage dispatch architecture without introducing a
second transport protocol. A non-cached CPU-synchronized launch first runs the
producer and transport stages, publishes exact receive and expanded counts,
and synchronizes that communication-stream stage with the host. Count-only
runtime validation requires the FP8 input SF pointer and positive input
strides, but does not require output pointers.

After the counts are known, the host allocates `recv_x`, `recv_sf`, routing,
weight, and metadata outputs with their exact logical extents. `recv_sf` keeps
the input SF dtype. Row-major output uses `[sf_packs, 1]`; column-major output
uses `[1, align(output_tokens, 4)]`. For an empty column-major output this is
exactly `[1, 0]`, and a null data pointer is valid because the epilogue copies
zero output records. For a non-empty FP8 output, epilogue validation requires a
non-null SF pointer and positive strides before launching the device copy.

Cached dispatch already knows the logical receive and expanded counts. It
allocates the FP8 SF result using that exact output count, records the input and
output SF tensors on the communication stream, and returns the SF tensor in
both synchronous and asynchronous modes. The returned async event records the
launch that writes payload and SF. Input SF, output SF, payload, routing,
metadata, descriptor, and predecessor resources remain retained until event
completion. Descriptor publication remains deferred through `AsyncBufferState`
and follows the same completion generation as BF16.

The Python collective preflight and C++ boundary admit stream overlap only for
pure scale-up cached dispatch or pure scale-up non-cached dispatch with CPU
count synchronization. FP8 does not weaken the existing requirements that a
`previous_event` must pair with communication-stream allocation and that one
operation at a time owns a buffer.

## Device Data Flow

The producer copies FP8 payload bytes and each four-byte SF pack into the local
record using the input strides. Transport puts, hybrid ingress, forwarding,
generation publication, and acquire checks copy the complete record and do not
interpret either representation.

The epilogue copies payload bytes to the contiguous output and SF packs through
the output strides. Expanded routing duplicates both payload and SF for each
valid expert lane. Zero padding clears payload, SF, and optional weights for
every alignment-only slot. Cached routing uses the existing validated slots
and performs the same payload/SF copies.

No device conversion is performed. E4M3 payload and both SF representations
are transported bitwise. Numerical acceptance dequantizes with an independent
PyTorch reference: FP32 factors multiply each 128-value group; packed UE8M0x4
factors are decoded from exponent bytes before multiplication. Dispatch bytes
and SF values are checked exactly. BF16 combine is compared after reproducing
the runtime's BF16-input, FP32-accumulation, BF16-output quantization order.

## Failure Model

Malformed payload/SF pairing, dtype, shape, stride, pack count, output layout,
or cached handle fails in Python collective preflight and again at the C++
boundary. Capacity and internal pointer inconsistencies fail in runtime
validation. Device protocol diagnostics retain rank, peer, operation, channel,
command, and generation fields.

An admitted operation that fails continues to poison the buffer through the
existing lease semantics. No FP8-specific resource is owned, so teardown order
and retry behavior are unchanged.

## Testing And Acceptance

Host tests cover:

- exact BF16 and FP8 record geometry and overflow;
- FP8 dispatch runtime admission and missing-pointer rejection;
- FP8 combine rejection;
- public FP8 size hints;
- Python preflight acceptance for float32 and packed int32 SF;
- malformed dtype, shape, stride, pairing, and cross-rank contracts;
- ordinary and column-major output stride forwarding; and
- preservation of platform-source isolation.

Async host coverage additionally requires:

- Python admission for cached and non-cached FP8 stream modes while deferred
  pre-epilogue, hybrid, scale-out, graph-capture, and non-CPU-sync modes remain
  rejected;
- runtime admission for FP8 count-only and epilogue launches, including
  missing input SF, missing non-empty output SF, and empty column-major output;
- exact FP8 SF allocation and stride forwarding for normal and expanded
  outputs;
- cached FP8 return-shape and representation transitions between BF16 and FP8;
  and
- SF stream recording, retention, event drop, completion mismatch, pending
  destruction, predecessor ordering, and repeated generations.

The production matrix uses public PyTorch NPU APIs and an independent
all-gather reference. It covers normal, expanded, padded, cached, weighted,
empty, asymmetric, and `-1` routing, exact payload/SF comparison, BF16 combine
after FP8 dispatch, repeated generations, and bounded malformed input. The
focused matrix runs first on two NPUs; supported four-rank and eight-rank smoke
tests then verify rank-parameterized behavior. The serialized TaskQueue
acceptance sequence combines a clean production build, dependency audit, host
tests, BF16 regressions, and the complete selected FP8 runtime matrix.

Async acceptance promotes the 60 deferred FP8 functional benchmark rows only
after the public matrix passes at two ranks and focused smoke coverage passes
at four and eight ranks. The matrix covers FP32 and packed UE8M0x4 SF, row- and
column-major SF output, cached and non-cached normal and expanded dispatch,
communication-stream allocation, async completion, predecessor ordering,
empty and asymmetric routes, representation changes, and lifecycle failures.
BF16 functional rows must remain green.

### Qualification Record

Task `task_20260819_171408_412650329832` established a clean CANN 9.2.0
production build/import without CUDA, NCCL, or NVSHMEM dependencies. It passed
176 host tests with 2 skipped across 23 subtests, the 12-case two-rank FP8
matrix, 100 barrier generations, 14 BF16 dispatch cases, and 24 BF16 combine
cases before exposing a benchmark-only E4M3 indexing limitation. Commit
`fa30444` changed the benchmark verifier to index payload bytes.

Task `task_20260819_173647_53443028121` passed the repaired two-rank BF16/FP8
benchmark and exposed a four-rank BF16 reference-order error. Commit `72de60f`
made the reference quantize each BF16 contribution before FP32 accumulation
and added a host regression.

Final task `task_20260819_184826_108102819968` ran on devices 0-7 at commit
`72de60f` and exited 0. It passed 56 focused host tests plus 20 subtests, both
two-rank benchmark cases, and all 12 FP8 runtime cases at four and eight ranks.
Combined with the clean-build and broad two-rank evidence, this completes the
synchronous Phase 3F acceptance boundary.

Compile coverage or an enabled benchmark case alone is not completion, and the
qualification above does not claim performance, physical multi-host,
FP8-combine, or async/stream-overlap support.

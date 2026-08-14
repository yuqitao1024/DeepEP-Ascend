# EPv2 Ascend SIMT URMA Transport

## Status

This document defines Phase 2D of the Ascend 950 EPv2 port. It replaces the
Phase 2B empty device transport stubs with a minimal device-initiated SIMT URMA
implementation for the communication semantics already exercised by the Phase
2C barrier, dispatch, and combine kernels.

Phase 2D validates the transport primitives independently. It does not claim
that the Phase 2C placeholder communication schedules constitute complete
multi-rank EPv2 operators, and it does not enable production capabilities until
the corresponding end-to-end semantics have passed multi-NPU tests.

This design builds on:

- `epv2-ascend-transport-contract.md`;
- `epv2-ascend-simt-transport-stub.md`;
- `epv2-ascend-stub-core-operators.md`.

## Goal

Implement the smallest CANN 9.2.0, Ascend 950 SIMT communication layer needed
by the current core kernels while preserving the operator-facing
`DeviceTransportFacade` and all existing CUDA code.

The first functional transport supports:

- symmetric-window offset and remote-address resolution;
- device-initiated remote write;
- 64-bit inline remote value write;
- 64-bit remote fetch-add used for release publication and signals;
- channel drain and completion polling;
- local signal load and wait;
- a single-host team barrier.

## Scope

### In scope

- CANN 9.2.0 on `dav-3510`;
- single-host, multi-NPU communication;
- the public HCCL team, window, and channel resource APIs;
- UBC_CTP channels owned by the AIV communication engine;
- one device-visible symmetric window per transport instance;
- one SIMT producer per selected channel;
- device compilation and two-rank runtime validation on node 20002;
- structured host errors and device-side diagnostic status;
- explicit CANN version and ABI compatibility checks.

### Deferred

- cross-host scale-out and RoCE;
- direct peer pointer dereference;
- device `get`;
- asynchronous request creation and wait;
- hybrid device/CPU memory;
- pipeline and engram transport;
- multiple SIMT producers sharing one submission queue;
- performance tuning and communication/computation overlap;
- production enablement of complete multi-rank barrier, dispatch, or combine.

Deferred facade methods remain present and compile, but continue to have stub
behavior and no corresponding capability bits.

## CANN Reference Findings

CANN 9.2.0 exposes host resource creation through:

- `HcclWorldTeamCreate`;
- `HcclTeamWindowRegister`;
- `HcclTeamChannelsCreate`;
- the matching deregistration and destruction functions.

The generated team and window handles refer to device-visible descriptors.
AIN resolves a peer channel from the team's contiguous channel table and
resolves a remote address from a peer entry in the registered window.

The AIN implementation maps the required semantics to Hcomm as follows:

| Required semantic | CANN reference operation |
| --- | --- |
| remote write | `WriteNbi` |
| inline value write | `WriteValueNbi` |
| remote add and signal | `AtomicFAA` |
| flush | `Drain` |
| barrier | FAA to every peer followed by local signal polling |

The V310 Hcomm URMA implementation constructs SQEs, writes them to a device
submission queue, rings a device doorbell, and polls CQEs. Its implementation
is `__aicore__`, uses UB and SIMD pipe operations, and cannot be called from a
SIMT VF.

CANN's public PTO URMA implementation provides a second useful reference. It
already owns minimal Hcomm-compatible POD descriptions and directly assembles
URMA work requests in GM. It remains AICore-only, but its structure confirms
that a small compatibility layer is sufficient and that the full AIN/Hcomm
class stack is unnecessary.

## Chosen Approach

DeepEP owns a minimal SIMT URMA implementation below the existing facade.
CANN continues to own connection establishment, memory registration, channel
allocation, authentication tokens, queue allocation, and teardown.

Production kernels do not include CANN internal Hcomm implementation headers.
The backend defines only the device-visible POD layouts and operations it
consumes. Host-side ABI probes compare those layouts with the CANN 9.2.0
definitions available in the installed package.

This approach is preferred over the alternatives:

1. Including Hcomm internal implementation headers does not solve the SIMT
   qualifier mismatch and creates an unstable production dependency.
2. A SIMT command queue serviced later by an AICore AIN phase would require a
   new execution protocol and substantial operator restructuring. It remains a
   separate fallback architecture, not an automatic silent fallback.

## Layering

```text
Phase 2C Ascend SIMT kernels
            |
DeviceTransportFacade
            |
Ascend SIMT transport policy
            |
address/channel resolver -- URMA WQE builder -- SQ/CQ operations
            |
CANN-created team, window, channel, SQ, CQ, and memory registration
```

Operator code remains independent of AIN, Hcomm, HCCL, and URMA types. CUDA
sources and NCCL Gin remain unchanged.

## Components

### CANN ABI compatibility definitions

A backend-local compatibility header contains the minimum device-visible
layouts required to read:

- team membership, self rank, channel base, per-member channel counts, and
  synchronization memory;
- window peer memory addresses and sizes;
- channel protocol, registered buffer tables, SQ/CQ contexts, and queue
  producer/consumer counters;
- UBC_CTP SQE, SGE, and CQE fields used by write, inline write, and FAA.

The definitions are versioned for CANN 9.2.0 and `dav-3510`. Tests enforce
structure sizes, alignments, and offsets against the CANN package. A mismatch
is a build or initialization failure, never a best-effort continuation.

The production device source includes the compatibility header but does not
include files from CANN's `asc/impl` tree.

### Host transport

The real host transport consumes an existing `HcclComm` passed as the
communicator handle. Torch NPU exposes the process-group communicator through
`ProcessGroupHCCL.get_hccl_comm`, allowing the existing Python transport-handle
flow to remain structurally similar to CUDA.

Initialization proceeds in this order:

1. Validate communicator rank and size against `TransportConfig`.
2. Create a world team with the synchronization memory required by channel zero
   signals and barrier session zero.
3. Register the symmetric device buffer as the team window.
4. Create at least one AIV UBC_CTP channel per non-local peer.
5. Export the device team handle, window handle, local window base, topology,
   and diagnostic buffer through `DeviceTransportContext`.
6. Keep production capability bits disabled until runtime validation passes.

Teardown reverses the resource order and is idempotent. Partial initialization
uses the same reverse-order cleanup path.

### SIMT primitive layer

The primitive layer is qualified for use from `__simt_vf__`. It owns:

- channel selection;
- local-pointer to window-offset conversion;
- peer-window address resolution;
- registered-buffer lookup;
- SQ slot reservation for a single producer;
- WQE and SGE construction;
- queue memory publication and doorbell writes;
- CQE polling, validation, and consumer update;
- timeout and diagnostic reporting.

It does not own connection setup, queue allocation, or memory registration.

## Device Context

`DeviceTransportContext` remains the facade-level ABI. Its existing opaque
fields are populated as follows for the real transport:

| Field | Meaning |
| --- | --- |
| `local_window_base` | local symmetric-window base address |
| `peer_address_table` | device-visible CANN window handle |
| `channel_table` | device-visible CANN team handle |
| `backend_context` | backend-owned device diagnostic and version context |

The backend context contains its own ABI version, expected CANN compatibility
version, an error record address, and any transport state not represented by
the CANN team or window.

## Address and Channel Resolution

Facade operands remain logical symmetric addresses. For a remote operation:

1. Validate that the local operand lies inside the registered symmetric
   window.
2. Compute `offset = operand - local_window_base`.
3. Read the peer memory base from the window descriptor.
4. Form `remote_address = peer_base + offset` and validate the byte range.
5. Resolve the channel index by summing per-member channel counts before the
   peer and adding the facade channel index.

The resolved remote address is valid as an URMA target. It is not advertised as
a directly dereferenceable peer pointer, so `kDirectPeerPointer` remains off.

## URMA Submission

The initial implementation supports UBC_CTP only. A submission performs:

1. Resolve and validate the remote registered-buffer entry.
2. Read SQ head, CQ head, queue depth, WQE size, authentication tokens, TP id,
   and remote EID from the channel.
3. Poll completions before the queue approaches exhaustion.
4. Populate the required SQE and optional SGE in the selected SQ slot.
5. Publish the complete request with the validated SIMT device fence/cache
   sequence.
6. Advance the channel SQ/CQ producer counters.
7. Ring the SQ device doorbell with the validated device-store primitive.

Write and inline write consume one WQE base block. FAA consumes two base blocks
and supplies a registered local fetch-result address from the team shadow
memory.

The initial configuration follows CANN's strong-order, fenced, completion-
generating behavior. Request aggregation and delayed doorbell commit remain
disabled even when the facade receives the aggregate option.

## Required Operations

### `put`

`put` posts an URMA write from a local registered source to the resolved peer
window address. Phase 2D accepts participant scope, device memory, default
options, and no remote action. Unsupported variants remain non-production and
must not be covered by enabled capabilities.

### `put_value`

`put_value` posts an inline 64-bit URMA write. The current Phase 2C call site
uses exactly eight bytes. Other widths remain stubbed and do not contribute to
capability enablement.

### `remote_add_release`

`remote_add_release` posts a 64-bit FAA. A SIMT system fence publishes earlier
ordinary writes before the FAA is submitted. The FAA completion is generated
and may be drained by `flush`.

### `signal`

Signal index zero uses the team's synchronization memory and a 64-bit FAA.
Signal increment adds one; signal add uses the requested value. Signal ranges
are checked against the synchronization-memory requirement created by the host.

### `flush`

`flush` polls every non-local peer channel selected by the facade channel index
until its consumed CQ count reaches the submitted CQ count. It validates owner,
status, and substatus, updates the CQ consumer, and rings the CQ doorbell.

The initial workgroup and device scopes use the same single-producer channel
drain because only thread zero issues communication in the Phase 2C kernels.

### Signal load and wait

Signal reads use a device load that observes remote FAA updates. Wait loops poll
the local signal address, enforce an optional retry bound, and apply an acquire
fence after the target is observed.

### Device barrier

Barrier session zero follows the AIN algorithm:

1. Apply a release fence.
2. FAA the calling rank's barrier slot on every peer.
3. Wait for the expected generation from every peer in local synchronization
   memory.
4. Apply an acquire fence before returning.

The local shadow memory tracks each peer's expected generation so repeated
barriers do not require clearing the remote counters.

## SIMT Hardware-Primitive Gate

Correct URMA submission requires more than ordinary GM loads and stores. Before
the implementation is accepted, a minimal `dav-3510` probe must prove that a
SIMT VF can perform all of the following:

- publish a fully written SQE before the doorbell;
- issue the device doorbell store accepted by the URMA queue;
- observe changing CQE owner and status fields without stale cache data;
- update the CQ consumer doorbell;
- provide release and acquire ordering around payload and signal operations.

The probe first tests the compiler-supported `ld_dev`/`st_dev` path and SIMT
fences. Runtime tests then validate actual queue progress and ordering. A plain
cached GM assignment is not an acceptable substitute for a device doorbell or
CQ cache invalidation.

If these primitives cannot be expressed from SIMT with the installed compiler,
direct SIMT URMA is considered blocked. The code remains capability-gated and
the separate AICore communication-service architecture must receive a new
design before use.

## Concurrency Model

Phase 2D matches the existing Phase 2C launch contract:

- one block per core operator kernel;
- only `threadIdx.x == 0` performs communication;
- facade channel zero is the sole producer channel.

Under this model, the channel's SQ and CQ counters have one producer and one
consumer. The implementation does not use an atomic SQ reservation algorithm.

Multiple blocks or SIMT threads may not share a channel. A later phase must
either allocate independent channels or add a proven multi-producer queue
protocol before relaxing this restriction.

## Ordering and Completion

The required publication sequence is:

```text
payload stores
  -> SIMT release fence
  -> URMA write or FAA submission
  -> SQ doorbell
  -> remote visibility
  -> CQ completion
  -> flush / signal observation
  -> SIMT acquire fence
  -> payload loads
```

CQ completion proves completion of the corresponding strongly ordered request.
Signal observation is only used as publication when the sender issued a release
fence and the receiver applies an acquire fence. Tests must verify the payload,
not merely the signal value.

## Error Handling

Host API failures retain `TransportStatus` with operation name, CANN/HCCL code,
rank, and a concise diagnostic.

Device failures record the first error in backend-owned GM diagnostics. The
record includes an operation code, peer, channel, queue indices, CQ status, and
timeout indication. Device functions preserve the facade signature; internal
runtime probes inspect the record after synchronization.

Invalid ABI, protocol, rank, channel, address range, queue layout, or value
width fails before a doorbell is written. Timeouts never cause a capability to
be enabled.

## Capability Policy

Capability bits describe validated end-to-end behavior. Adding code or passing
a compile test is insufficient.

The implementation initially exports zero production capabilities. Bits may be
enabled individually only after their two-rank semantic tests pass:

- `kSymmetricWindow`;
- `kDevicePut`;
- `kDevicePutValue` for the supported 64-bit contract;
- `kRemoteAtomicAddRelease`;
- `kRemoteSignal`;
- `kSystemMemoryOrdering`;
- `kDeviceBarrier`;
- `kScaleUpTeam`.

`kDeviceGet`, `kDirectPeerPointer`, `kAsyncCompletion`, and `kScaleOutTeam`
remain disabled in this phase.

Enabling the public barrier, dispatch, or combine APIs additionally requires
their complete operator-level multi-rank tests and is not automatic when the
primitive capability tests pass.

## Build Boundaries

Ascend host code links the CANN ACL and HCCL libraries required by the public
resource APIs. Ascend device objects compile for `dav-3510` using the existing
mixed AICore/SIMT mode.

Backend selection remains compile-time:

```text
DEEP_EP_PLATFORM_CUDA   -> existing CUDA and NCCL Gin implementation
DEEP_EP_PLATFORM_ASCEND -> Ascend host transport and SIMT URMA implementation
```

No CUDA source, link option, or runtime behavior changes in Phase 2D.

## Testing

### Source and host tests

- backend-neutral facade ABI remains unchanged;
- production device code has no `asc/impl` include;
- compatibility layouts match CANN 9.2.0 sizes and offsets;
- communicator, team, window, and channel validation reports structured errors;
- partial initialization and repeated destruction release resources once;
- deferred capability bits remain disabled.

### SIMT compile probes

- all required device functions compile from `__simt_vf__` for `dav-3510`;
- WQE construction compiles without calling `__aicore__` functions;
- the selected device load, store, fence, and atomic primitives compile without
  qualifier-loss warnings;
- unsupported facade methods continue to compile as stubs.

### Single-NPU structural probes

- address and channel resolution use the expected descriptors;
- invalid ranges, protocols, widths, and channel indices set diagnostics;
- WQE fields, owner transition, block count, and queue wrap arithmetic match
  CANN's UBC_CTP reference behavior.

### Two-NPU semantic probes on node 20002

- put transfers payload bytes in both directions;
- 64-bit inline put transfers the exact value;
- FAA returns completion and updates the remote value exactly once;
- standalone signal add and increment reach the expected generation;
- flush waits for all submitted work on the selected channel;
- payload-before-signal release/acquire ordering holds under repetition;
- barrier completes repeatedly and orders data across both ranks;
- queue wraparound does not overwrite unconsumed WQEs;
- invalid CQ status and bounded waits surface diagnostics rather than hanging;
- resource teardown succeeds after both normal and failed runs.

GPU runtime validation remains deferred by project decision.

## Acceptance Criteria

Phase 2D's minimal transport layer is complete when:

1. host resources are created and destroyed exclusively through supported CANN
   host APIs;
2. the backend-owned compatibility layouts pass CANN 9.2.0 ABI checks;
3. the required SIMT hardware primitives are proven by compile and runtime
   probes;
4. put, 64-bit put-value, 64-bit FAA, signal, flush, and barrier pass the
   two-NPU semantic tests on node 20002;
5. ordering and repeated queue progress are validated, not inferred;
6. deferred methods and capabilities remain disabled;
7. existing CUDA production sources remain unchanged;
8. public EPv2 operators remain gated until their own complete multi-rank
   correctness tests pass.

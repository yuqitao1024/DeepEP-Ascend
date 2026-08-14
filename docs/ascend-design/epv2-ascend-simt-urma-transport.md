# EPv2 Ascend SIMT URMA Transport

## Status

This document defines Phase 2D of the Ascend 950 EPv2 port. It replaces the
Phase 2B empty device transport stubs with a minimal SIMT-fronted URMA
implementation for the communication semantics already exercised by the Phase
2C barrier, dispatch, and combine kernels. SIMT records transport commands;
an AICore service phase submits those commands through the required device
doorbell instructions.

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
- one SIMT command producer and one AICore queue producer per selected channel;
- device compilation on node 20002 and two-rank runtime validation on a
  single host exposing at least two Ascend 950 devices;
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

### CANN 9.2 SIMT feasibility result

The `dav-3510` mixed-language probe on node 20002 established these compiler
boundaries:

- `asc_threadfence`, non-cacheable SIMT `__ldg`, and non-cacheable SIMT
  `__stg` compile from `__simt_vf__`;
- the public `ld_dev` and `st_dev` aliases are not declared in the SIMT API;
- calling `__builtin_cce_st_dev` directly from `__simt_vf__` consistently
  crashes Bisheng in the `HiTPE Stall Cycle Refactor` code-generation pass;
- CANN's Hcomm, PTO URMA, and MoE communication implementations all use
  `st_dev` for SQ and CQ doorbells; none uses ordinary GM stores for them.

`__stg` and `st_dev` are distinct compiler builtins with different device
semantics. A compiling non-cacheable `__stg` is therefore not accepted as a
doorbell substitute. Direct SIMT URMA is blocked with CANN 9.2.0.

## Chosen Approach

DeepEP owns a minimal staged transport below the existing facade. SIMT-facing
facade calls append fixed-size commands to a backend-owned GM command buffer.
After the VF returns, the enclosing AICore kernel invokes a service routine
that validates and executes the batch with `st_dev`, `ld_dev`, and the CANN
9.2 URMA layouts. CANN continues to own connection establishment, memory
registration, channel allocation, authentication tokens, queue allocation,
and teardown.

Production kernels do not include CANN internal Hcomm implementation headers.
The backend defines only the device-visible POD layouts and operations it
consumes. Host-side ABI probes compare those layouts with the CANN 9.2.0
definitions available in the installed package.

This approach is preferred over the alternatives:

1. Including Hcomm internal implementation headers does not solve the SIMT
   qualifier mismatch and creates an unstable production dependency.
2. Treating non-cacheable `__stg` as a doorbell is not supported by CANN's
   reference code and would risk silent queue failure.
3. A persistent concurrent AICore service could preserve synchronous facade
   calls, but adds kernel residency, scheduling, shutdown, and deadlock risks
   that the current phase does not need.
4. Waiting for a vendor SIMT communication patch remains a future replacement
   path. The command ABI isolates that change without blocking current work.

The cost of the staged design is explicit: facade calls enqueue work, and the
work becomes complete only at the next AICore service boundary. A VF must not
consume communication results after `flush` or `device_barrier` in the same VF.
Operator integration splits such consumers into a later VF phase. Current
Phase 2C communication calls are at VF tails and remain production-gated, so
Phase 2D can validate the boundary without claiming complete operator support.

## Layering

```text
Phase 2C Ascend SIMT kernels
            |
DeviceTransportFacade
            |
SIMT command encoder -- backend-owned GM command buffer
            |
AICore service boundary
            |
address/channel resolver -- URMA WQE builder -- SQ/CQ operations -- st_dev
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
5. Allocate and zero the command buffer, service state, and diagnostic buffer.
6. Export the device team handle, window handle, local window base, topology,
   and backend service context through `DeviceTransportContext`.
7. Keep production capability bits disabled until runtime validation passes.

Teardown reverses the resource order and is idempotent. Partial initialization
uses the same reverse-order cleanup path.

FAA fetch-result SGEs use the team-owned shadow synchronization memory. An
ordinary backend `aclrtMalloc` allocation is not exported as a fetch-result
slot because it is absent from the channel's registered local-buffer table and
therefore has no valid local token.

### SIMT command layer

The command layer is qualified for use from `__simt_vf__`. It owns:

- facade argument capture into fixed-size, trivially-copyable commands;
- single-producer command reservation;
- non-cacheable command publication and a release fence;
- local signal reads that use supported SIMT load and fence primitives;
- command overflow and unsupported-operation diagnostics.

It does not access SQ/CQ doorbells and does not include CANN internal headers.

### AICore service layer

The service layer runs in the enclosing `__global__ __vector__` function after
an `asc_vf_call` returns. It owns:

- command-buffer reset and cache visibility at phase boundaries;
- channel and registered-buffer resolution;
- WQE/SGE construction and SQ slot reservation;
- SQ publication, `st_dev` doorbell writes, CQ polling, and CQ doorbells;
- signal and barrier expansion;
- bounded completion waits and first-error diagnostics.

The service consumes a complete command batch synchronously before the kernel
enters another VF or returns. It never runs concurrently with its SIMT producer.

## Device Context

`DeviceTransportContext` remains the facade-level ABI. Its existing opaque
fields are populated as follows for the real transport:

| Field | Meaning |
| --- | --- |
| `local_window_base` | local symmetric-window base address |
| `peer_address_table` | device-visible CANN window handle |
| `channel_table` | device-visible CANN team handle |
| `backend_context` | backend-owned command, service, diagnostic, and version context |

The backend context contains its own ABI version, expected CANN compatibility
version, command-buffer address and capacity, an error record address, and any
transport state not represented by the CANN team or window.

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

The initial implementation supports UBC_CTP only. At the service boundary, a
command submission performs:

1. Resolve and validate the remote registered-buffer entry.
2. Read SQ head, CQ head, queue depth, WQE size, authentication tokens, TP id,
   and remote EID from the channel.
3. Poll completions before the queue approaches exhaustion.
4. Populate the required SQE and optional SGE in the selected SQ slot.
5. Publish the complete request with the CANN AICore fence/cache sequence.
6. Advance the channel SQ/CQ producer counters.
7. Ring the SQ device doorbell with `st_dev`.

Write and inline write consume one WQE base block. FAA consumes two base blocks
and supplies a registered local fetch-result address from the team shadow
memory.

The initial configuration follows CANN's strong-order, fenced, completion-
generating behavior. Request aggregation and delayed doorbell commit remain
disabled even when the facade receives the aggregate option.

## Required Operations

### `put`

`put` records an URMA write command from a local registered source to the
resolved peer window address. The following AICore service boundary posts it.
Phase 2D accepts participant scope, device memory, default options, and no
remote action. Unsupported variants remain non-production and must not be
covered by enabled capabilities.

### `put_value`

`put_value` records an inline 64-bit URMA write. The current Phase 2C call site
uses exactly eight bytes. Other widths remain stubbed and do not contribute to
capability enablement.

### `remote_add_release`

`remote_add_release` records a 64-bit FAA after a SIMT release fence. The
AICore service submits the FAA in command order. The FAA completion is
generated and drained before the service boundary returns when requested by a
later flush command.

### `signal`

Signal index zero uses the team's synchronization memory and a 64-bit FAA.
Signal increment adds one; signal add uses the requested value. Signal ranges
are checked against the synchronization-memory requirement created by the host.

### `flush`

`flush` records a batch-ordering marker. The AICore service polls every
non-local peer channel selected by the facade channel index until its consumed
CQ count reaches the submitted CQ count. It validates owner, status, and
substatus, updates the CQ consumer, and rings the CQ doorbell with `st_dev`.

The initial workgroup and device scopes use the same single-producer channel
drain because only thread zero issues communication in the Phase 2C kernels.

### Signal load and wait

Signal reads and waits that occur in a continuation VF use non-cacheable SIMT
loads, a finite retry bound, and an acquire fence after the target is observed.
The continuation runs only after the AICore service has completed the preceding
batch.

### Device barrier

Barrier session zero is one service command and follows the AIN algorithm in
the AICore service:

1. Apply a release fence.
2. FAA the calling rank's barrier slot on every peer.
3. Wait for the expected generation from every peer in local synchronization
   memory.
4. Apply an acquire fence before returning.

The local shadow memory tracks each peer's expected generation so repeated
barriers do not require clearing the remote counters.

## Device Hardware-Primitive Gate

Correct URMA submission requires more than ordinary GM loads and stores. The
direct-SIMT probe has proven that CANN 9.2 cannot issue the required doorbell
from a VF. The replacement mixed-phase probe must prove all of the following:

- a SIMT VF publishes a complete command before returning;
- the AICore phase observes the command after cache maintenance;
- AICore publishes a fully written SQE before an `st_dev` doorbell;
- AICore observes changing CQE owner and status fields and updates the CQ
  consumer doorbell;
- a continuation VF observes service results with acquire ordering.

Runtime tests then validate actual queue progress and ordering. No GM store,
cached or non-cacheable, substitutes for an SQ or CQ device doorbell.

## Concurrency Model

Phase 2D initially uses this launch contract:

- one block per core operator kernel;
- only `threadIdx.x == 0` records communication commands;
- facade channel zero is the sole command and URMA queue producer channel;
- SIMT command production and AICore service execution are sequential phases.

Under this model, the channel's SQ and CQ counters have one producer and one
consumer. The implementation does not use an atomic SQ reservation algorithm.

Multiple blocks or SIMT threads may not share a channel. A later phase must
either allocate independent channels or add a proven multi-producer queue
protocol before relaxing this restriction.

## Ordering and Completion

The required publication sequence is:

```text
payload stores / command fields
  -> SIMT release fence and command publication
  -> VF return
  -> AICore cache observation
  -> URMA write or FAA submission
  -> st_dev SQ doorbell
  -> remote visibility
  -> CQ completion
  -> flush / signal observation
  -> optional continuation VF acquire fence
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
mixed AICore/SIMT mode. Each production kernel that uses real transport must
call the service boundary after its command-producing VF and before any
consuming VF.

The two-rank runner is an ASC shared library loaded into each `torchrun`
process. It is not a child executable: `ProcessGroupHCCL.get_hccl_comm`
returns a process-local communicator pointer, so the runner must execute in the
same process and NPU context. The CANN 9.2 link set is `hcomm`, `ascendcl`, and
`c_sec`; `c_sec` is explicit because the public team descriptor initializers
inline calls to `memset_s` into the consumer object.

Backend selection remains compile-time:

```text
DEEP_EP_PLATFORM_CUDA   -> existing CUDA and NCCL Gin implementation
DEEP_EP_PLATFORM_ASCEND -> Ascend host transport and staged AICore URMA service
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

### Mixed-phase compile probes

- all facade command functions compile from `__simt_vf__` for `dav-3510`;
- the command-producing VF returns before the AICore service is called;
- WQE construction and `ld_dev`/`st_dev` compile only in the AICore service;
- the selected command publication, cache, fence, and doorbell primitives
  compile without qualifier-loss warnings;
- unsupported facade methods continue to compile as stubs.

### Single-NPU structural probes

- address and channel resolution use the expected descriptors;
- invalid ranges, protocols, widths, and channel indices set diagnostics;
- WQE fields, owner transition, block count, and queue wrap arithmetic match
  CANN's UBC_CTP reference behavior.

### Two-NPU semantic probes

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

### Validation record

Validation performed on 2026-08-14 against CANN 9.2.0 and `dav-3510` on
`root@121.41.199.170:20002` established the following:

- all CANN compatibility, WQE, command, service, and host lifecycle probes
  passed;
- the primitive, command encoder, AICore service, and runtime shared-library
  targets compiled with Bisheng for `dav-3510`;
- the runtime shared library linked using public ACL/HCCL interfaces only;
- a device runtime smoke completed the SIMT producer, AICore observation, and
  SIMT consumer phases on `Ascend950PR_9589` with diagnostic `kNone`;
- host lifecycle failure injection covered every team, window, channel,
  allocation, zero, and copy boundary, including idempotent reverse cleanup.

The 20002 execution container currently exposes only `/dev/davinci0`, and
`torch.npu.device_count()` returns one. A two-process launch therefore failed
at `torch.npu.set_device(1)` with CANN error `107001` before communicator or
transport initialization. Port 20001 is a different single-NPU host; using it
would turn this phase into an unplanned scale-out test rather than the required
single-host UBC_CTP test.

Consequently, no two-NPU put, put-value, FAA, signal, flush, ordering, barrier,
or queue-wrap result is recorded yet. The runtime harness for those cases is
present, but production capabilities remain exactly zero and Phase 2D does not
meet its completion criteria until the harness passes on a single host exposing
at least two Ascend 950 devices.

## Acceptance Criteria

Phase 2D's minimal transport layer is complete when:

1. host resources are created and destroyed exclusively through supported CANN
   host APIs;
2. the backend-owned compatibility layouts pass CANN 9.2.0 ABI checks;
3. the mixed SIMT-command/AICore-service boundary and its required hardware
   primitives are proven by compile and runtime probes;
4. put, 64-bit put-value, 64-bit FAA, signal, flush, and barrier pass the
   two-NPU semantic tests on a single host exposing at least two Ascend 950
   devices;
5. ordering and repeated queue progress are validated, not inferred;
6. deferred methods and capabilities remain disabled;
7. existing CUDA production sources remain unchanged;
8. public EPv2 operators remain gated until their own complete multi-rank
   correctness tests pass.

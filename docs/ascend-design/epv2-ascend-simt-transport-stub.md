# EPv2 Ascend SIMT Transport Facade and Stub

## Status

This document defines Phase 2B of the Ascend 950 EPv2 port and records the
following Phase 2C and Phase 2D direction. It builds on
`epv2-ascend-transport-contract.md` and defines the communication interface
needed before operator development and a real SIMT implementation can proceed.

Phase 2B is an interface and compile-probe phase. It does not port the core
operators or make multi-rank Ascend EPv2 communication functional.

## Context

EPv2 uses NCCL Gin as a device-side communication layer. Its kernels issue
one-sided transfers, publish counters and signals, wait for completion, and
select direct scale-up addressing when available. The Ascend port needs the
same semantic surface, but CANN 9.2.0 does not currently provide a public
SIMT-qualified equivalent of that complete interface.

CANN 9.2.0 does provide a public AICORE communication API under:

```text
include/comm_api/aicore/ain/ain.h
include/comm_api/aicore/ain/ain_common.h
```

`AscendC::Ain` is close to Gin at the semantic level, but its functions are
qualified with `__aicore__`, not the SIMT qualifiers used by Ascend 950 SIMT
code. Its public memory-order API also exposes only relaxed signal ordering.
Those gaps are known and are deliberately deferred to Phase 2D.

Waiting for the final communication implementation would unnecessarily block
the later operator port. Phase 2B therefore introduces a stable backend-neutral
SIMT transport facade, provides empty device stubs for it, and validates the
surface with minimal synthetic consumers. Phase 2C can then port operator code
against that facade while the public Python operations remain guarded by the
zero-capability host transport.

## Decisions

1. Phase 2B targets the communication abstraction required by later
   single-host, multi-NPU scale-up operator development.
2. The operator-facing transport facade follows the functional decomposition
   and semantics of upstream `NCCLGin`: addressing, one-sided transfer, remote
   action, completion, ordering, team selection, cooperation scope, request,
   segment, and option handling remain separate concepts. Ascend function and
   type names do not need to match NCCL names.
3. Backend-neutral public types remain owned by
   `csrc/backends/ascend/transport`; NCCL and CANN types do not enter common
   EPv2 interfaces.
4. Every communication operation needed by current and experimental EPv2
   kernels receives a SIMT-callable stub seam in Phase 2B.
5. Stub operations have no communication side effects and advertise no
   capabilities. They exist to compile and type-check consumers, not to emulate
   a working transport.
6. Production `barrier`, `dispatch`, and `combine` continue to fail at the
   host capability gate before allocation or kernel launch.
7. A real AIN/Hcomm binding, a communication-team patch, or a custom SIMT
   implementation replaces only the lowest device transport layer in Phase 2D.
8. Scale-out transport, HCCL resource initialization, performance tuning, and
   GPU runtime verification are outside Phase 2B.

## Goals

Phase 2B will:

- define a backend-neutral Ascend device facade for all EPv2 communication
  call sites, with a one-to-one semantic mapping to the upstream Gin surface;
- define empty SIMT VF stubs for every required device communication primitive;
- keep team, cooperation scope, request, option, and remote-action concepts
  explicit at every call site;
- compile minimal synthetic consumers that cover every required semantic
  operation without depending on full EPv2 operator kernels;
- keep the real communication replacement boundary small and testable;
- preserve the current Python API and compile-time GPU/Ascend backend split;
- keep CUDA production files unchanged;
- retain exact host-side unsupported diagnostics until real capabilities are
  validated end to end.

## Non-goals

Phase 2B will not:

- port complete Ascend barrier, dispatch, or combine operators;
- implement operator tiling, workspace layouts, local data paths, or host
  launch plumbing;
- report successful multi-rank Ascend dispatch, combine, or barrier behavior;
- include or call `comm_api/aicore/ain` from the SIMT stub implementation;
- include CANN internal implementation headers;
- call Hcomm directly from operator code;
- initialize HCCL teams, channels, symmetric windows, or device contexts;
- claim release/acquire or system memory ordering from relaxed AIN signals;
- implement a direct peer pointer by reaching into AIN or Hcomm internal data
  structures;
- use `HcclAlltoAllV` or another collective fallback;
- implement or validate cross-host scale-out communication;
- change upstream CUDA Gin wrappers or CUDA kernel behavior;
- treat a compile-only stub as communication correctness evidence.

## Deployment Domains

The contract retains three logical teams:

| Team | Meaning in EPv2 | Phase 2B status |
| --- | --- | --- |
| `world` | All EP ranks | Represented by the facade, not functional |
| `scale_up` | Ranks in one local high-bandwidth domain | Represented and compile-probed, not functional |
| `scale_out` | Corresponding ranks across hosts or scale-up domains | Represented only, deferred |

Single-host communication is not assumed to imply uniform direct peer access.
The eventual transport may choose direct scale-up addressing for some peers and
AIN/Hcomm channels for others. That routing decision belongs below the facade.

## CANN AIN API Inventory

The installed CANN 9.2.0 public AIN interface provides the following device
operations.

### Resources and addressing

- `HcommWindowHandle`: opaque symmetric-window handle;
- `HcommTeamHandle`: opaque team handle;
- window-relative offsets for local and remote operands;
- a context index used to select a per-peer channel;
- `AinDescriptorUbuf`, which supplies UB workspace, byte size, and an event id.

AIN does not expose a public device function that returns a directly
dereferenceable peer pointer. It resolves window handles and offsets internally
before submitting Hcomm requests.

### One-sided operations

- `Ain::Put`;
- `Ain::PutValue`;
- `Ain::Get`;
- immediate and delayed commit through `AinCommitFlags`;
- optional signal-add or signal-increment remote actions on put operations.

### Synchronization and completion

- `Ain::Signal`;
- `Ain::ReadSignal`;
- `Ain::WaitSignal`;
- `Ain::Flush`;
- `Ain::FlushAsync`;
- `Ain::Wait`;
- `AinBarrierSession::Sync`, including a timeout overload.

`FlushAsync` returns a channel handle rather than an operation-specific event.
Waiting on that handle drains outstanding work on the channel. The Phase 2B
request abstraction must permit that coarser completion model.

### AIN constraints relevant to EPv2

- `AIN_DEVICE` expands to an always-inline `__aicore__` qualifier. It is not a
  public SIMT device-function qualifier.
- Public AIN examples using it from an Ascend 950 SIMT VF were not found in the
  installed CANN tree.
- `AinMemoryOrder` currently contains only `AIN_MEMORY_ORDER_RELAX`.
- The public header states that relaxed signal access does not order ordinary
  data accesses before or after the signal.
- `Put`, `Get`, and `Signal` require valid UB workspace. The underlying Hcomm
  URMA implementation currently needs a small per-caller workspace, observed
  as 512 bytes in CANN 9.2.0 implementation constants.
- Host-side team, window, channel, and device-visible context construction is
  provided separately by HCCL resource APIs, not by `comm_api/aicore/ain`.
- Device methods return `void` at the AIN layer and its current implementation
  does not propagate every Hcomm submission or drain failure to the caller.

## Capability Assessment

Capabilities describe validated end-to-end semantics, not the presence of an
API name.

| Transport capability | CANN API evidence | Phase 2B advertisement |
| --- | --- | --- |
| `symmetric_window` | AIN window handles and offsets; HCCL host registration exists | Disabled |
| `direct_peer_pointer` | No public AIN device pointer lookup | Disabled |
| `device_put` | `Ain::Put` | Disabled |
| `device_get` | `Ain::Get` | Disabled |
| `device_put_value` | `Ain::PutValue` | Disabled |
| `remote_atomic_add_release` | `Ain::Signal` uses atomic fetch-add, but release ordering is not promised | Disabled |
| `remote_signal` | `Signal`, `ReadSignal`, and `WaitSignal` | Disabled |
| `async_completion` | `FlushAsync` and channel-level `Wait` | Disabled |
| `system_memory_ordering` | Only relaxed signal ordering is public | Disabled |
| `device_barrier` | `AinBarrierSession`, but only relaxed ordering is public | Disabled |
| `scale_up_team` | AIN consumes an Hcomm team; host provisioning is separate | Disabled |
| `scale_out_team` | The shape may be reusable, but it is not in Phase 2B | Disabled |

All bits remain disabled because Phase 2B does not bind or validate AIN. The
table records future implementation candidates only.

## Upstream Gin Semantic Compatibility

Upstream EPv2 kernels consume `deep_ep::elastic::handle::NCCLGin`, not raw NCCL
Gin at every call site. The Ascend facade preserves the same semantic
boundaries, but uses backend-neutral names and types. Source-level function-name
compatibility is not a goal.

### Required semantic operations

The Ascend facade keeps distinct operations for:

```text
peer accessibility query
symmetric pointer-to-offset conversion
direct peer pointer lookup
one-sided remote read
one-sided remote write with an optional remote action
inline remote value write
remote atomic add with release publication
standalone remote signal
channel or scope flush
asynchronous peer completion request creation
completion request wait
```

The Phase 2A declarations already use names such as
`is_peer_directly_accessible`, `get_symmetric_offset`,
`get_symmetric_pointer`, `put`, `get`, `put_value`,
`remote_add_release`, `signal`, `flush`, `flush_async`, and `wait`. Phase 2B
may refine those names or group them into a facade class, but it must not merge
semantic units that upstream schedules or orders independently.

Kernel ports should remain structurally comparable: each upstream Gin
operation maps to one Ascend facade operation with the same source/destination
roles, peer selection, completion point, cooperation scope, and ordering
intent. This is a semantic mapping, not a textual API copy.

### Type mapping

| Upstream Gin concept | Ascend facade concept |
| --- | --- |
| `ncclTeamTagWorld` | `WorldTeam` |
| `ncclTeamTagLsa` | `ScaleUpTeam` |
| `ncclTeamTagRail` | `ScaleOutTeam` |
| `ncclCoopThread` | participant cooperation |
| `ncclCoopWarp` | workgroup cooperation |
| GPU-wide sharing | device cooperation |
| `ncclGinRequest_t` | `DeviceRequest` |
| `ncclGin_None` | no remote action |
| `ncclGin_VASignalAdd` | remote signal add |
| default options | `kDefaultOptions` |
| `ncclGinOptFlagsAggregateRequests` | aggregate-request option |
| `ncclGin_SegmentDevice` | device-memory segment |
| `ncclGin_SegmentMixed` | mixed or externally registered segment |

The mapping is compile-time and backend-local. CUDA types are not aliased or
included in Ascend code.

### Pointer model

Upstream methods accept symmetric pointers and derive offsets internally. The
Ascend facade preserves that responsibility boundary: operator code supplies a
logical symmetric operand, while the transport layer owns pointer-to-offset
conversion and backend window addressing. The concrete Ascend argument form
may be a pointer, offset, or small POD descriptor; it need not reproduce the
NCCL signature.

A distinct direct-peer lookup operation remains part of the facade because
upstream dispatch and combine use that semantic result to choose a direct
scale-up store path. Its Ascend name need not be `get_sym_ptr`. The Phase 2B
stub always reports that a non-local peer is not directly accessible and
returns no remote pointer. The local-rank identity case may return the original
pointer because it is not communication. A future implementation must use a
public and validated peer-address mechanism before enabling
`direct_peer_pointer`.

## Layering

```text
Phase 2B synthetic consumers / Phase 2C EPv2 kernels
                         |
      backend-neutral Ascend transport facade
                         |
        backend-neutral Phase 2A device contract
                         |
          SIMT VF stub implementation (Phase 2B)
                         |
      real SIMT communication implementation (Phase 2D)
```

### Operator-facing transport facade

The facade is the only communication API included by Ascend EPv2 kernels. It
owns team-tag mapping, cooperation scope, segment and option mapping,
symmetric-offset conversion, remote-action representation, and
`DeviceRequest` adaptation.

Operator sources do not include AIN, Hcomm, HCCL, URMA, or internal CANN
headers. They do not inspect backend context fields or channel tables.

### Backend-neutral device contract

The existing Phase 2A declarations remain the semantic contract. Phase 2B may
add option bits, remote-action PODs, and stub-observation data only when needed
by an actual upstream call site. Any additions must remain trivially copyable
and vendor independent.

### SIMT VF stub implementation

Every device declaration consumed by the facade receives an Ascend SIMT build
definition. The exact compiler qualifier spelling is isolated behind the
Ascend device-build boundary so host contract probes do not depend on CANN.

The stubs:

- compile in the intended Ascend 950 SIMT kernel mode;
- preserve parameter evaluation and type checking;
- perform no remote read, write, atomic, signal, wait, flush, or barrier;
- do not include AIN or Hcomm;
- do not allocate or initialize communication resources;
- do not set transport capability bits;
- never become reachable from a successful public multi-rank operation.

The facade and stub must not encode assumptions about whether the final
implementation calls AIN directly, uses an AIV communication service, or is
provided by an external SIMT communication patch.

## Stub Safety Model

An empty device function can look successful if it is launched accidentally.
Phase 2B prevents that in two independent layers.

1. `StubHostTransport::capabilities()` remains empty.
2. `ElasticBuffer` checks the complete operation capability mask before output
   allocation or kernel launch.

The existing explicit terminal unsupported failures remain in place even if a
test transport is later introduced. Enabling a production kernel requires a
separate reviewed change that both advertises validated capabilities and
removes the operation's Phase 2A/2B terminal gate.

Internal kernel compile tests may instantiate the facade and stubs directly.
Such tests are not public EPv2 execution and must not assert multi-rank data
correctness.

## Minimal Semantic Consumer Probes

Phase 2B uses small synthetic SIMT consumers to instantiate and type-check the
facade without introducing full operator structure. Together the probes cover:

- direct peer accessibility, direct pointer lookup, and the non-direct
  fallback path;
- one-sided put, get, and inline put-value operations;
- a put with an optional remote action;
- standalone signal, signal read, and signal wait operations;
- remote atomic add with release-publication intent;
- synchronous flush, asynchronous flush request creation, and request wait;
- the device barrier seam;
- all required team, cooperation-scope, segment, option, request, and
  remote-action variants.

The probes may verify overload selection, parameter flow, branch structure,
and compilation for the intended device mode. They must remain independent of
barrier, dispatch, and combine tiling, data paths, or launch plumbing.

The following cannot be validated against an empty stub:

- remote payload arrival;
- remote counter or signal visibility;
- release/acquire ordering;
- concurrent channel progress;
- peer fairness or timeout behavior;
- multi-rank output correctness;
- communication/computation overlap;
- throughput or latency.

Tests must label those properties as deferred rather than inferring them from a
successful compile or single-rank launch.

## Compile-Time Backend Boundary

The GPU implementation remains unchanged. Backend selection continues to use
the repository's platform macros and source ownership boundaries.

```text
DEEP_EP_PLATFORM_CUDA   -> existing NCCLGin and CUDA kernels
DEEP_EP_PLATFORM_ASCEND -> backend-neutral Ascend facade and SIMT VF stubs
```

No translation unit includes both NCCL Gin and Ascend communication device
headers. Shared Python names, method signatures, and return shapes remain
unchanged.

## Error Handling

Host transport errors keep the Phase 2A structured status format, including
operation, rank, status category, backend code when present, and diagnostic.

Phase 2B device stubs do not manufacture backend runtime error codes because no
backend is called. Public operations fail on the host before the stub can run.

Compile-only and internal launch probes must fail their test setup if they are
mistakenly used with a transport that advertises a real capability. This keeps
stub-only binaries from being accepted as production communication builds.

## Testing and Acceptance

Phase 2B acceptance requires all of the following.

### Source and ABI tests

- the facade exposes the exact approved method and type surface;
- backend-neutral PODs remain trivially copyable and retain their ABI checks;
- the stub covers every device declaration with no unresolved symbols;
- Phase 2B synthetic consumers include only the facade, not AIN/Hcomm directly;
- CUDA production sources remain unchanged;
- forbidden vendor includes remain absent from the backend-neutral contract.

### Host behavior tests

- normal and hybrid `barrier`, `dispatch`, and `combine` retain their exact
  unsupported capability diagnostics;
- no output allocation or kernel launch occurs before the gate;
- the stub transport advertises zero capabilities;
- destruction remains idempotent.

### Ascend 950 compile tests on node 20002

- every facade method and every SIMT VF stub compiles for `dav-3510`;
- minimal synthetic consumers instantiate every required semantic operation
  in the intended SIMT compilation mode;
- no direct AIN/Hcomm call is required for the stub build;
- the host extension and device objects contain no CUDA, NCCL, or NVSHMEM
  dependency;
- compiler warnings about invalid SIMT/AICORE calls are treated as failures.

### Semantic-probe tests

- probes cover direct accessibility and fallback, put/get/put-value, optional
  remote action, signal/read/wait, remote add release,
  flush/flush-async/wait, and the barrier seam;
- team, cooperation scope, segment, option, request, and remote-action variants
  are instantiated at least once;
- communication call shape may be checked structurally or with a test-only
  recording policy, without treating no-op stubs as data movement;
- no test claims multi-rank correctness until a real transport is installed.

GPU runtime validation remains deferred by project decision.

## Phase 2C: Stub-Based Core Operator Port

Phase 2C ports the single-host scale-up barrier, non-hybrid dispatch, and
non-hybrid combine paths against the Phase 2B facade and empty stubs. It adds
tiling, workspace management, local data paths, communication-call scheduling,
and host launch plumbing while production APIs remain capability-gated.

Phase 2C validates structure, compilation, and communication-independent local
behavior only. It makes no multi-rank correctness claim. Hybrid, pipeline, and
engram paths require interface coverage where needed, but their complete
operator implementations are outside Phase 2C.

## Phase 2D: Real SIMT Communication

Phase 2D replaces the empty device stubs with real communication and provisions
the required host teams, windows, channels, and device-visible context. The
implementation will either consume a supported communication-team SIMT patch
or build the required SIMT layer using public SIMD/AIN behavior as a reference.

This phase validates memory ordering, completion, timeouts, teardown,
single-host multi-NPU correctness, and performance before enabling any
production capability. Cross-host scale-out remains deferred unless a later
design explicitly brings it into scope.

## Phase 2D Implementation Options

The Phase 2B design deliberately does not choose between the two viable real
SIMT paths.

### Obtain a supported SIMT communication patch

The preferred path is to obtain a communication-team patch that provides
SIMT-qualified equivalents of the required AIN/Gin operations with documented
completion and memory-order semantics. The patch should be adapted below the
facade, leaving operator code unchanged.

This path minimizes dependence on CANN internals and gives the communication
owner responsibility for URMA channel, queue, cache, and ordering correctness.

### Implement a SIMT layer using public SIMD/AIN behavior as reference

If no supported patch is available, the project may implement the low-level
SIMT primitives using public interfaces and documented hardware behavior,
with the existing SIMD/AIN implementation used only as a behavioral reference.

This path requires dedicated validation for WQE submission, UB workspace,
channel ownership, completion queues, cache maintenance, remote atomic
ordering, signal visibility, timeouts, and concurrency. Internal CANN headers
must not become a production dependency without an explicit separate decision.

### Decision gate for Phase 2D

Before replacing the stubs, the chosen implementation must demonstrate:

- a supported SIMT calling convention;
- device put, get, put-value, signal, and channel completion;
- release publication and acquire observation for payload-plus-signal flows;
- a barrier that orders ordinary data as required by EPv2;
- host creation and teardown of teams, windows, channels, and device context;
- single-host multi-NPU correctness on Ascend 950;
- explicit behavior for direct peer addressing or a facade-preserving fallback;
- stable error reporting and timeout behavior.

Only capabilities proven by those tests may be enabled.

## Phase Boundary

Phase 2B ends when the backend-neutral facade is complete, every required empty
SIMT VF stub compiles for `dav-3510`, minimal synthetic consumers exercise the
full semantic surface, and all public multi-rank operations still fail safely
at the host gate.

Phase 2C then ports the three core scale-up operator paths against those stubs.
Phase 2D subsequently supplies real host resources and device communication
semantics without changing the operator-facing facade defined in Phase 2B.

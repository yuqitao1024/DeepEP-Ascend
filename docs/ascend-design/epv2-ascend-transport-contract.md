# EPv2 Ascend Device Transport Contract

## Status

This document defines Phase 2A of the EPv2 Ascend migration. The design has
been reviewed section by section. Phase 2A introduces a backend-neutral device
transport contract and an unsupported Ascend stub. It does not implement
device-initiated communication.

## Context

The CUDA EPv2 backend does not use a conventional collective as its primary
data plane. It registers symmetric memory, creates an NCCL device communicator
with Gin resources, selects communication contexts inside kernels, performs
one-sided remote operations, and coordinates completion and signaling without
returning to the host.

Ascend C support for the required SIMT-style, device-initiated communication
semantics has not yet been established for Ascend 950. Phase 2A therefore
separates the semantic contract required by EPv2 from the future CANN/HCCL/AIV
implementation. It must not assume that CUDA thread, warp, CTA, or grid
semantics have a direct Ascend equivalent.

## Goals

Phase 2A will:

1. define the minimum host and device transport capabilities required by the
   existing EPv2 kernels;
2. keep NCCL, Gin, CUDA, CANN, HCCL, and Ascend C types out of the common
   contract;
3. add an Ascend transport stub whose capabilities are discoverable;
4. fail clearly before kernel launch when a required capability is absent;
5. preserve the existing Python EPv2 API and compile-time backend boundary;
6. provide a concrete checklist for the later Ascend C communication phase.

## Non-goals

Phase 2A will not:

- implement Ascend C dispatch, combine, barrier, pipeline-parallel, or engram
  kernels;
- initialize ACL or HCCL;
- register real symmetric memory or acquire AIV channels;
- expose a private PyTorch HCCL communicator API;
- use `HcclAlltoAllV` as a fallback data plane;
- emulate successful communication with no-op operations;
- alter CUDA EPv2 source code or behavior;
- perform GPU runtime validation.

## Considered Approaches

### Backend-neutral Gin-equivalent contract

This is the selected approach. It models the semantics EPv2 consumes while
using backend-neutral names and opaque resources. CUDA execution terms are
represented by generic cooperation scopes. The Ascend implementation can later
map these semantics to the actual AIV/URMA programming model.

This approach costs an additional abstraction layer, but it keeps unavailable
Ascend mechanisms visible and avoids treating an HCCL collective as equivalent
to device-initiated communication.

### Copy the NCCL Gin API

A one-to-one copy would make the CUDA call-site mapping obvious, but would leak
NCCL teams, QPs, warp cooperation, CTA sharing, and Gin request layouts into the
Ascend backend. It was rejected because those concepts are not yet proven to
have direct Ascend C equivalents.

### Add only an empty Ascend communicator class

This would preserve the current stub with minimal code. It was rejected because
it would not identify the capabilities required by dispatch and combine, and
the interface would have to be redesigned when Ascend C communication work
begins.

## Architecture

The Ascend backend is divided into the following layers:

```text
Python EPv2 API
    |
Ascend ElasticBuffer
    |
Ascend runtime, stream, and event layer
    |
HostTransport contract
    |
DeviceTransportContext POD
    |
Device transport operation contract
    |
Unsupported transport stub in Phase 2A
    |
Future HCCL symmetric-window and AIV/URMA implementation
```

`ElasticBuffer` owns a `HostTransport`. Before an operation allocates output
state or launches a kernel, it asks the transport to verify the complete set of
required capabilities. Only a supported implementation may export a device
context and launch a communication kernel.

The transport contract is internal to the Ascend backend in Phase 2A. It does
not replace the CUDA Gin wrappers or create a cross-platform hot-path virtual
interface.

## Common Types

The contract uses C++17-compatible, vendor-independent types.

### Status

`TransportStatusCode` has four values:

- `success`: the operation completed;
- `unsupported_capability`: the backend cannot provide the requested semantic;
- `invalid_argument`: the caller supplied an invalid rank, size, handle, or
  configuration;
- `runtime_failure`: an available backend mechanism failed at runtime.

`TransportStatus` contains the code, operation name, backend error code when
available, and a diagnostic message. The stub never returns
`runtime_failure`, because it calls no vendor runtime.

### Teams

`TransportTeam` has the following values:

- `world`: all EP ranks;
- `scale_up`: ranks in the local high-bandwidth domain;
- `scale_out`: corresponding ranks across high-bandwidth domains.

The names describe logical EPv2 domains. They do not imply NVLink, NCCL LSA,
NCCL rail teams, HCCS, PCIe, or RoCE in the contract.

### Cooperation scope

`CooperationScope` has the following values:

- `participant`: one logical execution participant issues an operation;
- `workgroup`: the active workgroup cooperates on an operation;
- `device`: communication resources are shared across the device.

No contract API uses the terms thread, warp, CTA, SM, block, or grid. The later
Ascend C implementation must document how each supported scope maps to Ascend
execution and scheduling semantics.

### Capabilities

`TransportCapabilities` is a bit set containing:

- `symmetric_window`;
- `direct_peer_pointer`;
- `device_put`;
- `device_get`;
- `device_put_value`;
- `remote_atomic_add_release`;
- `remote_signal`;
- `async_completion`;
- `system_memory_ordering`;
- `device_barrier`;
- `scale_up_team`;
- `scale_out_team`.

Capabilities describe usable end-to-end semantics, not the presence of a header
or symbol. For example, `device_put` is false unless a launched Ascend C kernel
can issue and complete the operation correctly.

### Topology

`TransportTopology` stores world rank and size, scale-up rank and size,
scale-out rank and size, and whether direct peer addressing is available for
the scale-up team. Values are valid only after successful topology discovery.

### Device context and requests

`DeviceTransportContext` is a trivially copyable POD containing an ABI version,
structure size, capability bits, topology, local symmetric-window base,
peer-address table handle, channel table handle, and an opaque backend context
handle. Handles are integer-width opaque values and must refer to device-visible
resources when exported by a real implementation.

`DeviceRequest` is a trivially copyable, 16-byte-aligned opaque POD with 32
bytes of storage. A backend that needs larger completion state must store an
indirect device handle in this object. The common contract does not inspect its
contents.

## Host Transport Contract

`HostTransport` provides these operations:

- `capabilities()`: return the current capability bit set;
- `require_capabilities(required, operation)`: report every missing capability;
- `query_topology()`: discover and return logical transport domains;
- `register_symmetric_window(base, bytes)`: register one contiguous window;
- `unregister_symmetric_window()`: unregister the active window;
- `get_peer_base_pointer(team, rank)`: return a directly addressable peer base;
- `acquire_channels(count, scope)`: acquire device communication resources;
- `release_channels()`: release all acquired channels;
- `export_device_context()`: return a kernel-consumable POD context;
- `host_barrier()`: synchronize ranks during host-side setup or teardown;
- `destroy()`: release all resources and remain safe on repeated calls.

Only one symmetric window may be active per transport instance. Registration,
channel acquisition, and device-context export must be complete before any
communication kernel is launched.

## Device Transport Contract

The device contract defines the following semantic operations. Phase 2A adds
vendor-independent types and declarations only; it does not compile an Ascend C
implementation or launch a device transport operation.

### Addressing

- `is_peer_directly_accessible(context, team, rank)`;
- `get_symmetric_offset(context, local_pointer)`;
- `get_symmetric_pointer(context, team, rank, local_pointer)`.

### One-sided transfer

- `put(context, channel, team, destination_rank, destination, source, bytes,
  scope, options)`;
- `get(context, channel, team, source_rank, source, destination, bytes, scope,
  options)`;
- `put_value(context, channel, team, destination_rank, destination, value,
  options)`.

### Remote synchronization

- `remote_add_release(context, channel, team, destination_rank, destination,
  value)`;
- `signal(context, channel, team, destination_rank, signal_index, value)`.

### Completion and ordering

- `flush(context, channel, scope)`;
- `flush_async(context, channel, peer_rank, scope, request)`;
- `wait(context, request)`;
- `load_acquire(pointer)`;
- `store_release(pointer, value)`;
- `system_fence()`;
- `device_barrier(context, team_mask, workspace, timeout)`.

The `options` argument represents aggregation and remote-action requirements
without exposing NCCL flags. Concrete option bits will be introduced only when
an Ascend implementation can define their behavior.

## CUDA Gin Requirement Matrix

| EPv2 consumer | CUDA Gin behavior | Required contract capability |
| --- | --- | --- |
| Dispatch | symmetric peer address lookup | `direct_peer_pointer` |
| Dispatch | remote payload and metadata writes | `device_put` |
| Dispatch | remote counter publication | `device_put_value` |
| Combine | peer accessibility and address lookup | `direct_peer_pointer` |
| Combine | remote token writes | `device_put` |
| Hybrid dispatch | scale-up direct access and scale-out Gin writes | `scale_up_team`, `scale_out_team`, `device_put` |
| Hybrid dispatch | remote tail updates | `remote_atomic_add_release` |
| Hybrid combine | scale-out writes and completion | `device_put`, `async_completion` |
| Hybrid combine | remote tail updates and ordered publication | `remote_atomic_add_release`, `system_memory_ordering` |
| Barrier | flush, remote signal, acquire polling | `remote_signal`, `system_memory_ordering`, `device_barrier` |
| Pipeline send/receive | put with remote signal and signal polling | `device_put`, `remote_signal`, `system_memory_ordering` |
| Engram fetch | asynchronous remote reads | `device_get`, `async_completion` |

The pipeline-parallel and engram rows inventory experimental elastic APIs. They
do not expand Phase 2A into functional implementation work.

## Stub Behavior

`StubHostTransport` behaves as follows:

- construction succeeds for a valid rank, world size, zero communicator handle,
  empty CPU communicator, positive device buffer size, and zero CPU buffer size;
- all device communication capability bits are false;
- topology discovery, symmetric-window registration, peer-pointer lookup,
  channel acquisition, device-context export, and host barrier return
  `unsupported_capability`;
- nonzero communicator handles and otherwise invalid construction arguments
  return `invalid_argument`;
- release and destruction operations are idempotent;
- no operation performs memory writes, allocates communication resources, or
  reports false success.

`ElasticBuffer` maps `unsupported_capability` to Python
`NotImplementedError`. It maps `invalid_argument` to the existing argument
validation exception behavior. Future `runtime_failure` errors will retain the
failed API name, backend error code, rank, and diagnostic text.

Dispatch, combine, barrier, stream/event access, and size calculations that
still lack a Phase 2A implementation remain host-gated. Their messages identify
the missing operation and state that the Ascend device transport is not yet
implemented; they no longer claim that the repository is in Phase 1.

## Lifecycle and Data Flow

Construction follows this sequence:

1. `ElasticBuffer` validates backend-independent arguments.
2. The Ascend transport factory receives `TransportConfig`.
3. The stub validates the zero communicator and Phase 2A buffer constraints.
4. The stub returns a live transport object with no capabilities.

An operation follows this sequence:

1. `ElasticBuffer` computes the operation's required capability set.
2. `require_capabilities` reports all missing semantics.
3. On failure, the host raises before allocation or kernel launch.
4. A future supported backend obtains the device context and launches the
   corresponding Ascend C kernel.

Destruction releases channels, the symmetric window, the communicator, and the
runtime in reverse acquisition order. Repeated explicit destruction and a later
destructor call are safe.

## Source Layout

Phase 2A adds:

```text
csrc/backends/ascend/transport/types.hpp
csrc/backends/ascend/transport/host_transport.hpp
csrc/backends/ascend/transport/device_transport.hpp
csrc/backends/ascend/transport/stub_transport.hpp
```

It updates:

```text
csrc/backends/ascend/elastic_buffer.hpp
tests/ascend/
tests/platform/
```

The transport remains header-only in Phase 2A. Therefore the Ascend setuptools
and CMake builds add no source file, library, include path, or vendor runtime
dependency.

## Verification

Phase 2A verification covers:

1. a pure C++17 contract test that compiles without Torch, CUDA, CANN, or HCCL;
2. capability enumeration and missing-capability diagnostics;
3. valid stub creation, explicit destruction, repeated destruction, and
   destructor safety;
4. invalid rank, size, CPU buffer, CPU communicator, and communicator handle
   rejection;
5. rejection of topology, window, peer pointer, channel, device context, and
   host barrier operations as unsupported;
6. host gating for dispatch, combine, barrier, stream/event, and remaining
   unimplemented methods;
7. unchanged Python extension names, methods, and signatures;
8. clean Ascend setuptools and CMake builds on the Ascend 950 validation node;
9. extension dependency inspection proving that CUDA, NCCL, NVSHMEM, ACL, HCCL,
   and `torch_npu` are not linked;
10. source-boundary checks proving that the CUDA backend and kernels were not
    modified.

GPU runtime verification remains deferred by agreement.

## Acceptance Criteria

Phase 2A is complete when:

1. the full semantic interface listed above exists as vendor-independent C++17
   declarations and types;
2. every required capability is represented explicitly;
3. the Ascend stub reports no device communication capabilities;
4. unsupported operations fail before allocation or kernel launch;
5. no unsupported operation silently succeeds or falls back to a collective;
6. transport lifecycle operations are idempotent;
7. Python EPv2 API compatibility is preserved;
8. the Ascend extension builds and passes its contract tests on the Ascend 950
   validation node;
9. the Ascend extension gains no vendor communication dependency in Phase 2A;
10. CUDA production sources remain unchanged.

## Next Phase

The next design and implementation phase will investigate the Ascend 950 CANN
and Ascend C environment against this capability list. It will determine:

- whether symmetric memory registration produces device-usable peer addressing;
- whether AIV channels expose device-side resources;
- how participant, workgroup, and device cooperation scopes map to Ascend C;
- which one-sided put, get, signal, atomic, completion, and ordering primitives
  are available;
- whether device barriers can be composed with bounded timeout behavior;
- which capabilities require a newer CANN release or cannot be supported.

That phase may implement supported primitives behind this contract. It must not
claim a capability until an end-to-end kernel probe demonstrates the required
semantics on Ascend 950.

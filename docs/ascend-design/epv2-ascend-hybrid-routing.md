# EPv2 Ascend Hybrid Routing And Mapped CPU Memory

## Scope

This specification defines Phase 3D after the Phase 3C topology, peer mapping,
device window, command ordering, and diagnostics contracts exist. It adds
two-stage scale-out/scale-up routing and an optional mapped CPU segment behind
the existing EPv2 `ElasticBuffer` and handle APIs.

The public result must match direct routing for the same token graph. A device-
only configuration remains independent of CPU allocation, CPU handles, and
host-visible synchronization.

## Decisions

### Use two-stage fixed-shard routing

Hybrid dispatch sends each remote-domain token along its scale-up rail to the
peer with the same `scale_up_rank`, then forwards it within the destination
domain to the owning expert rank. Combine reverses the recorded route: local
contributors reduce/forward to the rail rank, then the rail rank returns the
result to the origin domain.

This matches upstream EPv2 semantics without copying CUDA's live-tail,
linked-list, warp, or QP implementation. Ascend keeps bounded fixed shards and
explicit service boundaries. The first correct implementation may use more
staging memory; later performance work may replace the schedule without
changing the handle or route contract.

### Record reverse routes in the handle

Top-k indices alone cannot reconstruct hybrid combine. The versioned Ascend
dispatch descriptor adds, per emitted row:

- origin world rank and source row;
- ingress rail world rank;
- destination world rank and local expert;
- ingress and forwarded slot identifiers;
- route generation and topology epoch; and
- completion state for both stages.

Cached dispatch reuses this exact route state. Combine rejects a direct/hybrid
mode mismatch, stale topology epoch, foreign buffer family, incomplete stage,
or out-of-range slot before publication.

### Keep address spaces structured

The logical elastic address space contains explicit segments rather than an
unchecked flat pointer:

```text
{segment_kind, owner_rank, offset, bytes, registration_epoch}
```

The device segment is the Phase 3C symmetric window. The optional CPU portion
contains one equally sized, 2 MiB-aligned segment for each local scale-up rank,
ordered by `scale_up_rank`. Checked arithmetic validates every range against
its segment before mapping or transport submission.

No Ascend implementation may reuse CUDA VMM or POSIX-FD assumptions. A CANN
provider must prove pinned allocation, NPU mapping, peer import, transport
registration, CPU cache visibility, and lifetime semantics before the CPU
segment capability is enabled.

### Separate capability gates

Hybrid device routing and mapped CPU memory are distinct capabilities.
Device-only hybrid routing can be developed and tested first. Nonzero CPU
bytes are admitted only when mapped allocation, registration, host-visible
completion, and teardown are all available. Unsupported routes fail before
launch; they never return fabricated empty tensors or silently fall back to a
local route.

The logical single-host 2x2 mode may test route construction and device-only
forwarding. It cannot qualify cross-host mapped CPU visibility.

## Preflight And Publication

Each rank constructs a compact preflight record containing topology identity,
mode, dtype, dimensions, device/CPU capacities, route-layout version, and local
resource status. The process-group boundary aggregates these records before
the C++ runtime publishes windows or launches a collective operation.

If any rank rejects, every rank returns the same bounded error naming the
first failing rank and reason. An operation generation is consumed only after
the collective preflight succeeds. A later local launch failure poisons that
generation and is surfaced through the existing rank-qualified diagnostics.

## Mapped CPU Ownership

The runtime owns a `MappedSegmentDescriptor` with allocation owner, host
pointer, device-visible address, bytes, registration/import handle, topology
epoch, and published state. Construction is transactional:

1. allocate aligned pinned host memory;
2. map it into the local NPU address space;
3. export/import handles only inside the local scale-up group;
4. register the mapped ranges required by the selected transport;
5. aggregate rank-local status; and
6. publish the segment directory to device and host consumers.

Teardown stops admission, completes or times out device work, publishes an
invalid epoch, deregisters transport ranges, unmaps peer segments, releases
imports/exports, and frees the owner allocation. Partial construction follows
the same reverse order and remains retryable.

CPU producers publish payload before a release generation marker. NPU or
remote consumers perform the CANN-defined acquire/cache operation before
reading. NPU producers establish transport and system visibility before a CPU
completion marker; CPU consumers acquire the marker before reading. A host
barrier alone is not accepted as a cache-coherency proof.

### Pinned CANN Capability Audit

The 2026-08-18 NPU8P login-node audit used only the sourced system CANN
`/usr/local/Ascend/cann-9.2.0` and the pinned project HCOMM package
`/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann`; it did not access
an NPU or submit a TaskQueue job. The audit is deliberately limited to public
headers and dynamic-library exports.

| Required edge | Public evidence | Decision |
| --- | --- | --- |
| Pinned host allocation and free | `acl/acl_rt.h` declares `aclrtMallocHost` and `aclrtFreeHost`; `/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libascendcl.so` exports both (with implementation symbols in `libruntime.so`). | Available. |
| Host allocation to NPU address mapping and unmapping | `aclrtMemMapNoAccess` maps an `aclrtDrvMemHandle` into a reserved virtual range, and `aclrtMemMapSelectedLink` maps one virtual address to another link. Neither public signature accepts an `aclrtMallocHost` allocation or returns a device address for it. No public host-pointer mapping/unmapping pair was found. | Missing. |
| Transport registration, deregistration, import, and release | Pinned `hccl/hccl_team.h` declares `HcclTeamWindowRegister`/`HcclTeamWindowDeregister`; `hcomm/hcomm_res.h` declares `HcommMemReg`/`HcommMemUnreg` and `HcommMemExport`/`HcommMemImport`/`HcommMemUnimport`. Pinned `aarch64-linux/lib64/libhcomm.so` exports the team-window and import symbols. | Individually available, but cannot register a proven mapped host-to-NPU segment. |
| Cache/system visibility acquire and release | `acl/acl_rt.h` declares `aclrtMemFlush` and `aclrtMemInvalidate` for `devPtr`. The audited public headers provide no documented CPU release marker plus NPU acquire, or NPU release plus CPU acquire, for an `aclrtMallocHost` mapping. | Missing. |
| CPU-visible completion | `aclrtSynchronizeStream` and `aclrtSynchronizeDevice` are declared in `acl/acl_rt.h` and exported by `libascendcl.so`, but they only establish task completion; without the missing mapped-address and visibility contracts they do not prove CPU-visible completion of shared mapped memory. | Insufficient. |

Consequently `mapped_cpu_memory_supported()` is false. A nonzero
`cpu_buffer_bytes` remains an explicit construction error; zero CPU bytes make
no mapped-provider calls, and device-only direct and hybrid behavior is
unchanged. `MappedSegmentOwner` retains the required transactional contract
for a future provider: allocate, map, register, publish a release generation,
provider-defined acquire, invalidate before teardown, then deregister, unmap,
and free. It is not a claim that the current CANN installation can implement
those operations.

## Hybrid Dispatch

1. Validate the public inputs and dispatch handle mode collectively.
2. Build deterministic source routes and reserve fixed ingress slots.
3. Send remote-domain payload and route metadata to the same-rail peer using
   the Phase 3C scale-out path.
4. Flush, publish ingress generation/count, signal, and bounded-wait.
5. Each ingress rank scans source domains in world-rank order and forwards rows
   to local destination ranks through scale-up.
6. Flush, publish forwarding generation/count, signal, and bounded-wait.
7. Destination ranks compact rows using the existing expert/rank ordering and
   return a handle containing the complete reverse-route descriptor.

Local and direct scale-up rows bypass ingress but still receive route records
with explicit stage markers. Empty domains participate in both bounded control
stages with zero counts.

## Hybrid Combine

1. Collectively attest the dispatch descriptor and current topology.
2. Reduce local duplicate contributions in deterministic contributor order.
3. Forward remote-origin rows from destination ranks to their ingress rail
   rank using the recorded reverse route.
4. Flush, publish local-stage control, signal, and bounded-wait.
5. Return each staged row over scale-out to its origin rail peer.
6. Flush, publish return-stage control, signal, and bounded-wait.
7. The origin rank validates the recorded source row and completes the final
   deterministic reduction/output placement.

Both `allow_multiple_reduction` modes retain their existing public semantics.
No remote floating-point atomic is required by this schedule.

## Failure Semantics

- Pre-publication failure is aggregated and leaves no peer waiting.
- Stage-one success followed by stage-two failure poisons the operation
  generation; cached state from it is unusable.
- Every wait has a finite timeout and identifies stage, topology epoch, team,
  logical/world peer, command, channel, and generation.
- A CPU mapping or visibility failure disables only configurations requesting
  CPU bytes; device-only direct and hybrid routes continue to initialize.
- Teardown after timeout invalidates publication before freeing any mapped or
  registered memory.

## Validation

### Host and model tests

- route classification for local, direct scale-up, direct scale-out, and
  diagonal hybrid paths over the 2x2 topology;
- dispatch-to-combine reverse-route round trips and descriptor rejection;
- empty/asymmetric domains, duplicate target ranks, capacity boundaries, and
  stale stage generations;
- collective preflight agreement and injected rank-local rejection;
- device/CPU segment bounds, registration epochs, partial construction, and
  reverse teardown through a fake mapped-memory provider; and
- proof that zero CPU bytes never invoke mapped-memory APIs.

### NPU8P logical two-host tests

A serialized four-rank job maps two complete device pairs to logical hosts and
validates device-only hybrid routing for local, rail, and diagonal token
graphs. The final qualification used snapshot `53fec72`, devices `2,3` as
logical host A and `4,5` as logical host B, and task
`task_20260819_122044_73205324377`. It compared dispatch and combine with an
independent exact BF16 reference; covered fresh and cached handles, empty and
asymmetric routes, both multiple-reduction policies, and repeated generations;
and completed bounded failure plus poisoned teardown. The task printed
`PASS logical-single-host device-only hybrid smoke` and exited zero. Its
production build/import prerequisite was
`task_20260819_121747_7237407472`, and the tracked source archive SHA-256 was
`c4a62f050bee82adb7bf99d49815fce09930bd5d69b67634735a42c45bf6aaa6`.

This evidence is labeled `logical-single-host` and does not qualify CPU
mapping across hosts or physical RoCE.

If the installed CANN exposes a supportable local pinned/mapped-memory API, a
separate bounded same-host test may validate allocation, NPU visibility,
host-visible completion, and teardown. It is evidence for local API behavior
only.

### Real multi-host gate

Phase 3D is production-complete only when a real two-host 2x2 topology passes
direct scale-up, direct scale-out, and diagonal hybrid parity; mapped CPU and
device segment bounds; asymmetric preflight rejection; bounded CPU/NPU
visibility; repeated generations; and cleanup after injected failures. Until
then, logical device-only routing may land behind its explicit mode, while
physical hybrid and mapped-CPU capability bits remain disabled.

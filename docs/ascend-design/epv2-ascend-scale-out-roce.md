# EPv2 Ascend Scale-Out And RoCE

## Scope

This specification defines Phase 3C for the Ascend EPv2 `ElasticBuffer` path.
It preserves the public EPv2 API and the accepted BF16 fixed-shard schedules
from Phases 2F and 2G. Legacy DeepEP V1, GPU validation, FP8, asynchronous
events, communication-stream overlap, pipeline, Engram, and AGRS are outside
this phase.

Phase 3C adds a device-memory scale-out transport and direct scale-out
barrier, dispatch, and combine. Mapped CPU memory and two-stage hybrid routing
belong to Phase 3D.

The available NPU8P host is not a physical multi-host environment. Devices
`0,1` form logical host A and devices `6,7` form logical host B. This setup is
valid for topology, route-selection, ordering, result, and teardown tests. It
does not qualify physical RoCE addressing, NIC selection, link failures, or
cross-host performance.

## Decisions

### Preserve semantic layers, not CUDA mechanisms

The Ascend implementation preserves the upstream EPv2 topology, ordering,
handle, and result semantics. It does not reproduce NCCL Gin, CUDA VMM, QPs,
TMA, warp roles, or SM scheduling. CANN/HCOMM resources remain behind
`HostTransport`; SIMT code appends bounded commands; the AICore service owns
the transport submission and completion boundary.

### Use one owning world communicator

The process group and its HCOMM world resource own all ranks. Logical teams
are views over that world:

| World rank | Logical device | Scale-up rank/team | Scale-out rank/team |
| --- | --- | --- | --- |
| 0 | 0 | 0 in `[0,1]` | 0 in `[0,2]` |
| 1 | 1 | 1 in `[0,1]` | 0 in `[1,3]` |
| 2 | 6 | 0 in `[2,3]` | 1 in `[0,2]` |
| 3 | 7 | 1 in `[2,3]` | 1 in `[1,3]` |

Ranks use row-major decomposition:

```text
world_rank = scale_out_rank * scale_up_size + scale_up_rank
scale_out_rank = world_rank / scale_up_size
scale_up_rank = world_rank % scale_up_size
```

For a logical peer:

```text
scale-up world peer = local_scale_out_rank * scale_up_size + peer_scale_up_rank
scale-out world peer = peer_scale_out_rank * scale_up_size + local_scale_up_rank
```

This mapping works with a real world communicator spanning hosts and with the
single-host 2x2 simulation. It avoids depending on an undocumented HCOMM
subteam ABI. A future qualified HCOMM subteam implementation may replace the
resource lookup without changing device or operator semantics.

### Make topology explicit and versioned

`TransportTopology` is extended with a topology kind and epoch. Supported
kinds are flat scale-up, physical 2D, and logical simulation. A checked helper
constructs the row-major topology from `world_rank`, `world_size`, and
`scale_up_size` and rejects:

- non-positive dimensions;
- a world size not equal to `scale_up_size * scale_out_size`;
- ranks outside the world;
- integer overflow; and
- topology changes while an operation or handle is live.

The default remains flat scale-up. Ascend obtains the selected scale-up size
from a backend configuration value and validates it on every rank before
publishing transport resources. The Python boundary aggregates the value over
the process group so asymmetric configuration fails before construction.

The single-host validation runner explicitly selects logical simulation. A
production physical topology may be selected only by launcher-derived
configuration and a real multi-host communicator.

### Translate logical peers at the command boundary

Operators and the device facade address peers in their selected logical team.
The command encoder validates the logical peer and translates it to a world
rank before queue publication. Commands retain the requested logical team for
diagnostics. The staged transport context exports one world HCOMM team,
window, and channel resource plus the checked topology. The AICore service
submits against the world resource using the translated peer.

The translation is the only place allowed to flatten a logical team. Operators
must not perform ad hoc rank arithmetic. Invalid team, peer, epoch, or address
fails before queue publication.

### Capability policy

`kScaleOutTeam` means a physically qualified scale-out data path. It remains
disabled in logical simulation. Simulation has a distinct capability/status
marker used by tests and diagnostics; it cannot satisfy production RoCE
preflight.

Device get and direct peer pointers remain disabled. The selected fixed-shard
dispatch/combine protocol requires put, put-value, signal, flush/order, and a
bounded barrier, but does not require remote floating-point atomics.

## Transport Contract

### Addressing and registration

The device segment is one symmetric, 2 MiB-aligned window per rank. All remote
references use checked symmetric offsets; raw process pointers are never sent
between ranks. Window size, topology epoch, and descriptor identity are equal
across all ranks before publication.

The production resource order is:

1. validate communicator rank/size and topology agreement;
2. create or acquire the owning world team;
3. allocate and zero device memory;
4. collectively register the symmetric window;
5. create channels and publish the device context; and
6. admit operator launches.

Teardown reverses publication: stop admission, drain or time out commands,
clear the device context, release channels, deregister the window, destroy the
team, then free memory. Every stage is idempotent and retryable after a partial
failure.

### Ordering

For every fixed shard, the producer performs:

1. payload and metadata puts;
2. payload completion/flush;
3. generation-tagged count or control publication;
4. release signal; and
5. bounded team completion.

The consumer acquire-polls the generation signal, validates generation and
bounds, then reads the shard. A failed or timed-out attempt still consumes its
generation. No later operation may consume records from that attempt.

### Diagnostics and failures

All waits are finite. The first device diagnostic contains operation,
generation, logical team, logical peer, translated world peer, channel,
command index/opcode, completion state, and backend status. Host errors also
identify local rank and topology epoch.

Asymmetric topology, capacity, dtype, shape, or handle failures are aggregated
before collective publication. Unreachable peer, link error, stale
registration, and remote process loss require real multi-host qualification;
fake-host tests cover their state transitions but are not hardware evidence.

## Operator Schedules

### Barrier

Flat scale-up uses the existing scale-up barrier without acquiring scale-out
resources. Direct scale-out first drains outstanding payload commands, then
signals and acquire-waits across the selected scale-out rail. A world barrier
is an ordered scale-out barrier followed by a scale-up barrier. Logical
simulation uses the same team mapping over the world resource.

### Dispatch

Phase 3C reuses source-owned fixed shards. Each source writes the destination
rank's receive shard, publishes a generation-tagged count, signals completion,
and participates in a bounded barrier. The destination scans contributors in
world-rank order and compacts payload, top-k data, weights, and source metadata
using the existing deterministic reference rules.

A route is local, scale-up, or scale-out when source and destination share the
same rank, logical host, or rail respectively. A diagonal route is a direct
world route in Phase 3C. Two-stage forwarding is introduced only in Phase 3D.

### Combine

Combine reuses contributor-owned staging and origin receive shards. Each
contributor returns rows to the origin selected by the dispatch handle,
publishes control, and signals completion. The origin reduces contributors in
deterministic world-rank order. The handle attests topology kind, epoch,
dimensions, buffer family, generation, routing mode, and capacity before any
remote write.

## Validation

### Host and model tests

- all valid row-major mappings and reciprocal team-peer translations;
- malformed dimensions, overflow, wrong ranks, and topology-epoch mismatch;
- command translation for world, scale-up, and scale-out teams;
- logical simulation never advertising physical RoCE qualification;
- payload-before-control ordering and finite completion;
- partial initialization, stale registration, injected link/completion error,
  retryable teardown, and no scale-out dependency for flat scale-up; and
- BF16 barrier/dispatch/combine reference parity in the 2x2 model.

### NPU8P logical two-host tests

One serialized four-rank job uses devices `0,1,6,7` with
`scale_up_size=2`. It covers both rails (`0<->6`, `1<->7`), both local pairs,
diagonal routes, empty ranks, asymmetric routing, cached handles, and at least
100 barrier generations. Every task is bounded to five minutes and no later
task is submitted until the previous task completes.

The report labels this evidence `logical-single-host`. It proves functional
2D routing through HCOMM on one host, not physical RoCE.

Final NPU8P evidence for commit `4913fb6`:

- `task_20260818_133032_194277931616`: Bisheng production-extension rebuild
  with the pinned `hcomm-deepep-current` package, `exit=0`;
- `task_20260818_133336_197028830316`: four-rank `0,1,6,7` logical 2x2 smoke,
  `exit=0`, including 100 barrier generations and the independent BF16
  dispatch/combine reference; and
- the runner released all four device locks after completion. The CANN
  ownership and HCCL timeout messages in the task log are environment warnings,
  not qualification failures.

### Real multi-host gate

Phase 3C is production-complete only after a coordinated two-host, four-rank
run passes the same public BF16 reference, 100 generations, bounded failures,
and teardown while using actual RoCE endpoints. Until that environment exists,
the code and logical simulation may land, but the roadmap status remains
partially qualified and `kScaleOutTeam` stays physically unqualified.

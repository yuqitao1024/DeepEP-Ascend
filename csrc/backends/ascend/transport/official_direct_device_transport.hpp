#pragma once

// Direct facade mapping from DeviceTransportFacade to the official HCOMM SIMT
// API. This experimental backend keeps the staged host resource layout only so
// diagnostics, request generations, and stage profiling remain ABI-compatible.
// The command queue is intentionally not used by the producer hot path.
#include "c_api/sync/sync.h"
#include "asc_comm_master/hcomm/hcomm_simt.h"
#include "comm_api/hcomm/hcomm_team_entity_defs.h"
#include "device_topology.hpp"
#include "simt_intrinsics.hpp"
#include "sync_layout.hpp"
#include "transport_commands.hpp"

namespace deep_ep::ascend::transport::device {
namespace detail {

DEEP_EP_ASCEND_SIMT_CALLEE int local_rank(
    const DeviceTransportContext& context, TransportTeam team) {
    switch (team) {
        case TransportTeam::kWorld:
            return context.topology.world_rank;
        case TransportTeam::kScaleUp:
            return context.topology.scale_up_rank;
        case TransportTeam::kScaleOut:
            return context.topology.scale_out_rank;
    }
    return -1;
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ StagedTransportContext* staged_context(
    const DeviceTransportContext& context) {
    if (context.abi_version != kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(DeviceTransportContext) ||
        context.backend_context == 0)
        return nullptr;
    auto* staged = reinterpret_cast<__gm__ StagedTransportContext*>(
        context.backend_context);
    if (simt::load_observed(&staged->abi_version) !=
            kTransportCommandAbiVersion ||
        simt::load_observed(&staged->struct_size) !=
            sizeof(StagedTransportContext) ||
        simt::load_observed(&staged->cann_compatibility) !=
            kStagedTransportCannCompatibility)
        return nullptr;
    return staged;
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ TransportCommandQueue* command_queue(
    const DeviceTransportContext& context) {
    auto* staged = staged_context(context);
    if (staged == nullptr)
        return nullptr;
    return reinterpret_cast<__gm__ TransportCommandQueue*>(
        simt::load_observed(&staged->command_queue));
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ TransportServiceState* service_state(
    __gm__ TransportCommandQueue* queue) {
    if (queue == nullptr)
        return nullptr;
    return reinterpret_cast<__gm__ TransportServiceState*>(
        simt::load_observed(&queue->service_state));
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ DeviceTransportDiagnostic* diagnostic(
    __gm__ TransportCommandQueue* queue) {
    if (queue == nullptr)
        return nullptr;
    return reinterpret_cast<__gm__ DeviceTransportDiagnostic*>(
        simt::load_observed(&queue->diagnostic));
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ DeviceChannelTable* channel_table(
    const DeviceTransportContext& context) {
    if (context.channel_table == 0)
        return nullptr;
    auto* table = reinterpret_cast<__gm__ DeviceChannelTable*>(
        context.channel_table);
    if (simt::load_observed(&table->abi_version) != kTransportCommandAbiVersion ||
        simt::load_observed(&table->struct_size) != sizeof(DeviceChannelTable))
        return nullptr;
    return table;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool record_error(
    __gm__ DeviceTransportDiagnostic* output,
    DeviceTransportError error,
    TransportCommandOpcode opcode,
    TransportTeam team,
    int peer,
    int world_peer,
    DeviceChannel channel,
    std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    if (output == nullptr ||
        simt::load_observed(reinterpret_cast<__gm__ std::uint32_t*>(
                                &output->error)) !=
            static_cast<std::uint32_t>(DeviceTransportError::kNone))
        return false;
    simt::store_published(reinterpret_cast<__gm__ std::uint32_t*>(
                              &output->abi_version),
                          kTransportCommandAbiVersion);
    simt::store_published(&output->command_index,
                          static_cast<std::uint32_t>(0));
    simt::store_published(reinterpret_cast<__gm__ std::uint32_t*>(&output->opcode),
                          static_cast<std::uint32_t>(opcode));
    simt::store_published(&output->peer,
                          static_cast<std::uint32_t>(peer));
    simt::store_published(&output->world_peer, world_peer);
    simt::store_published(reinterpret_cast<__gm__ std::uint8_t*>(&output->team),
                          static_cast<std::uint8_t>(team));
    simt::store_published(&output->channel, channel);
    simt::store_published(&output->backend_status, backend_status);
    simt::store_published(&output->reserved, reserved);
    simt::system_fence();
    simt::store_published(reinterpret_cast<__gm__ std::uint32_t*>(&output->error),
                          static_cast<std::uint32_t>(error));
    simt::system_fence();
    return true;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool record_error(
    const DeviceTransportContext& context,
    DeviceTransportError error,
    TransportCommandOpcode opcode,
    TransportTeam team,
    int peer,
    int world_peer,
    DeviceChannel channel,
    std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    auto* queue = command_queue(context);
    if (queue == nullptr)
        return false;
    auto* state = service_state(queue);
    if (state == nullptr)
        return false;
    const auto generation = simt::load_observed(&queue->generation);
    auto* output = diagnostic(queue);
    if (output == nullptr ||
        !record_error(output, error, opcode, team, peer, world_peer, channel,
                      backend_status, reserved))
        return false;
    simt::store_published(reinterpret_cast<__gm__ std::uint64_t*>(
                              &output->generation),
                          generation);
    return true;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool record_error(
    __gm__ TransportCommandQueue* queue,
    DeviceTransportError error,
    TransportCommandOpcode opcode,
    TransportTeam team,
    int peer,
    int world_peer,
    DeviceChannel channel,
    std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    auto* output = diagnostic(queue);
    if (output == nullptr ||
        !record_error(output, error, opcode, team, peer, world_peer, channel,
                      backend_status, reserved))
        return false;
    simt::store_published(reinterpret_cast<__gm__ std::uint64_t*>(
                              &output->generation),
                          queue == nullptr ? 0 :
                              simt::load_observed(&queue->generation));
    return true;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool record_error(
    __gm__ TransportCommandQueue* queue,
    DeviceTransportError error,
    TransportCommandOpcode opcode,
    int peer,
    DeviceChannel channel,
    std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    return record_error(queue, error, opcode, TransportTeam::kWorld, peer,
                        peer, channel, backend_status, reserved);
}

DEEP_EP_ASCEND_SIMT_CALLEE bool translate_peer(
    const DeviceTransportContext& context,
    TransportTeam team,
    int peer,
    int* world_peer) {
    return checked_world_peer(context.topology, team, peer, world_peer);
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t channel_handle(
    const DeviceTransportContext& context,
    __gm__ DeviceChannelTable* table,
    int peer,
    std::uint32_t channel) {
    auto* staged = detail::staged_context(context);
    if (staged != nullptr && staged->team != 0) {
        if (peer < 0)
            return 0;
        auto* team = reinterpret_cast<__gm__ HcommTeam*>(staged->team);
        const auto member_count = simt::load_observed(&team->memberNum);
        const auto self = simt::load_observed(&team->selfMemberId);
        if (static_cast<std::uint32_t>(peer) >= member_count ||
            static_cast<std::uint32_t>(peer) == self)
            return 0;
        auto* accumulated = team->channelCntAccumulatePerMember;
        if (accumulated == nullptr)
            return 0;
        const auto base = simt::load_observed(
            reinterpret_cast<__gm__ std::uint32_t*>(
                reinterpret_cast<std::uintptr_t>(accumulated) +
                static_cast<std::uint64_t>(peer) * sizeof(std::uint32_t)));
        const auto channels = simt::load_observed(&team->channelsBaseAddr);
        return channels +
            static_cast<std::uint64_t>(base + channel) *
                sizeof(AscendC::simt::HcommSimtChannelEntity);
    }
    if (table == nullptr || peer < 0 ||
        static_cast<std::uint32_t>(peer) >=
            simt::load_observed(&table->member_count) ||
        static_cast<std::uint32_t>(peer) ==
            simt::load_observed(&table->self_member) ||
        channel >= simt::load_observed(&table->channel_count))
        return 0;
    auto* handles = reinterpret_cast<__gm__ std::uint64_t*>(
        simt::load_observed(&table->channels));
    if (handles == nullptr)
        return 0;
    return simt::load_observed(
        handles + static_cast<std::uint64_t>(peer) *
                      simt::load_observed(&table->channel_count) +
                  channel);
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t remote_target(
    const DeviceTransportContext& context,
    __gm__ DeviceChannelTable* table,
    int peer,
    DeviceAddress local_address,
    std::uint64_t bytes) {
    auto* staged = detail::staged_context(context);
    std::uint64_t base = 0;
    std::uint64_t window_bytes = 0;
    if (staged != nullptr && staged->window != 0) {
        auto* window = reinterpret_cast<__gm__ HcommWindow*>(staged->window);
        const auto lsa_base = simt::load_observed(&window->lsaWin.baseVa);
        const auto stride = simt::load_observed(&window->lsaWin.stride);
        window_bytes = simt::load_observed(&window->lsaWin.userSize);
        base = lsa_base + static_cast<std::uint64_t>(peer) * stride;
    } else if (table != nullptr) {
        auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(
            simt::load_observed(&table->remote_bases));
        if (bases != nullptr)
            base = simt::load_observed(bases + peer);
        window_bytes = simt::load_observed(&table->window_bytes);
    }
    if (local_address < context.local_window_base ||
        bytes == 0 || bytes > 0xffffffffULL)
        return 0;
    const auto offset = local_address - context.local_window_base;
    if (offset > window_bytes || bytes > window_bytes - offset)
        return 0;
    return base == 0 ? 0 : base + offset;
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t remote_sync_target(
    const DeviceTransportContext& context,
    __gm__ DeviceChannelTable* table,
    int peer,
    std::uint32_t signal_index) {
    auto* staged = detail::staged_context(context);
    if (staged != nullptr && staged->signal_window != 0) {
        if (peer < 0 || signal_index >= sync_layout::kLogicalSignalCount)
            return 0;
        auto* team = reinterpret_cast<__gm__ HcommTeam*>(staged->team);
        const auto member_count = simt::load_observed(&team->memberNum);
        const auto self = simt::load_observed(&team->selfMemberId);
        auto* window = reinterpret_cast<__gm__ HcommWindow*>(
            staged->signal_window);
        const auto base = simt::load_observed(&window->lsaWin.baseVa);
        const auto stride = simt::load_observed(&window->lsaWin.stride);
        const auto offset =
            (static_cast<std::uint64_t>(signal_index) * member_count + self) *
            sizeof(std::uint64_t);
        return base + static_cast<std::uint64_t>(peer) * stride + offset;
    }
    if (table == nullptr || peer < 0 ||
        static_cast<std::uint32_t>(peer) >=
            simt::load_observed(&table->member_count) ||
        signal_index >= sync_layout::kLogicalSignalCount)
        return 0;
    auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(
        simt::load_observed(&table->remote_sync_bases));
    if (bases == nullptr)
        return 0;
    const auto member_count = simt::load_observed(&table->member_count);
    const auto self = simt::load_observed(&table->self_member);
    const auto offset =
        (static_cast<std::uint64_t>(signal_index) * member_count + self) *
        sizeof(std::uint64_t);
    const auto sync_bytes = simt::load_observed(&table->sync_bytes);
    if (offset + sizeof(std::uint64_t) > sync_bytes)
        return 0;
    const auto base = simt::load_observed(bases + peer);
    return base == 0 ? 0 : base + offset;
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t fetch_result_target(
    __gm__ DeviceChannelTable* table,
    __gm__ StagedTransportContext* staged,
    std::uint32_t peer) {
    if (table == nullptr || peer >= simt::load_observed(&table->member_count))
        return 0;
    // Preserve the native executor's registered-fetch-storage priority. A
    // standalone barrier context can publish a dedicated fetch window; normal
    // transport contexts fall back to the first registered sync-memory row.
    const auto offset = static_cast<std::uint64_t>(peer) *
        sizeof(std::uint64_t);
    if (staged != nullptr) {
        const auto fetch_base = simt::load_observed(&staged->fetch_results);
        const auto fetch_bytes =
            simt::load_observed(&staged->fetch_result_bytes);
        if (fetch_base != 0 &&
            offset + sizeof(std::uint64_t) <= fetch_bytes)
            return fetch_base + offset;
    }
    const auto local_sync_base = simt::load_observed(&table->local_sync_base);
    if (local_sync_base == 0 ||
        offset + sizeof(std::uint64_t) >
            simt::load_observed(&table->sync_bytes))
        return 0;
    return local_sync_base + offset;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool checked_result(
    const DeviceTransportContext& context,
    std::int32_t result,
    TransportCommandOpcode opcode,
    TransportTeam team,
    int peer,
    int world_peer,
    DeviceChannel channel) {
    return result == 0 ||
           record_error(context, DeviceTransportError::kCompletionFailure,
                        opcode, team, peer, world_peer, channel,
                        static_cast<std::uint32_t>(result));
}

inline constexpr std::uint32_t kOfficialDirectBatchWorkspaceBytes = 256;
inline constexpr std::uint32_t kOfficialDirectMaxBatchLanes = 32;
inline constexpr std::uint32_t kOfficialDirectBatchTarget = 1;

// A single-lane batch is the correctness-first bridge from the facade API to
// the new cooperative Hcomm API.  It amortizes SQ reservation and DWQE
// publication while preserving the current producer ownership model.
struct OfficialDirectBackend {
    using HcommType = AscendC::simt::Hcomm<>;
    using BatchType = AscendC::simt::BatchHandle<AscendC::ChannelHandle>;
    HcommType hcomm;
    BatchType batch{};
    bool initialized = false;

    __simt_callee__ inline bool initialize(
        std::uint64_t channel, std::uint64_t remote_base,
        std::uint32_t item_bb, __ubuf__ std::uint8_t* workspace,
        std::uint32_t workspace_bytes) {
        initialized = hcomm.Init(nullptr, 0) == 0;
        if (!initialized)
            return false;
        if (workspace == nullptr ||
            workspace_bytes < kOfficialDirectBatchWorkspaceBytes)
            return false;
        batch = hcomm.MakeBatchHandle(
            channel, workspace, workspace_bytes,
            reinterpret_cast<__gm__ void*>(remote_base), item_bb);
        return batch.context != nullptr;
    }

    __simt_callee__ inline bool commit() {
        return batch.context != nullptr && hcomm.BatchCommit(batch) == 0;
    }

    __simt_callee__ inline bool drain(
        const DeviceTransportContext& context,
        __gm__ DeviceChannelTable* table,
        TransportCommandOpcode opcode = TransportCommandOpcode::kFlush) {
        if (table == nullptr)
            return false;
        const auto members = simt::load_observed(&table->member_count);
        const auto channels = simt::load_observed(&table->channel_count);
        for (std::uint32_t peer = 0; peer < members; ++peer) {
            if (peer == simt::load_observed(&table->self_member))
                continue;
            for (std::uint32_t channel = 0; channel < channels; ++channel) {
                const auto handle = channel_handle(
                    context, table, peer, channel);
                if (handle == 0) {
                    (void)record_error(
                        context, DeviceTransportError::kInvalidChannel, opcode,
                        TransportTeam::kWorld, static_cast<int>(peer),
                        static_cast<int>(peer), channel);
                    return false;
                }
                if (!checked_result(context, hcomm.Drain(handle), opcode,
                                    TransportTeam::kWorld,
                                    static_cast<int>(peer),
                                    static_cast<int>(peer), channel))
                    return false;
            }
        }
        simt::system_fence();
        return true;
    }
};

DEEP_EP_ASCEND_SIMT_CALLEE OfficialDirectBackend make_backend() {
    return OfficialDirectBackend{};
}

// The direct facade is synchronous-looking: a call returns only after its WQE
// is visible to the channel.  Use the immediate API for correctness first; the
// cooperative batch handle remains available for a later persistent producer.
enum class OfficialDirectPostKind : std::uint32_t {
    kWrite,
    kWriteValue,
    kAtomic
};

DEEP_EP_ASCEND_SIMT_CALLEE std::int32_t post_batch_item(
    OfficialDirectPostKind kind,
    OfficialDirectBackend& backend,
    std::uint64_t target,
    std::uint64_t source,
    std::uint64_t bytes,
    std::uint64_t fetch,
    std::uint64_t value) {
    switch (kind) {
        case OfficialDirectPostKind::kWrite:
            return backend.hcomm.WriteNbi(
                backend.batch, reinterpret_cast<__gm__ void*>(target),
                reinterpret_cast<__gm__ void*>(source), bytes);
        case OfficialDirectPostKind::kWriteValue:
            return backend.hcomm.template WriteValueNbi<std::uint64_t>(
                backend.batch, reinterpret_cast<__gm__ void*>(target),
                static_cast<std::uint64_t>(value));
        case OfficialDirectPostKind::kAtomic:
            return backend.hcomm.template AtomicFAA<std::uint64_t>(
                backend.batch, reinterpret_cast<__gm__ void*>(target),
                reinterpret_cast<__gm__ void*>(fetch),
                static_cast<std::uint64_t>(value));
    }
    return -1;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool post_single_item(
    std::uint64_t handle,
    OfficialDirectPostKind kind,
    std::uint64_t target,
    std::uint64_t source,
    std::uint64_t bytes,
    std::uint64_t fetch,
    std::uint64_t value) {
    auto backend = make_backend();
    auto& hcomm = backend.hcomm;
    if (hcomm.Init(nullptr, 0) != 0)
        return false;
    switch (kind) {
        case OfficialDirectPostKind::kWrite:
            return hcomm.WriteNbi(
                handle, reinterpret_cast<__gm__ void*>(target),
                reinterpret_cast<__gm__ void*>(source), bytes) == 0;
        case OfficialDirectPostKind::kWriteValue:
            return hcomm.template WriteValueNbi<std::uint64_t>(
                handle, reinterpret_cast<__gm__ void*>(target),
                static_cast<std::uint64_t>(value)) == 0;
        case OfficialDirectPostKind::kAtomic:
            return hcomm.template AtomicFAA<std::uint64_t>(
                handle, reinterpret_cast<__gm__ void*>(target),
                reinterpret_cast<__gm__ void*>(fetch),
                static_cast<std::uint64_t>(value)) == 0;
    }
    return false;
}

DEEP_EP_ASCEND_SIMT_CALLEE void post_direct(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportCommandOpcode opcode,
    TransportTeam team,
    int peer_rank,
    int world_peer,
    std::uint64_t remote_base,
    std::uint32_t item_bb,
    OfficialDirectPostKind kind,
    std::uint64_t target,
    std::uint64_t source,
    std::uint64_t bytes,
    std::uint64_t fetch,
    std::uint64_t value) {
    const auto handle = detail::channel_handle(
        context, detail::channel_table(context), world_peer, channel);
    // The immediate API resolves the remote registration from the channel and
    // destination address, so a BatchHandle remote base is not required here.
    (void)remote_base;
    (void)item_bb;
    if (handle == 0 ||
        !post_single_item(handle, kind, target, source, bytes, fetch, value)) {
        (void)detail::record_error(
            context, DeviceTransportError::kCompletionFailure, opcode, team,
            peer_rank, world_peer, channel);
        return;
    }
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t remote_payload_base(
    const DeviceTransportContext& context, int world_peer) {
    auto* staged = detail::staged_context(context);
    if (staged != nullptr && staged->window != 0) {
        auto* window = reinterpret_cast<__gm__ HcommWindow*>(staged->window);
        const auto base = simt::load_observed(&window->lsaWin.baseVa);
        const auto stride = simt::load_observed(&window->lsaWin.stride);
        return base + static_cast<std::uint64_t>(world_peer) * stride;
    }
    auto* table = detail::channel_table(context);
    if (table == nullptr)
        return 0;
    auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(
        simt::load_observed(&table->remote_bases));
    return bases == nullptr ? 0 : simt::load_observed(bases + world_peer);
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t remote_sync_base(
    const DeviceTransportContext& context, int world_peer) {
    auto* staged = detail::staged_context(context);
    if (staged != nullptr && staged->signal_window != 0) {
        auto* window = reinterpret_cast<__gm__ HcommWindow*>(
            staged->signal_window);
        const auto base = simt::load_observed(&window->lsaWin.baseVa);
        const auto stride = simt::load_observed(&window->lsaWin.stride);
        return base + static_cast<std::uint64_t>(world_peer) * stride;
    }
    auto* table = detail::channel_table(context);
    if (table == nullptr)
        return 0;
    auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(
        simt::load_observed(&table->remote_sync_bases));
    return bases == nullptr ? 0 : simt::load_observed(bases + world_peer);
}

DEEP_EP_ASCEND_SIMT_CALLEE void publish_request(
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request,
    std::uint64_t generation) {
    if (request == nullptr)
        return;
    if (request->abi_version != kDeviceRequestAbiVersion) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidAbi;
        return;
    }
    if (request->state == DeviceRequestState::kPending ||
        request->state < DeviceRequestState::kEmpty ||
        request->state > DeviceRequestState::kFailed) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidProtocol;
        return;
    }
    // The direct backend has no queued command. Keep the command interval
    // empty and carry a non-zero direct completion target. The persistent
    // release protocol treats target 0 as an unacknowledged batch, so the
    // service shim consumes this synthetic target after Drain().
    request->command_begin = 0;
    request->command_end = 0;
    request->queue_generation = generation;
    request->consumed_target = kOfficialDirectBatchTarget;
    request->terminal_error = DeviceTransportError::kNone;
    request->state = DeviceRequestState::kPending;
    simt::system_fence();
}

}  // namespace detail

DEEP_EP_ASCEND_SIMT_CALLEE bool is_peer_directly_accessible(
    const DeviceTransportContext& context, TransportTeam team, int rank) {
    return rank == detail::local_rank(context, team);
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint32_t channel_count(
    const DeviceTransportContext& context, TransportTeam team, int rank) {
    int world_peer = -1;
    auto* table = detail::channel_table(context);
    if (table == nullptr ||
        !detail::translate_peer(context, team, rank, &world_peer))
        return 0;
    if (world_peer < 0 ||
        static_cast<std::uint32_t>(world_peer) >=
            simt::load_observed(&table->member_count) ||
        static_cast<std::uint32_t>(world_peer) ==
            simt::load_observed(&table->self_member))
        return 0;
    const auto configured = context.channel_count != 0 ?
        context.channel_count :
        simt::load_observed(&table->channel_count);
    return configured != 0 && simt::load_observed(&table->channels) != 0 ?
        configured : 0;
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t get_symmetric_offset(
    const DeviceTransportContext& context, DeviceAddress local_address) {
    if (local_address == kNullDeviceAddress ||
        context.local_window_base == 0)
        return 0;
    return local_address - context.local_window_base;
}

DEEP_EP_ASCEND_SIMT_CALLEE DeviceAddress get_symmetric_pointer(
    const DeviceTransportContext& context, TransportTeam team, int rank,
    DeviceAddress local_address) {
    return is_peer_directly_accessible(context, team, rank) ? local_address :
        kNullDeviceAddress;
}

DEEP_EP_ASCEND_SIMT_CALLEE void put(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int destination_rank,
    DeviceAddress destination,
    DeviceAddress source,
    std::size_t bytes,
    CooperationScope scope,
    MemorySegment segment,
    DeviceOptions options,
    const RemoteAction& remote_action) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, destination_rank,
                                &world_peer)) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kPut, team, destination_rank, -1, channel);
        return;
    }
    if (destination == kNullDeviceAddress ||
        source == kNullDeviceAddress || bytes == 0 || bytes > 0xffffffffULL) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPut, team, destination_rank, world_peer,
            channel);
        return;
    }
    if (scope != CooperationScope::kParticipant ||
        segment != MemorySegment::kDevice || options != kDefaultOptions ||
        remote_action.kind != RemoteActionKind::kNone) {
        (void)detail::record_error(
            context, DeviceTransportError::kUnsupportedOperation,
            TransportCommandOpcode::kPut, team, destination_rank, world_peer,
            channel);
        return;
    }
    auto* table = detail::channel_table(context);
    const auto handle = detail::channel_handle(context, table, world_peer, channel);
    const auto target = detail::remote_target(context, table, world_peer,
                                              destination, bytes);
    if (handle == 0 || target == 0) {
        (void)detail::record_error(
            context, handle == 0 ? DeviceTransportError::kInvalidChannel :
                                   DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPut, team, destination_rank, world_peer,
            channel);
        return;
    }
    const auto remote_base = detail::remote_payload_base(context, world_peer);
    detail::post_direct(
        context, channel, TransportCommandOpcode::kPut, team,
        destination_rank, world_peer, remote_base, 1,
        detail::OfficialDirectPostKind::kWrite, target, source, bytes, 0, 0);
}

DEEP_EP_ASCEND_SIMT_CALLEE void get(
    const DeviceTransportContext&,
    DeviceChannel,
    TransportTeam,
    int,
    DeviceAddress,
    DeviceAddress,
    std::size_t,
    CooperationScope,
    MemorySegment,
    DeviceOptions) {}

DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int destination_rank,
    DeviceAddress destination,
    std::uint64_t value,
    std::uint32_t value_bytes,
    DeviceOptions options) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, destination_rank,
                                &world_peer)) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kPutValue64, team, destination_rank, -1,
            channel);
        return;
    }
    if (destination == kNullDeviceAddress ||
        value_bytes != sizeof(std::uint64_t) || options != kDefaultOptions) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPutValue64, team, destination_rank,
            world_peer, channel);
        return;
    }
    auto* table = detail::channel_table(context);
    const auto handle = detail::channel_handle(context, table, world_peer, channel);
    const auto target = detail::remote_target(context, table, world_peer,
                                              destination, sizeof(value));
    if (handle == 0 || target == 0) {
        (void)detail::record_error(
            context, handle == 0 ? DeviceTransportError::kInvalidChannel :
                                   DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPutValue64, team, destination_rank,
            world_peer, channel);
        return;
    }
    detail::post_direct(
        context, channel, TransportCommandOpcode::kPutValue64, team,
        destination_rank, world_peer,
        detail::remote_payload_base(context, world_peer), 1,
        detail::OfficialDirectPostKind::kWriteValue, target, 0, sizeof(value),
        0, value);
}

DEEP_EP_ASCEND_SIMT_CALLEE void put_control_slot(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int destination_rank,
    DeviceAddress destination,
    DeviceAddress source,
    DeviceOptions options) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, destination_rank,
                                &world_peer)) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kPutControlSlot, team, destination_rank,
            -1, channel);
        return;
    }
    if (destination == kNullDeviceAddress ||
        source == kNullDeviceAddress || options != kDefaultOptions) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPutControlSlot, team, destination_rank,
            world_peer, channel);
        return;
    }
    auto* table = detail::channel_table(context);
    const auto handle = detail::channel_handle(context, table, world_peer, channel);
    constexpr auto bytes = 2 * sizeof(std::uint64_t);
    const auto target = detail::remote_target(context, table, world_peer,
                                              destination, bytes);
    if (handle == 0 || target == 0) {
        (void)detail::record_error(
            context, handle == 0 ? DeviceTransportError::kInvalidChannel :
                                   DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPutControlSlot, team, destination_rank,
            world_peer, channel);
        return;
    }
    detail::post_direct(
        context, channel, TransportCommandOpcode::kPutControlSlot, team,
        destination_rank, world_peer,
        detail::remote_payload_base(context, world_peer), 1,
        detail::OfficialDirectPostKind::kWrite, target, source, bytes, 0, 0);
}

DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int destination_rank,
    DeviceAddress destination,
    std::int64_t value) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, destination_rank,
                                &world_peer)) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kRemoteAdd64, team, destination_rank, -1,
            channel);
        return;
    }
    if (destination == kNullDeviceAddress) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kRemoteAdd64, team, destination_rank,
            world_peer, channel);
        return;
    }
    auto* table = detail::channel_table(context);
    const auto handle = detail::channel_handle(context, table, world_peer, channel);
    const auto target = detail::remote_target(
        context, table, world_peer, destination, sizeof(std::uint64_t));
    if (handle == 0 || target == 0) {
        (void)detail::record_error(
            context, handle == 0 ? DeviceTransportError::kInvalidChannel :
                                   DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kRemoteAdd64, team, destination_rank,
            world_peer, channel);
        return;
    }
    const auto fetch = detail::fetch_result_target(
        table, detail::staged_context(context),
        static_cast<std::uint32_t>(world_peer));
    if (fetch == 0)
        return;
    detail::post_direct(
        context, channel, TransportCommandOpcode::kRemoteAdd64, team,
        destination_rank, world_peer,
        detail::remote_payload_base(context, world_peer), 2,
        detail::OfficialDirectPostKind::kAtomic, target, 0, sizeof(value),
        fetch, static_cast<std::uint64_t>(value));
}

DEEP_EP_ASCEND_SIMT_CALLEE void signal(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int destination_rank,
    const RemoteAction& remote_action) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, destination_rank,
                                &world_peer)) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kSignal, team, destination_rank, -1,
            channel);
        return;
    }
    if (remote_action.kind != RemoteActionKind::kSignalAdd &&
        remote_action.kind != RemoteActionKind::kSignalIncrement &&
        remote_action.kind != RemoteActionKind::kSignalSet) {
        (void)detail::record_error(
            context, DeviceTransportError::kUnsupportedOperation,
            TransportCommandOpcode::kSignal, team, destination_rank,
            world_peer, channel);
        return;
    }
    auto* table = detail::channel_table(context);
    const auto handle = detail::channel_handle(context, table, world_peer, channel);
    if (handle == 0) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidChannel,
            TransportCommandOpcode::kSignal, team, destination_rank,
            world_peer, channel);
        return;
    }
    if (remote_action.kind == RemoteActionKind::kSignalSet) {
        const auto target = detail::remote_sync_target(
            context, table, world_peer, remote_action.signal_index);
        if (target == 0) {
            (void)detail::record_error(
                context, DeviceTransportError::kInvalidAddress,
                TransportCommandOpcode::kSignal, team, destination_rank,
                world_peer, channel);
            return;
        }
        detail::post_direct(
            context, channel, TransportCommandOpcode::kSignal, team,
            destination_rank, world_peer,
            detail::remote_sync_base(context, world_peer), 1,
            detail::OfficialDirectPostKind::kWriteValue, target, 0,
            sizeof(std::uint64_t), 0, remote_action.value);
        return;
    }
    std::uint64_t target = 0;
    if (remote_action.kind == RemoteActionKind::kSignalAdd) {
        target = detail::remote_target(
            context, table, world_peer,
            context.local_window_base + remote_action.symmetric_offset,
            sizeof(std::uint64_t));
    } else {
        target = detail::remote_sync_target(
            context, table, world_peer, remote_action.signal_index);
    }
    if (target == 0) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kSignal, team, destination_rank,
            world_peer, channel);
        return;
    }
    const auto fetch = detail::fetch_result_target(
        table, detail::staged_context(context),
        static_cast<std::uint32_t>(world_peer));
    if (fetch == 0)
        return;
    detail::post_direct(
        context, channel, TransportCommandOpcode::kSignal, team,
        destination_rank, world_peer,
        remote_action.kind == RemoteActionKind::kSignalAdd ?
            detail::remote_payload_base(context, world_peer) :
            detail::remote_sync_base(context, world_peer),
        2, detail::OfficialDirectPostKind::kAtomic, target, 0,
        sizeof(std::uint64_t), fetch, remote_action.value);
}

DEEP_EP_ASCEND_SIMT_CALLEE SignalValue read_signal(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int source_rank, std::uint32_t signal_index) {
    if (channel != 0)
        return 0;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, source_rank, &world_peer))
        return 0;
    auto* table = detail::channel_table(context);
    if (table == nullptr ||
        static_cast<std::uint32_t>(world_peer) >=
            simt::load_observed(&table->member_count) ||
        signal_index >= sync_layout::kLogicalSignalCount)
        return 0;
    const auto local_sync = simt::load_observed(&table->local_sync_base);
    const auto member_count = simt::load_observed(&table->member_count);
    const auto offset =
        (static_cast<std::uint64_t>(signal_index) * member_count +
         static_cast<std::uint32_t>(world_peer)) *
        sizeof(std::uint64_t);
    if (local_sync == 0 ||
        offset + sizeof(std::uint64_t) >
            simt::load_observed(&table->sync_bytes))
        return 0;
    const auto value = simt::load_observed(
        reinterpret_cast<__gm__ std::uint64_t*>(local_sync + offset));
    simt::system_fence();
    return value;
}

DEEP_EP_ASCEND_SIMT_CALLEE void wait_signal(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int source_rank,
    std::uint32_t signal_index,
    SignalValue target,
    std::uint64_t timeout_cycles) {
    if (channel != 0)
        return;
    int world_peer = -1;
    if (!detail::translate_peer(context, team, source_rank, &world_peer)) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kSignal, team, source_rank, -1, channel);
        return;
    }
    const auto limit = timeout_cycles == 0 ? std::uint64_t{1000000} :
                                             timeout_cycles;
    SignalValue observed = 0;
    std::uint64_t retry = 0;
    const auto begin = static_cast<std::uint64_t>(clock());
    do {
        observed = read_signal(context, channel, team, source_rank,
                               signal_index);
        if (observed >= target)
            break;
        simt::poll_nop();
        ++retry;
        const bool timed_out = timeout_cycles == 0 ?
            retry >= limit :
            static_cast<std::uint64_t>(clock()) - begin >= timeout_cycles;
        if (timed_out) {
            (void)detail::record_error(
                context, DeviceTransportError::kCompletionTimeout,
                TransportCommandOpcode::kSignal, team, source_rank,
                world_peer, channel);
            return;
        }
    } while (true);
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE void flush(
    const DeviceTransportContext& context, DeviceChannel channel,
    CooperationScope) {
    if (threadIdx.x != 0 || channel != 0)
        return;
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    auto* output = detail::diagnostic(queue);
    auto* table = detail::channel_table(context);
    if (queue == nullptr || state == nullptr || output == nullptr ||
        table == nullptr)
        return;
    const auto generation = simt::load_observed(&queue->generation);
    if (generation == 0)
        return;
    const bool success = detail::make_backend().drain(
        context, table, TransportCommandOpcode::kFlush);
    const auto error = static_cast<DeviceTransportError>(
        simt::load_observed(reinterpret_cast<__gm__ std::uint32_t*>(
            &output->error)));
    if (success && error == DeviceTransportError::kNone)
        simt::store_published(&state->consumed_generation, generation);
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE void flush_async(
    const DeviceTransportContext& context,
    DeviceChannel channel,
    TransportTeam team,
    int peer_rank,
    CooperationScope,
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) {
    if (threadIdx.x != 0)
        return;
    if (channel != 0) {
        if (request != nullptr) {
            request->state = DeviceRequestState::kFailed;
            request->terminal_error = DeviceTransportError::kInvalidChannel;
        }
        return;
    }
    int world_peer = -1;
    if (!detail::translate_peer(context, team, peer_rank, &world_peer)) {
        if (request != nullptr) {
            request->state = DeviceRequestState::kFailed;
            request->terminal_error = DeviceTransportError::kInvalidRank;
        }
        return;
    }
    if (request == nullptr) {
        (void)detail::record_error(
            context, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kFlush, team, peer_rank, world_peer,
            channel);
        return;
    }
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    if (queue == nullptr || state == nullptr) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidQueue;
        return;
    }
    if (detail::channel_table(context) == nullptr) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidChannel;
        return;
    }
    const auto generation = simt::load_observed(&queue->generation);
    if (generation == 0) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidProtocol;
        return;
    }
    simt::store_published(&state->consumed_generation, std::uint64_t{0});
    detail::publish_request(request, generation);
}

DEEP_EP_ASCEND_SIMT_CALLEE void wait(
    const DeviceTransportContext& context,
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) {
    if (threadIdx.x != 0 || request == nullptr)
        return;
    if (request->state == DeviceRequestState::kCompleted ||
        request->state == DeviceRequestState::kFailed)
        return;
    if (request->abi_version != kDeviceRequestAbiVersion) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidAbi;
        return;
    }
    if (request->state != DeviceRequestState::kPending) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidProtocol;
        return;
    }
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    auto* output = detail::diagnostic(queue);
    if (state == nullptr || output == nullptr) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidQueue;
        return;
    }
    auto* table = detail::channel_table(context);
    if (table == nullptr) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidChannel;
        return;
    }
    const bool success = detail::make_backend().drain(
        context, table, TransportCommandOpcode::kFlush);
    const auto error = static_cast<DeviceTransportError>(
        simt::load_observed(reinterpret_cast<__gm__ std::uint32_t*>(
            &output->error)));
    if (!success || error != DeviceTransportError::kNone) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error =
            !success &&
                    error == DeviceTransportError::kNone ?
                DeviceTransportError::kCompletionFailure :
                error;
    } else {
        simt::store_published(&state->consumed_generation,
                              request->queue_generation);
        request->state = DeviceRequestState::kCompleted;
        request->terminal_error = DeviceTransportError::kNone;
    }
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t load_acquire(
    DeviceAddress address) {
    if (address == kNullDeviceAddress)
        return 0;
    const auto value = simt::load_observed(
        reinterpret_cast<const __gm__ std::uint64_t*>(address));
    simt::system_fence();
    return value;
}

DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
    DeviceAddress address, std::uint64_t value) {
    if (address == kNullDeviceAddress)
        return;
    simt::system_fence();
    simt::store_published(reinterpret_cast<__gm__ std::uint64_t*>(address),
                          value);
}

DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() {
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t consumed_generation(
    const DeviceTransportContext& context) {
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    if (state == nullptr)
        return 0;
    return simt::load_observed(&state->consumed_generation);
}

DEEP_EP_ASCEND_SIMT_CALLEE bool barrier_peer_in_phase(
    int peer,
    std::uint32_t self,
    std::uint32_t scale_up_size,
    std::uint32_t phase) {
    if (peer < 0 || static_cast<std::uint32_t>(peer) == self)
        return false;
    return phase == 0 ?
        static_cast<std::uint32_t>(peer) % scale_up_size ==
            self % scale_up_size :
        static_cast<std::uint32_t>(peer) / scale_up_size ==
            self / scale_up_size;
}

DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
    const DeviceTransportContext& context,
    std::uint32_t team_mask,
    DeviceAddress,
    std::uint64_t timeout_cycles) {
    if (threadIdx.x != 0)
        return;
    if ((team_mask & ~kWorldTeamMask) != 0)
        return;
    auto* table = detail::channel_table(context);
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    if (table == nullptr || state == nullptr)
        return;
    const auto members = simt::load_observed(&table->member_count);
    const auto self = simt::load_observed(&table->self_member);
    const auto scale_up_size = static_cast<std::uint32_t>(
        context.topology.scale_up_size);
    if (members > 64 || scale_up_size == 0 ||
        simt::load_observed(&table->sync_bytes) <
            sync_layout::sync_window_bytes(members))
        return;
    auto backend = detail::make_backend();
    const auto generation = simt::load_observed(&state->barrier_generation) + 1;
    for (std::uint32_t phase = 0; phase < 2; ++phase) {
        const auto mask = phase == 0 ? kScaleOutTeamMask :
                                       kScaleUpTeamMask;
        const auto size = phase == 0 ?
            static_cast<std::uint32_t>(context.topology.scale_out_size) :
            static_cast<std::uint32_t>(context.topology.scale_up_size);
        if ((team_mask & mask) == 0 || size <= 1)
            continue;
        const auto row =
            (static_cast<std::uint64_t>(sync_layout::kLogicalSignalCount) +
             phase) * members;
        std::uint64_t pending = 0;
        for (std::uint32_t peer = 0; peer < members; ++peer) {
            if (!barrier_peer_in_phase(static_cast<int>(peer), self,
                                       scale_up_size, phase))
                continue;
            pending |= std::uint64_t{1} << peer;
            auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(
                simt::load_observed(&table->remote_sync_bases));
            const auto address = simt::load_observed(bases + peer) +
                (row + self) * sizeof(std::uint64_t);
            const auto handle = detail::channel_handle(context, table, peer, 0);
            const auto fetch = detail::fetch_result_target(
                table, detail::staged_context(context), peer);
            if (handle == 0 || fetch == 0) {
                (void)detail::record_error(
                    context, DeviceTransportError::kInvalidChannel,
                    TransportCommandOpcode::kBarrier,
                    TransportTeam::kWorld, static_cast<int>(peer),
                    static_cast<int>(peer), 0);
                return;
            }
            // Match the official SIMT executor's barrier protocol: post an
            // immediate atomic on the ChannelHandle, then Drain.  A batch
            // handle resolves its remote registration once and is intended
            // for a producer's homogeneous payload stream; using it for this
            // standalone barrier caused channel Drain completion timeouts.
            backend.hcomm.Init(nullptr, 0);
            const auto result = backend.hcomm.template AtomicFAA<
                std::uint64_t>(
                handle, reinterpret_cast<__gm__ void*>(address),
                reinterpret_cast<__gm__ void*>(fetch), 1);
            if (!detail::checked_result(
                context, result, TransportCommandOpcode::kBarrier,
                TransportTeam::kWorld, static_cast<int>(peer),
                static_cast<int>(peer), 0)) {
                return;
            }
        }
        if (!backend.drain(context, table,
                           TransportCommandOpcode::kBarrier))
            return;
        const auto begin = static_cast<std::uint64_t>(clock());
        const auto retries =
            simt::load_observed(&state->default_retry_limit) == 0 ?
                std::uint64_t{1000000} :
                simt::load_observed(&state->default_retry_limit);
        std::uint64_t retry = 0;
        while (pending != 0) {
            simt::poll_nop();
            for (std::uint32_t peer = 0; peer < members; ++peer) {
                if ((pending & (std::uint64_t{1} << peer)) == 0)
                    continue;
                const auto address =
                    simt::load_observed(&table->local_sync_base) +
                    (row + peer) * sizeof(std::uint64_t);
                if (simt::load_observed(
                        reinterpret_cast<__gm__ std::uint64_t*>(address)) >=
                    generation)
                    pending &= ~(std::uint64_t{1} << peer);
            }
            if (pending == 0)
                break;
            ++retry;
            const bool timed_out = timeout_cycles == 0 ?
                retry >= retries :
                static_cast<std::uint64_t>(clock()) - begin >= timeout_cycles;
            if (timed_out) {
                for (std::uint32_t peer = 0; peer < members; ++peer) {
                    if ((pending & (std::uint64_t{1} << peer)) == 0)
                        continue;
                    (void)detail::record_error(
                        context, DeviceTransportError::kCompletionTimeout,
                        TransportCommandOpcode::kBarrier,
                        TransportTeam::kWorld, static_cast<int>(peer),
                        static_cast<int>(peer), 0);
                }
                return;
            }
        }
        simt::system_fence();
    }
    simt::store_published(&state->barrier_generation, generation);
    simt::system_fence();
}

}  // namespace deep_ep::ascend::transport::device

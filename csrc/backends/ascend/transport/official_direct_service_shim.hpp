#pragma once

#include <cstdint>

#include "aicore_intrinsics.hpp"
#include "transport_commands.hpp"

namespace deep_ep::ascend::transport::service {

namespace detail {

__aicore__ inline __gm__ TransportCommandQueue* command_queue(
    const DeviceTransportContext& context) {
    if (context.abi_version != kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(DeviceTransportContext) ||
        context.backend_context == 0)
        return nullptr;
    auto* staged = reinterpret_cast<__gm__ StagedTransportContext*>(
        context.backend_context);
    if (staged->command_queue == 0)
        return nullptr;
    return reinterpret_cast<__gm__ TransportCommandQueue*>(
        staged->command_queue);
}

__aicore__ inline __gm__ StagedTransportContext* staged_context(
    const DeviceTransportContext& context) {
    return reinterpret_cast<__gm__ StagedTransportContext*>(
        context.backend_context);
}

__aicore__ inline __gm__ DeviceTransportDiagnostic* diagnostic(
    __gm__ TransportCommandQueue* queue) {
    return queue == nullptr ? nullptr :
        reinterpret_cast<__gm__ DeviceTransportDiagnostic*>(
            queue->diagnostic);
}

template <bool ProfileEnabled = true>
__aicore__ inline __gm__ TransportStageProfile* profile_buffer(
    const DeviceTransportContext& context) {
    if constexpr (!ProfileEnabled)
        return nullptr;
    auto* staged = staged_context(context);
    if (staged == nullptr)
        return nullptr;
    aicore::flush_cacheline(staged);
    if (staged->abi_version != kTransportCommandAbiVersion ||
        staged->struct_size != sizeof(StagedTransportContext) ||
        staged->cann_compatibility != kStagedTransportCannCompatibility ||
        staged->stage_profile == 0 ||
        staged->stage_profile_bytes != sizeof(TransportStageProfile))
        return nullptr;
    return reinterpret_cast<__gm__ TransportStageProfile*>(
        staged->stage_profile);
}

__aicore__ inline void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    TransportTeam transport_team, int peer, int world_peer,
    std::uint32_t channel, std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    auto* output = detail::diagnostic(queue);
    if (output == nullptr)
        return;
    aicore::flush_cacheline(output);
    if (output->error != DeviceTransportError::kNone)
        return;
    output->command_index = command_index;
    output->opcode = opcode;
    output->peer = static_cast<std::uint32_t>(peer);
    output->world_peer = world_peer;
    output->team = transport_team;
    output->channel = channel;
    output->backend_status = backend_status;
    output->reserved = reserved;
    output->error = error;
    aicore::system_fence();
    aicore::flush_cacheline(output);
}

__aicore__ inline void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    int world_peer, std::uint32_t channel,
    std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    record_error(queue, error, command_index, opcode,
                 TransportTeam::kWorld, world_peer, world_peer, channel,
                 backend_status, reserved);
}

__aicore__ inline __gm__ TransportServiceState* service_state(
    __gm__ TransportCommandQueue* queue) {
    return queue == nullptr ? nullptr :
        reinterpret_cast<__gm__ TransportServiceState*>(
            queue->service_state);
}

__aicore__ inline void complete_profile(
    const DeviceTransportContext& context, std::uint64_t generation) {
    auto* profile = profile_buffer(context);
    if (profile == nullptr || profile->generation != generation)
        return;
    profile->completion_generation = generation;
    aicore::system_fence();
    for (std::uint32_t phase = 0;
         phase < kTransportProfileBarrierPhaseCount; ++phase)
        for (std::uint32_t peer = 0;
             peer < kTransportProfileMaxBarrierPeers; ++peer)
            if (profile->barrier_peers[phase][peer].valid != 0)
                aicore::flush_cacheline(
                    &profile->barrier_peers[phase][peer]);
    aicore::flush_stage_profile_header(profile);
}

}  // namespace detail

// In official-direct mode the staged AICore service is not executed. Keep the
// launch-side symbols available so operator code can compile unchanged; reset
// only maintains the generation ABI used by the direct facade.
template <bool ProfileEnabled = true>
__aicore__ inline bool reset(
    const DeviceTransportContext& context, std::uint64_t generation) {
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    auto* output = detail::diagnostic(queue);
    if (queue == nullptr || state == nullptr || output == nullptr)
        return false;
    aicore::store_device(&state->consumed_generation, std::uint64_t{0});
    aicore::store_device(&queue->count, std::uint32_t{0});
    aicore::store_device(&output->abi_version, kTransportCommandAbiVersion);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(&output->error),
        static_cast<std::uint32_t>(DeviceTransportError::kNone));
    aicore::store_device(&output->generation, generation);
    aicore::store_device(&queue->generation, generation);
    aicore::system_fence();
    return true;
}

template <bool ProfileEnabled = true>
__aicore__ inline void execute(
    const DeviceTransportContext& context) {
    // The producer kernel resets the staged command queue for the generation
    // ABI even though official-direct mode posts work directly to Hcomm.  A
    // barrier therefore has no queued commands, but its continuation still
    // expects the empty generation to be consumed.  Publish that completion
    // here without launching the staged transport service.
    auto* queue = detail::command_queue(context);
    auto* state = detail::service_state(queue);
    auto* output = detail::diagnostic(queue);
    if (queue == nullptr || state == nullptr || output == nullptr)
        return;
    aicore::flush_cacheline(queue);
    aicore::flush_cacheline(output);
    const auto count = aicore::load_published(&queue->count);
    const auto generation = aicore::load_published(&queue->generation);
    const auto error = static_cast<DeviceTransportError>(
        aicore::load_published(reinterpret_cast<__gm__ const std::uint32_t*>(
            &output->error)));
    if (count != 0 || error != DeviceTransportError::kNone)
        return;
    aicore::store_device(
        &state->consumed_generation, generation);
    aicore::system_fence();
}

template <bool ProfileEnabled = true>
__aicore__ inline void record_stage_start(
    const DeviceTransportContext& context,
    TransportProfileOperation operation, std::uint64_t generation,
    std::uint32_t stage, std::uint32_t block,
    std::uint32_t block_count) {
    if constexpr (!ProfileEnabled)
        return;
    auto* profile = detail::profile_buffer<ProfileEnabled>(context);
    if (profile == nullptr || stage >= kTransportProfileStageCount ||
        block >= kTransportProfileMaxBlocks)
        return;
    profile->operation = operation;
    profile->generation = generation;
    auto* cycles = &profile->stages[stage].blocks[block];
    cycles->start = record_transport_stage_start(
        cycles->start, static_cast<std::uint64_t>(AscendC::GetSystemCycle()));
    if (block == 0)
        profile->stages[stage].block_count = block_count;
}

template <bool ProfileEnabled = true>
__aicore__ inline void record_stage_end(
    const DeviceTransportContext& context,
    TransportProfileOperation operation, std::uint64_t generation,
    std::uint32_t stage, std::uint32_t block, bool complete_operation) {
    if constexpr (!ProfileEnabled)
        return;
    auto* profile = detail::profile_buffer<ProfileEnabled>(context);
    if (profile == nullptr || profile->generation != generation ||
        stage >= kTransportProfileStageCount ||
        block >= kTransportProfileMaxBlocks)
        return;
    auto* cycles = &profile->stages[stage].blocks[block];
    cycles->end = record_transport_stage_end(
        cycles->start, cycles->end,
        static_cast<std::uint64_t>(AscendC::GetSystemCycle()));
    aicore::system_fence();
    aicore::flush_cacheline(&profile->stages[stage].blocks[block]);
    if (block == 0)
        aicore::flush_cacheline(&profile->stages[stage]);
    if (block == 0 && complete_operation)
        detail::complete_profile(context, generation);
}

}  // namespace deep_ep::ascend::transport::service

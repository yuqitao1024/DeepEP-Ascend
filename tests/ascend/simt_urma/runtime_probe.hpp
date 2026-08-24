#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "csrc/backends/ascend/transport/device_transport.hpp"
#include "csrc/backends/ascend/transport/transport_commands.hpp"
#include "csrc/backends/ascend/transport/types.hpp"

namespace deep_ep::ascend::transport::runtime_probe {

inline constexpr std::uint32_t kMaxQueueWrapBatchOperations = 64;
inline constexpr std::uint32_t kMixedProfileCommandCount = 6;
inline constexpr std::uint32_t kMixedProfilePutCommandCount = 1;
inline constexpr std::uint64_t kMixedProfilePayloadBytes = 24;

constexpr std::uint32_t queue_wrap_batch_operations(
    std::uint32_t command_capacity) noexcept {
    constexpr std::uint32_t kBarrierCommandCount = 2;
    if (command_capacity <= kBarrierCommandCount)
        return 0;
    const auto available = command_capacity - kBarrierCommandCount;
    return available < kMaxQueueWrapBatchOperations ?
        available : kMaxQueueWrapBatchOperations;
}

enum class RuntimeCase : std::uint32_t {
    kPut,
    kPutValue64,
    kFaa64,
    kSignal,
    kSignalSet,
    kFlush,
    kPayloadSignalOrder,
    kBarrierRepeat,
    kQueueWrap,
    kProfileMixed,
    kPhaseBoundary,
};

constexpr bool runtime_case_records_transport_profile(
    RuntimeCase runtime_case) noexcept {
    return runtime_case != RuntimeCase::kPhaseBoundary;
}

template <typename Stream, typename Reset, typename Synchronize, typename Launch>
bool reset_synchronize_and_launch(
    bool synchronize_ranks, Stream stream, Reset&& reset,
    Synchronize&& synchronize, Launch&& launch) {
    if (!reset())
        return false;
    if (synchronize_ranks && !synchronize(stream))
        return false;
    return launch(stream);
}

template <typename Transport>
DEEP_EP_ASCEND_SIMT_CALLEE void enqueue_profile_mixed_final_commands(
    Transport& transport, int peer, DeviceAddress destination,
    DeviceAddress source, std::uint64_t source_value,
    DeviceAddress atomic_value, const TeamPeer& signal_route,
    std::uint64_t generation, std::uint64_t barrier_timeout) {
    transport.put(
        TransportTeam::kWorld, peer, destination, source,
        sizeof(std::uint64_t), CooperationScope::kParticipant,
        MemorySegment::kDevice, kDefaultOptions, RemoteAction::none());
    transport.put_value(
        TransportTeam::kWorld, peer, destination, source_value,
        sizeof(std::uint64_t), kDefaultOptions);
    transport.remote_add_release(
        TransportTeam::kWorld, peer, atomic_value, 1);
    transport.signal(
        signal_route.team, signal_route.peer,
        RemoteAction::signal_set(0, generation));
    transport.device_barrier(
        kWorldTeamMask, kNullDeviceAddress, barrier_timeout);
    transport.flush(CooperationScope::kParticipant);
}

struct alignas(64) RuntimeState {
    std::uint64_t source = 0;
    std::uint64_t destination = 0;
    std::uint64_t atomic_value = 0;
    std::uint64_t observed = 0;
    std::uint64_t phase_payload = 0;
    std::uint32_t phase_sequence = 0;
    std::uint32_t success = 0;
    std::uint64_t generation = 0;
    DeviceTransportDiagnostic diagnostic{};
};

static_assert(std::is_trivially_copyable_v<RuntimeState>);
static_assert(alignof(RuntimeState) == 64);

}  // namespace deep_ep::ascend::transport::runtime_probe

extern "C" int deep_ep_ascend_urma_launch_runtime_probe(
    deep_ep::ascend::transport::runtime_probe::RuntimeState* state,
    deep_ep::ascend::transport::DeviceTransportContext context,
    deep_ep::ascend::transport::runtime_probe::RuntimeCase runtime_case,
    std::uint32_t peer, std::uint64_t generation,
    std::uint32_t operation_count, bool finalize_profile_pressure,
    void* stream);

extern "C" int deep_ep_ascend_urma_run_case(
    std::int64_t communicator_handle, std::uint32_t rank,
    std::uint32_t world_size, const char* case_name,
    std::uint64_t iterations, char* error, std::size_t error_capacity);

extern "C" int deep_ep_ascend_urma_run_local_phase_boundary(
    char* error, std::size_t error_capacity);

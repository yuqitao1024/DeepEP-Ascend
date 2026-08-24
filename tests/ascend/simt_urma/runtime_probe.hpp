#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "csrc/backends/ascend/transport/transport_commands.hpp"
#include "csrc/backends/ascend/transport/types.hpp"

namespace deep_ep::ascend::transport::runtime_probe {

inline constexpr std::uint32_t kMaxQueueWrapBatchOperations = 64;

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
    kPhaseBoundary,
};

constexpr bool runtime_case_records_transport_profile(
    RuntimeCase runtime_case) noexcept {
    return runtime_case != RuntimeCase::kPhaseBoundary;
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
    std::uint32_t operation_count, void* stream);

extern "C" int deep_ep_ascend_urma_run_case(
    std::int64_t communicator_handle, std::uint32_t rank,
    std::uint32_t world_size, const char* case_name,
    std::uint64_t iterations, char* error, std::size_t error_capacity);

extern "C" int deep_ep_ascend_urma_run_local_phase_boundary(
    char* error, std::size_t error_capacity);

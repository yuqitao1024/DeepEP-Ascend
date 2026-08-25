#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "csrc/backends/ascend/transport/transport_commands.hpp"

namespace deep_ep::ascend::hccs_benchmark {

struct alignas(64) BenchmarkControl {
    std::uint64_t producer_cycles = 0;
    std::uint64_t service_cycles = 0;
    std::uint64_t total_cycles = 0;
    transport::DeviceTransportDiagnostic diagnostic{};
};

static_assert(std::is_trivially_copyable_v<BenchmarkControl>);
static_assert(alignof(BenchmarkControl) == 64);

struct BenchmarkCycles {
    std::uint64_t producer_cycles = 0;
    std::uint64_t service_cycles = 0;
    std::uint64_t total_cycles = 0;
};

}  // namespace deep_ep::ascend::hccs_benchmark

extern "C" int deep_ep_hccs_benchmark_launch(
    void* window, std::uint64_t peer_stride,
    const std::uint64_t* peer_bytes,
    deep_ep::ascend::transport::DeviceTransportContext context,
    std::uint64_t generation,
    deep_ep::ascend::hccs_benchmark::BenchmarkControl* control,
    void* stream);

extern "C" std::uint64_t
deep_ep_hccs_benchmark_representative_record_bytes();

extern "C" void* deep_ep_hccs_benchmark_create(
    std::int64_t communicator, std::uint32_t rank,
    std::uint32_t world_size, std::uint64_t peer_stride,
    char* error, std::size_t error_capacity);

extern "C" int deep_ep_hccs_benchmark_reset(
    void* handle, char* error, std::size_t error_capacity);

extern "C" int deep_ep_hccs_benchmark_run(
    void* handle, const std::uint64_t* peer_bytes,
    std::uint64_t generation,
    deep_ep::ascend::hccs_benchmark::BenchmarkCycles* cycles,
    char* error, std::size_t error_capacity);

extern "C" int deep_ep_hccs_benchmark_verify(
    void* handle, const std::uint64_t* expected_bytes_by_sender,
    char* error, std::size_t error_capacity);

extern "C" int deep_ep_hccs_benchmark_destroy(
    void* handle, char* error, std::size_t error_capacity);

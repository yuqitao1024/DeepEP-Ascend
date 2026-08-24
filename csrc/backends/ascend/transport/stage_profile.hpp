#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace deep_ep::ascend::transport {

inline constexpr std::uint32_t kTransportStageProfileAbiVersion = 1;
inline constexpr std::uint32_t kTransportProfileStageCount = 16;
inline constexpr std::uint32_t kTransportProfileMaxBlocks = 72;

enum class TransportProfileOperation : std::uint32_t {
    kNone,
    kDispatch,
    kCombine,
    kBarrier,
};

struct TransportStageBlockCycles {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

struct TransportStageCycles {
    std::uint32_t block_count = 0;
    std::uint32_t reserved = 0;
    TransportStageBlockCycles blocks[kTransportProfileMaxBlocks]{};
};

struct alignas(64) TransportStageProfile {
    std::uint32_t abi_version = kTransportStageProfileAbiVersion;
    std::uint32_t struct_size = sizeof(TransportStageProfile);
    TransportProfileOperation operation = TransportProfileOperation::kNone;
    std::uint32_t flags = 0;
    std::uint64_t generation = 0;
    std::uint64_t completion_generation = 0;
    std::uint64_t valid_stage_mask = 0;
    std::uint32_t command_count = 0;
    std::uint32_t put_command_count = 0;
    std::uint32_t sq_depth = 0;
    std::uint32_t cq_depth = 0;
    std::uint32_t sq_high_watermark = 0;
    std::uint32_t cq_high_watermark = 0;
    std::uint64_t command_bytes = 0;
    std::uint64_t service_start_cycles = 0;
    std::uint64_t service_end_cycles = 0;
    std::uint64_t wait_cycles = 0;
    std::uint64_t reserved[3]{};
    TransportStageCycles stages[kTransportProfileStageCount]{};
};

static_assert(sizeof(TransportStageBlockCycles) == 16);
static_assert(alignof(TransportStageProfile) == 64);
static_assert(sizeof(TransportStageProfile) % 64 == 0);
static_assert(std::is_trivially_copyable_v<TransportStageBlockCycles>);
static_assert(std::is_trivially_copyable_v<TransportStageCycles>);
static_assert(std::is_trivially_copyable_v<TransportStageProfile>);

}  // namespace deep_ep::ascend::transport

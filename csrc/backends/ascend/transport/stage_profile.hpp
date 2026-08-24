#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace deep_ep::ascend::transport {

inline constexpr std::uint32_t kTransportStageProfileAbiVersion = 1;
inline constexpr std::uint32_t kTransportProfileStageCount = 16;
inline constexpr std::uint32_t kTransportProfileMaxBlocks = 72;
inline constexpr std::size_t kTransportStageProfileCacheLineBytes = 64;

enum class TransportProfileOperation : std::uint32_t {
    kNone,
    kDispatch,
    kCombine,
    kBarrier,
};

struct alignas(64) TransportStageBlockCycles {
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

inline constexpr std::size_t kTransportStageProfileHeaderBytes =
    offsetof(TransportStageProfile, stages);
inline constexpr std::size_t kTransportStageProfileHeaderCacheLineCount =
    (kTransportStageProfileHeaderBytes +
     kTransportStageProfileCacheLineBytes - 1) /
    kTransportStageProfileCacheLineBytes;

enum class TransportStageProfileMaskStatus : std::uint32_t {
    kValid,
    kNoStages,
    kInvalidMask,
    kPartialMask,
    kInvalidOperation,
};

inline constexpr std::uint64_t kTransportStageProfileFullMask = 1;
inline constexpr std::uint64_t kTransportDispatchPipelineStageMask =
    ((std::uint64_t{1} << 14U) - 1U) & ~kTransportStageProfileFullMask;
inline constexpr std::uint64_t kTransportCombinePipelineStageMask =
    ((std::uint64_t{1} << 12U) - 1U) & ~kTransportStageProfileFullMask;

inline constexpr TransportStageProfileMaskStatus stage_profile_mask_status(
    TransportProfileOperation operation, std::uint64_t stage_mask) {
    const std::uint64_t pipeline_mask = operation ==
            TransportProfileOperation::kDispatch ?
        kTransportDispatchPipelineStageMask : operation ==
            TransportProfileOperation::kCombine ?
        kTransportCombinePipelineStageMask : 0;
    if (pipeline_mask == 0)
        return TransportStageProfileMaskStatus::kInvalidOperation;
    if (stage_mask == 0)
        return TransportStageProfileMaskStatus::kNoStages;
    const std::uint64_t allowed_mask =
        pipeline_mask | kTransportStageProfileFullMask;
    if ((stage_mask & ~allowed_mask) != 0)
        return TransportStageProfileMaskStatus::kInvalidMask;
    if (stage_mask == kTransportStageProfileFullMask)
        return TransportStageProfileMaskStatus::kValid;
    if ((stage_mask & kTransportStageProfileFullMask) != 0)
        return TransportStageProfileMaskStatus::kPartialMask;
    return stage_mask == pipeline_mask ?
        TransportStageProfileMaskStatus::kValid :
        TransportStageProfileMaskStatus::kPartialMask;
}

inline constexpr bool transport_stage_profile_service_cycles_valid(
    std::uint64_t service_start_cycles, std::uint64_t service_end_cycles,
    std::uint64_t wait_cycles) {
    return service_start_cycles != 0 && service_end_cycles != 0 &&
        service_end_cycles >= service_start_cycles &&
        wait_cycles <= service_end_cycles - service_start_cycles;
}

struct TransportStageProfilePhaseCycles {
    std::uint64_t producer = 0;
    std::uint64_t publication = 0;
    std::uint64_t service_submit = 0;
    std::uint64_t cq_wait = 0;
    std::uint64_t consumer_wait = 0;
    std::uint64_t consumer_compute = 0;
    std::uint64_t epilogue = 0;
};

inline TransportStageProfilePhaseCycles derive_stage_profile_phase_cycles(
    TransportProfileOperation operation, std::uint64_t stage_mask,
    const std::uint64_t* stage_spans, std::uint64_t service_start_cycles,
    std::uint64_t service_end_cycles, std::uint64_t wait_cycles) {
    TransportStageProfilePhaseCycles phases{};
    if (stage_spans == nullptr ||
        stage_profile_mask_status(operation, stage_mask) !=
            TransportStageProfileMaskStatus::kValid ||
        !transport_stage_profile_service_cycles_valid(
            service_start_cycles, service_end_cycles, wait_cycles))
        return phases;
    if (stage_mask == kTransportStageProfileFullMask) {
        // A one-block launch has no independently measurable pipeline phases.
        phases.producer = stage_spans[0];
        return phases;
    }

    const auto sum_stages = [stage_spans](
        std::uint32_t first, std::uint32_t last) {
        std::uint64_t total = 0;
        for (auto stage = first; stage <= last; ++stage)
            total += stage_spans[stage];
        return total;
    };
    const std::uint64_t service_cycles =
        service_end_cycles - service_start_cycles;
    const std::uint64_t release_cycles = stage_spans[5];
    phases.producer = sum_stages(1, 4);
    phases.publication = release_cycles > service_cycles ?
        release_cycles - service_cycles : 0;
    phases.service_submit = service_cycles - wait_cycles;
    phases.cq_wait = wait_cycles;
    phases.consumer_wait = sum_stages(6, 8);
    phases.consumer_compute = operation == TransportProfileOperation::kDispatch ?
        sum_stages(9, 12) : sum_stages(9, 10);
    phases.epilogue = stage_spans[
        operation == TransportProfileOperation::kDispatch ? 13 : 11];
    return phases;
}

static_assert(sizeof(TransportStageBlockCycles) == 64);
static_assert(alignof(TransportStageProfile) == 64);
static_assert(sizeof(TransportStageProfile) % 64 == 0);
static_assert(kTransportStageProfileHeaderBytes ==
              2 * kTransportStageProfileCacheLineBytes);
static_assert(kTransportStageProfileHeaderCacheLineCount == 2);
static_assert(std::is_trivially_copyable_v<TransportStageBlockCycles>);
static_assert(std::is_trivially_copyable_v<TransportStageCycles>);
static_assert(std::is_trivially_copyable_v<TransportStageProfile>);

}  // namespace deep_ep::ascend::transport

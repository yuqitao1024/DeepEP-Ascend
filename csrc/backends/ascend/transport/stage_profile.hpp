#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
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

struct TransportQueueDepthSnapshot {
    std::uint32_t sq_depth = 0;
    std::uint32_t cq_depth = 0;
};

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE
#define DEEP_EP_ASCEND_PROFILE_INLINE __aicore__ inline
#else
#define DEEP_EP_ASCEND_PROFILE_INLINE inline constexpr
#endif

struct TransportServiceIntervalAccumulation {
    bool valid = false;
    std::uint32_t command_count = 0;
    std::uint64_t service_start_cycles = 0;
    std::uint64_t service_end_cycles = 0;
};

DEEP_EP_ASCEND_PROFILE_INLINE TransportServiceIntervalAccumulation
accumulate_transport_service_interval(
    std::uint32_t existing_command_count,
    std::uint64_t existing_service_start_cycles,
    std::uint64_t existing_service_end_cycles,
    std::uint32_t command_begin, std::uint32_t command_end,
    std::uint64_t service_start_cycles,
    std::uint64_t service_end_cycles) noexcept {
    if (command_begin > command_end ||
        service_start_cycles == 0 ||
        service_end_cycles < service_start_cycles)
        return {};
    const auto command_delta = command_end - command_begin;
    if (existing_command_count >
        std::numeric_limits<std::uint32_t>::max() - command_delta)
        return {};
    const auto accumulated_start = existing_service_start_cycles == 0 ||
            service_start_cycles < existing_service_start_cycles ?
        service_start_cycles : existing_service_start_cycles;
    const auto accumulated_end = service_end_cycles >
            existing_service_end_cycles ?
        service_end_cycles : existing_service_end_cycles;
    return {true, static_cast<std::uint32_t>(
                      existing_command_count + command_delta),
            accumulated_start, accumulated_end};
}

DEEP_EP_ASCEND_PROFILE_INLINE std::uint64_t record_transport_stage_start(
    std::uint64_t existing_start, std::uint64_t observed_start) noexcept {
    return existing_start == 0 && observed_start != 0 ?
        observed_start : existing_start;
}

DEEP_EP_ASCEND_PROFILE_INLINE std::uint64_t record_transport_stage_end(
    std::uint64_t start, std::uint64_t existing_end,
    std::uint64_t observed_end) noexcept {
    return start != 0 && observed_end >= start && observed_end > existing_end ?
        observed_end : existing_end;
}

#undef DEEP_EP_ASCEND_PROFILE_INLINE

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

enum class TransportStageProfileCommandMetricsStatus : std::uint32_t {
    kValid,
    kPutCommandCountExceedsCommandCount,
    kSqDepthExceedsHighWatermark,
    kCqDepthExceedsHighWatermark,
    kQueueDepthMismatch,
    kQueueHighWatermarkMismatch,
    kQueueActivityWithoutCommands,
    kPutCommandsWithoutPayload,
    kPayloadWithoutQueueActivity,
    kCompletedServiceHasOutstandingRequests,
};

inline constexpr TransportStageProfileCommandMetricsStatus
transport_stage_profile_command_metrics_status(
    const TransportStageProfile& profile, bool service_completed) noexcept {
    if (profile.put_command_count > profile.command_count)
        return TransportStageProfileCommandMetricsStatus::
            kPutCommandCountExceedsCommandCount;
    if (profile.sq_depth > profile.sq_high_watermark)
        return TransportStageProfileCommandMetricsStatus::
            kSqDepthExceedsHighWatermark;
    if (profile.cq_depth > profile.cq_high_watermark)
        return TransportStageProfileCommandMetricsStatus::
            kCqDepthExceedsHighWatermark;
    if (profile.sq_depth != profile.cq_depth)
        return TransportStageProfileCommandMetricsStatus::kQueueDepthMismatch;
    if (profile.sq_high_watermark != profile.cq_high_watermark)
        return TransportStageProfileCommandMetricsStatus::
            kQueueHighWatermarkMismatch;
    if (profile.command_count == 0 &&
        (profile.put_command_count != 0 || profile.command_bytes != 0 ||
         profile.sq_depth != 0 || profile.cq_depth != 0 ||
         profile.sq_high_watermark != 0 ||
         profile.cq_high_watermark != 0 || profile.wait_cycles != 0))
        return TransportStageProfileCommandMetricsStatus::
            kQueueActivityWithoutCommands;
    if (profile.put_command_count != 0 && profile.command_bytes == 0)
        return TransportStageProfileCommandMetricsStatus::
            kPutCommandsWithoutPayload;
    if (profile.command_bytes != 0 && profile.sq_high_watermark == 0)
        return TransportStageProfileCommandMetricsStatus::
            kPayloadWithoutQueueActivity;
    if (service_completed &&
        (profile.sq_depth != 0 || profile.cq_depth != 0))
        return TransportStageProfileCommandMetricsStatus::
            kCompletedServiceHasOutstandingRequests;
    return TransportStageProfileCommandMetricsStatus::kValid;
}

inline constexpr const char* transport_stage_profile_command_metrics_reason(
    TransportStageProfileCommandMetricsStatus status) noexcept {
    switch (status) {
        case TransportStageProfileCommandMetricsStatus::kValid:
            return nullptr;
        case TransportStageProfileCommandMetricsStatus::
                kPutCommandCountExceedsCommandCount:
            return "put_command_count_exceeds_command_count";
        case TransportStageProfileCommandMetricsStatus::
                kSqDepthExceedsHighWatermark:
            return "sq_depth_exceeds_high_watermark";
        case TransportStageProfileCommandMetricsStatus::
                kCqDepthExceedsHighWatermark:
            return "cq_depth_exceeds_high_watermark";
        case TransportStageProfileCommandMetricsStatus::kQueueDepthMismatch:
            return "queue_depth_mismatch";
        case TransportStageProfileCommandMetricsStatus::
                kQueueHighWatermarkMismatch:
            return "queue_high_watermark_mismatch";
        case TransportStageProfileCommandMetricsStatus::
                kQueueActivityWithoutCommands:
            return "queue_activity_without_commands";
        case TransportStageProfileCommandMetricsStatus::
                kPutCommandsWithoutPayload:
            return "put_commands_without_payload";
        case TransportStageProfileCommandMetricsStatus::
                kPayloadWithoutQueueActivity:
            return "payload_without_queue_activity";
        case TransportStageProfileCommandMetricsStatus::
                kCompletedServiceHasOutstandingRequests:
            return "completed_service_has_outstanding_requests";
    }
    return "invalid_command_metrics";
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
static_assert(std::is_trivially_copyable_v<TransportQueueDepthSnapshot>);

}  // namespace deep_ep::ascend::transport

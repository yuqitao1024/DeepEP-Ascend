#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace deep_ep::ascend::transport {

inline constexpr std::uint32_t kTransportStageProfileAbiVersion = 2;
inline constexpr std::uint32_t kTransportProfileStageCount = 16;
inline constexpr std::uint32_t kTransportProfileMaxBlocks = 72;
inline constexpr std::size_t kTransportStageProfileCacheLineBytes = 64;
inline constexpr std::uint32_t kTransportStageProfileReleaseAblation = 1U;

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
    std::uint64_t payload_command_cycles = 0;
    std::uint64_t control_command_cycles = 0;
    std::uint64_t flush_command_cycles = 0;
    std::uint64_t barrier_command_cycles = 0;
    std::uint64_t barrier_poll_cycles = 0;
    std::uint64_t reserved[7]{};
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

enum class TransportServiceCycleClass : std::uint32_t {
    kUnclassified,
    kPayload,
    kControl,
    kFlush,
    kBarrier,
    kBarrierPoll,
};


struct TransportServiceCycleBreakdown {
    std::uint64_t payload_command_cycles = 0;
    std::uint64_t control_command_cycles = 0;
    std::uint64_t flush_command_cycles = 0;
    std::uint64_t barrier_command_cycles = 0;
    std::uint64_t barrier_poll_cycles = 0;
};

struct TransportServiceCycleAccumulation {
    bool valid = false;
    TransportServiceCycleBreakdown cycles{};
};

DEEP_EP_ASCEND_PROFILE_INLINE std::uint64_t
accumulate_transport_service_counter(
    std::uint64_t existing, std::uint64_t start_cycles,
    std::uint64_t end_cycles) noexcept {
    if (start_cycles == 0 || end_cycles < start_cycles)
        return existing;
    const auto interval = end_cycles - start_cycles;
    if (existing > std::numeric_limits<std::uint64_t>::max() - interval)
        return existing;
    return existing + interval;
}

DEEP_EP_ASCEND_PROFILE_INLINE std::uint32_t
accumulate_transport_service_command_count(
    std::uint32_t existing, std::uint32_t command_begin,
    std::uint32_t command_end) noexcept {
    if (command_begin > command_end)
        return existing;
    const auto command_delta = command_end - command_begin;
    if (existing >
        std::numeric_limits<std::uint32_t>::max() - command_delta)
        return existing;
    return existing + command_delta;
}

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

DEEP_EP_ASCEND_PROFILE_INLINE TransportServiceCycleAccumulation
record_transport_service_cycles(
    TransportServiceCycleBreakdown existing,
    TransportServiceCycleClass cycle_class,
    std::uint64_t start_cycles, std::uint64_t end_cycles) noexcept {
    if (cycle_class == TransportServiceCycleClass::kUnclassified ||
        start_cycles == 0 || end_cycles < start_cycles)
        return {false, existing};
    const auto interval = end_cycles - start_cycles;
    std::uint64_t current = 0;
    switch (cycle_class) {
        case TransportServiceCycleClass::kUnclassified:
            break;
        case TransportServiceCycleClass::kPayload:
            current = existing.payload_command_cycles;
            break;
        case TransportServiceCycleClass::kControl:
            current = existing.control_command_cycles;
            break;
        case TransportServiceCycleClass::kFlush:
            current = existing.flush_command_cycles;
            break;
        case TransportServiceCycleClass::kBarrier:
            current = existing.barrier_command_cycles;
            break;
        case TransportServiceCycleClass::kBarrierPoll:
            current = existing.barrier_poll_cycles;
            break;
    }
    if (current > std::numeric_limits<std::uint64_t>::max() - interval)
        return {false, existing};
    const auto accumulated = current + interval;
    switch (cycle_class) {
        case TransportServiceCycleClass::kUnclassified:
            break;
        case TransportServiceCycleClass::kPayload:
            existing.payload_command_cycles = accumulated;
            break;
        case TransportServiceCycleClass::kControl:
            existing.control_command_cycles = accumulated;
            break;
        case TransportServiceCycleClass::kFlush:
            existing.flush_command_cycles = accumulated;
            break;
        case TransportServiceCycleClass::kBarrier:
            existing.barrier_command_cycles = accumulated;
            break;
        case TransportServiceCycleClass::kBarrierPoll:
            existing.barrier_poll_cycles = accumulated;
            break;
    }
    return {true, existing};
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
inline constexpr std::uint64_t kTransportDispatchReleaseAblationStageMask =
    ((std::uint64_t{1} << 16U) - 1U) & ~kTransportStageProfileFullMask;
inline constexpr std::uint64_t kTransportCombineReleaseAblationStageMask =
    ((std::uint64_t{1} << 15U) - 1U) & ~kTransportStageProfileFullMask;

inline constexpr std::uint64_t transport_stage_profile_pipeline_mask(
    TransportProfileOperation operation, bool release_ablation) noexcept {
    if (operation == TransportProfileOperation::kDispatch)
        return release_ablation ? kTransportDispatchReleaseAblationStageMask :
                                  kTransportDispatchPipelineStageMask;
    if (operation == TransportProfileOperation::kCombine)
        return release_ablation ? kTransportCombineReleaseAblationStageMask :
                                  kTransportCombinePipelineStageMask;
    return 0;
}

inline constexpr TransportStageProfileMaskStatus stage_profile_mask_status(
    TransportProfileOperation operation, std::uint64_t stage_mask) {
    const std::uint64_t pipeline_mask =
        transport_stage_profile_pipeline_mask(operation, false);
    const std::uint64_t release_ablation_mask =
        transport_stage_profile_pipeline_mask(operation, true);
    if (pipeline_mask == 0)
        return TransportStageProfileMaskStatus::kInvalidOperation;
    if (stage_mask == 0)
        return TransportStageProfileMaskStatus::kNoStages;
    const std::uint64_t allowed_mask =
        release_ablation_mask | kTransportStageProfileFullMask;
    if ((stage_mask & ~allowed_mask) != 0)
        return TransportStageProfileMaskStatus::kInvalidMask;
    if (stage_mask == kTransportStageProfileFullMask)
        return TransportStageProfileMaskStatus::kValid;
    if ((stage_mask & kTransportStageProfileFullMask) != 0)
        return TransportStageProfileMaskStatus::kPartialMask;
    return stage_mask == pipeline_mask || stage_mask == release_ablation_mask ?
        TransportStageProfileMaskStatus::kValid :
        TransportStageProfileMaskStatus::kPartialMask;
}

inline constexpr bool transport_stage_profile_service_cycles_valid(
    std::uint64_t service_start_cycles, std::uint64_t service_end_cycles,
    std::uint64_t wait_cycles,
    std::uint64_t barrier_poll_cycles = 0) {
    if (service_start_cycles == 0 || service_end_cycles == 0 ||
        service_end_cycles < service_start_cycles)
        return false;
    const auto service_cycles = service_end_cycles - service_start_cycles;
    return wait_cycles <= service_cycles &&
        barrier_poll_cycles <= service_cycles - wait_cycles;
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
    kBarrierPollExceedsBarrierCommand,
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
         profile.cq_high_watermark != 0 || profile.wait_cycles != 0 ||
         profile.payload_command_cycles != 0 ||
         profile.control_command_cycles != 0 ||
         profile.flush_command_cycles != 0 ||
         profile.barrier_command_cycles != 0 ||
         profile.barrier_poll_cycles != 0))
        return TransportStageProfileCommandMetricsStatus::
            kQueueActivityWithoutCommands;
    if (profile.put_command_count != 0 && profile.command_bytes == 0)
        return TransportStageProfileCommandMetricsStatus::
            kPutCommandsWithoutPayload;
    if (profile.command_bytes != 0 && profile.sq_high_watermark == 0)
        return TransportStageProfileCommandMetricsStatus::
            kPayloadWithoutQueueActivity;
    if (profile.barrier_poll_cycles > profile.barrier_command_cycles)
        return TransportStageProfileCommandMetricsStatus::
            kBarrierPollExceedsBarrierCommand;
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
                kBarrierPollExceedsBarrierCommand:
            return "barrier_poll_exceeds_barrier_command";
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
    std::uint64_t barrier_wait = 0;
    std::uint64_t consumer_wait = 0;
    std::uint64_t consumer_compute = 0;
    std::uint64_t epilogue = 0;
};

inline TransportStageProfilePhaseCycles derive_stage_profile_phase_cycles(
    TransportProfileOperation operation, std::uint64_t stage_mask,
    const std::uint64_t* stage_spans, std::uint64_t service_start_cycles,
    std::uint64_t service_end_cycles, std::uint64_t wait_cycles,
    std::uint64_t barrier_poll_cycles = 0) {
    TransportStageProfilePhaseCycles phases{};
    if (stage_spans == nullptr ||
        stage_profile_mask_status(operation, stage_mask) !=
            TransportStageProfileMaskStatus::kValid ||
        !transport_stage_profile_service_cycles_valid(
            service_start_cycles, service_end_cycles, wait_cycles,
            barrier_poll_cycles))
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
    const bool release_ablation = stage_mask ==
        transport_stage_profile_pipeline_mask(operation, true);
    const std::uint64_t release_cycles = stage_spans[5] +
        (release_ablation ? stage_spans[
            operation == TransportProfileOperation::kDispatch ? 14 : 12] : 0) +
        (release_ablation ? stage_spans[
            operation == TransportProfileOperation::kDispatch ? 15 : 13] : 0) +
        (release_ablation &&
             operation == TransportProfileOperation::kCombine ?
             stage_spans[14] : 0);
    phases.producer = sum_stages(1, 4);
    phases.publication = release_cycles > service_cycles ?
        release_cycles - service_cycles : 0;
    phases.service_submit =
        service_cycles - wait_cycles - barrier_poll_cycles;
    phases.cq_wait = wait_cycles;
    phases.barrier_wait = barrier_poll_cycles;
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
              3 * kTransportStageProfileCacheLineBytes);
static_assert(kTransportStageProfileHeaderCacheLineCount == 3);
static_assert(std::is_trivially_copyable_v<TransportStageBlockCycles>);
static_assert(std::is_trivially_copyable_v<TransportStageCycles>);
static_assert(std::is_trivially_copyable_v<TransportStageProfile>);
static_assert(std::is_trivially_copyable_v<TransportQueueDepthSnapshot>);

}  // namespace deep_ep::ascend::transport

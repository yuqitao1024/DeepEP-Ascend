#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace deep_ep::ascend::runtime {

enum class HostTimelinePhase : std::uint8_t {
    kDispatchPrelaunchSetup,
    kDispatchSynchronize,
    kDispatchCountsToHost,
    kDispatchHostPrefix,
    kDispatchPrefixToDevice,
    kDispatchOutputAllocation,
    kDispatchEpilogueSetup,
    kDispatchEpilogueSubmit,
    kDispatchCompletionRecord,
    kDispatchCompletionWait,
    kCombinePrelaunchSetup,
    kCombineHandleToHost,
    kCombineMetadataToHost,
    kCombineHostValidation,
    kCombineStreamSetup,
    kCombineOutputAllocation,
    kCombineSubmit,
    kCombineCompletionRecord,
    kCombineCompletionWait,
    kCount,
};

inline constexpr const char* host_timeline_phase_name(
    HostTimelinePhase phase) noexcept {
    switch (phase) {
        case HostTimelinePhase::kDispatchPrelaunchSetup:
            return "dispatch_prelaunch_setup";
        case HostTimelinePhase::kDispatchSynchronize:
            return "dispatch_synchronize";
        case HostTimelinePhase::kDispatchCountsToHost:
            return "dispatch_counts_to_host";
        case HostTimelinePhase::kDispatchHostPrefix:
            return "dispatch_host_prefix";
        case HostTimelinePhase::kDispatchPrefixToDevice:
            return "dispatch_prefix_to_device";
        case HostTimelinePhase::kDispatchOutputAllocation:
            return "dispatch_output_allocation";
        case HostTimelinePhase::kDispatchEpilogueSetup:
            return "dispatch_epilogue_setup";
        case HostTimelinePhase::kDispatchEpilogueSubmit:
            return "dispatch_epilogue_submit";
        case HostTimelinePhase::kDispatchCompletionRecord:
            return "dispatch_completion_record";
        case HostTimelinePhase::kDispatchCompletionWait:
            return "dispatch_completion_wait";
        case HostTimelinePhase::kCombinePrelaunchSetup:
            return "combine_prelaunch_setup";
        case HostTimelinePhase::kCombineHandleToHost:
            return "combine_handle_to_host";
        case HostTimelinePhase::kCombineMetadataToHost:
            return "combine_metadata_to_host";
        case HostTimelinePhase::kCombineHostValidation:
            return "combine_host_validation";
        case HostTimelinePhase::kCombineStreamSetup:
            return "combine_stream_setup";
        case HostTimelinePhase::kCombineOutputAllocation:
            return "combine_output_allocation";
        case HostTimelinePhase::kCombineSubmit:
            return "combine_submit";
        case HostTimelinePhase::kCombineCompletionRecord:
            return "combine_completion_record";
        case HostTimelinePhase::kCombineCompletionWait:
            return "combine_completion_wait";
        case HostTimelinePhase::kCount:
            break;
    }
    return "unknown";
}

inline std::uint64_t host_timestamp_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct HostTimelineProfile {
    static constexpr std::size_t kPhaseCount =
        static_cast<std::size_t>(HostTimelinePhase::kCount);

    std::uint64_t generation = 0;
    std::array<std::uint64_t, kPhaseCount> phase_durations_ns{};

    void reset(std::uint64_t next_generation) noexcept {
        generation = next_generation;
        phase_durations_ns.fill(0);
    }

    bool bind_generation(std::uint64_t next_generation) noexcept {
        if (next_generation == 0 || generation != 0)
            return false;
        generation = next_generation;
        return true;
    }

    bool record(
        HostTimelinePhase phase, std::uint64_t start_ns,
        std::uint64_t end_ns) noexcept {
        const auto index = static_cast<std::size_t>(phase);
        if (index >= phase_durations_ns.size() || end_ns < start_ns)
            return false;
        const auto duration = end_ns - start_ns;
        auto& accumulated = phase_durations_ns[index];
        if (duration > std::numeric_limits<std::uint64_t>::max() - accumulated)
            accumulated = std::numeric_limits<std::uint64_t>::max();
        else
            accumulated += duration;
        return true;
    }

    std::uint64_t phase_ns(HostTimelinePhase phase) const noexcept {
        const auto index = static_cast<std::size_t>(phase);
        return index < phase_durations_ns.size() ?
            phase_durations_ns[index] : 0;
    }

    std::uint64_t total_ns() const noexcept {
        std::uint64_t total = 0;
        for (const auto duration : phase_durations_ns) {
            if (duration > std::numeric_limits<std::uint64_t>::max() - total)
                return std::numeric_limits<std::uint64_t>::max();
            total += duration;
        }
        return total;
    }
};

}  // namespace deep_ep::ascend::runtime

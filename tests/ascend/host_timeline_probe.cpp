#include <cstdint>

#include "csrc/backends/ascend/runtime/host_timeline.hpp"

int main() {
    using deep_ep::ascend::runtime::HostTimelinePhase;
    using deep_ep::ascend::runtime::HostTimelineProfile;

    HostTimelineProfile profile{};
    profile.reset(0);
    if (profile.generation != 0 || profile.total_ns() != 0)
        return 1;
    if (!profile.record(HostTimelinePhase::kDispatchPrelaunchSetup, 10, 40) ||
        !profile.record(HostTimelinePhase::kDispatchSynchronize, 100, 160) ||
        !profile.record(HostTimelinePhase::kDispatchSynchronize, 200, 225) ||
        !profile.record(HostTimelinePhase::kDispatchOutputAllocation, 300, 340) ||
        !profile.record(HostTimelinePhase::kDispatchCompletionRecord, 400, 430) ||
        !profile.record(HostTimelinePhase::kCombinePrelaunchSetup, 480, 495) ||
        !profile.record(HostTimelinePhase::kCombineHandleToHost, 500, 520) ||
        !profile.record(HostTimelinePhase::kCombineMetadataToHost, 530, 570) ||
        !profile.record(HostTimelinePhase::kCombineHostValidation, 580, 650) ||
        !profile.record(HostTimelinePhase::kCombineStreamSetup, 660, 690) ||
        !profile.record(HostTimelinePhase::kCombineOutputAllocation, 700, 725) ||
        !profile.record(HostTimelinePhase::kCombineSubmit, 730, 750) ||
        !profile.record(HostTimelinePhase::kCombineCompletionRecord, 760, 770) ||
        !profile.record(HostTimelinePhase::kCombineCompletionWait, 780, 880))
        return 2;
    if (!profile.bind_generation(19) || profile.generation != 19 ||
        profile.phase_ns(HostTimelinePhase::kDispatchPrelaunchSetup) != 30 ||
        profile.phase_ns(HostTimelinePhase::kDispatchSynchronize) != 85 ||
        profile.phase_ns(HostTimelinePhase::kDispatchOutputAllocation) != 40 ||
        profile.phase_ns(HostTimelinePhase::kDispatchCompletionRecord) != 30 ||
        profile.phase_ns(HostTimelinePhase::kCombinePrelaunchSetup) != 15 ||
        profile.phase_ns(HostTimelinePhase::kCombineHandleToHost) != 20 ||
        profile.phase_ns(HostTimelinePhase::kCombineMetadataToHost) != 40 ||
        profile.phase_ns(HostTimelinePhase::kCombineHostValidation) != 70 ||
        profile.phase_ns(HostTimelinePhase::kCombineStreamSetup) != 30 ||
        profile.phase_ns(HostTimelinePhase::kCombineOutputAllocation) != 25 ||
        profile.phase_ns(HostTimelinePhase::kCombineSubmit) != 20 ||
        profile.phase_ns(HostTimelinePhase::kCombineCompletionRecord) != 10 ||
        profile.phase_ns(HostTimelinePhase::kCombineCompletionWait) != 100 ||
        profile.total_ns() != 515)
        return 3;
    if (profile.record(HostTimelinePhase::kDispatchEpilogueSubmit, 10, 9) ||
        profile.record(HostTimelinePhase::kCount, 1, 2) ||
        profile.bind_generation(20) || profile.total_ns() != 515)
        return 4;
    profile.reset(0);
    return profile.generation == 0 && profile.total_ns() == 0 ? 0 : 5;
}

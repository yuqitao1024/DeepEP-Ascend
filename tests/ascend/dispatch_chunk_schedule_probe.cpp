// Exercise the actual host launcher with an event/stage recorder. This checks
// scheduling and cleanup only; device memory ordering requires NPU validation.
#include <cassert>
#include <cstdint>
#include <vector>

#include "csrc/backends/ascend/elastic/dispatch_pipeline_config.hpp"

using aclrtStream = void*;
struct Event {
    unsigned index;
    bool recorded = false;
};
using aclrtEvent = Event*;
constexpr unsigned ACL_EVENT_SYNC = 0;
static std::vector<Event*> events;
static unsigned records, releases, epilogues, controls, barriers, syncs, destroyed;
static unsigned ready_waited, done_waited;
static unsigned completion_fences;
static bool source_launch = true;
static int fail_chunk = -1;
static void* const producer = reinterpret_cast<void*>(1);
static void* const communication = reinterpret_cast<void*>(2);

int aclrtCreateEventWithFlag(aclrtEvent* event, unsigned) {
    *event = new Event{static_cast<unsigned>(events.size())};
    events.push_back(*event);
    return 0;
}
int aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    assert(stream == (event->index % 2 == 0 ? producer : communication));
    const unsigned chunk = event->index / 2;
    assert((event->index % 2 == 0 ? records : releases) == chunk + 1);
    event->recorded = true;
    return 0;
}
int aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    assert(event->recorded);
    assert(stream == (event->index % 2 == 0 ? communication : producer));
    if (stream == communication)
        ready_waited |= 1U << (event->index / 2);
    else
        done_waited |= 1U << (event->index / 2);
    return 0;
}
int aclrtSynchronizeStream(aclrtStream) { ++syncs; return 0; }
int aclrtDestroyEvent(aclrtEvent event) {
    // The existing launcher synchronizes both streams before releasing events.
    assert(syncs == 2);
    ++destroyed;
    delete event;
    return 0;
}

namespace deep_ep::ascend::elastic {

int launch_dispatch_kernel(
    DispatchArguments, CoreTiling, void*, CoreLaunchShape,
    DirectDispatchStage, std::uint32_t) { return -91; }

int launch_persistent_dispatch_source_pipeline(
    DispatchArguments, CoreTiling, void*, void*, const DispatchSourceChunkPlan&) {
    return -77;  // The old path cannot publish scalar/transport progress.
}

int launch_direct_dispatch_stage(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    DirectDispatchStage stage, std::uint32_t copy_outputs) {
    if (stage == DirectDispatchStage::kProducerRecord ||
        stage == DirectDispatchStage::kProducerRelease) {
        if (!source_launch) {
            assert(stream == producer);
            if (stage == DirectDispatchStage::kProducerRecord)
                ++records;
            else
                ++releases;
            return 0;
        }
        const unsigned chunk = arguments.pipeline_chunk_index;
        assert(arguments.pipeline_source_chunk == 1);
        assert(arguments.pipeline_chunk_begin == chunk * arguments.pipeline_chunk_tiles);
        assert(arguments.pipeline_chunk_end <= tiling.num_tokens / 4);
        if (stage == DirectDispatchStage::kProducerRecord) {
            assert(stream == producer);
            if (chunk >= kDispatchPipelineSlotCount)
                assert(done_waited & (1U << (chunk - kDispatchPipelineSlotCount)));
            if (static_cast<int>(chunk) == fail_chunk)
                return -31;
            ++records;
        } else {
            assert(stream == communication);
            assert(ready_waited & (1U << chunk));
            ++releases;
        }
    } else if (stage == DirectDispatchStage::kProducerReleaseControl ||
               stage == DirectDispatchStage::kProducerReleaseBarrier) {
        assert(stream == (source_launch ? communication : producer));
        assert(arguments.pipeline_final_chunk == 1);
        if (stage == DirectDispatchStage::kProducerReleaseControl)
            ++controls;
        else
            ++barriers;
    } else if (direct_dispatch_epilogue_stage(stage)) {
        assert(stream == producer && copy_outputs == 1);
        if (source_launch)
            assert(done_waited & (1U << (releases - 1)));
        assert(arguments.pipeline_source_chunk == 0);
        assert(arguments.pipeline_chunk_begin == 0 && arguments.pipeline_chunk_end == 0);
        ++epilogues;
    } else {
        assert(stream == producer);
    }
    return 0;
}
}  // namespace deep_ep::ascend::elastic

extern "C" int deep_ep_ascend_launch_dispatch_consumed_barrier(
    deep_ep::ascend::elastic::DispatchArguments arguments,
    deep_ep::ascend::elastic::CoreTiling, void* stream) {
    if (arguments.source_completion_fence == 0)
        return 0;
    assert(stream == producer && epilogues == 7 && syncs == 0);
    ++completion_fences;
    return 0;
}

// Generated from dispatch.asc by the test, without rewriting the launcher.
#include "dispatch_pipeline_under_test.hpp"

int main() {
    using namespace deep_ep::ascend::elastic;
    for (bool profile : {false, true}) {
        for (unsigned chunks : {2U, 4U, 8U}) {
            for (int failure : {-1, 1}) {
                records = releases = epilogues = controls = barriers = syncs = destroyed = 0;
                ready_waited = done_waited = 0;
                completion_fences = 0;
                events.clear();
                fail_chunk = failure;
                DispatchArguments arguments{};
                arguments.pipeline_chunk_tiles = 2048 / chunks;
                arguments.source_completion_fence = 1;
                CoreTiling tiling{};
                tiling.num_tokens = 8192;
                tiling.mode_flags = mode_bit(CoreMode::kPipeline);
                tiling.element_kind = ElementKind::kFloat8E4M3;
                if (profile)
                    tiling.transport_context.capabilities =
                        deep_ep::ascend::transport::capability_bit(
                            deep_ep::ascend::transport::TransportCapability::kStageProfile);
                const int status = deep_ep_ascend_launch_dispatch_pipeline(
                    arguments, tiling, producer, communication);
                assert(status == (failure < 0 ? 0 : -31));
                assert(destroyed == chunks * 2 && syncs == 2);
                assert(records == (failure < 0 ? chunks : 1));
                assert(releases == records);
                assert(epilogues == (failure < 0 ? 7U : 0U));
                assert(controls == (profile && failure < 0 ? 1U : 0U));
                assert(barriers == controls);
                assert(completion_fences == (failure < 0 ? 1U : 0U));
            }
        }
    }
    // Local fallback still joins, even though this rank has no chunk events.
    source_launch = false;
    records = releases = epilogues = controls = barriers = syncs = 0;
    completion_fences = 0;
    DispatchArguments arguments{};
    arguments.source_completion_fence = 1;
    CoreTiling tiling{};
    tiling.num_tokens = 1;
    assert(deep_ep_ascend_launch_dispatch(arguments, tiling, producer) == 0);
    assert(completion_fences == 1 && epilogues == 7 && syncs == 0);
}

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "csrc/backends/ascend/elastic/dispatch_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    static_assert(std::is_standard_layout_v<DispatchControlSlot>);
    static_assert(std::is_trivially_copyable_v<DispatchControlSlot>);
    static_assert(sizeof(DispatchControlSlot) == 16);
    static_assert(std::is_standard_layout_v<DispatchHandleDescriptor>);
    static_assert(std::is_trivially_copyable_v<DispatchHandleDescriptor>);

    DispatchSequence sequence;
    {
        DispatchAttempt first(sequence);
        CHECK(first.valid());
        CHECK(first.generation() == 1);
        first.complete();
    }
    CHECK(!sequence.poisoned());

    {
        DispatchAttempt incomplete(sequence);
        CHECK(incomplete.valid());
        CHECK(incomplete.generation() == 2);
    }
    CHECK(sequence.poisoned());
    DispatchAttempt rejected(sequence);
    CHECK(!rejected.valid());
    CHECK(rejected.generation() == 0);

    CoreTopology topology{};
    topology.world_size = 2;
    topology.scale_up_size = 2;
    DispatchHandleDescriptor expected = make_dispatch_handle_descriptor(
        17, topology, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    CHECK(expected.abi_version == kDispatchHandleDescriptorAbiVersion);
    CHECK(expected.struct_size == sizeof(DispatchHandleDescriptor));
    CHECK(validate_dispatch_handle(expected, expected).ok());

    auto mismatch = expected;
    mismatch.family += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.topology.scale_up_rank = 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_tokens += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.hidden += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_experts += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_topk += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.expert_alignment += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_max_tokens_per_rank += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.mode_flags = 0;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    return 0;
}

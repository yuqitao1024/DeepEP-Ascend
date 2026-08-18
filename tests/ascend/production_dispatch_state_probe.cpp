#include <cstddef>
#include <cstdint>
#include <limits>
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

    DispatchSequence exhausted(std::numeric_limits<std::uint64_t>::max());
    DispatchAttempt exhausted_attempt(exhausted);
    CHECK(!exhausted_attempt.valid());
    CHECK(exhausted_attempt.generation() == 0);
    CHECK(exhausted.poisoned());

    CoreTopology topology{};
    topology.world_size = 2;
    topology.scale_up_size = 2;
    DispatchHandleDescriptor expected = make_dispatch_handle_descriptor(
        17, topology, 11, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    CHECK(expected.abi_version == kDispatchHandleDescriptorAbiVersion);
    CHECK(expected.struct_size == sizeof(DispatchHandleDescriptor));
    CHECK(expected.generation == 11);
    CHECK(validate_dispatch_handle(expected, expected).ok());

    auto zero_padded = expected;
    zero_padded.mode_flags |= mode_bit(CoreMode::kZeroPadding);
    CHECK(validate_dispatch_handle(expected, zero_padded).ok());
    CHECK(validate_dispatch_handle(zero_padded, expected).ok());

    const auto attested_expanded = make_attested_dispatch_handle_descriptor(
        17, topology, 11, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    const auto attested_zero_padded =
        make_attested_dispatch_handle_descriptor(
            17, topology, 11, 3, 128, 8, 2, 4, 256,
            mode_bit(CoreMode::kExpanded) |
                mode_bit(CoreMode::kZeroPadding));
    CHECK(attested_expanded.family == attested_zero_padded.family);
    CHECK(validate_dispatch_handle(
        attested_expanded, attested_zero_padded).ok());

    auto mismatch = expected;
    mismatch.family += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.topology.scale_up_rank = 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.generation = 10;
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

    const auto attested = make_attested_dispatch_handle_descriptor(
        99, topology, 11, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    const auto stale = make_attested_dispatch_handle_descriptor(
        99, topology, 10, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    CHECK(attested.family != stale.family);
    return 0;
}

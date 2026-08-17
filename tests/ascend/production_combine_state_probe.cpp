#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/combine_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    static_assert(std::is_standard_layout_v<CombineControlSlot>);
    static_assert(std::is_trivially_copyable_v<CombineControlSlot>);
    static_assert(sizeof(CombineControlSlot) == 16);
    static_assert(std::is_standard_layout_v<CombineRecordHeader>);
    static_assert(std::is_trivially_copyable_v<CombineRecordHeader>);
    static_assert(sizeof(CombineRecordHeader) == 24);

    CombineSequence sequence;
    {
        CombineAttempt first(sequence);
        CHECK(first.valid());
        CHECK(first.generation() == 1);
        first.complete();
    }
    CHECK(!sequence.poisoned());
    {
        CombineAttempt incomplete(sequence);
        CHECK(incomplete.valid());
        CHECK(incomplete.generation() == 2);
    }
    CHECK(sequence.poisoned());
    CombineAttempt rejected(sequence);
    CHECK(!rejected.valid());
    CHECK(rejected.generation() == 0);

    CombineSequence exhausted(std::numeric_limits<std::uint64_t>::max());
    CombineAttempt exhausted_attempt(exhausted);
    CHECK(!exhausted_attempt.valid());
    CHECK(exhausted.poisoned());

    const CombineRecordHeader header{};
    CHECK(header.abi_version == kCombineRecordAbiVersion);
    CHECK(header.struct_size == sizeof(CombineRecordHeader));
    CHECK(header.origin_token == -1);
    CHECK(header.contributor_rank == -1);
    CHECK(header.master_lane == -1);
    CHECK(header.contribution_lane == -1);
    // Rank 1 owns one token but may contribute rank 0's second token.
    CHECK(is_valid_combine_source_identity(1, 0, 1, 4, 1));
    CHECK(!is_valid_combine_source_identity(5, 1, 1, 4, 1));
    CHECK(!is_valid_combine_source_identity(4, 0, 1, 4, 1));
    CHECK(!is_valid_combine_origin_token(1, 1, 4));

    CHECK(is_valid_combine_source_identity(0, 0, 0, 4, 1));
    CHECK(!is_valid_combine_source_identity(1, 0, 0, 4, 1));
    CHECK(is_valid_combine_source_identity(4, 1, 0, 4, 1));
    CHECK(is_valid_combine_source_identity(5, 1, 0, 4, 1));

    const std::int64_t routes[] = {1, 0, 2, -1};
    CHECK(is_valid_combine_record_lanes(
        routes, 4, 4, 2, 0, 0, 0, false, false));
    CHECK(!is_valid_combine_record_lanes(
        routes, 4, 4, 2, 0, 1, 1, false, false));
    CHECK(!is_valid_combine_record_lanes(
        routes, 4, 4, 2, 0, 0, 1, true, true));
    CHECK(is_valid_combine_record_lanes(
        routes, 4, 4, 2, 0, 0, 1, true, false));
    CHECK(!is_valid_combine_record_lanes(
        routes, 4, 4, 2, 0, 0, 2, true, false));
    CHECK(is_valid_combine_record_lanes(
        routes, 4, 4, 2, 1, 2, 2, true, true));
    return 0;
}

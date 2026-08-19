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
    static_assert(std::is_standard_layout_v<HybridCombineRouteMetadata>);
    static_assert(std::is_trivially_copyable_v<HybridCombineRouteMetadata>);
    static_assert(sizeof(HybridCombineRouteMetadata) == 16);
    static_assert(kDirectCombineRecordTrailerBytes == 32);
    static_assert(kHybridCombineRecordTrailerBytes == 64);

    const auto direct_trailer = combine_record_trailer_layout(96, false);
    CHECK(direct_trailer.valid);
    CHECK(direct_trailer.header_offset == 64);
    CHECK(!direct_trailer.has_route_metadata);
    CHECK(direct_trailer.route_metadata_offset == 0);
    const auto hybrid_trailer = combine_record_trailer_layout(128, true);
    CHECK(hybrid_trailer.valid);
    CHECK(hybrid_trailer.header_offset == 64);
    CHECK(hybrid_trailer.has_route_metadata);
    CHECK(hybrid_trailer.route_metadata_offset == 88);
    CHECK(hybrid_trailer.route_metadata_offset +
              sizeof(HybridCombineRouteMetadata) <= 128);
    CHECK(!combine_record_trailer_layout(32, true).valid);

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

#include <cstdint>

#include "csrc/backends/ascend/elastic/combine_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

float bfloat16_to_float(std::uint16_t bits) {
    union {
        std::uint32_t integer;
        float value;
    } conversion{static_cast<std::uint32_t>(bits) << 16U};
    return conversion.value;
}

std::uint16_t float_to_bfloat16(float value) {
    union {
        float value;
        std::uint32_t integer;
    } conversion{value};
    const std::uint32_t round_to_nearest_even =
        0x7fffU + ((conversion.integer >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        (conversion.integer + round_to_nearest_even) >> 16U);
}

struct HostOriginRecord {
    CombineRecordHeader header{};
    float activations[3]{};
    const float* routing_weights = nullptr;
};

struct HostOriginRecordRange {
    const HostOriginRecord* records = nullptr;
    std::uint64_t count = 0;
};

struct HostOriginRecords {
    const HostOriginRecordRange* ranges;

    std::uint64_t record_count(std::uint64_t contributor_rank) const {
        return ranges[contributor_rank].count;
    }

    CombineOriginRecordView record_at(
        std::uint64_t contributor_rank, std::uint64_t slot,
        std::uint64_t hidden) const {
        const HostOriginRecord& record = ranges[contributor_rank].records[slot];
        return {record.header, record.activations[hidden],
                record.routing_weights};
    }
};

}  // namespace

int main() {
    // BF16 2^24 is the FP32 integer-precision boundary. Each column below
    // has a distinct hand-derived ordering or bias-placement result.
    const std::int64_t contributor_indices[] = {0, 1, 2, -1};
    const std::int64_t lane_indices[] = {0, 1, 2, -1};
    const std::int64_t bias_indices[] = {0, 1, -1, -1};
    const float rank_zero_weights[] = {0.5F, 0.25F, 0.0F, 0.0F};
    const float rank_one_weights[] = {0.0F, 0.0F, 0.75F, 0.0F};
    const float padding_weights[] = {99.0F, 99.0F, 99.0F, 99.0F};

    const CombineRecordHeader rank_zero_lane_zero_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 0, 0, 0};
    const CombineRecordHeader rank_zero_lane_one_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 0, 1, 1};
    const CombineRecordHeader rank_zero_lane_two_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 0, 2, 2};
    const CombineRecordHeader rank_one_lane_two_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 1, 2, 2};
    const CombineRecordHeader padding_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 0, 2, 2};

    // Physical rank-0 slots are lane 0, lane 2, lane 1. The three BF16
    // columns respectively cover contributor order, lane order, and biases.
    const HostOriginRecord rank_zero_records[] = {
        {rank_zero_lane_zero_header,
         {bfloat16_to_float(0x4b80U), bfloat16_to_float(0x4b80U),
          bfloat16_to_float(0x4b80U)}, rank_zero_weights},
        {rank_zero_lane_two_header,
         {0.0F, bfloat16_to_float(0xcb80U), 0.0F}, rank_zero_weights},
        {rank_zero_lane_one_header,
         {bfloat16_to_float(0x3f80U), bfloat16_to_float(0x3f80U),
          bfloat16_to_float(0xcb80U)}, rank_zero_weights},
        {padding_header,
         {bfloat16_to_float(0x7f80U), bfloat16_to_float(0x7f80U),
          bfloat16_to_float(0x7f80U)}, padding_weights},
    };
    const HostOriginRecord rank_one_records[] = {
        {rank_one_lane_two_header,
         {bfloat16_to_float(0xcb80U), 0.0F, 0.0F}, rank_one_weights},
        {padding_header,
         {bfloat16_to_float(0x7f80U), bfloat16_to_float(0x7f80U),
          bfloat16_to_float(0x7f80U)}, padding_weights},
    };
    const HostOriginRecordRange ranges[] = {
        {rank_zero_records, 3},
        {rank_one_records, 1},
    };

    float output_weights[] = {9.0F, 9.0F, 9.0F, 9.0F};
    const HostOriginRecords records{ranges};
    const float contributor_order = combine_reduce_origin_records(
        records, 2, 0, contributor_indices, 4, 4, 2, false, false, 0,
        0.0F, 0.0F, output_weights);
    const float logical_lane_order = combine_reduce_origin_records(
        records, 1, 0, lane_indices, 4, 4, 4, false, false, 1,
        0.0F, 0.0F, nullptr);
    const float biases_after_reduction = combine_reduce_origin_records(
        records, 1, 0, bias_indices, 4, 4, 4, false, false, 2,
        bfloat16_to_float(0x3e80U), bfloat16_to_float(0x3e80U), nullptr);

    // Required results are respectively 0, 0, and 0.5. A contributor-order
    // or physical-slot scan yields 1 for the first two; bias before the large
    // cancellation yields 0, and duplicated biases yield 1 for the third.
    CHECK(float_to_bfloat16(contributor_order) == 0x0000U);
    CHECK(float_to_bfloat16(logical_lane_order) == 0x0000U);
    CHECK(float_to_bfloat16(biases_after_reduction) == 0x3f00U);
    CHECK(output_weights[0] == 0.5F);
    CHECK(output_weights[1] == 0.25F);
    CHECK(output_weights[2] == 0.75F);
    CHECK(output_weights[3] == 0.0F);
    return 0;
}

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

struct HostOriginRecords {
    const CombineOriginRecordRange* ranges;

    std::uint64_t record_count(std::uint64_t contributor_rank) const {
        return ranges[contributor_rank].count;
    }

    CombineOriginRecordView record_at(
        std::uint64_t contributor_rank, std::uint64_t slot,
        std::uint64_t /* hidden */) const {
        return ranges[contributor_rank].records[slot];
    }
};

}  // namespace

int main() {
    // The intended rank/lane order is -65536 + 65536 + 1 == 1.  Reversing
    // ranks or the physical lane-2/lane-0 record order loses the final one.
    const std::int64_t combined_topk_indices[] = {2, 0, 3, -1};
    const float rank_zero_weights[] = {0.0F, 0.25F, 0.0F, 0.0F};
    const float rank_one_weights[] = {0.5F, 0.0F, 0.75F, 0.0F};
    const float padding_weights[] = {99.0F, 99.0F, 99.0F, 99.0F};

    const CombineRecordHeader rank_zero_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 0, 1, 1};
    const CombineRecordHeader rank_one_lane_two_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 1, 2, 2};
    const CombineRecordHeader rank_one_lane_zero_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 1, 0, 0};
    const CombineRecordHeader padding_header{
        kCombineRecordAbiVersion, sizeof(CombineRecordHeader), 0, 1, 2, 2};

    const CombineOriginRecordView rank_zero_records[] = {
        {rank_zero_header, bfloat16_to_float(0xc780U), rank_zero_weights},
        {padding_header, bfloat16_to_float(0x7f80U), padding_weights},
    };
    // Physical slot order is deliberately lane 2 then lane 0; semantics are
    // contributor-rank then logical lane, not transport slot order.
    const CombineOriginRecordView rank_one_records[] = {
        {rank_one_lane_two_header,
         bfloat16_to_float(0x3f80U), rank_one_weights},
        {rank_one_lane_zero_header,
         bfloat16_to_float(0x4780U), rank_one_weights},
        {padding_header, bfloat16_to_float(0x7f80U), padding_weights},
    };
    const CombineOriginRecordRange ranges[] = {
        {rank_zero_records, 1},
        {rank_one_records, 2},
    };

    float output_weights[] = {9.0F, 9.0F, 9.0F, 9.0F};
    const float output = combine_reduce_origin_records(
        HostOriginRecords{ranges}, 2, 0, combined_topk_indices, 4, 4, 2,
        false, false, 0, bfloat16_to_float(0x3f80U),
        bfloat16_to_float(0x3f00U), output_weights);

    // Biases are each placed once after the full FP32 activation reduction,
    // and the only conversion to the BF16-compatible result is here.
    CHECK(float_to_bfloat16(output) == 0x4020U);
    CHECK(output_weights[0] == 0.5F);
    CHECK(output_weights[1] == 0.25F);
    CHECK(output_weights[2] == 0.75F);
    CHECK(output_weights[3] == 0.0F);
    return 0;
}

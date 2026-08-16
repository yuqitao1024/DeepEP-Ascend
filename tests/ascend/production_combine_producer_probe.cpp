#include <cstdint>
#include <cstdio>

#include "csrc/backends/ascend/elastic/combine_state.hpp"

using namespace deep_ep::ascend::elastic;

namespace {

float float_from_bits(std::uint32_t bits) {
    union {
        std::uint32_t bits;
        float value;
    } conversion{bits};
    return conversion.value;
}

std::uint32_t float_bits(float value) {
    union {
        float value;
        std::uint32_t bits;
    } conversion{value};
    return conversion.bits;
}

bool check_record(
    const char* name, const float* source_weights, std::uint64_t source_row,
    std::int32_t master_lane, const float* expected) {
    constexpr std::uint64_t kNumTopk = 2;
    for (std::int32_t lane = 0; lane < static_cast<std::int32_t>(kNumTopk);
         ++lane) {
        const float actual = combine_normal_record_routing_weight(
            source_weights, source_row, lane, master_lane, kNumTopk);
        if (float_bits(actual) != float_bits(expected[lane])) {
            std::fprintf(
                stderr,
                "%s lane %d: expected %.9g, got %.9g\n",
                name, lane, expected[lane], actual);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    // Contributor rank 1 returns rank 0 token 0 with master lane 1.
    const float rank_one_source_weights[] = {
        0.125F, 0.25F,
        0.5F, 0.625F,
        0.75F, 1.0F,
    };
    const float rank_zero_token_zero[] = {0.125F, 0.25F};
    bool valid = check_record(
        "rank 0 token 0", rank_one_source_weights, 0, 1,
        rank_zero_token_zero);

    // Contributor rank 0 returns rank 1 token 1 from compact source row 2.
    const float rank_zero_source_weights[] = {
        0.125F, 0.25F,
        0.375F, 0.875F,
        0.75F, 1.0F,
    };
    const float rank_one_token_one[] = {0.75F, 1.0F};
    valid = check_record(
        "rank 1 token 1", rank_zero_source_weights, 2, 1,
        rank_one_token_one) && valid;
    if (!valid)
        return 1;

    const float exact_fp32_source_weights[] = {
        float_from_bits(0x3dcccccdU), float_from_bits(0x80000000U),
    };
    const float exact_fp32_expected[] = {
        float_from_bits(0x3dcccccdU), float_from_bits(0x80000000U),
    };
    if (!check_record(
            "exact FP32 row", exact_fp32_source_weights, 0, 0,
            exact_fp32_expected))
        return 2;

    if (combine_normal_record_routing_weight(
            nullptr, 0, 1, 1, 2) != 0.0F ||
        combine_normal_record_routing_weight(
            rank_one_source_weights, 0, -1, 1, 2) != 0.0F ||
        combine_normal_record_routing_weight(
            rank_one_source_weights, 0, 2, 1, 2) != 0.0F ||
        combine_normal_record_routing_weight(
            rank_one_source_weights, 0, 1, -1, 2) != 0.0F ||
        combine_normal_record_routing_weight(
            rank_one_source_weights, 0, 1, 2, 2) != 0.0F)
        return 3;
    return 0;
}

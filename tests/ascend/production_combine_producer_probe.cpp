#include <array>
#include <cstdint>
#include <cstdio>

#include "csrc/backends/ascend/elastic/combine_state.hpp"
#include "csrc/backends/ascend/elastic/layout.hpp"

using namespace deep_ep::ascend::elastic;

namespace {

constexpr std::uint64_t kHiddenElements = 3;
constexpr std::uint64_t kHiddenBytes = kHiddenElements * sizeof(std::uint16_t);
constexpr std::uint64_t kNumTopk = 2;
constexpr std::size_t kRecordStorageBytes = 128;
constexpr std::uint8_t kRecordSentinel = 0xa5U;

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
    std::int32_t master_lane, const float* expected,
    std::uint64_t record_bytes, std::uint64_t weight_offset) {
    alignas(kAscendElasticAlignment)
        std::array<std::uint8_t, kRecordStorageBytes> record{};
    record.fill(kRecordSentinel);
    combine_fill_normal_record_routing_weights(
        source_weights, source_row, master_lane, kNumTopk,
        record.data(), weight_offset);
    const auto* actual_weights = reinterpret_cast<const float*>(
        record.data() + weight_offset);
    for (std::uint64_t lane = 0; lane < kNumTopk; ++lane) {
        const float actual = actual_weights[lane];
        if (float_bits(actual) != float_bits(expected[lane])) {
            std::fprintf(
                stderr,
                "%s lane %llu: expected %.9g, got %.9g\n",
                name, static_cast<unsigned long long>(lane),
                expected[lane], actual);
            return false;
        }
    }
    for (std::uint64_t offset = 0; offset < kHiddenBytes; ++offset) {
        if (record[offset] != kRecordSentinel) {
            std::fprintf(stderr, "%s mutated activation byte %llu\n",
                         name, static_cast<unsigned long long>(offset));
            return false;
        }
    }
    for (std::uint64_t offset = kHiddenBytes; offset < weight_offset; ++offset) {
        if (record[offset] != kRecordSentinel) {
            std::fprintf(stderr, "%s mutated alignment byte %llu\n",
                         name, static_cast<unsigned long long>(offset));
            return false;
        }
    }
    const std::uint64_t weight_end =
        weight_offset + kNumTopk * sizeof(float);
    for (std::uint64_t offset = weight_end; offset < record_bytes; ++offset) {
        if (record[offset] != kRecordSentinel) {
            std::fprintf(stderr, "%s mutated record byte %llu\n",
                         name, static_cast<unsigned long long>(offset));
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    SymmetricWindowLayout layout{};
    const LayoutStatus layout_status = build_symmetric_window_layout(
        {2, 4, kHiddenElements, kNumTopk, sizeof(std::uint16_t), false,
         false},
        &layout);
    if (!layout_status.ok() ||
        layout.combine_record_bytes > kRecordStorageBytes ||
        layout.combine_weight_offset <= kHiddenBytes ||
        layout.combine_weight_offset % alignof(float) != 0)
        return 10;

    // Contributor rank 1 returns rank 0 token 0 with master lane 1.
    const float rank_one_source_weights[] = {
        0.125F, 0.25F,
        0.5F, 0.625F,
        0.75F, 1.0F,
    };
    const float rank_zero_token_zero[] = {0.125F, 0.25F};
    bool valid = check_record(
        "rank 0 token 0", rank_one_source_weights, 0, 1,
        rank_zero_token_zero, layout.combine_record_bytes,
        layout.combine_weight_offset);

    // Contributor rank 0 returns rank 1 token 1 from compact source row 2.
    const float rank_zero_source_weights[] = {
        0.125F, 0.25F,
        0.375F, 0.875F,
        0.75F, 1.0F,
    };
    const float rank_one_token_one[] = {0.75F, 1.0F};
    valid = check_record(
        "rank 1 token 1", rank_zero_source_weights, 2, 1,
        rank_one_token_one, layout.combine_record_bytes,
        layout.combine_weight_offset) && valid;
    if (!valid)
        return 1;

    // Both lanes belong to one contributor; lane 1 is not the master lane.
    const float same_contributor_source_weights[] = {0.375F, 0.875F};
    const float same_contributor_expected[] = {0.375F, 0.875F};
    if (!check_record(
            "same contributor row", same_contributor_source_weights, 0, 0,
            same_contributor_expected, layout.combine_record_bytes,
            layout.combine_weight_offset))
        return 2;

    const float exact_fp32_source_weights[] = {
        float_from_bits(0x3dcccccdU), float_from_bits(0x80000000U),
    };
    const float exact_fp32_expected[] = {
        float_from_bits(0x3dcccccdU), float_from_bits(0x80000000U),
    };
    if (!check_record(
            "exact FP32 row", exact_fp32_source_weights, 0, 0,
            exact_fp32_expected, layout.combine_record_bytes,
            layout.combine_weight_offset))
        return 3;

    const float zero_weights[] = {0.0F, 0.0F};
    if (!check_record(
            "missing weights", nullptr, 0, 1, zero_weights,
            layout.combine_record_bytes, layout.combine_weight_offset) ||
        !check_record(
            "negative master lane", rank_one_source_weights, 0, -1,
            zero_weights, layout.combine_record_bytes,
            layout.combine_weight_offset) ||
        !check_record(
            "out-of-range master lane", rank_one_source_weights, 0, 2,
            zero_weights, layout.combine_record_bytes,
            layout.combine_weight_offset))
        return 4;
    combine_fill_normal_record_routing_weights(
        rank_one_source_weights, 0, 1, kNumTopk, nullptr,
        layout.combine_weight_offset);
    return 0;
}

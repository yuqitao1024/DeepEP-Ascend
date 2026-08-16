#include <cstdint>

#include "csrc/backends/ascend/elastic/combine_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    // Two valid lanes deliberately map around BF16 padding sentinels.
    const std::int32_t metadata[] = {0, 0, 2, 5, -1};
    const float input[] = {777.0F, 777.0F, 1.0F, 777.0F, 777.0F,
                           0.00390625F};
    const float routing[] = {0.125F, 0.625F};

    CHECK(combine_expanded_record_count(metadata + 2, 3, true) == 1);
    CHECK(combine_expanded_record_count(metadata + 2, 3, false) == 2);
    CHECK(combine_expanded_record_lane(metadata + 2, 3, 0) == 0);
    CHECK(combine_expanded_record_lane(metadata + 2, 3, 1) == 1);
    CHECK(combine_expanded_record_lane(metadata + 2, 3, 2) == -1);
    CHECK(combine_expanded_input_row_is_valid(2, 6));
    CHECK(combine_expanded_input_row_is_valid(-1, 6));
    CHECK(!combine_expanded_input_row_is_valid(-2, 6));
    CHECK(!combine_expanded_input_row_is_valid(6, 6));

    const float local = combine_reduce_expanded_lanes(
        input, metadata + 2, 3, 6, 1, 0);
    CHECK(local == 1.00390625F);
    const float lane_zero = combine_reduce_expanded_lanes(
        input, metadata + 2, 1, 6, 1, 0);
    const float lane_one = combine_reduce_expanded_lanes(
        input, metadata + 3, 1, 6, 1, 0);
    CHECK(lane_zero == 1.0F);
    CHECK(lane_one == 0.00390625F);

    // Routing values return unchanged and do not weight the activation.
    const float remote_contributor = 2.0F;
    CHECK(combine_routing_weight(routing, 0) == 0.125F);
    CHECK(combine_routing_weight(routing, 1) == 0.625F);
    CHECK(combine_apply_biases(local + remote_contributor, 1.0F, 0.5F) ==
          4.50390625F);
    return 0;
}

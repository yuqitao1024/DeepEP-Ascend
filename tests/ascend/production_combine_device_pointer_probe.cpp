#include <cstdint>

#define DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL volatile
#include "csrc/backends/ascend/elastic/combine_state.hpp"

#define CHECK(expression) do { if (!(expression)) return __LINE__; } while (0)

namespace {

using deep_ep::ascend::elastic::CombineOriginRecordView;
using deep_ep::ascend::elastic::CombineRecordHeader;

struct RecordSource {
    volatile float routing_weights[2]{0.25F, 0.75F};

    std::uint64_t record_count(std::uint64_t contributor_rank) const {
        return contributor_rank == 0 ? 1 : 0;
    }

    CombineOriginRecordView record_at(
        std::uint64_t contributor_rank, std::uint64_t slot,
        std::uint64_t hidden) const {
        if (contributor_rank != 0 || slot != 0 || hidden != 0)
            return {};
        CombineRecordHeader header{};
        header.origin_token = 0;
        header.contributor_rank = 0;
        header.master_lane = 0;
        header.contribution_lane = 0;
        return {header, 3.5F, routing_weights};
    }
};

}  // namespace

int main() {
    using namespace deep_ep::ascend::elastic;

    volatile std::int32_t input_rows[2]{0, 1};
    CHECK(combine_expanded_record_count(input_rows, 2, false) == 2);
    CHECK(combine_expanded_record_lane(input_rows, 2, 1) == 1);

    volatile float expanded_input[4]{1.0F, 2.0F, 3.0F, 4.0F};
    CHECK(combine_reduce_expanded_lanes(
              expanded_input, input_rows, 2, 2, 2, 0) == 4.0F);
    CHECK(combine_routing_weight(expanded_input, 3) == 4.0F);

    volatile CombineRecordHeader global_header{};
    global_header.origin_token = 7;
    global_header.contributor_rank = 1;
    global_header.master_lane = 2;
    global_header.contribution_lane = 3;
    const CombineRecordHeader loaded =
        load_combine_record_header(&global_header);
    CHECK(loaded.abi_version == kCombineRecordAbiVersion);
    CHECK(loaded.struct_size == sizeof(CombineRecordHeader));
    CHECK(loaded.origin_token == 7);
    CHECK(loaded.contributor_rank == 1);
    CHECK(loaded.master_lane == 2);
    CHECK(loaded.contribution_lane == 3);

    const RecordSource source{};
    volatile std::int64_t combined_topk_indices[2]{0, -1};
    volatile float output_weights[2]{-1.0F, -1.0F};
    const float activation = combine_reduce_origin_records(
        source, 2, 0, combined_topk_indices, 2, 4, 2, false, false, 0,
        1.0F, 2.0F, output_weights);
    CHECK(activation == 6.5F);
    CHECK(output_weights[0] == 0.25F);
    CHECK(output_weights[1] == 0.0F);
    return 0;
}

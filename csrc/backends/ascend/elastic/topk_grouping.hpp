#pragma once

#include <cstddef>
#include <cstdint>

namespace deep_ep::ascend::elastic {

struct CombineContributorEntry {
    std::int32_t contributor_rank = -1;
    std::int32_t contribution_lane = -1;
    std::int32_t receive_slot = -1;
};

static_assert(sizeof(CombineContributorEntry) == 12);
static_assert(alignof(CombineContributorEntry) == alignof(std::int32_t));

template <std::size_t MaxTopk>
struct TopkGroupingResult {
    std::uint32_t contributor_count = 0;
    CombineContributorEntry entries[MaxTopk]{};
    std::int32_t resolved_slots[MaxTopk]{};
};

template <std::size_t MaxTopk>
constexpr TopkGroupingResult<MaxTopk>
group_combine_contributors_reference(
    const std::int32_t* contributor_keys,
    const std::int32_t* observed_slots,
    std::uint32_t num_topk) noexcept {
    static_assert(MaxTopk > 0);
    TopkGroupingResult<MaxTopk> result{};
    for (std::size_t lane = 0; lane < MaxTopk; ++lane)
        result.resolved_slots[lane] = -1;
    if (contributor_keys == nullptr || observed_slots == nullptr ||
        num_topk > MaxTopk)
        return result;

    for (std::uint32_t lane = 0; lane < num_topk; ++lane) {
        const std::int32_t key = contributor_keys[lane];
        if (key < 0)
            continue;
        std::uint32_t owner_lane = lane;
        for (std::uint32_t candidate = 0; candidate < lane; ++candidate) {
            if (contributor_keys[candidate] == key) {
                owner_lane = candidate;
                break;
            }
        }
        if (owner_lane != lane || observed_slots[owner_lane] < 0)
            continue;

        CombineContributorEntry entry{
            key, static_cast<std::int32_t>(owner_lane),
            observed_slots[owner_lane]};
        std::uint32_t position = result.contributor_count;
        while (position > 0) {
            const auto& prior = result.entries[position - 1];
            if (prior.contributor_rank < entry.contributor_rank ||
                (prior.contributor_rank == entry.contributor_rank &&
                 prior.contribution_lane < entry.contribution_lane))
                break;
            result.entries[position] = prior;
            --position;
        }
        result.entries[position] = entry;
        ++result.contributor_count;
    }

    for (std::uint32_t lane = 0; lane < num_topk; ++lane) {
        const std::int32_t key = contributor_keys[lane];
        if (key < 0)
            continue;
        std::uint32_t owner_lane = lane;
        for (std::uint32_t candidate = 0; candidate < lane; ++candidate) {
            if (contributor_keys[candidate] == key) {
                owner_lane = candidate;
                break;
            }
        }
        if (observed_slots[owner_lane] >= 0)
            result.resolved_slots[lane] = observed_slots[owner_lane];
    }
    return result;
}

}  // namespace deep_ep::ascend::elastic

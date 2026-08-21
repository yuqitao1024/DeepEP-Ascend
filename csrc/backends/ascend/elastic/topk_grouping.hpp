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

#if defined(DEEP_EP_ASCEND_TOPK_GROUPING_DEVICE)

inline constexpr std::uint32_t kTopkSubgroupWidth = 32;
using TopkSubgroupMask = std::uint32_t;

struct TopkSubgroupGroup {
    TopkSubgroupMask active_mask = 0;
    TopkSubgroupMask group_mask = 0;
    TopkSubgroupMask unique_owner_mask = 0;
    std::int32_t owner_lane = -1;
    std::uint32_t owner_ordinal = 0;
    bool is_owner = false;
};

__SIMT_DEVICE_FUNCTIONS_DECL__ inline std::int32_t
topk_first_set_lane(TopkSubgroupMask mask) noexcept {
    return mask == 0 ? -1 : __ffs(static_cast<std::int32_t>(mask)) - 1;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline TopkSubgroupGroup
group_topk_subgroup(std::int32_t key, bool logically_active) noexcept {
    const std::uint32_t lane = static_cast<std::uint32_t>(laneid());
    const bool active = logically_active && key >= 0;
    TopkSubgroupGroup result{};
    result.active_mask = static_cast<TopkSubgroupMask>(
        asc_ballot(active ? 1 : 0));
    TopkSubgroupMask remaining = result.active_mask;
    while (remaining != 0) {
        const std::int32_t leader_lane = topk_first_set_lane(remaining);
        const std::int32_t leader_key = asc_shfl(
            key, static_cast<std::uint32_t>(leader_lane),
            kTopkSubgroupWidth);
        const TopkSubgroupMask peers = static_cast<TopkSubgroupMask>(
            asc_ballot(active && key == leader_key ? 1 : 0));
        if ((peers & (TopkSubgroupMask{1} << lane)) != 0)
            result.group_mask = peers;
        remaining &= ~peers;
    }
    result.owner_lane = topk_first_set_lane(result.group_mask);
    result.is_owner = result.owner_lane >= 0 &&
        lane == static_cast<std::uint32_t>(result.owner_lane);
    result.unique_owner_mask = static_cast<TopkSubgroupMask>(
        asc_ballot(result.is_owner ? 1 : 0));
    if (result.owner_lane >= 0) {
        const TopkSubgroupMask lower_lanes = static_cast<TopkSubgroupMask>(
            lanemask_lt());
        const std::uint32_t owner_prefix = asc_shfl(
            static_cast<std::uint32_t>(__popc(
                result.unique_owner_mask & lower_lanes)),
            static_cast<std::uint32_t>(result.owner_lane),
            kTopkSubgroupWidth);
        result.owner_ordinal = owner_prefix;
    }
    return result;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline std::int32_t
topk_broadcast_owner_value(
    const TopkSubgroupGroup& group, std::int32_t value,
    std::int32_t invalid_value) noexcept {
    return group.owner_lane < 0 ? invalid_value : asc_shfl(
        value, static_cast<std::uint32_t>(group.owner_lane),
        kTopkSubgroupWidth);
}

#endif

}  // namespace deep_ep::ascend::elastic

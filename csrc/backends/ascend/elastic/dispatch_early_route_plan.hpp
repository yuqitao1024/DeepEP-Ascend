#pragma once

#include <cstdint>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kDispatchEarlyRoutePlanMaximumExperts = 256;
inline constexpr std::uint64_t kDispatchRoutePlanMaximumTopk = 8;
inline constexpr int kDispatchRoutePlanMaximumWorldSize = 8;

enum class DispatchEarlyRoutePlanConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchEarlyRoutePlanConfig {
    bool enabled = false;
};

struct DispatchRoutePlanCoordinate {
    std::uint32_t destination_rank = 0;
    std::uint32_t local_expert = 0;
    bool valid = false;
};

inline constexpr DispatchRoutePlanCoordinate dispatch_route_plan_coordinate(
    std::int64_t expert, std::uint64_t num_experts,
    std::uint32_t world_size) noexcept {
    DispatchRoutePlanCoordinate coordinate{};
    if (expert < 0 || num_experts == 0 || world_size == 0 ||
        num_experts % world_size != 0 ||
        static_cast<std::uint64_t>(expert) >= num_experts)
        return coordinate;
    const std::uint64_t local_experts = num_experts / world_size;
    coordinate.destination_rank = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(expert) / local_experts);
    coordinate.local_expert = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(expert) % local_experts);
    coordinate.valid = true;
    return coordinate;
}

constexpr bool
handoff_dispatch_route_source_count(
    bool early_route_plan, std::uint64_t planned_count,
    std::uint64_t observed_count, std::uint64_t shard_capacity,
    std::uint64_t* canonical_count) noexcept {
    if (canonical_count == nullptr || observed_count > shard_capacity ||
        (early_route_plan && planned_count != observed_count))
        return false;
    *canonical_count = observed_count;
    return true;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_handoff_dispatch_route_source_count(
    bool early_route_plan, std::uint64_t planned_count,
    std::uint64_t observed_count, std::uint64_t shard_capacity,
    std::uint64_t* canonical_count) noexcept {
    if (canonical_count == nullptr || observed_count > shard_capacity ||
        (early_route_plan && planned_count != observed_count))
        return false;
    *canonical_count = observed_count;
    return true;
}
#endif

inline DispatchEarlyRoutePlanConfigStatus
select_dispatch_early_route_plan_config(
    const char* value, bool grouping_enabled, bool fp8_dispatch,
    bool cached_mode, bool expanded, bool hybrid_mode, bool stream_mode,
    bool pipeline_enabled, std::uint64_t num_experts,
    std::uint64_t num_topk, int world_size,
    DispatchEarlyRoutePlanConfig* output) noexcept {
    if (output == nullptr)
        return DispatchEarlyRoutePlanConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr || (value[0] == '0' && value[1] == '\0'))
        return DispatchEarlyRoutePlanConfigStatus::kDisabled;
    if (value[0] != '1' || value[1] != '\0')
        return DispatchEarlyRoutePlanConfigStatus::kInvalid;
    if (!grouping_enabled || !fp8_dispatch || cached_mode || expanded ||
        hybrid_mode || stream_mode || pipeline_enabled || num_experts == 0 ||
        num_experts > kDispatchEarlyRoutePlanMaximumExperts ||
        num_topk == 0 ||
        num_topk > kDispatchRoutePlanMaximumTopk || world_size < 1 ||
        world_size > kDispatchRoutePlanMaximumWorldSize ||
        num_experts % static_cast<std::uint64_t>(world_size) != 0)
        return DispatchEarlyRoutePlanConfigStatus::kDisabled;

    output->enabled = true;
    return DispatchEarlyRoutePlanConfigStatus::kEnabled;
}

}  // namespace deep_ep::ascend::elastic

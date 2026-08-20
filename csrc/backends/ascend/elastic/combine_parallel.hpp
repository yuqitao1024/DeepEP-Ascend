#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

constexpr bool combine_record_slot_index(
    std::uint64_t token, std::uint64_t lane, std::uint64_t num_tokens,
    std::uint64_t num_topk, std::uint64_t* index) noexcept {
    if (index == nullptr || token >= num_tokens || lane >= num_topk ||
        (num_topk != 0 &&
         token > (std::numeric_limits<std::uint64_t>::max() - lane) /
             num_topk))
        return false;
    *index = token * num_topk + lane;
    return true;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
combine_simt_record_slot_index(
    std::uint64_t token, std::uint64_t lane, std::uint64_t num_tokens,
    std::uint64_t num_topk, std::uint64_t* index) noexcept {
    if (index == nullptr || token >= num_tokens || lane >= num_topk ||
        (num_topk != 0 &&
         token > (std::numeric_limits<std::uint64_t>::max() - lane) /
             num_topk))
        return false;
    *index = token * num_topk + lane;
    return true;
}

#endif

}  // namespace deep_ep::ascend::elastic

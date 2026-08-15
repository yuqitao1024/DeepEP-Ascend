#pragma once

#include <cstddef>
#include <cstdint>

#include "cann_compat.hpp"

#ifndef DEEP_EP_ASCEND_AICORE_WQE_CALLEE
#define DEEP_EP_ASCEND_AICORE_WQE_CALLEE
#define DEEP_EP_ASCEND_AICORE_WQE_CALLEE_LOCAL 1
#endif

namespace deep_ep::ascend::transport::urma {

inline constexpr std::uint32_t kStrongOrdering = 5U;
inline constexpr std::uint32_t kFence = 1U;
inline constexpr std::uint32_t kCompletionEntry = 1U;

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline std::uint64_t pack_sq_head(
    std::uint32_t position, std::uint32_t request_count) {
    return static_cast<std::uint64_t>(position) |
           (static_cast<std::uint64_t>(request_count) << 32U);
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline std::uint32_t sq_position(
    std::uint64_t head) {
    return static_cast<std::uint32_t>(head);
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline std::uint32_t sq_request_count(
    std::uint64_t head) {
    return static_cast<std::uint32_t>(head >> 32U);
}

struct WriteRequest {
    cann_abi::UrmaSqe sqe;
    cann_abi::UrmaSge sge;
};

struct InlineWrite64Request {
    cann_abi::UrmaSqe sqe;
    std::uint64_t value;
    std::uint64_t reserved;
};

struct Faa64Request {
    cann_abi::UrmaSqe sqe;
    cann_abi::UrmaSge fetch_result;
    std::uint64_t add_value;
    std::uint64_t reserved[7];
};

static_assert(sizeof(WriteRequest) == 64);
static_assert(sizeof(InlineWrite64Request) == 64);
static_assert(sizeof(Faa64Request) == 128);

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline std::uint32_t sq_slot(
    std::uint32_t head, std::uint32_t depth) {
    return depth == 0 ? 0 : head % depth;
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline std::uint32_t owner_for(
    std::uint32_t head, std::uint32_t depth) {
    return depth == 0 ? 0 : 1U ^ ((head / depth) & 1U);
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline bool cqe_owner_valid(
    std::uint32_t owner, std::uint32_t tail, std::uint32_t depth) {
    return owner == owner_for(tail, depth);
}

namespace detail {

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline std::uint64_t load_eid_word(
    const std::uint8_t* eid, std::size_t start) {
    std::uint64_t result = 0;
    for (std::size_t byte = 0; byte < sizeof(result); ++byte)
        result |= static_cast<std::uint64_t>(eid[start + byte]) << (byte * 8);
    return result;
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline cann_abi::UrmaSqe make_sqe(
    const cann_abi::SqContext& sq,
    const cann_abi::RegisteredBuffer& remote_memory,
    std::uint32_t head, std::uint64_t remote_address,
    std::uint32_t opcode, bool inline_data) {
    cann_abi::UrmaSqe result{};
    const std::uint32_t flag =
        kStrongOrdering | (kFence << 3U) | (kCompletionEntry << 5U) |
        (static_cast<std::uint32_t>(inline_data) << 6U);
    result.word0 =
        (sq_slot(head, sq.depth) & 0xffffU) | (flag << 16U) |
        (1U << 28U) | (1U << 29U) | (owner_for(head, sq.depth) << 31U);
    result.word1 =
        ((opcode & 0xffU) << 8U) |
        ((inline_data ? sizeof(std::uint64_t) : 0U) << 22U);
    result.word2 =
        (sq.transport_path_id & 0x00ffffffU) |
        ((inline_data ? 0U : 1U) << 24U);
    result.word3 = remote_memory.token_id & 0x000fffffU;
    result.remote_eid_low = load_eid_word(sq.remote_eid, 0);
    result.remote_eid_high = load_eid_word(sq.remote_eid, 8);
    result.remote_token_value = remote_memory.token_value;
    result.remote_address_low = static_cast<std::uint32_t>(remote_address);
    result.remote_address_high =
        static_cast<std::uint32_t>(remote_address >> 32U);
    return result;
}

}  // namespace detail

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline WriteRequest make_write(
    const cann_abi::SqContext& sq,
    const cann_abi::RegisteredBuffer& remote_memory,
    std::uint32_t head, std::uint64_t remote_address,
    std::uint64_t local_address, std::uint32_t bytes,
    std::uint32_t local_token_id) {
    WriteRequest result{};
    result.sqe = detail::make_sqe(
        sq, remote_memory, head, remote_address,
        cann_abi::kUrmaWriteOpcode, false);
    result.sge.bytes = bytes;
    result.sge.token_id = local_token_id;
    result.sge.address = local_address;
    return result;
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline InlineWrite64Request make_inline_write64(
    const cann_abi::SqContext& sq,
    const cann_abi::RegisteredBuffer& remote_memory,
    std::uint32_t head, std::uint64_t remote_address,
    std::uint64_t value) {
    InlineWrite64Request result{};
    result.sqe = detail::make_sqe(
        sq, remote_memory, head, remote_address,
        cann_abi::kUrmaWriteOpcode, true);
    result.value = value;
    return result;
}

DEEP_EP_ASCEND_AICORE_WQE_CALLEE inline Faa64Request make_faa64(
    const cann_abi::SqContext& sq,
    const cann_abi::RegisteredBuffer& remote_memory,
    std::uint32_t head, std::uint64_t remote_address,
    std::uint64_t fetch_result_address, std::uint64_t add_value) {
    Faa64Request result{};
    result.sqe = detail::make_sqe(
        sq, remote_memory, head, remote_address,
        cann_abi::kUrmaFaaOpcode, false);
    result.fetch_result.bytes = sizeof(std::uint64_t);
    result.fetch_result.address = fetch_result_address;
    result.add_value = add_value;
    return result;
}

}  // namespace deep_ep::ascend::transport::urma

#if defined(DEEP_EP_ASCEND_AICORE_WQE_CALLEE_LOCAL)
#undef DEEP_EP_ASCEND_AICORE_WQE_CALLEE_LOCAL
#undef DEEP_EP_ASCEND_AICORE_WQE_CALLEE
#endif

#pragma once

#include <cstddef>
#include <cstdint>

namespace deep_ep::ascend::transport::cann_abi {

inline constexpr std::uint32_t kCompatibilityVersion = 0x00090200U;
inline constexpr std::uint32_t kUbcCtpProtocol = 4U;
inline constexpr std::uint32_t kDefaultQueueIndex = 0U;
inline constexpr std::uint32_t kUrmaWriteOpcode = 3U;
inline constexpr std::uint32_t kUrmaFaaOpcode = 0xBU;

struct AbiHeader {
    std::uint32_t version;
    std::uint32_t magic_word;
    std::uint32_t struct_size;
    std::uint32_t reserved;
};

struct Memory {
    std::int32_t type;
    std::uint32_t reserved0;
    std::uint64_t address;
    std::uint64_t bytes;
};

struct RegisteredBuffer {
    std::int32_t type;
    std::uint32_t reserved0;
    std::uint64_t address;
    std::uint64_t bytes;
    std::int32_t protection_type;
    std::uint32_t token_id;
    std::uint32_t token_value;
    std::uint8_t reserved1[28];
};

struct SqContext {
    std::int32_t type;
    std::uint32_t reserved0;
    std::uint64_t base;
    std::uint64_t head;
    std::uint64_t tail;
    std::uint64_t doorbell;
    std::uint32_t queue_id;
    std::uint32_t entry_bytes;
    std::uint32_t depth;
    std::uint32_t transport_path_id;
    std::uint8_t remote_eid[16];
    std::uint8_t reserved1[56];
};

struct CqContext {
    std::int32_t type;
    std::uint32_t reserved0;
    std::uint64_t base;
    std::uint64_t head;
    std::uint64_t tail;
    std::uint64_t doorbell;
    std::uint32_t queue_id;
    std::uint32_t entry_bytes;
    std::uint32_t depth;
    std::uint8_t reserved1[76];
};

struct Channel {
    AbiHeader header;
    std::int32_t engine;
    std::int32_t protocol;
    std::uint32_t local_notify_count;
    std::uint32_t remote_notify_count;
    std::uint32_t local_buffer_count;
    std::uint32_t remote_buffer_count;
    std::uint32_t sq_count;
    std::uint32_t cq_count;
    std::uint64_t local_notifies;
    std::uint64_t remote_notifies;
    std::uint64_t local_buffers;
    std::uint64_t remote_buffers;
    std::uint64_t sq_contexts;
    std::uint64_t cq_contexts;
    std::uint8_t reserved[160];
};

struct Window {
    AbiHeader header;
    std::uint32_t memory_count;
    std::uint32_t reserved0;
    std::uint64_t memories;
    std::uint64_t world_team;
    std::uint32_t reserved1[8];
};

struct Team {
    AbiHeader header;
    std::int32_t engine;
    std::uint32_t member_count;
    std::uint32_t self_member;
    std::uint32_t reserved0;
    std::uint64_t channels;
    std::uint64_t channel_counts;
    std::uint32_t network_layer;
    std::uint32_t reserved1;
    std::uint64_t world_team_ids;
    std::uint64_t remote_sync_memories;
    std::uint32_t remote_sync_memory_count;
    std::uint32_t reserved2;
    Memory shadow_sync_memory;
    std::uint32_t signal_count;
    std::uint32_t counter_count;
    std::uint32_t barrier_count;
    std::uint32_t sync_requirement_reserved[5];
    std::uint64_t sync_memory_bytes;
    std::uint32_t sync_reserved[5];
    std::uint32_t reserved3;
    std::uint32_t reserved4[8];
};

struct UrmaSqe {
    std::uint32_t word0;
    std::uint32_t word1;
    std::uint32_t word2;
    std::uint32_t word3;
    std::uint64_t remote_eid_low;
    std::uint64_t remote_eid_high;
    std::uint32_t remote_token_value;
    std::uint32_t word9;
    std::uint32_t remote_address_low;
    std::uint32_t remote_address_high;
};

struct UrmaSge {
    std::uint32_t bytes;
    std::uint32_t token_id;
    std::uint64_t address;
};

struct UrmaCqe {
    std::uint32_t words[16];
};

static_assert(sizeof(AbiHeader) == 16);
static_assert(sizeof(Memory) == 24);
static_assert(sizeof(RegisteredBuffer) == 64);
static_assert(sizeof(SqContext) == 128);
static_assert(sizeof(CqContext) == 128);
static_assert(sizeof(Channel) == 256);
static_assert(sizeof(Window) == 72);
static_assert(sizeof(Team) == 200);
static_assert(sizeof(UrmaSqe) == 48);
static_assert(sizeof(UrmaSge) == 16);
static_assert(sizeof(UrmaCqe) == 64);

}  // namespace deep_ep::ascend::transport::cann_abi

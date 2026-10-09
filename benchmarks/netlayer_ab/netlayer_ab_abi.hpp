#pragma once

#include <cstddef>
#include <cstdint>

namespace netlayer_ab {

// CANN 9.3 HCOMM ABI subset used by this benchmark.  The layout mirrors the
// installed ChannelEntity/SqContext/CqContext definitions; it intentionally
// does not include DeepEP runtime headers so the benchmark can be copied to a
// clean machine.
struct AbiHeader {
    std::uint32_t version;
    std::uint32_t magic_word;
    std::uint32_t struct_size;
    std::uint32_t reserved;
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

struct alignas(64) WriteRequest {
    UrmaSqe sqe;
    UrmaSge sge;
};

static_assert(sizeof(AbiHeader) == 16);
static_assert(sizeof(RegisteredBuffer) == 64);
static_assert(sizeof(SqContext) == 128);
static_assert(sizeof(CqContext) == 128);
static_assert(sizeof(Channel) == 256);
static_assert(sizeof(UrmaSqe) == 48);
static_assert(sizeof(UrmaSge) == 16);
static_assert(sizeof(WriteRequest) == 64);

inline constexpr std::uint32_t kUbcCtpProtocol = 4;
inline constexpr std::uint32_t kUbcTpProtocol = 3;
inline constexpr std::uint32_t kUrmaWriteOpcode = 3;

struct alignas(64) DeviceControl {
    std::uint64_t local_payload;
    std::uint64_t local_generation;
    std::uint64_t payload_bytes;
    std::uint64_t chunk_bytes;
    std::uint32_t rank;
    std::uint32_t world_size;
    std::uint64_t remote_addresses;
    std::uint64_t channels;
    std::uint32_t timeout_cycles;
    std::uint32_t success;
    std::uint32_t failed_peer;
    std::uint32_t failed_stage;
    std::uint32_t backend_status;
    std::uint32_t reserved0;
    std::uint64_t cycle_counter;
    std::uint64_t cycle_frequency;
    std::uint64_t reserved1[4];
};

static_assert(sizeof(DeviceControl) == 128);

}  // namespace netlayer_ab

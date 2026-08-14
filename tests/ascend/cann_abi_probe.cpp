#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "csrc/backends/ascend/transport/cann_compat.hpp"

#include <hcomm/hcomm_res_entity_defs.h>
#include <hcomm/hcomm_team_entity_defs.h>
#include <pto/comm/async/urma/urma_types.hpp>
#include <adv_api/detail/hcomm/common/hcomm_inner_def.h>

namespace cann_abi = deep_ep::ascend::transport::cann_abi;
namespace cann_urma = pto::comm::urma;

namespace {

int failures = 0;

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": "                \
                      << #expression << '\n';                                 \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

template <typename Root, typename Member>
std::size_t member_offset(Root& root, Member& member) {
    return static_cast<std::size_t>(
        reinterpret_cast<std::uint8_t*>(&member) -
        reinterpret_cast<std::uint8_t*>(&root));
}

#define CHECK_OFFSET(our_type, our_member, package_object, package_member)    \
    CHECK(offsetof(our_type, our_member) ==                                   \
          member_offset(package_object, package_object.package_member))

static_assert(cann_abi::kCompatibilityVersion == 0x00090200U);
static_assert(cann_abi::kUbcCtpProtocol == 4U);
static_assert(cann_abi::kDefaultQueueIndex == 0U);
static_assert(cann_abi::kUrmaWriteOpcode == 3U);
static_assert(cann_abi::kUrmaFaaOpcode == 0xBU);

static_assert(sizeof(cann_abi::RegisteredBuffer) ==
              sizeof(::RegedBufferEntity));
static_assert(sizeof(cann_abi::SqContext) == sizeof(::SqContext));
static_assert(sizeof(cann_abi::CqContext) == sizeof(::CqContext));
static_assert(sizeof(cann_abi::Channel) == sizeof(::ChannelEntity));
static_assert(sizeof(cann_abi::Window) == sizeof(::HcommWindow));
static_assert(sizeof(cann_abi::Team) == sizeof(::HcommTeam));
static_assert(sizeof(cann_abi::UrmaSqe) == sizeof(cann_urma::UrmaSqeCtx));
static_assert(sizeof(cann_abi::UrmaSqe) == 48);
static_assert(sizeof(cann_abi::UrmaSge) == sizeof(cann_urma::UrmaSgeCtx));
static_assert(sizeof(cann_abi::UrmaSge) == 16);
static_assert(sizeof(cann_abi::UrmaCqe) ==
              sizeof(cann_urma::UrmaJfcCqeCtx));
static_assert(sizeof(cann_abi::UrmaCqe) == 64);
static_assert(std::is_standard_layout_v<cann_abi::Team>);
static_assert(std::is_standard_layout_v<cann_abi::Window>);
static_assert(std::is_standard_layout_v<cann_abi::Channel>);
static_assert(std::is_trivially_copyable_v<cann_abi::Team>);
static_assert(std::is_trivially_copyable_v<cann_abi::Window>);
static_assert(std::is_trivially_copyable_v<cann_abi::Channel>);

void check_registered_buffer() {
    ::RegedBufferEntity package{};
    CHECK_OFFSET(cann_abi::RegisteredBuffer, type, package, type);
    CHECK_OFFSET(cann_abi::RegisteredBuffer, address, package,
                 bufferInfo.rma.addr);
    CHECK_OFFSET(cann_abi::RegisteredBuffer, bytes, package,
                 bufferInfo.rma.size);
    CHECK_OFFSET(cann_abi::RegisteredBuffer, protection_type, package,
                 bufferInfo.rma.protectionInfo.type);
    CHECK_OFFSET(cann_abi::RegisteredBuffer, token_id, package,
                 bufferInfo.rma.protectionInfo.memInfo.ub.tokenId);
    CHECK_OFFSET(cann_abi::RegisteredBuffer, token_value, package,
                 bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue);
}

void check_sq_context() {
    ::SqContext package{};
    CHECK_OFFSET(cann_abi::SqContext, type, package, type);
    CHECK_OFFSET(cann_abi::SqContext, base, package, contextInfo.ubJfs.sqVa);
    CHECK_OFFSET(cann_abi::SqContext, head, package,
                 contextInfo.ubJfs.headAddr);
    CHECK_OFFSET(cann_abi::SqContext, tail, package,
                 contextInfo.ubJfs.tailAddr);
    CHECK_OFFSET(cann_abi::SqContext, doorbell, package,
                 contextInfo.ubJfs.dbVa);
    CHECK_OFFSET(cann_abi::SqContext, queue_id, package,
                 contextInfo.ubJfs.jfsID);
    CHECK_OFFSET(cann_abi::SqContext, entry_bytes, package,
                 contextInfo.ubJfs.wqeSize);
    CHECK_OFFSET(cann_abi::SqContext, depth, package,
                 contextInfo.ubJfs.sqDepth);
    CHECK_OFFSET(cann_abi::SqContext, transport_path_id, package,
                 contextInfo.ubJfs.tpID);
    CHECK_OFFSET(cann_abi::SqContext, remote_eid, package,
                 contextInfo.ubJfs.remoteEID);
}

void check_cq_context() {
    ::CqContext package{};
    CHECK_OFFSET(cann_abi::CqContext, type, package, type);
    CHECK_OFFSET(cann_abi::CqContext, base, package, contextInfo.ubJfc.scqVa);
    CHECK_OFFSET(cann_abi::CqContext, head, package,
                 contextInfo.ubJfc.headAddr);
    CHECK_OFFSET(cann_abi::CqContext, tail, package,
                 contextInfo.ubJfc.tailAddr);
    CHECK_OFFSET(cann_abi::CqContext, doorbell, package,
                 contextInfo.ubJfc.dbVa);
    CHECK_OFFSET(cann_abi::CqContext, queue_id, package,
                 contextInfo.ubJfc.jfcID);
    CHECK_OFFSET(cann_abi::CqContext, entry_bytes, package,
                 contextInfo.ubJfc.cqeSize);
    CHECK_OFFSET(cann_abi::CqContext, depth, package,
                 contextInfo.ubJfc.cqDepth);
}

void check_channel() {
    ::ChannelEntity package{};
    CHECK_OFFSET(cann_abi::Channel, header, package, abiHeader);
    CHECK_OFFSET(cann_abi::Channel, engine, package, engine);
    CHECK_OFFSET(cann_abi::Channel, protocol, package, protocol);
    CHECK_OFFSET(cann_abi::Channel, local_buffer_count, package,
                 localBufferNum);
    CHECK_OFFSET(cann_abi::Channel, remote_buffer_count, package,
                 remoteBufferNum);
    CHECK_OFFSET(cann_abi::Channel, sq_count, package, sqNum);
    CHECK_OFFSET(cann_abi::Channel, cq_count, package, cqNum);
    CHECK_OFFSET(cann_abi::Channel, local_buffers, package, localBufferAddr);
    CHECK_OFFSET(cann_abi::Channel, remote_buffers, package,
                 remoteBufferAddr);
    CHECK_OFFSET(cann_abi::Channel, sq_contexts, package, sqContextAddr);
    CHECK_OFFSET(cann_abi::Channel, cq_contexts, package, cqContextAddr);

    AscendC::ChannelEntity aicore_package{};
    CHECK_OFFSET(cann_abi::Channel, sq_head, aicore_package, sqHead);
    CHECK_OFFSET(cann_abi::Channel, sq_tail, aicore_package, sqTail);
    CHECK_OFFSET(cann_abi::Channel, cq_head, aicore_package, cqHead);
    CHECK_OFFSET(cann_abi::Channel, cq_tail, aicore_package, cqTail);
}

void check_team_and_window() {
    ::HcommWindow package_window{};
    CHECK_OFFSET(cann_abi::Window, header, package_window, header);
    CHECK_OFFSET(cann_abi::Window, memory_count, package_window, memsNum);
    CHECK_OFFSET(cann_abi::Window, memories, package_window, mems);
    CHECK_OFFSET(cann_abi::Window, world_team, package_window, worldTeam);

    ::HcommTeam package_team{};
    CHECK_OFFSET(cann_abi::Team, header, package_team, header);
    CHECK_OFFSET(cann_abi::Team, engine, package_team, engine);
    CHECK_OFFSET(cann_abi::Team, member_count, package_team, memberNum);
    CHECK_OFFSET(cann_abi::Team, self_member, package_team, selfMemberId);
    CHECK_OFFSET(cann_abi::Team, channels, package_team, channelsBaseAddr);
    CHECK_OFFSET(cann_abi::Team, channel_counts, package_team,
                 channelNumPerMember);
    CHECK_OFFSET(cann_abi::Team, network_layer, package_team, netLayer);
    CHECK_OFFSET(cann_abi::Team, world_team_ids, package_team, worldTeamIds);
    CHECK_OFFSET(cann_abi::Team, remote_sync_memories, package_team,
                 syncMem.remoteMems);
    CHECK_OFFSET(cann_abi::Team, remote_sync_memory_count, package_team,
                 syncMem.remoteMemsNum);
    CHECK_OFFSET(cann_abi::Team, shadow_sync_memory, package_team,
                 syncMem.shadowMem);
    CHECK_OFFSET(cann_abi::Team, signal_count, package_team,
                 syncMem.syncMemReq.signalCount);
    CHECK_OFFSET(cann_abi::Team, counter_count, package_team,
                 syncMem.syncMemReq.counterCount);
    CHECK_OFFSET(cann_abi::Team, barrier_count, package_team,
                 syncMem.syncMemReq.barrierCount);
    CHECK_OFFSET(cann_abi::Team, sync_memory_bytes, package_team,
                 syncMem.syncMemSize);
}

void check_urma_entries() {
    CHECK(offsetof(cann_abi::UrmaSqe, remote_eid_low) ==
          cann_urma::kUrmaSqeRmtEidLOffset);
    CHECK(offsetof(cann_abi::UrmaSqe, remote_eid_high) ==
          cann_urma::kUrmaSqeRmtEidHOffset);
    CHECK(offsetof(cann_abi::UrmaSqe, remote_address_low) ==
          cann_urma::kUrmaSqeRmtAddrLOffset);
    CHECK(offsetof(cann_abi::UrmaSqe, remote_address_high) ==
          cann_urma::kUrmaSqeRmtAddrHOffset);

    cann_urma::UrmaSgeCtx package{};
    CHECK_OFFSET(cann_abi::UrmaSge, bytes, package, len);
    CHECK_OFFSET(cann_abi::UrmaSge, token_id, package, tokenId);
    CHECK_OFFSET(cann_abi::UrmaSge, address, package, va);
}

}  // namespace

int main() {
    check_registered_buffer();
    check_sq_context();
    check_cq_context();
    check_channel();
    check_team_and_window();
    check_urma_entries();
    return failures == 0 ? 0 : 1;
}

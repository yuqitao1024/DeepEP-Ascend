#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "csrc/backends/ascend/transport/aicore_transport_service.hpp"
#include "csrc/backends/ascend/transport/transport_commands.hpp"

namespace transport = deep_ep::ascend::transport;

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

static_assert(sizeof(transport::TransportCommand) == 128);
static_assert(alignof(transport::TransportCommand) == 64);
static_assert(offsetof(transport::TransportCommand, opcode) == 0);
static_assert(offsetof(transport::TransportCommand, team) == 4);
static_assert(offsetof(transport::TransportCommand, peer) == 8);
static_assert(offsetof(transport::TransportCommand, channel) == 12);
static_assert(offsetof(transport::TransportCommand, options) == 16);
static_assert(offsetof(transport::TransportCommand, value_bytes) == 20);
static_assert(offsetof(transport::TransportCommand, signal_index) == 24);
static_assert(offsetof(transport::TransportCommand, world_peer) == 28);
static_assert(offsetof(transport::TransportCommand, source) == 32);
static_assert(offsetof(transport::TransportCommand, destination) == 40);
static_assert(offsetof(transport::TransportCommand, bytes) == 48);
static_assert(offsetof(transport::TransportCommand, value) == 56);
static_assert(offsetof(transport::TransportCommand, symmetric_offset) == 64);
static_assert(offsetof(transport::TransportCommand, timeout_cycles) == 72);
static_assert(sizeof(transport::TransportCommandQueue) == 64);
static_assert(sizeof(transport::TransportServiceState) == 64);
static_assert(sizeof(transport::DeviceTransportDiagnostic) == 64);
static_assert(offsetof(transport::DeviceTransportDiagnostic, peer) == 16);
static_assert(offsetof(transport::DeviceTransportDiagnostic, world_peer) == 48);
static_assert(offsetof(transport::DeviceTransportDiagnostic, team) == 52);
static_assert(sizeof(transport::StagedTransportContext) == 64);

constexpr transport::TransportTopology kBarrierTopology{
    transport::kTransportTopologyAbiVersion,
    sizeof(transport::TransportTopology),
    1,
    4,
    1,
    2,
    0,
    2,
    false,
    transport::TransportTopologyKind::kLogicalSimulation,
    1,
};
static_assert(transport::command::barrier_team_enabled(
    kBarrierTopology, transport::kWorldTeamMask,
    transport::TransportTeam::kScaleOut));
static_assert(transport::command::barrier_peer_in_team(
    kBarrierTopology, transport::TransportTeam::kScaleOut, 3));

void check_team_peer_translation() {
    transport::TransportTopology topology{};
    CHECK(transport::build_transport_topology(
        1, 4, 2, transport::TransportTopologyKind::kLogicalSimulation,
        1, &topology).ok());

    int world_peer = -1;
    CHECK(transport::command::checked_world_peer(
        topology, transport::TransportTeam::kWorld, 3, &world_peer));
    CHECK(world_peer == 3);
    CHECK(transport::command::checked_world_peer(
        topology, transport::TransportTeam::kScaleUp, 0, &world_peer));
    CHECK(world_peer == 0);
    CHECK(transport::command::checked_world_peer(
        topology, transport::TransportTeam::kScaleOut, 1, &world_peer));
    CHECK(world_peer == 3);

    CHECK(!transport::command::checked_world_peer(
        topology, transport::TransportTeam::kScaleOut, 2, &world_peer));
    CHECK(!transport::command::checked_world_peer(
        topology, static_cast<transport::TransportTeam>(99), 0, &world_peer));
    CHECK(!transport::command::checked_world_peer(
        topology, transport::TransportTeam::kWorld, 0, nullptr));

    auto malformed = topology;
    malformed.world_size = 5;
    CHECK(!transport::command::checked_world_peer(
        malformed, transport::TransportTeam::kScaleOut, 1, &world_peer));
    malformed = topology;
    malformed.scale_up_rank = 0;
    CHECK(!transport::command::checked_world_peer(
        malformed, transport::TransportTeam::kScaleUp, 0, &world_peer));

    transport::TransportTopology flat{};
    CHECK(transport::build_transport_topology(
        2, 4, 4, transport::TransportTopologyKind::kFlatScaleUp,
        1, &flat).ok());
    CHECK(transport::command::checked_world_peer(
        flat, transport::TransportTeam::kScaleUp, 3, &world_peer));
    CHECK(world_peer == 3);
    CHECK(transport::command::checked_world_peer(
        flat, transport::TransportTeam::kScaleOut, 0, &world_peer));
    CHECK(world_peer == 2);
}

void check_factories() {
    const auto put = transport::command::make_put(
        transport::TransportTeam::kWorld, 2, 2, 0, 0x1110, 0x2220, 96,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    CHECK(put.opcode == transport::TransportCommandOpcode::kPut);
    CHECK(put.team == transport::TransportTeam::kWorld);
    CHECK(put.peer == 2);
    CHECK(put.world_peer == 2);
    CHECK(put.source == 0x2220);
    CHECK(put.destination == 0x1110);
    CHECK(put.bytes == 96);

    const auto put_value = transport::command::make_put_value64(
        transport::TransportTeam::kScaleUp, 1, 3, 0, 0x3330,
        0xa5a5a5a55a5a5a5aULL, transport::kDefaultOptions);
    CHECK(put_value.opcode == transport::TransportCommandOpcode::kPutValue64);
    CHECK(put_value.world_peer == 3);
    CHECK(put_value.value_bytes == 8);
    CHECK(put_value.destination == 0x3330);
    CHECK(put_value.value == 0xa5a5a5a55a5a5a5aULL);

    const auto faa = transport::command::make_remote_add64(
        transport::TransportTeam::kWorld, 1, 1, 0, 0x4440, -7);
    CHECK(faa.opcode == transport::TransportCommandOpcode::kRemoteAdd64);
    CHECK(faa.destination == 0x4440);
    CHECK(static_cast<std::int64_t>(faa.value) == -7);

    const auto signal = transport::command::make_signal(
        transport::TransportTeam::kWorld, 1, 1, 0,
        transport::RemoteAction::signal_add(0x88, 9));
    CHECK(signal.opcode == transport::TransportCommandOpcode::kSignal);
    CHECK(signal.action_kind == transport::RemoteActionKind::kSignalAdd);
    CHECK(signal.symmetric_offset == 0x88);
    CHECK(signal.value == 9);

    const auto flush = transport::command::make_flush(
        0, transport::CooperationScope::kWorkgroup);
    const auto barrier = transport::command::make_barrier(1, 1234);
    CHECK(flush.opcode == transport::TransportCommandOpcode::kFlush);
    CHECK(barrier.opcode == transport::TransportCommandOpcode::kBarrier);
    CHECK(barrier.options == 1);
    CHECK(barrier.timeout_cycles == 1234);
}

void check_queue_model() {
    transport::TransportCommand commands[3]{};
    transport::TransportServiceState service{};
    transport::DeviceTransportDiagnostic diagnostic{};
    auto queue = transport::command::make_queue(
        commands, 3, &service, &diagnostic);
    transport::StagedTransportContext staged{};
    staged.command_queue = reinterpret_cast<std::uintptr_t>(&queue);
    staged.reserved = transport::command::registration_cookie(
        staged.command_queue, queue.commands, queue.service_state,
        queue.diagnostic, queue.capacity);

    CHECK(transport::command::checked_reset(
        staged, queue, service, diagnostic, 7));
    CHECK(queue.generation == 7);
    CHECK(queue.count == 0);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);

    CHECK(transport::command::append(
        queue, transport::command::make_put(
            transport::TransportTeam::kWorld, 1, 1, 0, 0x1000, 0x2000, 64,
            transport::CooperationScope::kParticipant,
            transport::MemorySegment::kDevice,
            transport::kDefaultOptions)));
    CHECK(transport::command::append(
        queue, transport::command::make_flush(
            0, transport::CooperationScope::kParticipant)));
    CHECK(transport::command::append(
        queue, transport::command::make_barrier(1, 55)));
    CHECK(queue.count == 3);
    CHECK(commands[0].opcode == transport::TransportCommandOpcode::kPut);
    CHECK(commands[1].opcode == transport::TransportCommandOpcode::kFlush);
    CHECK(commands[2].opcode == transport::TransportCommandOpcode::kBarrier);

    CHECK(!transport::command::append(
        queue, transport::command::make_flush(
            0, transport::CooperationScope::kParticipant)));
    CHECK(queue.count == 3);
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCommandOverflow);
    CHECK(diagnostic.command_index == 3);
    CHECK(diagnostic.team == transport::TransportTeam::kWorld);
    CHECK(diagnostic.peer == 0);
    CHECK(diagnostic.world_peer == 0);

    transport::command::record_first_error(
        diagnostic, transport::DeviceTransportError::kInvalidRank, 1,
        transport::TransportCommandOpcode::kPut, 99, 7);
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCommandOverflow);

    CHECK(transport::command::checked_reset(
        staged, queue, service, diagnostic, 8));
    CHECK(queue.count == 0);
    CHECK(queue.generation == 8);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);
}

void check_checked_queue_reset() {
    transport::TransportCommand commands[2]{};
    transport::TransportServiceState service{};
    service.consumed_count = 2;
    service.active = 1;
    service.consumed_generation = 6;
    transport::DeviceTransportDiagnostic diagnostic{};
    diagnostic.error = transport::DeviceTransportError::kCompletionFailure;
    auto queue = transport::command::make_queue(
        commands, 2, &service, &diagnostic);
    queue.count = 2;
    queue.generation = 6;
    transport::StagedTransportContext staged{};
    staged.command_queue = reinterpret_cast<std::uintptr_t>(&queue);
    staged.reserved = transport::command::registration_cookie(
        staged.command_queue, queue.commands, queue.service_state,
        queue.diagnostic, queue.capacity);

    CHECK(transport::command::checked_reset(
        staged, queue, service, diagnostic, 7));
    CHECK(queue.count == 0);
    CHECK(queue.generation == 7);
    CHECK(service.consumed_count == 0);
    CHECK(service.active == 0);
    CHECK(service.consumed_generation == 0);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);
    CHECK(diagnostic.generation == 7);

    const auto valid_staged = staged;
    const auto valid_queue = queue;
    const auto valid_service = service;
    const auto valid_diagnostic = diagnostic;
    const auto reject_without_mutation = [&](auto corrupt) {
        staged = valid_staged;
        queue = valid_queue;
        service = valid_service;
        diagnostic = valid_diagnostic;
        corrupt();
        const auto staged_before = staged;
        const auto queue_before = queue;
        const auto service_before = service;
        const auto diagnostic_before = diagnostic;
        CHECK(!transport::command::checked_reset(
            staged, queue, service, diagnostic, 8));
        CHECK(std::memcmp(&staged, &staged_before, sizeof(staged)) == 0);
        CHECK(std::memcmp(&queue, &queue_before, sizeof(queue)) == 0);
        CHECK(std::memcmp(&service, &service_before, sizeof(service)) == 0);
        CHECK(std::memcmp(
            &diagnostic, &diagnostic_before, sizeof(diagnostic)) == 0);
    };

    reject_without_mutation([&] { ++staged.abi_version; });
    reject_without_mutation([&] { staged.reserved ^= 1; });
    reject_without_mutation([&] { ++queue.abi_version; });
    reject_without_mutation([&] { --queue.struct_size; });
    reject_without_mutation([&] { queue.commands = 1; });
    reject_without_mutation([&] { queue.service_state = 1; });
    reject_without_mutation([&] { queue.diagnostic = 1; });
    reject_without_mutation([&] { queue.count = queue.capacity + 1; });
    reject_without_mutation([&] { ++service.abi_version; });
    reject_without_mutation([&] { ++diagnostic.abi_version; });
}

void check_service_entry_contract() {
    transport::TransportCommand commands[2]{};
    transport::TransportServiceState service{};
    transport::DeviceTransportDiagnostic diagnostic{};
    auto queue = transport::command::make_queue(
        commands, 2, &service, &diagnostic);
    transport::StagedTransportContext staged{};
    staged.command_queue = reinterpret_cast<std::uintptr_t>(&queue);

    CHECK(transport::command::valid_staged_context_header(
        staged.abi_version, staged.struct_size, staged.cann_compatibility,
        staged.command_queue));
    CHECK(transport::command::valid_command_queue_header(
        queue.abi_version, queue.struct_size, queue.commands, queue.capacity,
        queue.count, queue.service_state, queue.diagnostic));

    CHECK(!transport::command::valid_staged_context_header(
        staged.abi_version + 1, staged.struct_size,
        staged.cann_compatibility, staged.command_queue));
    CHECK(!transport::command::valid_staged_context_header(
        staged.abi_version, staged.struct_size - 1,
        staged.cann_compatibility, staged.command_queue));
    CHECK(!transport::command::valid_staged_context_header(
        staged.abi_version, staged.struct_size,
        staged.cann_compatibility + 1, staged.command_queue));
    CHECK(!transport::command::valid_staged_context_header(
        staged.abi_version, staged.struct_size, staged.cann_compatibility, 0));

    CHECK(!transport::command::valid_command_queue_header(
        queue.abi_version + 1, queue.struct_size, queue.commands,
        queue.capacity, queue.count, queue.service_state, queue.diagnostic));
    CHECK(!transport::command::valid_command_queue_header(
        queue.abi_version, queue.struct_size - 1, queue.commands,
        queue.capacity, queue.count, queue.service_state, queue.diagnostic));
    CHECK(!transport::command::valid_command_queue_header(
        queue.abi_version, queue.struct_size, 0, queue.capacity, queue.count,
        queue.service_state, queue.diagnostic));
    CHECK(!transport::command::valid_command_queue_header(
        queue.abi_version, queue.struct_size, queue.commands, queue.capacity,
        queue.capacity + 1, queue.service_state, queue.diagnostic));
}

void check_barrier_poll_timeout() {
    using deep_ep::ascend::transport::service::barrier_poll_timed_out;

    CHECK(!barrier_poll_timed_out(100, 104, 5, 99, 1));
    CHECK(barrier_poll_timed_out(100, 105, 5, 0, 1000));
    CHECK(!barrier_poll_timed_out(100, 1000, 0, 2, 3));
    CHECK(barrier_poll_timed_out(100, 100, 0, 3, 3));
    CHECK(barrier_poll_timed_out(UINT64_MAX - 2, 2, 5, 0, 1000));
}

void check_signal_address_layout_diagnostics() {
    using Failure = transport::sync_layout::SignalAddressFailure;
    using transport::sync_layout::classify_signal_address_layout;

    CHECK(classify_signal_address_layout(4, 0, 0, 0, 6, 1, 1, 0) ==
          Failure::kNone);
    CHECK(classify_signal_address_layout(4, 0, 0, 0, 6, 1, -1, 0) ==
          Failure::kInvalidSourceMember);
    CHECK(classify_signal_address_layout(4, 4, 0, 0, 6, 1, 1, 0) ==
          Failure::kInvalidSelfMember);
    CHECK(classify_signal_address_layout(4, 0, 1, 0, 6, 1, 1, 0) ==
          Failure::kInvalidSignalCount);
    CHECK(classify_signal_address_layout(4, 0, 0, 1, 6, 1, 1, 0) ==
          Failure::kInvalidCounterCount);
    CHECK(classify_signal_address_layout(4, 0, 0, 0, 5, 1, 1, 0) ==
          Failure::kInvalidBarrierCount);
    CHECK(classify_signal_address_layout(4, 0, 0, 0, 6, 1, 1, 4) ==
          Failure::kInvalidSignalIndex);
    CHECK(classify_signal_address_layout(4, 0, 0, 0, 6, 0, 1, 0) ==
          Failure::kMissingRemoteSyncMemories);
    CHECK(classify_signal_address_layout(0, 0, 0, 0, 6, 1, 0, 0) ==
          Failure::kInvalidSourceMember);
}

}  // namespace

int main() {
    check_team_peer_translation();
    check_factories();
    check_queue_model();
    check_checked_queue_reset();
    check_service_entry_contract();
    check_barrier_poll_timeout();
    check_signal_address_layout_diagnostics();
    return failures == 0 ? 0 : 1;
}

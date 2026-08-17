#include <cstdint>
#include <iostream>

#include "csrc/backends/ascend/transport/aicore_transport_service.hpp"
#include "csrc/backends/ascend/transport/sync_layout.hpp"

namespace transport = deep_ep::ascend::transport;
namespace service = deep_ep::ascend::transport::service;
namespace sync_layout = deep_ep::ascend::transport::sync_layout;

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

void check_order_flush_and_barrier() {
    CHECK(sync_layout::kLogicalSignalCount == 4);
    CHECK(sync_layout::kLogicalBarrierCount == 1);
    CHECK(sync_layout::kWorldTeamSignalCount == 0);
    CHECK(sync_layout::kWorldTeamCounterCount == 0);
    CHECK(sync_layout::kWorldTeamBarrierCount == 5);
    CHECK(sync_layout::has_required_world_team_layout(0, 0, 5));
    CHECK(!sync_layout::has_required_world_team_layout(4, 0, 1));
    CHECK(sync_layout::signal_offset(4, 0, 0) == 0);
    CHECK(sync_layout::signal_offset(4, 2, 3) == 88);
    CHECK(sync_layout::barrier_offset(4, 0, 0) == 128);
    CHECK(sync_layout::barrier_offset(4, 0, 3) == 152);
    CHECK(service::fetch_result_offset(3) == 24);

    transport::TransportCommand commands[5]{};
    commands[0] = transport::command::make_put(
        transport::TransportTeam::kWorld, 1, 0, 0x1000, 0x2000, 64,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    commands[1] = transport::command::make_remote_add64(
        transport::TransportTeam::kWorld, 2, 0, 0x3000, 1);
    commands[2] = transport::command::make_flush(
        0, transport::CooperationScope::kDevice);
    commands[3] = transport::command::make_signal(
        transport::TransportTeam::kWorld, 1, 0,
        transport::RemoteAction::signal_increment(2));
    commands[4] = transport::command::make_barrier(1, 64);

    auto state = service::model::make_state(3, 0, 64);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(service::model::execute(commands, 5, state, diagnostic));
    CHECK(state.executed_count == 5);
    CHECK(state.event_count == 5);
    for (std::uint32_t index = 0; index < 5; ++index)
        CHECK(state.events[index] == commands[index].opcode);
    CHECK(state.outstanding == 0);
    CHECK(state.completed == state.submitted);
    CHECK(state.barrier_generation == 1);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);
}

void check_validation_stops_before_later_commands() {
    transport::TransportCommand commands[2]{};
    commands[0] = transport::command::make_put_value64(
        transport::TransportTeam::kWorld, 7, 0, 0x1000, 9,
        transport::kDefaultOptions);
    commands[1] = transport::command::make_flush(
        0, transport::CooperationScope::kParticipant);

    auto state = service::model::make_state(2, 0, 8);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(!service::model::execute(commands, 2, state, diagnostic));
    CHECK(state.executed_count == 0);
    CHECK(state.event_count == 0);
    CHECK(diagnostic.error == transport::DeviceTransportError::kInvalidRank);
    CHECK(diagnostic.command_index == 0);

    transport::command::record_first_error(
        diagnostic, transport::DeviceTransportError::kCommandOverflow, 9,
        transport::TransportCommandOpcode::kFlush, 0, 0);
    CHECK(diagnostic.error == transport::DeviceTransportError::kInvalidRank);
}

void check_completion_timeout_is_finite() {
    transport::TransportCommand commands[2]{};
    commands[0] = transport::command::make_remote_add64(
        transport::TransportTeam::kWorld, 1, 0, 0x1000, 1);
    commands[1] = transport::command::make_flush(
        0, transport::CooperationScope::kParticipant);

    auto state = service::model::make_state(2, 0, 3);
    state.completions_enabled = false;
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(!service::model::execute(commands, 2, state, diagnostic));
    CHECK(state.executed_count == 1);
    CHECK(state.retry_count == 3);
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCompletionTimeout);
    CHECK(diagnostic.command_index == 1);
}

void check_service_uses_translated_world_peer() {
    transport::TransportTopology topology{};
    topology.world_rank = 1;
    topology.world_size = 4;
    topology.scale_up_rank = 1;
    topology.scale_up_size = 2;
    topology.scale_out_rank = 0;
    topology.scale_out_size = 2;
    auto command = transport::command::make_put_value64(
        transport::TransportTeam::kScaleOut, 1, 0, 0x1000, 9,
        transport::kDefaultOptions);
    command.world_peer = 3;

    auto state = service::model::make_state(topology, 8);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(service::model::execute(&command, 1, state, diagnostic));
    CHECK(state.executed_count == 1);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);

    command.world_peer = 2;
    state = service::model::make_state(topology, 8);
    diagnostic = {};
    CHECK(!service::model::execute(&command, 1, state, diagnostic));
    CHECK(diagnostic.error == transport::DeviceTransportError::kInvalidRank);
    CHECK(diagnostic.team == transport::TransportTeam::kScaleOut);
    CHECK(diagnostic.peer == 1);
    CHECK(diagnostic.world_peer == 2);
}

}  // namespace

int main() {
    check_order_flush_and_barrier();
    check_validation_stops_before_later_commands();
    check_completion_timeout_is_finite();
    check_service_uses_translated_world_peer();
    return failures == 0 ? 0 : 1;
}

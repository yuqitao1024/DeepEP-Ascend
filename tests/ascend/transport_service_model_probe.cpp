#include <cstdint>
#include <iostream>

#include "csrc/backends/ascend/elastic/release_protocol.hpp"
#include "csrc/backends/ascend/transport/aicore_transport_service.hpp"
#include "csrc/backends/ascend/transport/sync_layout.hpp"

namespace transport = deep_ep::ascend::transport;
namespace service = deep_ep::ascend::transport::service;
namespace sync_layout = deep_ep::ascend::transport::sync_layout;
namespace release_protocol = deep_ep::ascend::elastic::release_protocol;

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
    CHECK(sync_layout::kLogicalBarrierCount == 2);
    CHECK(sync_layout::kWorldTeamSignalCount == 0);
    CHECK(sync_layout::kWorldTeamCounterCount == 0);
    CHECK(sync_layout::kWorldTeamBarrierCount == 6);
    CHECK(sync_layout::has_required_world_team_layout(0, 0, 6));
    CHECK(!sync_layout::has_required_world_team_layout(4, 0, 1));
    CHECK(sync_layout::signal_offset(4, 0, 0) == 0);
    CHECK(sync_layout::signal_offset(4, 2, 3) == 88);
    CHECK(sync_layout::barrier_offset(4, 0, 0) == 128);
    CHECK(sync_layout::barrier_offset(4, 0, 3) == 152);
    CHECK(service::fetch_result_offset(3) == 24);

    transport::TransportCommand commands[5]{};
    commands[0] = transport::command::make_put(
        transport::TransportTeam::kWorld, 1, 1, 0, 0x1000, 0x2000, 64,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    commands[1] = transport::command::make_remote_add64(
        transport::TransportTeam::kWorld, 2, 2, 0, 0x3000, 1);
    commands[2] = transport::command::make_flush(
        0, transport::CooperationScope::kDevice);
    commands[3] = transport::command::make_signal(
        transport::TransportTeam::kWorld, 1, 1, 0,
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
        transport::TransportTeam::kWorld, 7, 7, 0, 0x1000, 9,
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
        transport::TransportTeam::kWorld, 1, 1, 0, 0x1000, 1);
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

void check_barrier_failure_preserves_failed_world_peer() {
    auto barrier = transport::command::make_barrier(1, 64);
    auto state = service::model::make_state(4, 0, 3);
    state.completions_enabled = false;
    state.completion_failure_world_peer = 3;
    transport::DeviceTransportDiagnostic diagnostic{};

    CHECK(!service::model::execute(&barrier, 1, state, diagnostic));
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCompletionTimeout);
    CHECK(diagnostic.team == transport::TransportTeam::kWorld);
    CHECK(diagnostic.peer == 3);
    CHECK(diagnostic.world_peer == 3);

    transport::command::record_first_error(
        diagnostic, transport::DeviceTransportError::kInvalidQueue, 9,
        transport::TransportCommandOpcode::kBarrier,
        transport::TransportTeam::kScaleOut, 1, 2, 0);
    CHECK(diagnostic.team == transport::TransportTeam::kWorld);
    CHECK(diagnostic.peer == 3);
    CHECK(diagnostic.world_peer == 3);
}

void check_service_uses_translated_world_peer() {
    transport::TransportTopology topology{};
    CHECK(transport::build_transport_topology(
        1, 4, 2, transport::TransportTopologyKind::kLogicalSimulation,
        1, &topology).ok());
    auto command = transport::command::make_put_value64(
        transport::TransportTeam::kScaleOut, 1, 3, 0, 0x1000, 9,
        transport::kDefaultOptions);

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

void check_protocol_validation_precedes_dispatch() {
    auto check_rejected = [](
                              transport::TransportCommand command,
                              transport::DeviceTransportError expected) {
        auto state = service::model::make_state(2, 0, 8);
        transport::DeviceTransportDiagnostic diagnostic{};
        CHECK(!service::model::execute(&command, 1, state, diagnostic));
        CHECK(state.executed_count == 0);
        CHECK(state.submitted == 0);
        CHECK(diagnostic.error == expected);
    };

    auto remote = transport::command::make_put(
        transport::TransportTeam::kWorld, 1, 1, 0, 0x1000, 0x2000, 64,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    remote.channel = 1;
    check_rejected(remote, transport::DeviceTransportError::kInvalidChannel);
    remote.channel = 0;
    remote.options = transport::kAggregateRequests;
    check_rejected(remote, transport::DeviceTransportError::kInvalidProtocol);
    remote.options = transport::kDefaultOptions;
    remote.scope = transport::CooperationScope::kWorkgroup;
    check_rejected(remote, transport::DeviceTransportError::kInvalidProtocol);

    auto flush = transport::command::make_flush(
        0, transport::CooperationScope::kDevice);
    flush.channel = 1;
    check_rejected(flush, transport::DeviceTransportError::kInvalidChannel);
    flush.channel = 0;
    flush.options = transport::kAggregateRequests;
    check_rejected(flush, transport::DeviceTransportError::kInvalidProtocol);

    auto barrier = transport::command::make_barrier(1, 64);
    barrier.channel = 1;
    check_rejected(barrier, transport::DeviceTransportError::kInvalidChannel);
    barrier.channel = 0;
    barrier.options = 2;
    check_rejected(barrier, transport::DeviceTransportError::kInvalidProtocol);

    auto signal = transport::command::make_signal(
        transport::TransportTeam::kWorld, 1, 1, 0,
        transport::RemoteAction::signal_increment(1));
    signal.action_kind = transport::RemoteActionKind::kNone;
    check_rejected(signal, transport::DeviceTransportError::kInvalidProtocol);
}

struct ReleaseProtocolModel {
    bool staging_fenced = false;
    bool put_observed_staging_fence = false;
    bool payload_visible = false;
    bool control_published_after_payload = false;
    std::uint64_t release_generation = 0;
    std::uint64_t acquired_generation = 0;

    void system_fence() {
        staging_fenced = true;
    }

    void put(
        transport::TransportTeam, int, transport::DeviceAddress,
        transport::DeviceAddress, std::size_t,
        transport::CooperationScope, transport::MemorySegment,
        transport::DeviceOptions, const transport::RemoteAction&) {
        put_observed_staging_fence = staging_fenced;
    }

    void flush(transport::CooperationScope scope) {
        CHECK(scope == transport::CooperationScope::kDevice);
        payload_visible = true;
    }

    void put_value(
        transport::TransportTeam, int, transport::DeviceAddress,
        std::uint64_t, std::uint32_t, transport::DeviceOptions) {
        control_published_after_payload = payload_visible;
    }

    void signal(
        transport::TransportTeam, int,
        const transport::RemoteAction& action) {
        CHECK(action.kind == transport::RemoteActionKind::kSignalSet);
        release_generation = action.value;
    }

    void wait_signal(
        transport::TransportTeam, int, std::uint32_t,
        transport::SignalValue target, std::uint64_t) {
        if (release_generation >= target)
            acquired_generation = target;
    }

    transport::SignalValue read_signal(
        transport::TransportTeam, int, std::uint32_t) const {
        return release_generation;
    }
};

void check_release_acquire_and_selected_barrier_sequence() {
    transport::TransportTopology topology{};
    CHECK(transport::build_transport_topology(
        0, 4, 2, transport::TransportTopologyKind::kLogicalSimulation,
        1, &topology).ok());

    ReleaseProtocolModel protocol;
    const transport::TeamPeer rail{
        transport::TransportTeam::kScaleOut, 1, 2};
    release_protocol::put_staged_payload(
        protocol, rail, 0x2000, 0x3000, 64);
    CHECK(protocol.put_observed_staging_fence);
    release_protocol::flush_payload(protocol);
    release_protocol::publish_control_and_release(
        protocol, rail, 0x1000, 3, 0x1008, 7,
        sync_layout::kDispatchReleaseSignalIndex);
    CHECK(protocol.control_published_after_payload);
    CHECK(release_protocol::acquire_release(
        protocol, topology, 2, sync_layout::kDispatchReleaseSignalIndex,
        7, 64));
    CHECK(protocol.acquired_generation == 7);

    const auto barrier = transport::command::make_barrier(
        transport::kWorldTeamMask, 64);
    auto state = service::model::make_state(topology, 8);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(service::model::execute(&barrier, 1, state, diagnostic));
    CHECK(state.barrier_phase_count == 2);
    CHECK(state.barrier_teams[0] == transport::TransportTeam::kScaleOut);
    CHECK(state.barrier_world_peers[0] == 2);
    CHECK(state.barrier_teams[1] == transport::TransportTeam::kScaleUp);
    CHECK(state.barrier_world_peers[1] == 1);
    CHECK(state.submitted == 2);
}

}  // namespace

int main() {
    check_order_flush_and_barrier();
    check_validation_stops_before_later_commands();
    check_completion_timeout_is_finite();
    check_barrier_failure_preserves_failed_world_peer();
    check_service_uses_translated_world_peer();
    check_protocol_validation_precedes_dispatch();
    check_release_acquire_and_selected_barrier_sequence();
    return failures == 0 ? 0 : 1;
}

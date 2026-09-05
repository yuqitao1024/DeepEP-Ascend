#include <cstdint>
#include <iostream>

#include "csrc/backends/ascend/elastic/kernels.hpp"
#include "csrc/backends/ascend/elastic/release_protocol.hpp"
#include "csrc/backends/ascend/elastic/tiling.hpp"
#include "csrc/backends/ascend/transport/aicore_transport_service.hpp"
#include "csrc/backends/ascend/transport/sync_layout.hpp"

namespace elastic = deep_ep::ascend::elastic;
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

void check_incremental_append_does_not_replay() {
    transport::TransportCommand commands[4]{};
    commands[0] = transport::command::make_put(
        transport::TransportTeam::kWorld, 1, 1, 0, 0x1000, 0x2000, 64,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    commands[1] = transport::command::make_flush(
        0, transport::CooperationScope::kParticipant);
    commands[2] = transport::command::make_put(
        transport::TransportTeam::kWorld, 1, 1, 0, 0x1040, 0x2040, 32,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    commands[3] = transport::command::make_flush(
        0, transport::CooperationScope::kParticipant);

    auto state = service::model::make_state(2, 0, 8);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(service::model::execute(commands, 2, state, diagnostic));
    CHECK(state.consumed_count == 2);
    CHECK(state.executed_count == 2);
    CHECK(state.event_count == 2);

    CHECK(service::model::execute(commands, 4, state, diagnostic));
    CHECK(state.consumed_count == 4);
    CHECK(state.executed_count == 4);
    CHECK(state.event_count == 4);
    CHECK(state.events[0] == transport::TransportCommandOpcode::kPut);
    CHECK(state.events[1] == transport::TransportCommandOpcode::kFlush);
    CHECK(state.events[2] == transport::TransportCommandOpcode::kPut);
    CHECK(state.events[3] == transport::TransportCommandOpcode::kFlush);

    CHECK(service::model::execute(commands, 4, state, diagnostic));
    CHECK(state.consumed_count == 4);
    CHECK(state.executed_count == 4);
    CHECK(state.event_count == 4);
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

void check_terminal_completion_precedes_success() {
    auto command = transport::command::make_put_value64(
        transport::TransportTeam::kWorld, 1, 1, 0, 0x1000, 9,
        transport::kDefaultOptions);

    auto state = service::model::make_state(2, 0, 3);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(service::model::execute(&command, 1, state, diagnostic));
    CHECK(state.consumed_count == 1);
    CHECK(state.outstanding == 0);
    CHECK(state.completed == state.submitted);

    state = service::model::make_state(2, 0, 3);
    state.completions_enabled = false;
    diagnostic = {};
    CHECK(!service::model::execute(&command, 1, state, diagnostic));
    CHECK(state.consumed_count == 1);
    CHECK(state.outstanding == 1);
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCompletionTimeout);
    CHECK(diagnostic.command_index == 1);
    CHECK(diagnostic.opcode == transport::TransportCommandOpcode::kFlush);
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
    transport::TransportTeam waited_team = transport::TransportTeam::kWorld;
    int waited_peer = -1;
    std::uint32_t waited_signal_index = 0;
    int wait_count = 0;
    int event_index = 0;
    int fence_event = 0;
    int wait_event = 0;
    int first_load_event = 0;
    transport::DeviceAddress store_addresses[2]{};
    std::uint64_t store_values[2]{};
    int store_events[2]{};
    int store_count = 0;

    void system_fence() {
        staging_fenced = true;
        fence_event = ++event_index;
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
        transport::TransportTeam team, int peer, std::uint32_t signal_index,
        transport::SignalValue target, std::uint64_t) {
        waited_team = team;
        waited_peer = peer;
        waited_signal_index = signal_index;
        ++wait_count;
        wait_event = ++event_index;
        if (release_generation >= target)
            acquired_generation = target;
    }

    transport::SignalValue read_signal(
        transport::TransportTeam, int, std::uint32_t) const {
        return release_generation;
    }

    std::uint64_t load_acquire(transport::DeviceAddress address) {
        const int event = ++event_index;
        if (first_load_event == 0)
            first_load_event = event;
        return *reinterpret_cast<const std::uint64_t*>(address);
    }

    void store_release(
        transport::DeviceAddress address, std::uint64_t value) {
        CHECK(store_count < 2);
        if (store_count < 2) {
            store_addresses[store_count] = address;
            store_values[store_count] = value;
            store_events[store_count] = ++event_index;
            ++store_count;
        }
        *reinterpret_cast<std::uint64_t*>(address) = value;
    }
};

struct ReleaseControlFixture {
    std::uint64_t generation = 0;
    std::uint64_t count = 0;
};

void reset_release_fixture(
    ReleaseProtocolModel* protocol, ReleaseControlFixture* slots,
    int canonical_slot) {
    *protocol = {};
    protocol->release_generation = 7;
    for (int slot = 0; slot < 4; ++slot) {
        slots[slot].generation = 90 + static_cast<std::uint64_t>(slot);
        slots[slot].count = 100 + static_cast<std::uint64_t>(slot);
    }
    slots[canonical_slot].generation = 7;
    slots[canonical_slot].count = 0;
}

void check_wait_peer(
    const ReleaseProtocolModel& protocol,
    const transport::TransportTopology& topology, int expected_world_sender,
    std::uint32_t expected_signal_index, bool should_wait) {
    if (!should_wait) {
        CHECK(protocol.wait_count == 0);
        return;
    }
    transport::TeamPeer expected{};
    CHECK(transport::device::detail::checked_device_team_peer_for_world_rank(
        topology, expected_world_sender, &expected));
    CHECK(protocol.wait_count == 1);
    CHECK(protocol.waited_team == expected.team);
    CHECK(protocol.waited_peer == expected.peer);
    CHECK(protocol.waited_signal_index == expected_signal_index);
}

void check_hybrid_release_observation_boundaries() {
    constexpr bool epilogue_waits[4][4] = {
        {false, true, true, false},
        {true, false, false, true},
        {true, false, false, true},
        {false, true, true, false},
    };

    for (int receiver = 0; receiver < 4; ++receiver) {
        transport::TransportTopology topology{};
        CHECK(transport::build_transport_topology(
            receiver, 4, 2,
            transport::TransportTopologyKind::kLogicalSimulation,
            1, &topology).ok());
        for (int source = 0; source < 4; ++source) {
            ReleaseProtocolModel protocol;
            ReleaseControlFixture slots[4]{};
            reset_release_fixture(&protocol, slots, source);
            const auto boundary = elastic::dispatch_release_boundary(
                source, receiver, 2);
            const auto observation = release_protocol::observe_release_control(
                protocol, topology, boundary, receiver, slots,
                sync_layout::kDispatchReleaseSignalIndex, 7, 64);
            CHECK(boundary.control_slot_world_rank == source);
            CHECK(boundary.remote_acquire_required ==
                  epilogue_waits[receiver][source]);
            CHECK(observation.acquired);
            CHECK(observation.generation == 7);
            CHECK(observation.count == 0);
            check_wait_peer(
                protocol, topology, source,
                sync_layout::kDispatchReleaseSignalIndex,
                epilogue_waits[receiver][source]);
        }

        for (int contributor = 0; contributor < 4; ++contributor) {
            ReleaseProtocolModel protocol;
            ReleaseControlFixture slots[4]{};
            reset_release_fixture(&protocol, slots, contributor);
            const auto boundary = elastic::combine_release_boundary(
                receiver, contributor, 2);
            const auto observation = release_protocol::observe_release_control(
                protocol, topology, boundary, receiver, slots,
                sync_layout::kCombineReleaseSignalIndex, 7, 64);
            CHECK(boundary.control_slot_world_rank == contributor);
            CHECK(boundary.remote_acquire_required ==
                  epilogue_waits[receiver][contributor]);
            CHECK(observation.acquired);
            CHECK(observation.generation == 7);
            CHECK(observation.count == 0);
            check_wait_peer(
                protocol, topology, contributor,
                sync_layout::kCombineReleaseSignalIndex,
                epilogue_waits[receiver][contributor]);
        }
    }
}

void check_hybrid_prepare_observes_ingress_before_control() {
    for (int receiver = 0; receiver < 4; ++receiver) {
        transport::TransportTopology topology{};
        CHECK(transport::build_transport_topology(
            receiver, 4, 2,
            transport::TransportTopologyKind::kLogicalSimulation,
            1, &topology).ok());
        for (int source = 0; source < 4; ++source) {
            const auto route = elastic::classify_world_route(
                source, receiver, 2);
            if (route.kind != elastic::WorldRouteKind::kDiagonal)
                continue;
            ReleaseProtocolModel protocol;
            ReleaseControlFixture slots[4]{};
            reset_release_fixture(&protocol, slots, source);
            const auto boundary =
                elastic::dispatch_prepare_release_boundary(
                    source, receiver, 2);
            const auto observation = release_protocol::observe_release_control(
                protocol, topology, boundary, receiver, slots,
                sync_layout::kDispatchReleaseSignalIndex, 7, 64);
            CHECK(boundary.control_slot_world_rank == source);
            CHECK(boundary.signal_sender_world_rank ==
                  route.ingress_world_rank);
            CHECK(boundary.remote_acquire_required);
            CHECK(observation.acquired);
            CHECK(observation.generation == 7);
            CHECK(observation.count == 0);
            CHECK(protocol.wait_event > 0);
            CHECK(protocol.first_load_event > protocol.wait_event);
            check_wait_peer(
                protocol, topology, route.ingress_world_rank,
                sync_layout::kDispatchReleaseSignalIndex, true);
        }

        for (int contributor = 0; contributor < 4; ++contributor) {
            const auto route = elastic::classify_world_route(
                receiver, contributor, 2);
            if (route.kind != elastic::WorldRouteKind::kDiagonal)
                continue;
            ReleaseProtocolModel protocol;
            ReleaseControlFixture slots[4]{};
            reset_release_fixture(&protocol, slots, contributor);
            const auto boundary =
                elastic::combine_prepare_release_boundary(
                    receiver, contributor, 2);
            const auto observation = release_protocol::observe_release_control(
                protocol, topology, boundary, receiver, slots,
                sync_layout::kCombineReleaseSignalIndex, 7, 64);
            CHECK(boundary.control_slot_world_rank == contributor);
            CHECK(boundary.signal_sender_world_rank ==
                  route.ingress_world_rank);
            CHECK(boundary.remote_acquire_required);
            CHECK(observation.acquired);
            CHECK(observation.generation == 7);
            CHECK(observation.count == 0);
            CHECK(protocol.wait_event > 0);
            CHECK(protocol.first_load_event > protocol.wait_event);
            check_wait_peer(
                protocol, topology, route.ingress_world_rank,
                sync_layout::kCombineReleaseSignalIndex, true);
        }
    }
}

void check_local_canonical_control_publication() {
    for (const std::uint64_t count : {std::uint64_t{0}, std::uint64_t{1}}) {
        ReleaseProtocolModel protocol;
        ReleaseControlFixture slot{99, 88};
        const auto generation_address =
            reinterpret_cast<transport::DeviceAddress>(&slot.generation);
        const auto count_address =
            reinterpret_cast<transport::DeviceAddress>(&slot.count);
        release_protocol::publish_local_control(
            protocol, count_address, count, generation_address, 7);
        CHECK(protocol.fence_event > 0);
        CHECK(protocol.store_count == 2);
        CHECK(protocol.store_addresses[0] == count_address);
        CHECK(protocol.store_values[0] == count);
        CHECK(protocol.store_addresses[1] == generation_address);
        CHECK(protocol.store_values[1] == 7);
        CHECK(protocol.store_events[0] > protocol.fence_event);
        CHECK(protocol.store_events[1] > protocol.store_events[0]);
        CHECK(slot.count == count);
        CHECK(slot.generation == 7);
    }
}

void check_outbound_ingress_counts_survive_reset_after_publish() {
    elastic::CoreTilingInput input{};
    input.operation = elastic::OperationKind::kDispatch;
    input.element_kind = elastic::ElementKind::kBFloat16;
    input.mode_flags = elastic::mode_bit(elastic::CoreMode::kHybrid);
    input.num_tokens = 4;
    input.hidden = 4;
    input.num_experts = 4;
    input.num_topk = 1;
    input.expert_alignment = 1;
    input.num_max_tokens_per_rank = 4;
    input.topology.world_rank = 2;
    input.topology.world_size = 4;
    input.topology.scale_up_rank = 0;
    input.topology.scale_up_size = 2;
    input.topology.scale_out_rank = 1;
    input.topology.scale_out_size = 2;
    input.topology.kind =
        transport::TransportTopologyKind::kLogicalSimulation;
    input.topology.epoch = 1;

    elastic::CoreTiling tiling{};
    CHECK(elastic::build_core_tiling(input, &tiling).ok());
    const auto& layout = tiling.workspace_layout;
    CHECK(layout.scratch_outbound_ingress_count == 4);
    CHECK(layout.scratch_outbound_ingress_counts_offset >=
          layout.scratch_offset);
    CHECK(layout.scratch_outbound_ingress_counts_offset +
              layout.scratch_outbound_ingress_count * sizeof(std::uint64_t) <=
          layout.scratch_offset + layout.scratch_bytes);

    alignas(elastic::kAscendElasticAlignment) std::uint8_t workspace[1024]{};
    ReleaseControlFixture receive_owned{7, 1};
    auto* outbound = reinterpret_cast<std::uint64_t*>(
        workspace + layout.scratch_outbound_ingress_counts_offset);

    // Model an early peer publication followed by this rank's late local reset.
    for (std::uint64_t rank = 0;
         rank < layout.scratch_outbound_ingress_count; ++rank)
        outbound[rank] = 0;
    CHECK(receive_owned.generation == 7);
    CHECK(receive_owned.count == 1);
}

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
    check_incremental_append_does_not_replay();
    check_validation_stops_before_later_commands();
    check_completion_timeout_is_finite();
    check_terminal_completion_precedes_success();
    check_barrier_failure_preserves_failed_world_peer();
    check_service_uses_translated_world_peer();
    check_protocol_validation_precedes_dispatch();
    check_hybrid_release_observation_boundaries();
    check_hybrid_prepare_observes_ingress_before_control();
    check_local_canonical_control_publication();
    check_outbound_ingress_counts_survive_reset_after_publish();
    check_release_acquire_and_selected_barrier_sequence();
    return failures == 0 ? 0 : 1;
}

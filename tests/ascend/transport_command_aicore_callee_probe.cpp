#define __aicore__ __attribute__((noinline))
#define DEEP_EP_ASCEND_AICORE_URMA_SERVICE 1

#include "csrc/backends/ascend/transport/transport_commands.hpp"

namespace transport = deep_ep::ascend::transport;
namespace command = deep_ep::ascend::transport::command;

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

constexpr transport::DeviceTransportError validate_host_barrier() {
    transport::TransportCommand barrier{};
    barrier.opcode = transport::TransportCommandOpcode::kBarrier;
    barrier.options = transport::kWorldTeamMask;
    return command::validate_for_dispatch(barrier, kBarrierTopology);
}

static_assert(!__builtin_has_attribute(
    command::barrier_team_enabled, noinline));
static_assert(!__builtin_has_attribute(
    command::barrier_peer_in_team, noinline));
static_assert(validate_host_barrier() == transport::DeviceTransportError::kNone);

static_assert(__builtin_has_attribute(
    command::aicore_barrier_team_enabled, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_barrier_peer_in_team, noinline));

__aicore__ bool aicore_barrier_predicate_probe() {
    return command::aicore_barrier_team_enabled(
               kBarrierTopology, transport::kWorldTeamMask,
               transport::TransportTeam::kScaleOut) &&
           command::aicore_barrier_peer_in_team(
               kBarrierTopology, transport::TransportTeam::kScaleOut, 3);
}

int main() {
    return aicore_barrier_predicate_probe() ? 0 : 1;
}

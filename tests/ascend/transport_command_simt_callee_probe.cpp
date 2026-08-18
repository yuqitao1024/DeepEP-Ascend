#include "csrc/backends/ascend/transport/transport_commands.hpp"

#define barrier_team_enabled host_barrier_team_must_not_be_called
#define __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((noinline))
#define DEEP_EP_ASCEND_SIMT_DEVICE 1
#include "csrc/backends/ascend/transport/execution_domain_helpers.hpp"
#undef barrier_team_enabled

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

static_assert(!__builtin_has_attribute(
    command::barrier_team_enabled, noinline));
static_assert(__builtin_has_attribute(
    command::simt_barrier_team_enabled, noinline));

__SIMT_DEVICE_FUNCTIONS_DECL__ bool simt_execution_helpers_probe() {
    return command::simt_barrier_team_enabled(
        kBarrierTopology, transport::kWorldTeamMask,
        transport::TransportTeam::kScaleOut);
}

int main() {
    return simt_execution_helpers_probe() ? 0 : 1;
}

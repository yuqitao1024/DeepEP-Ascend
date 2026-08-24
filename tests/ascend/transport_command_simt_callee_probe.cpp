#include "csrc/backends/ascend/transport/transport_commands.hpp"

#define barrier_team_enabled host_barrier_team_must_not_be_called
#define publish_request host_publish_request_must_not_be_called
#define observe_request host_observe_request_must_not_be_called
#define timeout_request host_timeout_request_must_not_be_called
#define __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((noinline))
#define DEEP_EP_ASCEND_SIMT_DEVICE 1
#define DEEP_EP_ASCEND_SIMT_GLOBAL
#include "csrc/backends/ascend/transport/execution_domain_helpers.hpp"
#undef timeout_request
#undef observe_request
#undef publish_request
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
static_assert(__builtin_has_attribute(
    command::simt_publish_request, noinline));
static_assert(__builtin_has_attribute(
    command::simt_observe_request, noinline));
static_assert(__builtin_has_attribute(
    command::simt_timeout_request, noinline));

__SIMT_DEVICE_FUNCTIONS_DECL__ bool simt_execution_helpers_probe() {
    transport::DeviceRequest completed{};
    if (!command::simt_publish_request(&completed, 3, 4, 17) ||
        !command::simt_observe_request(
            &completed, 17, 0, 4, 17, 4,
            transport::DeviceTransportError::kNone) ||
        completed.state != transport::DeviceRequestState::kCompleted)
        return false;

    transport::DeviceRequest failed{};
    if (!command::simt_publish_request(&failed, 5, 6, 18) ||
        !command::simt_observe_request(
            &failed, 18, 0, 5, 18, 5,
            transport::DeviceTransportError::kInvalidAddress) ||
        failed.state != transport::DeviceRequestState::kFailed ||
        failed.terminal_error !=
            transport::DeviceTransportError::kInvalidAddress)
        return false;

    transport::DeviceRequest timed_out{};
    if (!command::simt_publish_request(&timed_out, 6, 7, 19) ||
        !command::simt_timeout_request(&timed_out) ||
        timed_out.state != transport::DeviceRequestState::kFailed ||
        timed_out.terminal_error !=
            transport::DeviceTransportError::kCompletionTimeout)
        return false;

    return command::simt_barrier_team_enabled(
        kBarrierTopology, transport::kWorldTeamMask,
        transport::TransportTeam::kScaleOut);
}

int main() {
    return simt_execution_helpers_probe() ? 0 : 1;
}

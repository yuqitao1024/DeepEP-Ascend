#define __aicore__ __attribute__((noinline))
#define DEEP_EP_ASCEND_AICORE_URMA_SERVICE 1

#include "csrc/backends/ascend/transport/sync_layout.hpp"
#include "csrc/backends/ascend/transport/transport_commands.hpp"

#define valid_staged_context_header host_staged_header_must_not_be_called
#define valid_command_queue_header host_queue_header_must_not_be_called
#define valid_service_state_header host_service_header_must_not_be_called
#define valid_diagnostic_header host_diagnostic_header_must_not_be_called
#define mix_registration_cookie host_cookie_mix_must_not_be_called
#define registration_cookie host_cookie_must_not_be_called
#define valid_registration_cookie host_cookie_validation_must_not_be_called
#define barrier_team_enabled host_barrier_team_must_not_be_called
#define barrier_peer_in_team host_barrier_peer_must_not_be_called
#define barrier_offset host_barrier_offset_must_not_be_called
#include "csrc/backends/ascend/transport/execution_domain_helpers.hpp"
#undef barrier_offset
#undef barrier_peer_in_team
#undef barrier_team_enabled
#undef valid_registration_cookie
#undef registration_cookie
#undef mix_registration_cookie
#undef valid_diagnostic_header
#undef valid_service_state_header
#undef valid_command_queue_header
#undef valid_staged_context_header

namespace transport = deep_ep::ascend::transport;
namespace command = deep_ep::ascend::transport::command;
namespace sync_layout = deep_ep::ascend::transport::sync_layout;

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

constexpr bool validate_host_helpers() {
    constexpr std::uintptr_t kQueue = 0x1000;
    constexpr std::uintptr_t kCommands = 0x2000;
    constexpr std::uintptr_t kService = 0x3000;
    constexpr std::uintptr_t kDiagnostic = 0x4000;
    constexpr std::uint32_t kCapacity = 6;
    constexpr auto cookie = command::registration_cookie(
        kQueue, kCommands, kService, kDiagnostic, kCapacity);
    return command::valid_staged_context_header(
               transport::kTransportCommandAbiVersion,
               sizeof(transport::StagedTransportContext),
               transport::kStagedTransportCannCompatibility, kQueue) &&
           command::valid_command_queue_header(
               transport::kTransportCommandAbiVersion,
               sizeof(transport::TransportCommandQueue), kCommands,
               kCapacity, kCapacity, kService, kDiagnostic) &&
           command::valid_service_state_header(
               transport::kTransportCommandAbiVersion,
               sizeof(transport::TransportServiceState)) &&
           command::valid_diagnostic_header(
               transport::kTransportCommandAbiVersion) &&
           command::valid_registration_cookie(
               cookie, kQueue, kCommands, kService, kDiagnostic,
               kCapacity) &&
           sync_layout::barrier_offset(4, 0, 0) == 128 &&
           command::barrier_team_enabled(
               kBarrierTopology, transport::kWorldTeamMask,
               transport::TransportTeam::kScaleOut) &&
           command::barrier_peer_in_team(
               kBarrierTopology, transport::TransportTeam::kScaleOut, 3);
}

static_assert(validate_host_helpers());
static_assert(!__builtin_has_attribute(
    command::valid_staged_context_header, noinline));
static_assert(!__builtin_has_attribute(
    command::valid_command_queue_header, noinline));
static_assert(!__builtin_has_attribute(
    command::valid_service_state_header, noinline));
static_assert(!__builtin_has_attribute(
    command::valid_diagnostic_header, noinline));
static_assert(!__builtin_has_attribute(
    command::mix_registration_cookie, noinline));
static_assert(!__builtin_has_attribute(
    command::registration_cookie, noinline));
static_assert(!__builtin_has_attribute(
    command::valid_registration_cookie, noinline));
static_assert(!__builtin_has_attribute(
    command::barrier_team_enabled, noinline));
static_assert(!__builtin_has_attribute(
    command::barrier_peer_in_team, noinline));
static_assert(!__builtin_has_attribute(
    sync_layout::barrier_offset, noinline));

static_assert(__builtin_has_attribute(
    command::aicore_valid_staged_context_header, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_valid_command_queue_header, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_valid_service_state_header, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_valid_diagnostic_header, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_mix_registration_cookie, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_registration_cookie, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_valid_registration_cookie, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_barrier_team_enabled, noinline));
static_assert(__builtin_has_attribute(
    command::aicore_barrier_peer_in_team, noinline));
static_assert(__builtin_has_attribute(
    sync_layout::aicore_barrier_offset, noinline));

__aicore__ bool aicore_execution_helpers_probe() {
    constexpr std::uintptr_t kQueue = 0x1000;
    constexpr std::uintptr_t kCommands = 0x2000;
    constexpr std::uintptr_t kService = 0x3000;
    constexpr std::uintptr_t kDiagnostic = 0x4000;
    constexpr std::uint32_t kCapacity = 6;
    const auto mixed = command::aicore_mix_registration_cookie(1, 2);
    const auto cookie = command::aicore_registration_cookie(
        kQueue, kCommands, kService, kDiagnostic, kCapacity);
    return mixed != 0 &&
           command::aicore_valid_staged_context_header(
               transport::kTransportCommandAbiVersion,
               sizeof(transport::StagedTransportContext),
               transport::kStagedTransportCannCompatibility, kQueue) &&
           command::aicore_valid_command_queue_header(
               transport::kTransportCommandAbiVersion,
               sizeof(transport::TransportCommandQueue), kCommands,
               kCapacity, kCapacity, kService, kDiagnostic) &&
           command::aicore_valid_service_state_header(
               transport::kTransportCommandAbiVersion,
               sizeof(transport::TransportServiceState)) &&
           command::aicore_valid_diagnostic_header(
               transport::kTransportCommandAbiVersion) &&
           command::aicore_valid_registration_cookie(
               cookie, kQueue, kCommands, kService, kDiagnostic,
               kCapacity) &&
           sync_layout::aicore_barrier_offset(4, 0, 0) == 128 &&
           command::aicore_barrier_team_enabled(
               kBarrierTopology, transport::kWorldTeamMask,
               transport::TransportTeam::kScaleOut) &&
           command::aicore_barrier_peer_in_team(
               kBarrierTopology, transport::TransportTeam::kScaleOut, 3);
}

int main() {
    return aicore_execution_helpers_probe() ? 0 : 1;
}

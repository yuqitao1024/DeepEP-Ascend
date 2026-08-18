#include "csrc/backends/ascend/transport/types.hpp"

#define checked_team_peer_for_world_rank host_only_mapping_must_not_be_used
#define __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((noinline))
#define DEEP_EP_ASCEND_SIMT_DEVICE 1
#include "csrc/backends/ascend/elastic/release_protocol.hpp"
#undef checked_team_peer_for_world_rank

namespace transport = deep_ep::ascend::transport;
namespace release_protocol = deep_ep::ascend::elastic::release_protocol;

static_assert(__builtin_has_attribute(
    transport::device::detail::checked_device_team_peer_for_world_rank,
    noinline));

struct RouteProbeFacade {
    transport::TransportTeam team = transport::TransportTeam::kWorld;
    int peer = -1;
    std::uint32_t signal_index = 0;
    std::uint64_t generation = 0;

    void wait_signal(
        transport::TransportTeam route_team, int route_peer,
        std::uint32_t route_signal_index, std::uint64_t target,
        std::uint64_t) {
        team = route_team;
        peer = route_peer;
        signal_index = route_signal_index;
        generation = target;
    }

    transport::SignalValue read_signal(
        transport::TransportTeam, int, std::uint32_t) {
        return generation;
    }
};

int main() {
    transport::TransportTopology topology{};
    if (!transport::build_transport_topology(
            0, 4, 2,
            transport::TransportTopologyKind::kLogicalSimulation,
            1, &topology).ok())
        return 1;

    RouteProbeFacade facade;
    constexpr std::uint32_t kSignalIndex = 3;
    if (!release_protocol::acquire_release(
            facade, topology, 2, kSignalIndex, 7, 1000))
        return 2;
    return facade.team == transport::TransportTeam::kScaleOut &&
            facade.peer == 1 && facade.signal_index == kSignalIndex ? 0 : 3;
}

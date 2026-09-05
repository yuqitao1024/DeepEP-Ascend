#include <cstdlib>

#include "csrc/backends/ascend/transport/channel_config.hpp"
#include "csrc/backends/ascend/transport/topology_config.hpp"
#include "csrc/backends/ascend/elastic/runtime.hpp"

using namespace deep_ep::ascend::transport;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    TransportTopology flat{};
    CHECK(build_transport_topology(
              3, 4, 4, TransportTopologyKind::kFlatScaleUp, 1, &flat)
              .ok());
    CHECK(flat.abi_version == kTransportTopologyAbiVersion);
    CHECK(flat.struct_size == sizeof(TransportTopology));
    CHECK(flat.kind == TransportTopologyKind::kFlatScaleUp);
    CHECK(flat.epoch == 1);
    CHECK(flat.world_rank == 3 && flat.world_size == 4);
    CHECK(flat.scale_up_rank == 3 && flat.scale_up_size == 4);
    CHECK(flat.scale_out_rank == 0 && flat.scale_out_size == 1);

    TransportTopology topology{};
    CHECK(build_transport_topology(
              3, 4, 2, TransportTopologyKind::kLogicalSimulation, 17,
              &topology)
              .ok());
    CHECK(topology.scale_up_rank == 1 && topology.scale_up_size == 2);
    CHECK(topology.scale_out_rank == 1 && topology.scale_out_size == 2);
    const auto core_topology =
        deep_ep::ascend::elastic::core_topology_from_transport(topology);
    CHECK(core_topology.kind == TransportTopologyKind::kLogicalSimulation);
    CHECK(core_topology.epoch == 17);

    const auto logical_physical_domain =
        physical_transport_domain_size(topology, kNoCapabilities);
    CHECK(logical_physical_domain.first == 1 &&
          logical_physical_domain.second == 4);
    const auto flat_physical_domain =
        physical_transport_domain_size(flat, kNoCapabilities);
    CHECK(flat_physical_domain.first == 1 &&
          flat_physical_domain.second == 4);

    TransportTopology physical{};
    CHECK(build_transport_topology(
              3, 4, 2, TransportTopologyKind::kPhysical2D, 17, &physical)
              .ok());
    const auto disabled_physical_domain =
        physical_transport_domain_size(physical, kNoCapabilities);
    CHECK(disabled_physical_domain.first == 1 &&
          disabled_physical_domain.second == 4);
    const auto enabled_physical_domain = physical_transport_domain_size(
        physical, capability_bit(TransportCapability::kScaleOutTeam));
    CHECK(enabled_physical_domain.first == 2 &&
          enabled_physical_domain.second == 2);

    int world_peer = -1;
    CHECK(checked_team_world_rank(
        topology, TransportTeam::kWorld, 0, &world_peer));
    CHECK(world_peer == 0);
    CHECK(checked_team_world_rank(
        topology, TransportTeam::kScaleUp, 0, &world_peer));
    CHECK(world_peer == 2);
    CHECK(checked_team_world_rank(
        topology, TransportTeam::kScaleOut, 0, &world_peer));
    CHECK(world_peer == 1);
    CHECK(!checked_team_world_rank(
        topology, TransportTeam::kScaleUp, 2, &world_peer));

    struct RouteFixture {
        int source;
        int destination;
        TransportTeam team;
        int peer;
    };
    constexpr RouteFixture routes[] = {
        {0, 1, TransportTeam::kScaleUp, 1},
        {0, 2, TransportTeam::kScaleOut, 1},
        {0, 3, TransportTeam::kWorld, 3},
        {1, 0, TransportTeam::kScaleUp, 0},
        {1, 2, TransportTeam::kWorld, 2},
        {1, 3, TransportTeam::kScaleOut, 1},
        {2, 0, TransportTeam::kScaleOut, 0},
        {2, 1, TransportTeam::kWorld, 1},
        {2, 3, TransportTeam::kScaleUp, 1},
        {3, 0, TransportTeam::kWorld, 0},
        {3, 1, TransportTeam::kScaleOut, 0},
        {3, 2, TransportTeam::kScaleUp, 0},
    };
    for (const auto& fixture : routes) {
        TransportTopology source_topology{};
        CHECK(build_transport_topology(
                  fixture.source, 4, 2,
                  TransportTopologyKind::kLogicalSimulation, 17,
                  &source_topology)
                  .ok());
        TeamPeer route{};
        CHECK(checked_team_peer_for_world_rank(
            source_topology, fixture.destination, &route));
        CHECK(route.team == fixture.team);
        CHECK(route.peer == fixture.peer);
        CHECK(route.world_peer == fixture.destination);
        CHECK(checked_team_world_rank(
            source_topology, route.team, route.peer, &world_peer));
        CHECK(world_peer == fixture.destination);
    }
    TeamPeer route{};
    CHECK(!checked_team_peer_for_world_rank(topology, -1, &route));
    CHECK(!checked_team_peer_for_world_rank(topology, 4, &route));
    auto malformed_topology = topology;
    malformed_topology.struct_size = 0;
    CHECK(!checked_team_peer_for_world_rank(
        malformed_topology, 0, &route));

    for (int source = 0; source < 4; ++source) {
        TransportTopology source_topology{};
        CHECK(build_transport_topology(
                  source, 4, 4, TransportTopologyKind::kFlatScaleUp, 1,
                  &source_topology)
                  .ok());
        for (int destination = 0; destination < 4; ++destination) {
            CHECK(checked_team_peer_for_world_rank(
                source_topology, destination, &route));
            if (destination == source)
                CHECK(route.team == TransportTeam::kWorld &&
                      route.peer == source);
            else
                CHECK(route.team == TransportTeam::kScaleUp &&
                      route.peer == destination);
            CHECK(route.world_peer == destination);
        }
    }

    TransportTopology identity_peer{};
    CHECK(build_transport_topology(
              1, 4, 2, TransportTopologyKind::kLogicalSimulation, 17,
              &identity_peer)
              .ok());
    CHECK(same_transport_topology_identity(topology, identity_peer));
    identity_peer.epoch = 18;
    CHECK(!same_transport_topology_identity(topology, identity_peer));
    identity_peer = topology;
    identity_peer.kind = TransportTopologyKind::kPhysical2D;
    CHECK(!same_transport_topology_identity(topology, identity_peer));

    for (const auto invalid : {
             build_transport_topology(
                 0, 0, 1, TransportTopologyKind::kFlatScaleUp, 1, &flat),
             build_transport_topology(
                 4, 4, 4, TransportTopologyKind::kFlatScaleUp, 1, &flat),
             build_transport_topology(
                 0, 4, 0, TransportTopologyKind::kLogicalSimulation, 1,
                 &flat),
             build_transport_topology(
                 0, 4, 3, TransportTopologyKind::kLogicalSimulation, 1,
                 &flat),
             build_transport_topology(
                 0, 4, 2, TransportTopologyKind::kFlatScaleUp, 1, &flat),
             build_transport_topology(
                 0, 4, 4, TransportTopologyKind::kLogicalSimulation, 1,
                 &flat),
             build_transport_topology(
                 0, 4, 2, TransportTopologyKind::kLogicalSimulation, 0,
                 &flat),
         })
        CHECK(invalid.code == TransportStatusCode::kInvalidArgument);
    CHECK(build_transport_topology(
              0, 4, 2, TransportTopologyKind::kLogicalSimulation, 1,
              nullptr)
              .code == TransportStatusCode::kInvalidArgument);

    unsetenv("DEEP_EP_ASCEND_SCALE_UP_SIZE");
    unsetenv("DEEP_EP_ASCEND_LOGICAL_SIMULATION");
    unsetenv("DEEP_EP_ASCEND_TOPOLOGY_EPOCH");
    TransportConfig config{};
    config.rank = 3;
    config.world_size = 4;
    CHECK(configure_transport_topology_from_environment(&config).ok());
    CHECK(config.scale_up_size == 4);
    CHECK(config.topology_kind == TransportTopologyKind::kFlatScaleUp);
    CHECK(config.topology_epoch == 1);

    setenv("DEEP_EP_ASCEND_SCALE_UP_SIZE", "2", 1);
    setenv("DEEP_EP_ASCEND_LOGICAL_SIMULATION", "1", 1);
    setenv("DEEP_EP_ASCEND_TOPOLOGY_EPOCH", "17", 1);
    CHECK(configure_transport_topology_from_environment(&config).ok());
    CHECK(config.scale_up_size == 2);
    CHECK(config.topology_kind ==
          TransportTopologyKind::kLogicalSimulation);
    CHECK(config.topology_epoch == 17);

    setenv("DEEP_EP_ASCEND_SCALE_UP_SIZE", "3", 1);
    CHECK(!configure_transport_topology_from_environment(&config).ok());
    setenv("DEEP_EP_ASCEND_SCALE_UP_SIZE", "2", 1);
    setenv("DEEP_EP_ASCEND_LOGICAL_SIMULATION", "true", 1);
    CHECK(!configure_transport_topology_from_environment(&config).ok());
    unsetenv("DEEP_EP_ASCEND_SCALE_UP_SIZE");
    setenv("DEEP_EP_ASCEND_LOGICAL_SIMULATION", "1", 1);
    CHECK(!configure_transport_topology_from_environment(&config).ok());

    unsetenv("DEEP_EP_ASCEND_SCALE_UP_SIZE");
    unsetenv("DEEP_EP_ASCEND_LOGICAL_SIMULATION");
    unsetenv("DEEP_EP_ASCEND_TOPOLOGY_EPOCH");

    unsetenv("DEEP_EP_ASCEND_CHANNELS");
    config.requested_channels = 0;
    CHECK(configure_transport_channels_from_environment(&config).ok());
    CHECK(config.requested_channels == 1);
    setenv("DEEP_EP_ASCEND_CHANNELS", "4", 1);
    CHECK(configure_transport_channels_from_environment(&config).ok());
    CHECK(config.requested_channels == 4);
    setenv("DEEP_EP_ASCEND_CHANNELS", "0", 1);
    CHECK(!configure_transport_channels_from_environment(&config).ok());
    setenv("DEEP_EP_ASCEND_CHANNELS", "5", 1);
    CHECK(!configure_transport_channels_from_environment(&config).ok());
    setenv("DEEP_EP_ASCEND_CHANNELS", "2x", 1);
    CHECK(!configure_transport_channels_from_environment(&config).ok());
    unsetenv("DEEP_EP_ASCEND_CHANNELS");
    return 0;
}

#pragma once

#include "types.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_DEVICE_TOPOLOGY_CALLEE \
    __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#else
#define DEEP_EP_ASCEND_DEVICE_TOPOLOGY_CALLEE inline
#endif

namespace deep_ep::ascend::transport::device::detail {

DEEP_EP_ASCEND_DEVICE_TOPOLOGY_CALLEE bool checked_world_peer(
    const TransportTopology& topology, TransportTeam team, int peer,
    int* world_peer) {
    if (world_peer == nullptr || peer < 0 ||
        topology.abi_version != kTransportTopologyAbiVersion ||
        topology.struct_size != sizeof(TransportTopology) ||
        topology.epoch == 0 || topology.world_size <= 0 ||
        topology.world_rank < 0 ||
        topology.world_rank >= topology.world_size ||
        topology.scale_up_size <= 0 || topology.scale_out_size <= 0 ||
        topology.scale_up_rank < 0 ||
        topology.scale_up_rank >= topology.scale_up_size ||
        topology.scale_out_rank < 0 ||
        topology.scale_out_rank >= topology.scale_out_size ||
        static_cast<std::int64_t>(topology.scale_up_size) *
                topology.scale_out_size != topology.world_size ||
        topology.scale_up_rank !=
            topology.world_rank % topology.scale_up_size ||
        topology.scale_out_rank !=
            topology.world_rank / topology.scale_up_size)
        return false;
    if (topology.kind == TransportTopologyKind::kFlatScaleUp) {
        if (topology.scale_up_size != topology.world_size ||
            topology.scale_out_size != 1)
            return false;
    } else if (topology.kind == TransportTopologyKind::kPhysical2D ||
               topology.kind == TransportTopologyKind::kLogicalSimulation) {
        if (topology.scale_out_size < 2)
            return false;
    } else {
        return false;
    }

    int translated = -1;
    switch (team) {
        case TransportTeam::kWorld:
            if (peer >= topology.world_size)
                return false;
            translated = peer;
            break;
        case TransportTeam::kScaleUp:
            if (peer >= topology.scale_up_size)
                return false;
            translated = topology.scale_out_rank * topology.scale_up_size +
                peer;
            break;
        case TransportTeam::kScaleOut:
            if (peer >= topology.scale_out_size)
                return false;
            translated = peer * topology.scale_up_size +
                topology.scale_up_rank;
            break;
        default: return false;
    }
    if (translated < 0 || translated >= topology.world_size)
        return false;
    *world_peer = translated;
    return true;
}

DEEP_EP_ASCEND_DEVICE_TOPOLOGY_CALLEE bool valid_topology(
    const TransportTopology& topology) {
    int translated = -1;
    return checked_world_peer(
               topology, TransportTeam::kWorld, topology.world_rank,
               &translated) &&
           translated == topology.world_rank;
}

DEEP_EP_ASCEND_DEVICE_TOPOLOGY_CALLEE bool
checked_device_team_peer_for_world_rank(
    const TransportTopology& topology, int world_peer,
    TeamPeer* team_peer) {
    if (team_peer == nullptr || world_peer < 0 ||
        world_peer >= topology.world_size || !valid_topology(topology))
        return false;

    TeamPeer result{};
    result.world_peer = world_peer;
    if (world_peer == topology.world_rank) {
        result.team = TransportTeam::kWorld;
        result.peer = world_peer;
    } else {
        const int peer_scale_up_rank = world_peer % topology.scale_up_size;
        const int peer_scale_out_rank = world_peer / topology.scale_up_size;
        if (peer_scale_out_rank == topology.scale_out_rank) {
            result.team = TransportTeam::kScaleUp;
            result.peer = peer_scale_up_rank;
        } else if (peer_scale_up_rank == topology.scale_up_rank) {
            result.team = TransportTeam::kScaleOut;
            result.peer = peer_scale_out_rank;
        } else {
            result.team = TransportTeam::kWorld;
            result.peer = world_peer;
        }
    }

    int translated = -1;
    if (!checked_world_peer(
            topology, result.team, result.peer, &translated) ||
        translated != world_peer)
        return false;
    *team_peer = result;
    return true;
}

}  // namespace deep_ep::ascend::transport::device::detail

#undef DEEP_EP_ASCEND_DEVICE_TOPOLOGY_CALLEE

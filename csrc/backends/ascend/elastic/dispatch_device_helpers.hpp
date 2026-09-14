#pragma once

DEEP_EP_ASCEND_SIMT_CALLEE transport::DeviceTransportContext
make_hybrid_dispatch_context(
    std::uint32_t abi_version, std::uint32_t struct_size,
    std::uint64_t capabilities, std::uintptr_t local_window_base,
    std::uintptr_t channel_table, std::uintptr_t peer_address_table,
    std::uint32_t topology_abi_version,
    std::uint32_t topology_struct_size,
    int world_rank, int world_size, int scale_up_rank, int scale_up_size,
    int scale_out_rank, int scale_out_size,
    std::uint32_t scale_up_direct, std::uint32_t topology_kind,
    std::uint64_t topology_epoch, std::uintptr_t backend_context) {
    transport::DeviceTransportContext result{};
    result.abi_version = abi_version;
    result.struct_size = struct_size;
    result.capabilities = capabilities;
    result.topology.abi_version = topology_abi_version;
    result.topology.struct_size = topology_struct_size;
    result.topology.world_rank = world_rank;
    result.topology.world_size = world_size;
    result.topology.scale_up_rank = scale_up_rank;
    result.topology.scale_up_size = scale_up_size;
    result.topology.scale_out_rank = scale_out_rank;
    result.topology.scale_out_size = scale_out_size;
    result.topology.scale_up_direct = scale_up_direct != 0;
    result.topology.kind =
        static_cast<transport::TransportTopologyKind>(topology_kind);
    result.topology.epoch = topology_epoch;
    result.local_window_base = local_window_base;
    result.channel_table = channel_table;
    result.peer_address_table = peer_address_table;
    result.backend_context = backend_context;
    return result;
}

#include <cstdint>
#include <cstring>

#include "csrc/backends/ascend/transport/device_transport_facade.hpp"
#include "csrc/backends/ascend/transport/device_transport_stub.hpp"
#include "csrc/backends/ascend/transport/stub_transport.hpp"

using namespace deep_ep::ascend::transport;

int main() {
    std::uint64_t storage[8] = {11, 22, 33, 44, 55, 66, 77, 88};
    std::uint64_t storage_snapshot[8];
    std::memcpy(storage_snapshot, storage, sizeof(storage));

    auto context = make_device_transport_context();
    context.topology.world_rank = 5;
    context.topology.scale_up_rank = 1;
    context.topology.scale_out_rank = 2;
    context.local_window_base = reinterpret_cast<std::uintptr_t>(storage);
    const DeviceAddress storage_address =
        reinterpret_cast<DeviceAddress>(storage);

    DeviceTransportFacade transport(context, 3);
    if (!transport.is_peer_directly_accessible(TransportTeam::kWorld, 5))
        return 1;
    if (transport.is_peer_directly_accessible(TransportTeam::kWorld, 4))
        return 2;
    if (transport.get_symmetric_pointer(
            TransportTeam::kScaleUp, 1, storage_address) != storage_address)
        return 3;
    if (transport.get_symmetric_pointer(
            TransportTeam::kScaleUp, 0, storage_address) != kNullDeviceAddress)
        return 4;
    if (transport.get_symmetric_offset(
            reinterpret_cast<DeviceAddress>(storage + 2)) !=
        2 * sizeof(std::uint64_t))
        return 5;

    DeviceRequest request{};
    const DeviceRequest request_snapshot = request;

    transport.put(
        TransportTeam::kWorld, 0, storage_address, storage_address,
        sizeof(storage),
        CooperationScope::kParticipant, MemorySegment::kDevice,
        kDefaultOptions, RemoteAction::none());
    transport.get(
        TransportTeam::kScaleOut, 0, storage_address, storage_address,
        sizeof(storage),
        CooperationScope::kWorkgroup, MemorySegment::kMixed,
        kAggregateRequests);
    transport.put_value(
        TransportTeam::kScaleUp, 0, storage_address, 7,
        sizeof(std::uint64_t),
        kDefaultOptions);
    transport.remote_add_release(
        TransportTeam::kWorld, 0, storage_address, 1);
    transport.signal(
        TransportTeam::kScaleOut, 0, RemoteAction::signal_increment(1));
    if (transport.read_signal(TransportTeam::kWorld, 0, 1) != 0)
        return 6;
    transport.wait_signal(TransportTeam::kWorld, 0, 1, 1, 1000);
    transport.flush(CooperationScope::kDevice);
    transport.flush_async(
        TransportTeam::kWorld, 0, CooperationScope::kParticipant, &request);
    transport.wait(&request);
    if (transport.load_acquire(storage_address) != 0)
        return 7;
    transport.store_release(storage_address, 99);
    transport.system_fence();
    transport.device_barrier(1, storage_address, 1000);

    if (std::memcmp(storage_snapshot, storage, sizeof(storage)) != 0)
        return 8;
    if (std::memcmp(&request_snapshot, &request, sizeof(request)) != 0)
        return 9;

    StubHostTransport host_transport;
    if (host_transport.capabilities() != kNoCapabilities)
        return 10;

    return 0;
}

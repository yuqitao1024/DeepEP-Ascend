#include <cstdint>
#include <type_traits>

#include "csrc/backends/ascend/transport/device_transport.hpp"

using namespace deep_ep::ascend::transport;

static_assert(std::is_trivially_copyable_v<TransportTopology>);
static_assert(std::is_trivially_copyable_v<DeviceTransportContext>);
static_assert(std::is_trivially_copyable_v<DeviceRequest>);
static_assert(alignof(DeviceRequest) == 16);
static_assert(sizeof(DeviceRequest) == 32);
static_assert(kNoCapabilities == 0);
static_assert(capability_bit(TransportCapability::kDevicePut) != 0);
static_assert(capability_bit(TransportCapability::kDeviceGet) !=
              capability_bit(TransportCapability::kDevicePut));

int main() {
    const auto missing = capability_bit(TransportCapability::kDevicePut) |
                         capability_bit(TransportCapability::kRemoteSignal);
    if (capability_names(missing) != "device_put, remote_signal")
        return 1;

    const auto status = TransportStatus::unsupported(
        "dispatch", "missing device transport capabilities: " +
                    capability_names(missing));
    if (status.ok() ||
        status.code != TransportStatusCode::kUnsupportedCapability ||
        status.operation != "dispatch" || status.backend_code != 0)
        return 2;

    DeviceTransportContext context = make_device_transport_context();
    if (context.abi_version != kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(DeviceTransportContext))
        return 3;
    return 0;
}

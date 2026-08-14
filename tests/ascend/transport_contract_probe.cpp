#include <cstdint>
#include <string>
#include <type_traits>

#include "csrc/backends/ascend/transport/device_transport.hpp"
#include "csrc/backends/ascend/transport/stub_transport.hpp"

using namespace deep_ep::ascend::transport;

TransportConfig valid_config() {
    TransportConfig config;
    config.rank = 0;
    config.world_size = 1;
    config.communicator_handle = 0;
    config.cpu_communicator_empty = true;
    config.device_buffer_bytes = 4096;
    config.cpu_buffer_bytes = 0;
    config.requested_channels = 3;
    return config;
}

static_assert(std::is_trivially_copyable_v<TransportTopology>);
static_assert(std::is_trivially_copyable_v<DeviceTransportContext>);
static_assert(std::is_trivially_copyable_v<DeviceRequest>);
static_assert(alignof(DeviceRequest) == 16);
static_assert(sizeof(DeviceRequest) == 32);
static_assert(std::is_trivially_copyable_v<RemoteAction>);
static_assert(kDefaultOptions == 0);
static_assert((kAggregateRequests & kDefaultOptions) == 0);
static_assert(kNoCapabilities == 0);
static_assert(capability_bit(TransportCapability::kDevicePut) != 0);
static_assert(capability_bit(TransportCapability::kDeviceGet) !=
              capability_bit(TransportCapability::kDevicePut));

using DevicePut = void (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    void*, const void*, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions, const RemoteAction&);
using DeviceGet = void (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    const void*, void*, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions);
using ReadSignal = SignalValue (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    std::uint32_t);
using WaitSignal = void (*)(
    const DeviceTransportContext&, device::DeviceChannel, TransportTeam, int,
    std::uint32_t, SignalValue, std::uint64_t);

static_assert(std::is_same_v<decltype(&device::put), DevicePut>);
static_assert(std::is_same_v<decltype(&device::get), DeviceGet>);
static_assert(std::is_same_v<decltype(&device::read_signal), ReadSignal>);
static_assert(std::is_same_v<decltype(&device::wait_signal), WaitSignal>);

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

    const auto no_action = RemoteAction::none();
    const auto signal_add = RemoteAction::signal_add(128, 7);
    const auto signal_increment = RemoteAction::signal_increment(3);
    if (no_action.kind != RemoteActionKind::kNone ||
        signal_add.kind != RemoteActionKind::kSignalAdd ||
        signal_add.symmetric_offset != 128 || signal_add.value != 7 ||
        signal_increment.kind != RemoteActionKind::kSignalIncrement ||
        signal_increment.signal_index != 3 || signal_increment.value != 1)
        return 4;

    auto created = make_stub_transport(valid_config());
    if (!created.status.ok() || !created.transport)
        return 10;
    if (created.transport->capabilities() != kNoCapabilities)
        return 11;

    const auto required = capability_bit(TransportCapability::kDevicePut) |
                          capability_bit(TransportCapability::kRemoteSignal);
    const auto requirement = created.transport->require_capabilities(
        required, "dispatch");
    if (requirement.code != TransportStatusCode::kUnsupportedCapability ||
        requirement.message.find("device_put, remote_signal") == std::string::npos)
        return 12;

    TransportTopology topology;
    if (created.transport->query_topology(&topology).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 13;
    if (created.transport->register_symmetric_window(nullptr, 4096).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 14;
    std::uintptr_t peer_pointer = 0;
    if (created.transport->get_peer_base_pointer(
            TransportTeam::kWorld, 0, &peer_pointer).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 15;
    if (created.transport->acquire_channels(
            3, CooperationScope::kWorkgroup).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 16;
    DeviceTransportContext exported = make_device_transport_context();
    if (created.transport->export_device_context(&exported).code !=
        TransportStatusCode::kUnsupportedCapability)
        return 17;
    if (created.transport->host_barrier().code !=
        TransportStatusCode::kUnsupportedCapability)
        return 18;
    if (!created.transport->unregister_symmetric_window().ok() ||
        !created.transport->release_channels().ok())
        return 19;
    if (!created.transport->destroy().ok() || !created.transport->destroy().ok())
        return 20;

    const auto expect_invalid = [](TransportConfig invalid,
                                   const std::string& fragment) {
        auto rejected = make_stub_transport(invalid);
        return !rejected.transport &&
               rejected.status.code == TransportStatusCode::kInvalidArgument &&
               rejected.status.message.find(fragment) != std::string::npos;
    };

    auto invalid = valid_config();
    invalid.communicator_handle = 7;
    if (!expect_invalid(invalid, "communicator_handle must be zero"))
        return 21;
    invalid = valid_config();
    invalid.world_size = 0;
    if (!expect_invalid(invalid, "world_size must be positive"))
        return 22;
    invalid = valid_config();
    invalid.rank = invalid.world_size;
    if (!expect_invalid(invalid, "rank must be in [0, world_size)"))
        return 23;
    invalid = valid_config();
    invalid.cpu_communicator_empty = false;
    if (!expect_invalid(invalid, "cpu_communicator must be empty"))
        return 24;
    invalid = valid_config();
    invalid.device_buffer_bytes = 0;
    if (!expect_invalid(invalid, "device_buffer_bytes must be positive"))
        return 25;
    invalid = valid_config();
    invalid.cpu_buffer_bytes = 4096;
    if (!expect_invalid(invalid, "cpu_buffer_bytes must be zero"))
        return 26;
    invalid = valid_config();
    invalid.requested_channels = -1;
    if (!expect_invalid(invalid, "requested_channels must be non-negative"))
        return 27;

    return 0;
}

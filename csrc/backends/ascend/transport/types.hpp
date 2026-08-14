#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace deep_ep::ascend::transport {

enum class TransportStatusCode : std::uint8_t {
    kSuccess,
    kUnsupportedCapability,
    kInvalidArgument,
    kRuntimeFailure,
};

struct TransportStatus {
    TransportStatusCode code = TransportStatusCode::kSuccess;
    std::string operation;
    int backend_code = 0;
    std::string message;

    bool ok() const noexcept { return code == TransportStatusCode::kSuccess; }

    static TransportStatus success() {
        return {};
    }

    static TransportStatus unsupported(std::string operation, std::string message) {
        return {TransportStatusCode::kUnsupportedCapability, std::move(operation), 0,
                std::move(message)};
    }

    static TransportStatus invalid(std::string operation, std::string message) {
        return {TransportStatusCode::kInvalidArgument, std::move(operation), 0,
                std::move(message)};
    }

    static TransportStatus runtime_failure(
        std::string operation, int backend_code, std::string message) {
        return {TransportStatusCode::kRuntimeFailure, std::move(operation), backend_code,
                std::move(message)};
    }
};

enum class TransportCapability : std::uint8_t {
    kSymmetricWindow,
    kDirectPeerPointer,
    kDevicePut,
    kDeviceGet,
    kDevicePutValue,
    kRemoteAtomicAddRelease,
    kRemoteSignal,
    kAsyncCompletion,
    kSystemMemoryOrdering,
    kDeviceBarrier,
    kScaleUpTeam,
    kScaleOutTeam,
};

using TransportCapabilities = std::uint64_t;
inline constexpr TransportCapabilities kNoCapabilities = 0;

constexpr TransportCapabilities capability_bit(TransportCapability capability) {
    return TransportCapabilities{1} << static_cast<std::uint8_t>(capability);
}

constexpr bool has_capability(TransportCapabilities capabilities,
                              TransportCapability capability) {
    return (capabilities & capability_bit(capability)) != 0;
}

inline const char* capability_name(TransportCapability capability) {
    switch (capability) {
        case TransportCapability::kSymmetricWindow: return "symmetric_window";
        case TransportCapability::kDirectPeerPointer: return "direct_peer_pointer";
        case TransportCapability::kDevicePut: return "device_put";
        case TransportCapability::kDeviceGet: return "device_get";
        case TransportCapability::kDevicePutValue: return "device_put_value";
        case TransportCapability::kRemoteAtomicAddRelease: return "remote_atomic_add_release";
        case TransportCapability::kRemoteSignal: return "remote_signal";
        case TransportCapability::kAsyncCompletion: return "async_completion";
        case TransportCapability::kSystemMemoryOrdering: return "system_memory_ordering";
        case TransportCapability::kDeviceBarrier: return "device_barrier";
        case TransportCapability::kScaleUpTeam: return "scale_up_team";
        case TransportCapability::kScaleOutTeam: return "scale_out_team";
    }
    return "unknown";
}

inline std::string capability_names(TransportCapabilities capabilities) {
    if (capabilities == kNoCapabilities)
        return "none";

    std::string names;
    for (std::uint8_t index = 0; index < 12; ++index) {
        const auto capability = static_cast<TransportCapability>(index);
        if (!has_capability(capabilities, capability))
            continue;
        if (!names.empty())
            names += ", ";
        names += capability_name(capability);
    }
    return names;
}

enum class TransportTeam : std::uint8_t { kWorld, kScaleUp, kScaleOut };
enum class CooperationScope : std::uint8_t {
    kParticipant,
    kWorkgroup,
    kDevice,
};
enum class MemorySegment : std::uint8_t { kDevice, kMixed };

using DeviceOptions = std::uint32_t;
inline constexpr DeviceOptions kDefaultOptions = 0;
inline constexpr DeviceOptions kAggregateRequests = DeviceOptions{1} << 0;

enum class RemoteActionKind : std::uint8_t {
    kNone,
    kSignalAdd,
    kSignalIncrement,
};

struct RemoteAction {
    RemoteActionKind kind = RemoteActionKind::kNone;
    std::uint32_t signal_index = 0;
    std::uint64_t symmetric_offset = 0;
    std::uint64_t value = 0;

    static constexpr RemoteAction none() {
        return {};
    }

    static constexpr RemoteAction signal_add(
        std::uint64_t offset, std::uint64_t addend) {
        return {RemoteActionKind::kSignalAdd, 0, offset, addend};
    }

    static constexpr RemoteAction signal_increment(std::uint32_t index) {
        return {RemoteActionKind::kSignalIncrement, index, 0, 1};
    }
};

using SignalValue = std::uint64_t;

struct TransportTopology {
    int world_rank = 0;
    int world_size = 1;
    int scale_up_rank = 0;
    int scale_up_size = 1;
    int scale_out_rank = 0;
    int scale_out_size = 1;
    bool scale_up_direct = false;
};

struct TransportConfig {
    int rank = 0;
    int world_size = 1;
    std::int64_t communicator_handle = 0;
    bool cpu_communicator_empty = true;
    std::int64_t device_buffer_bytes = 0;
    std::int64_t cpu_buffer_bytes = 0;
    bool allow_hybrid_mode = false;
    int service_level = 0;
    int requested_channels = 0;
};

inline constexpr std::uint32_t kDeviceTransportAbiVersion = 1;

struct DeviceTransportContext {
    std::uint32_t abi_version = kDeviceTransportAbiVersion;
    std::uint32_t struct_size = 0;
    TransportCapabilities capabilities = kNoCapabilities;
    TransportTopology topology{};
    std::uintptr_t local_window_base = 0;
    std::uintptr_t peer_address_table = 0;
    std::uintptr_t channel_table = 0;
    std::uintptr_t backend_context = 0;
};

inline DeviceTransportContext make_device_transport_context() {
    DeviceTransportContext context;
    context.struct_size = sizeof(DeviceTransportContext);
    return context;
}

struct alignas(16) DeviceRequest {
    std::array<std::uint64_t, 4> words{};
};

static_assert(std::is_trivially_copyable_v<TransportTopology>);
static_assert(std::is_trivially_copyable_v<DeviceTransportContext>);
static_assert(std::is_trivially_copyable_v<DeviceRequest>);
static_assert(std::is_trivially_copyable_v<RemoteAction>);

}  // namespace deep_ep::ascend::transport

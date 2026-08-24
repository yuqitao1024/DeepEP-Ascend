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
    kStageProfile,
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
        case TransportCapability::kStageProfile: return "stage_profile";
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

struct TeamPeer {
    TransportTeam team = TransportTeam::kWorld;
    int peer = -1;
    int world_peer = -1;
};
enum class CooperationScope : std::uint8_t {
    kParticipant,
    kWorkgroup,
    kDevice,
};
enum class MemorySegment : std::uint8_t { kDevice, kMixed, kMappedCpu };

enum class TransportTopologyKind : std::uint8_t {
    kFlatScaleUp,
    kPhysical2D,
    kLogicalSimulation,
};

using DeviceOptions = std::uint32_t;
inline constexpr DeviceOptions kDefaultOptions = 0;
inline constexpr DeviceOptions kAggregateRequests = DeviceOptions{1} << 0;

using DeviceAddress = std::uintptr_t;
inline constexpr DeviceAddress kNullDeviceAddress = 0;

enum class RemoteActionKind : std::uint8_t {
    kNone,
    kSignalAdd,
    kSignalIncrement,
    kSignalSet,
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

    static constexpr RemoteAction signal_set(
        std::uint32_t index, std::uint64_t value) {
        return {RemoteActionKind::kSignalSet, index, 0, value};
    }
};

using SignalValue = std::uint64_t;

inline constexpr std::uint32_t kTransportTopologyAbiVersion = 1;

struct TransportTopology {
    std::uint32_t abi_version = kTransportTopologyAbiVersion;
    std::uint32_t struct_size = 0;
    int world_rank = 0;
    int world_size = 1;
    int scale_up_rank = 0;
    int scale_up_size = 1;
    int scale_out_rank = 0;
    int scale_out_size = 1;
    bool scale_up_direct = false;
    TransportTopologyKind kind = TransportTopologyKind::kFlatScaleUp;
    std::uint64_t epoch = 1;
};

inline TransportStatus build_transport_topology(
    int world_rank, int world_size, int scale_up_size,
    TransportTopologyKind kind, std::uint64_t epoch,
    TransportTopology* topology) {
    if (topology == nullptr)
        return TransportStatus::invalid(
            "build_transport_topology", "topology must not be null");
    if (world_size <= 0 || world_rank < 0 || world_rank >= world_size)
        return TransportStatus::invalid(
            "build_transport_topology", "invalid world rank or size");
    if (scale_up_size <= 0 || world_size % scale_up_size != 0)
        return TransportStatus::invalid(
            "build_transport_topology",
            "scale_up_size must be a positive divisor of world_size");
    if (epoch == 0)
        return TransportStatus::invalid(
            "build_transport_topology", "topology epoch must be positive");

    const int scale_out_size = world_size / scale_up_size;
    switch (kind) {
        case TransportTopologyKind::kFlatScaleUp:
            if (scale_up_size != world_size)
                return TransportStatus::invalid(
                    "build_transport_topology",
                    "flat scale-up topology must contain the whole world");
            break;
        case TransportTopologyKind::kPhysical2D:
        case TransportTopologyKind::kLogicalSimulation:
            if (scale_out_size < 2)
                return TransportStatus::invalid(
                    "build_transport_topology",
                    "2D topology requires at least two scale-out ranks");
            break;
        default:
            return TransportStatus::invalid(
                "build_transport_topology", "unknown topology kind");
    }

    TransportTopology result;
    result.struct_size = sizeof(TransportTopology);
    result.world_rank = world_rank;
    result.world_size = world_size;
    result.scale_up_rank = world_rank % scale_up_size;
    result.scale_up_size = scale_up_size;
    result.scale_out_rank = world_rank / scale_up_size;
    result.scale_out_size = scale_out_size;
    result.kind = kind;
    result.epoch = epoch;
    *topology = result;
    return TransportStatus::success();
}

constexpr bool valid_transport_topology(
    const TransportTopology& topology) noexcept {
    if (topology.abi_version != kTransportTopologyAbiVersion ||
        topology.struct_size != sizeof(TransportTopology) ||
        topology.epoch == 0 || topology.world_size <= 0 ||
        topology.world_rank < 0 || topology.world_rank >= topology.world_size ||
        topology.scale_up_size <= 0 || topology.scale_out_size <= 0 ||
        topology.scale_up_rank < 0 ||
        topology.scale_up_rank >= topology.scale_up_size ||
        topology.scale_out_rank < 0 ||
        topology.scale_out_rank >= topology.scale_out_size ||
        static_cast<std::int64_t>(topology.scale_up_size) *
                topology.scale_out_size !=
            topology.world_size ||
        topology.scale_up_rank !=
            topology.world_rank % topology.scale_up_size ||
        topology.scale_out_rank !=
            topology.world_rank / topology.scale_up_size)
        return false;
    switch (topology.kind) {
        case TransportTopologyKind::kFlatScaleUp:
            return topology.scale_up_size == topology.world_size &&
                   topology.scale_out_size == 1;
        case TransportTopologyKind::kPhysical2D:
        case TransportTopologyKind::kLogicalSimulation:
            return topology.scale_out_size >= 2;
        default:
            return false;
    }
}

constexpr std::pair<int, int> physical_transport_domain_size(
    const TransportTopology& topology,
    TransportCapabilities capabilities) noexcept {
    if (topology.kind == TransportTopologyKind::kPhysical2D &&
        has_capability(capabilities, TransportCapability::kScaleOutTeam))
        return {topology.scale_out_size, topology.scale_up_size};
    return {1, topology.world_size};
}

constexpr bool checked_team_world_rank(
    const TransportTopology& topology, TransportTeam team, int peer,
    int* world_peer) noexcept {
    if (world_peer == nullptr || peer < 0 ||
        !valid_transport_topology(topology))
        return false;
    std::int64_t translated = -1;
    switch (team) {
        case TransportTeam::kWorld:
            if (peer >= topology.world_size)
                return false;
            translated = peer;
            break;
        case TransportTeam::kScaleUp:
            if (peer >= topology.scale_up_size)
                return false;
            translated =
                static_cast<std::int64_t>(topology.scale_out_rank) *
                    topology.scale_up_size + peer;
            break;
        case TransportTeam::kScaleOut:
            if (peer >= topology.scale_out_size)
                return false;
            translated = static_cast<std::int64_t>(peer) *
                             topology.scale_up_size +
                         topology.scale_up_rank;
            break;
        default:
            return false;
    }
    if (translated < 0 || translated >= topology.world_size)
        return false;
    *world_peer = static_cast<int>(translated);
    return true;
}

constexpr bool checked_team_peer_for_world_rank(
    const TransportTopology& topology, int world_peer,
    TeamPeer* team_peer) noexcept {
    if (team_peer == nullptr || !valid_transport_topology(topology) ||
        world_peer < 0 || world_peer >= topology.world_size)
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
    if (!checked_team_world_rank(
            topology, result.team, result.peer, &translated) ||
        translated != world_peer)
        return false;
    *team_peer = result;
    return true;
}

constexpr bool same_transport_topology_identity(
    const TransportTopology& lhs, const TransportTopology& rhs) noexcept {
    return lhs.abi_version == rhs.abi_version &&
           lhs.struct_size == rhs.struct_size && lhs.kind == rhs.kind &&
           lhs.epoch == rhs.epoch && lhs.world_size == rhs.world_size &&
           lhs.scale_up_size == rhs.scale_up_size &&
           lhs.scale_out_size == rhs.scale_out_size;
}

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
    int scale_up_size = 0;
    TransportTopologyKind topology_kind =
        TransportTopologyKind::kFlatScaleUp;
    std::uint64_t topology_epoch = 1;
    bool stage_profile_enabled = false;
};

inline TransportStatus build_configured_transport_topology(
    const TransportConfig& config, TransportTopology* topology) {
    const int scale_up_size =
        config.scale_up_size == 0 &&
                config.topology_kind == TransportTopologyKind::kFlatScaleUp
            ? config.world_size
            : config.scale_up_size;
    return build_transport_topology(
        config.rank, config.world_size, scale_up_size,
        config.topology_kind, config.topology_epoch, topology);
}

inline constexpr std::uint32_t kDeviceTransportAbiVersion = 2;
inline constexpr std::uint32_t kDeviceRequestAbiVersion = 1;

enum class DeviceRequestState : std::uint32_t {
    kEmpty,
    kPending,
    kCompleted,
    kFailed,
};

enum class DeviceTransportError : std::uint32_t {
    kNone,
    kInvalidAbi,
    kInvalidRank,
    kInvalidChannel,
    kInvalidAddress,
    kInvalidProtocol,
    kInvalidQueue,
    kUnsupportedOperation,
    kCommandOverflow,
    kCompletionTimeout,
    kCompletionFailure,
};

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
    context.topology.struct_size = sizeof(TransportTopology);
    return context;
}

struct alignas(16) DeviceRequest {
    std::uint32_t abi_version = kDeviceRequestAbiVersion;
    DeviceRequestState state = DeviceRequestState::kEmpty;
    std::uint32_t command_begin = 0;
    std::uint32_t command_end = 0;
    std::uint64_t queue_generation = 0;
    std::uint32_t consumed_target = 0;
    DeviceTransportError terminal_error = DeviceTransportError::kNone;
};

static_assert(std::is_trivially_copyable_v<TransportTopology>);
static_assert(std::is_trivially_copyable_v<DeviceTransportContext>);
static_assert(std::is_trivially_copyable_v<DeviceRequest>);
static_assert(std::is_trivially_copyable_v<RemoteAction>);

}  // namespace deep_ep::ascend::transport

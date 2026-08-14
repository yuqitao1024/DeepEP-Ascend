#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "types.hpp"

namespace deep_ep::ascend::transport {

inline constexpr std::uint32_t kTransportCommandAbiVersion = 1;
inline constexpr std::uint32_t kStagedTransportCannCompatibility =
    0x00090200U;

enum class TransportCommandOpcode : std::uint32_t {
    kNone,
    kPut,
    kPutValue64,
    kRemoteAdd64,
    kSignal,
    kFlush,
    kBarrier,
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

struct alignas(64) TransportCommand {
    TransportCommandOpcode opcode = TransportCommandOpcode::kNone;
    TransportTeam team = TransportTeam::kWorld;
    CooperationScope scope = CooperationScope::kParticipant;
    MemorySegment segment = MemorySegment::kDevice;
    RemoteActionKind action_kind = RemoteActionKind::kNone;
    std::int32_t peer = 0;
    std::uint32_t channel = 0;
    DeviceOptions options = kDefaultOptions;
    std::uint32_t value_bytes = 0;
    std::uint32_t signal_index = 0;
    std::uint32_t reserved0 = 0;
    DeviceAddress source = kNullDeviceAddress;
    DeviceAddress destination = kNullDeviceAddress;
    std::uint64_t bytes = 0;
    std::uint64_t value = 0;
    std::uint64_t symmetric_offset = 0;
    std::uint64_t timeout_cycles = 0;
    std::uint64_t reserved1[6]{};
};

struct alignas(64) TransportCommandQueue {
    std::uint32_t abi_version = kTransportCommandAbiVersion;
    std::uint32_t struct_size = sizeof(TransportCommandQueue);
    std::uint32_t capacity = 0;
    std::uint32_t count = 0;
    std::uint64_t generation = 0;
    std::uintptr_t commands = 0;
    std::uintptr_t service_state = 0;
    std::uintptr_t diagnostic = 0;
    std::uint64_t reserved[2]{};
};

struct alignas(64) TransportServiceState {
    std::uint32_t abi_version = kTransportCommandAbiVersion;
    std::uint32_t struct_size = sizeof(TransportServiceState);
    std::uint32_t consumed_count = 0;
    std::uint32_t active = 0;
    std::uint64_t consumed_generation = 0;
    std::uint64_t barrier_generation = 0;
    std::uint64_t default_retry_limit = 0;
    std::uint64_t reserved[3]{};
};

struct alignas(64) DeviceTransportDiagnostic {
    std::uint32_t abi_version = kTransportCommandAbiVersion;
    DeviceTransportError error = DeviceTransportError::kNone;
    std::uint32_t command_index = 0;
    TransportCommandOpcode opcode = TransportCommandOpcode::kNone;
    std::uint32_t peer = 0;
    std::uint32_t channel = 0;
    std::uint32_t sq_head = 0;
    std::uint32_t cq_head = 0;
    std::uint32_t cq_tail = 0;
    std::uint32_t backend_status = 0;
    std::uint64_t generation = 0;
    std::uint64_t reserved[2]{};
};

struct alignas(64) StagedTransportContext {
    std::uint32_t abi_version = kTransportCommandAbiVersion;
    std::uint32_t struct_size = sizeof(StagedTransportContext);
    std::uint32_t cann_compatibility = kStagedTransportCannCompatibility;
    std::uint32_t flags = 0;
    std::uintptr_t command_queue = 0;
    std::uintptr_t team = 0;
    std::uintptr_t window = 0;
    std::uintptr_t fetch_results = 0;
    std::uint64_t fetch_result_bytes = 0;
    std::uint64_t reserved = 0;
};

static_assert(sizeof(TransportCommand) == 128);
static_assert(sizeof(TransportCommandQueue) == 64);
static_assert(sizeof(TransportServiceState) == 64);
static_assert(sizeof(DeviceTransportDiagnostic) == 64);
static_assert(sizeof(StagedTransportContext) == 64);
static_assert(std::is_trivially_copyable_v<TransportCommand>);
static_assert(std::is_trivially_copyable_v<TransportCommandQueue>);
static_assert(std::is_trivially_copyable_v<TransportServiceState>);
static_assert(std::is_trivially_copyable_v<DeviceTransportDiagnostic>);
static_assert(std::is_trivially_copyable_v<StagedTransportContext>);

namespace command {

inline TransportCommand make_put(
    TransportTeam team, int peer, std::uint32_t channel,
    DeviceAddress destination, DeviceAddress source, std::uint64_t bytes,
    CooperationScope scope, MemorySegment segment, DeviceOptions options) {
    TransportCommand result;
    result.opcode = TransportCommandOpcode::kPut;
    result.team = team;
    result.scope = scope;
    result.segment = segment;
    result.peer = peer;
    result.channel = channel;
    result.options = options;
    result.source = source;
    result.destination = destination;
    result.bytes = bytes;
    return result;
}

inline TransportCommand make_put_value64(
    TransportTeam team, int peer, std::uint32_t channel,
    DeviceAddress destination, std::uint64_t value, DeviceOptions options) {
    TransportCommand result;
    result.opcode = TransportCommandOpcode::kPutValue64;
    result.team = team;
    result.peer = peer;
    result.channel = channel;
    result.options = options;
    result.value_bytes = sizeof(std::uint64_t);
    result.destination = destination;
    result.value = value;
    return result;
}

inline TransportCommand make_remote_add64(
    TransportTeam team, int peer, std::uint32_t channel,
    DeviceAddress destination, std::int64_t value) {
    TransportCommand result;
    result.opcode = TransportCommandOpcode::kRemoteAdd64;
    result.team = team;
    result.peer = peer;
    result.channel = channel;
    result.destination = destination;
    result.value = static_cast<std::uint64_t>(value);
    return result;
}

inline TransportCommand make_signal(
    TransportTeam team, int peer, std::uint32_t channel,
    const RemoteAction& action) {
    TransportCommand result;
    result.opcode = TransportCommandOpcode::kSignal;
    result.team = team;
    result.action_kind = action.kind;
    result.peer = peer;
    result.channel = channel;
    result.signal_index = action.signal_index;
    result.value = action.value;
    result.symmetric_offset = action.symmetric_offset;
    return result;
}

inline TransportCommand make_flush(
    std::uint32_t channel, CooperationScope scope) {
    TransportCommand result;
    result.opcode = TransportCommandOpcode::kFlush;
    result.scope = scope;
    result.channel = channel;
    return result;
}

inline TransportCommand make_barrier(
    std::uint32_t team_mask, std::uint64_t timeout_cycles) {
    TransportCommand result;
    result.opcode = TransportCommandOpcode::kBarrier;
    result.options = team_mask;
    result.timeout_cycles = timeout_cycles;
    return result;
}

inline TransportCommandQueue make_queue(
    TransportCommand* commands, std::uint32_t capacity,
    TransportServiceState* service_state,
    DeviceTransportDiagnostic* diagnostic) {
    TransportCommandQueue result;
    result.capacity = capacity;
    result.commands = reinterpret_cast<std::uintptr_t>(commands);
    result.service_state = reinterpret_cast<std::uintptr_t>(service_state);
    result.diagnostic = reinterpret_cast<std::uintptr_t>(diagnostic);
    return result;
}

inline void record_first_error(
    DeviceTransportDiagnostic& diagnostic, DeviceTransportError error,
    std::uint32_t command_index, TransportCommandOpcode opcode, int peer,
    std::uint32_t channel) {
    if (diagnostic.error != DeviceTransportError::kNone)
        return;
    diagnostic.error = error;
    diagnostic.command_index = command_index;
    diagnostic.opcode = opcode;
    diagnostic.peer = static_cast<std::uint32_t>(peer);
    diagnostic.channel = channel;
}

inline void reset(TransportCommandQueue& queue, std::uint64_t generation) {
    queue.count = 0;
    queue.generation = generation;
    auto* diagnostic = reinterpret_cast<DeviceTransportDiagnostic*>(
        queue.diagnostic);
    if (diagnostic != nullptr) {
        *diagnostic = DeviceTransportDiagnostic{};
        diagnostic->generation = generation;
    }
}

inline bool append(
    TransportCommandQueue& queue, const TransportCommand& transport_command) {
    auto* diagnostic = reinterpret_cast<DeviceTransportDiagnostic*>(
        queue.diagnostic);
    auto* commands = reinterpret_cast<TransportCommand*>(queue.commands);
    if (commands == nullptr || queue.count >= queue.capacity) {
        if (diagnostic != nullptr) {
            record_first_error(
                *diagnostic,
                commands == nullptr ? DeviceTransportError::kInvalidQueue
                                    : DeviceTransportError::kCommandOverflow,
                queue.count, transport_command.opcode,
                transport_command.peer, transport_command.channel);
        }
        return false;
    }
    commands[queue.count] = transport_command;
    ++queue.count;
    return true;
}

}  // namespace command

}  // namespace deep_ep::ascend::transport

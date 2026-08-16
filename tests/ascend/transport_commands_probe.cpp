#include <cstddef>
#include <cstdint>
#include <iostream>

#include "csrc/backends/ascend/transport/aicore_transport_service.hpp"
#include "csrc/backends/ascend/transport/transport_commands.hpp"

namespace transport = deep_ep::ascend::transport;

namespace {

int failures = 0;

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": "                \
                      << #expression << '\n';                                 \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

static_assert(sizeof(transport::TransportCommand) == 128);
static_assert(alignof(transport::TransportCommand) == 64);
static_assert(offsetof(transport::TransportCommand, opcode) == 0);
static_assert(offsetof(transport::TransportCommand, team) == 4);
static_assert(offsetof(transport::TransportCommand, peer) == 8);
static_assert(offsetof(transport::TransportCommand, channel) == 12);
static_assert(offsetof(transport::TransportCommand, options) == 16);
static_assert(offsetof(transport::TransportCommand, value_bytes) == 20);
static_assert(offsetof(transport::TransportCommand, signal_index) == 24);
static_assert(offsetof(transport::TransportCommand, source) == 32);
static_assert(offsetof(transport::TransportCommand, destination) == 40);
static_assert(offsetof(transport::TransportCommand, bytes) == 48);
static_assert(offsetof(transport::TransportCommand, value) == 56);
static_assert(offsetof(transport::TransportCommand, symmetric_offset) == 64);
static_assert(offsetof(transport::TransportCommand, timeout_cycles) == 72);
static_assert(sizeof(transport::TransportCommandQueue) == 64);
static_assert(sizeof(transport::TransportServiceState) == 64);
static_assert(sizeof(transport::DeviceTransportDiagnostic) == 64);
static_assert(sizeof(transport::StagedTransportContext) == 64);

void check_factories() {
    const auto put = transport::command::make_put(
        transport::TransportTeam::kWorld, 2, 0, 0x1110, 0x2220, 96,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions);
    CHECK(put.opcode == transport::TransportCommandOpcode::kPut);
    CHECK(put.team == transport::TransportTeam::kWorld);
    CHECK(put.peer == 2);
    CHECK(put.source == 0x2220);
    CHECK(put.destination == 0x1110);
    CHECK(put.bytes == 96);

    const auto put_value = transport::command::make_put_value64(
        transport::TransportTeam::kScaleUp, 1, 0, 0x3330,
        0xa5a5a5a55a5a5a5aULL, transport::kDefaultOptions);
    CHECK(put_value.opcode == transport::TransportCommandOpcode::kPutValue64);
    CHECK(put_value.value_bytes == 8);
    CHECK(put_value.destination == 0x3330);
    CHECK(put_value.value == 0xa5a5a5a55a5a5a5aULL);

    const auto faa = transport::command::make_remote_add64(
        transport::TransportTeam::kWorld, 1, 0, 0x4440, -7);
    CHECK(faa.opcode == transport::TransportCommandOpcode::kRemoteAdd64);
    CHECK(faa.destination == 0x4440);
    CHECK(static_cast<std::int64_t>(faa.value) == -7);

    const auto signal = transport::command::make_signal(
        transport::TransportTeam::kWorld, 1, 0,
        transport::RemoteAction::signal_add(0x88, 9));
    CHECK(signal.opcode == transport::TransportCommandOpcode::kSignal);
    CHECK(signal.action_kind == transport::RemoteActionKind::kSignalAdd);
    CHECK(signal.symmetric_offset == 0x88);
    CHECK(signal.value == 9);

    const auto flush = transport::command::make_flush(
        0, transport::CooperationScope::kWorkgroup);
    const auto barrier = transport::command::make_barrier(1, 1234);
    CHECK(flush.opcode == transport::TransportCommandOpcode::kFlush);
    CHECK(barrier.opcode == transport::TransportCommandOpcode::kBarrier);
    CHECK(barrier.options == 1);
    CHECK(barrier.timeout_cycles == 1234);
}

void check_queue_model() {
    transport::TransportCommand commands[3]{};
    transport::TransportServiceState service{};
    transport::DeviceTransportDiagnostic diagnostic{};
    auto queue = transport::command::make_queue(
        commands, 3, &service, &diagnostic);

    transport::command::reset(queue, 7);
    CHECK(queue.generation == 7);
    CHECK(queue.count == 0);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);

    CHECK(transport::command::append(
        queue, transport::command::make_put(
            transport::TransportTeam::kWorld, 1, 0, 0x1000, 0x2000, 64,
            transport::CooperationScope::kParticipant,
            transport::MemorySegment::kDevice,
            transport::kDefaultOptions)));
    CHECK(transport::command::append(
        queue, transport::command::make_flush(
            0, transport::CooperationScope::kParticipant)));
    CHECK(transport::command::append(
        queue, transport::command::make_barrier(1, 55)));
    CHECK(queue.count == 3);
    CHECK(commands[0].opcode == transport::TransportCommandOpcode::kPut);
    CHECK(commands[1].opcode == transport::TransportCommandOpcode::kFlush);
    CHECK(commands[2].opcode == transport::TransportCommandOpcode::kBarrier);

    CHECK(!transport::command::append(
        queue, transport::command::make_flush(
            0, transport::CooperationScope::kParticipant)));
    CHECK(queue.count == 3);
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCommandOverflow);
    CHECK(diagnostic.command_index == 3);

    transport::command::record_first_error(
        diagnostic, transport::DeviceTransportError::kInvalidRank, 1,
        transport::TransportCommandOpcode::kPut, 99, 7);
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCommandOverflow);

    transport::command::reset(queue, 8);
    CHECK(queue.count == 0);
    CHECK(queue.generation == 8);
    CHECK(diagnostic.error == transport::DeviceTransportError::kNone);
}

void check_barrier_poll_timeout() {
    using deep_ep::ascend::transport::service::barrier_poll_timed_out;

    CHECK(!barrier_poll_timed_out(100, 104, 5, 99, 1));
    CHECK(barrier_poll_timed_out(100, 105, 5, 0, 1000));
    CHECK(!barrier_poll_timed_out(100, 1000, 0, 2, 3));
    CHECK(barrier_poll_timed_out(100, 100, 0, 3, 3));
    CHECK(barrier_poll_timed_out(UINT64_MAX - 2, 2, 5, 0, 1000));
}

}  // namespace

int main() {
    check_factories();
    check_queue_model();
    check_barrier_poll_timeout();
    return failures == 0 ? 0 : 1;
}

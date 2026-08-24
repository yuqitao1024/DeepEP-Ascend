#include <cstdlib>
#include <vector>

#include "tests/ascend/simt_urma/runtime_probe.hpp"

namespace probe = deep_ep::ascend::transport::runtime_probe;
namespace transport = deep_ep::ascend::transport;

#define CHECK(condition)           \
    do {                           \
        if (!(condition))          \
            std::abort();          \
    } while (false)

namespace {

class RecordingTransport {
public:
    void device_barrier(
        std::uint32_t, transport::DeviceAddress, std::uint64_t) {
        commands.push_back(transport::TransportCommandOpcode::kBarrier);
    }

    void put(
        transport::TransportTeam, int, transport::DeviceAddress,
        transport::DeviceAddress, std::size_t,
        transport::CooperationScope, transport::MemorySegment,
        transport::DeviceOptions, const transport::RemoteAction&) {
        commands.push_back(transport::TransportCommandOpcode::kPut);
    }

    void put_value(
        transport::TransportTeam, int, transport::DeviceAddress,
        std::uint64_t, std::uint32_t, transport::DeviceOptions) {
        commands.push_back(transport::TransportCommandOpcode::kPutValue64);
    }

    void remote_add_release(
        transport::TransportTeam, int, transport::DeviceAddress,
        std::int64_t) {
        commands.push_back(transport::TransportCommandOpcode::kRemoteAdd64);
    }

    void signal(
        transport::TransportTeam, int, const transport::RemoteAction&) {
        commands.push_back(transport::TransportCommandOpcode::kSignal);
    }

    void flush(transport::CooperationScope) {
        commands.push_back(transport::TransportCommandOpcode::kFlush);
    }

    std::vector<transport::TransportCommandOpcode> commands;
};

}  // namespace

int main() {
    RecordingTransport transport;
    const transport::TeamPeer signal_route{
        transport::TransportTeam::kWorld, 1, 1};
    probe::enqueue_profile_mixed_final_commands(
        transport, 1, 0x1000, 0x2000, 0x1234, 0x3000,
        signal_route, 9, 1000);
    CHECK(transport.commands ==
          std::vector<transport::TransportCommandOpcode>({
              transport::TransportCommandOpcode::kPut,
              transport::TransportCommandOpcode::kPutValue64,
              transport::TransportCommandOpcode::kRemoteAdd64,
              transport::TransportCommandOpcode::kSignal,
              transport::TransportCommandOpcode::kBarrier,
              transport::TransportCommandOpcode::kFlush,
          }));
    return 0;
}

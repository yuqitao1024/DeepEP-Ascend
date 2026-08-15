#pragma once

#include <memory>

#include "host_transport.hpp"

namespace deep_ep::ascend::transport {

class StubHostTransport final : public HostTransport {
public:
    TransportCapabilities capabilities() const noexcept override {
        return kNoCapabilities;
    }

    TransportStatus query_topology(TransportTopology*) override {
        return unavailable("query_topology");
    }

    TransportStatus register_symmetric_window(void*, std::int64_t) override {
        return unavailable("register_symmetric_window");
    }

    TransportStatus unregister_symmetric_window() override {
        return TransportStatus::success();
    }

    TransportStatus get_peer_base_pointer(
        TransportTeam, int, std::uintptr_t*) override {
        return unavailable("get_peer_base_pointer");
    }

    TransportStatus acquire_channels(int, CooperationScope) override {
        return unavailable("acquire_channels");
    }

    TransportStatus release_channels() override {
        return TransportStatus::success();
    }

    TransportStatus export_device_context(DeviceTransportContext*) override {
        return unavailable("export_device_context");
    }

    TransportStatus read_diagnostic(
        DeviceTransportDiagnostic*) override {
        return unavailable("read_diagnostic");
    }

    TransportStatus host_barrier() override {
        return unavailable("host_barrier");
    }

    TransportStatus destroy() override {
        return TransportStatus::success();
    }

private:
    static TransportStatus unavailable(const char* operation) {
        return TransportStatus::unsupported(
            operation,
            "is unavailable until the Ascend device transport is implemented");
    }
};

inline TransportCreateResult make_stub_transport(const TransportConfig& config) {
    if (config.world_size <= 0)
        return {TransportStatus::invalid(
                    "make_stub_transport", "world_size must be positive"),
                nullptr};
    if (config.rank < 0 || config.rank >= config.world_size)
        return {TransportStatus::invalid(
                    "make_stub_transport", "rank must be in [0, world_size)"),
                nullptr};
    if (config.communicator_handle != 0)
        return {TransportStatus::invalid(
                    "make_stub_transport",
                    "communicator_handle must be zero in Phase 2A"),
                nullptr};
    if (!config.cpu_communicator_empty)
        return {TransportStatus::invalid(
                    "make_stub_transport",
                    "cpu_communicator must be empty in Phase 2A"),
                nullptr};
    if (config.device_buffer_bytes <= 0)
        return {TransportStatus::invalid(
                    "make_stub_transport", "device_buffer_bytes must be positive"),
                nullptr};
    if (config.cpu_buffer_bytes != 0)
        return {TransportStatus::invalid(
                    "make_stub_transport", "cpu_buffer_bytes must be zero in Phase 2A"),
                nullptr};
    if (config.requested_channels < 0)
        return {TransportStatus::invalid(
                    "make_stub_transport", "requested_channels must be non-negative"),
                nullptr};
    return {TransportStatus::success(), std::make_unique<StubHostTransport>()};
}

}  // namespace deep_ep::ascend::transport

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "transport_commands.hpp"
#include "stage_profile.hpp"
#include "types.hpp"

namespace deep_ep::ascend::transport {

class HostTransport {
public:
    virtual ~HostTransport() = default;
    virtual TransportCapabilities capabilities() const noexcept = 0;

    TransportStatus require_capabilities(
        TransportCapabilities required, const std::string& operation) const {
        const auto missing = required & ~capabilities();
        if (missing == kNoCapabilities)
            return TransportStatus::success();
        return TransportStatus::unsupported(
            operation,
            "requires unavailable device transport capabilities: " +
                capability_names(missing));
    }

    virtual TransportStatus query_topology(TransportTopology* topology) = 0;
    virtual TransportStatus register_symmetric_window(
        void* base, std::int64_t bytes) = 0;
    virtual TransportStatus unregister_symmetric_window() = 0;
    virtual TransportStatus get_peer_base_pointer(
        TransportTeam team, int rank, std::uintptr_t* pointer) = 0;
    virtual TransportStatus acquire_channels(
        int count, CooperationScope scope) = 0;
    virtual TransportStatus release_channels() = 0;
    virtual TransportStatus export_device_context(
        DeviceTransportContext* context) = 0;
    virtual TransportStatus read_diagnostic(
        DeviceTransportDiagnostic* diagnostic) = 0;
    virtual TransportStatus reset_stage_profile() = 0;
    virtual TransportStatus read_stage_profile(
        TransportStageProfile* profile) = 0;
    virtual TransportStatus host_barrier() = 0;
    virtual TransportStatus destroy() = 0;
};

struct TransportCreateResult {
    TransportStatus status;
    std::unique_ptr<HostTransport> transport;
};

}  // namespace deep_ep::ascend::transport

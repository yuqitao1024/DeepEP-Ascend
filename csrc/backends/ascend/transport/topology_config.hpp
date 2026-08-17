#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>
#include "types.hpp"

namespace deep_ep::ascend::transport {
namespace detail {

inline bool parse_positive_topology_integer(
    const char* value, int* parsed) noexcept {
    if (value == nullptr || parsed == nullptr || value[0] == '\0' ||
        value[0] == '0')
        return false;
    const char* end = value + std::strlen(value);
    int result = 0;
    const auto conversion = std::from_chars(value, end, result, 10);
    if (conversion.ec != std::errc{} || conversion.ptr != end || result <= 0)
        return false;
    *parsed = result;
    return true;
}

}  // namespace detail

inline TransportStatus configure_transport_topology_from_environment(
    TransportConfig* config) {
    if (config == nullptr)
        return TransportStatus::invalid(
            "configure_transport_topology", "config must not be null");
    if (config->world_size <= 0 || config->rank < 0 ||
        config->rank >= config->world_size)
        return TransportStatus::invalid(
            "configure_transport_topology", "invalid world rank or size");

    const char* simulation =
        std::getenv("DEEP_EP_ASCEND_LOGICAL_SIMULATION");
    if (simulation == nullptr)
        simulation = "0";
    if (std::strcmp(simulation, "0") != 0 &&
        std::strcmp(simulation, "1") != 0)
        return TransportStatus::invalid(
            "configure_transport_topology",
            "DEEP_EP_ASCEND_LOGICAL_SIMULATION must be 0 or 1");
    const bool logical_simulation = std::strcmp(simulation, "1") == 0;

    const char* scale_up = std::getenv("DEEP_EP_ASCEND_SCALE_UP_SIZE");
    int scale_up_size = config->world_size;
    if (scale_up != nullptr) {
        if (!detail::parse_positive_topology_integer(
                scale_up, &scale_up_size))
            return TransportStatus::invalid(
                "configure_transport_topology",
                "DEEP_EP_ASCEND_SCALE_UP_SIZE must be a positive integer");
    } else if (logical_simulation) {
        return TransportStatus::invalid(
            "configure_transport_topology",
            "logical simulation requires DEEP_EP_ASCEND_SCALE_UP_SIZE");
    }

    int topology_epoch = 1;
    const char* epoch = std::getenv("DEEP_EP_ASCEND_TOPOLOGY_EPOCH");
    if (epoch != nullptr &&
        !detail::parse_positive_topology_integer(epoch, &topology_epoch))
        return TransportStatus::invalid(
            "configure_transport_topology",
            "DEEP_EP_ASCEND_TOPOLOGY_EPOCH must be a positive integer");

    TransportConfig candidate = *config;
    candidate.scale_up_size = scale_up_size;
    candidate.topology_kind = logical_simulation
        ? TransportTopologyKind::kLogicalSimulation
        : TransportTopologyKind::kFlatScaleUp;
    candidate.topology_epoch = static_cast<std::uint64_t>(topology_epoch);
    TransportTopology topology;
    auto status = build_configured_transport_topology(candidate, &topology);
    if (!status.ok()) {
        status.operation = "configure_transport_topology";
        return status;
    }
    *config = candidate;
    return TransportStatus::success();
}

}  // namespace deep_ep::ascend::transport

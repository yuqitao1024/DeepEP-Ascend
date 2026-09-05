#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>

#include "types.hpp"

namespace deep_ep::ascend::transport {

inline bool valid_transport_channel_count(int count) noexcept {
    return count >= 1 && count <= kMaxTransportChannels;
}

inline TransportStatus configure_transport_channels_from_environment(
    TransportConfig* config) {
    if (config == nullptr)
        return TransportStatus::invalid(
            "configure_transport_channels", "config must not be null");

    const char* value = std::getenv("DEEP_EP_ASCEND_CHANNELS");
    if (value == nullptr) {
        config->requested_channels = 1;
        return TransportStatus::success();
    }
    if (value[0] == '\0')
        return TransportStatus::invalid(
            "configure_transport_channels",
            "DEEP_EP_ASCEND_CHANNELS must be an integer in [1, 4]");

    int count = 0;
    const char* end = value + std::strlen(value);
    const auto conversion = std::from_chars(value, end, count, 10);
    if (conversion.ec != std::errc{} || conversion.ptr != end ||
        !valid_transport_channel_count(count))
        return TransportStatus::invalid(
            "configure_transport_channels",
            "DEEP_EP_ASCEND_CHANNELS must be an integer in [1, 4]");
    config->requested_channels = count;
    return TransportStatus::success();
}

}  // namespace deep_ep::ascend::transport

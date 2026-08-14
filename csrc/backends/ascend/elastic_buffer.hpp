#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "transport/stub_transport.hpp"

namespace deep_ep::ascend {

[[noreturn]] inline void raise_unsupported(
    const char* operation, const std::string& detail) {
    const std::string message =
        std::string("DeepEP Ascend backend: ") + operation + " " + detail;
    PyErr_SetString(PyExc_NotImplementedError, message.c_str());
    throw pybind11::error_already_set();
}

[[noreturn]] inline void raise_transport_status(
    const transport::TransportStatus& status) {
    if (status.code == transport::TransportStatusCode::kUnsupportedCapability)
        raise_unsupported(status.operation.c_str(), status.message);
    TORCH_CHECK(false, "DeepEP Ascend backend: ", status.operation,
                " ", status.message);
}

struct EventHandle {
    void current_stream_wait() const {
        raise_unsupported(
            "current_stream_wait",
            "is unavailable until the Ascend device transport is implemented");
    }
};

class ElasticBuffer {
    int rank_idx_;
    int num_ranks_;
    int64_t num_buffer_bytes_;
    bool allow_hybrid_mode_;
    std::unique_ptr<transport::HostTransport> transport_;
    bool destroyed_ = false;

    static constexpr auto kDispatchCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDirectPeerPointer) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(
            transport::TransportCapability::kRemoteAtomicAddRelease) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier);

    static constexpr auto kCombineCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDirectPeerPointer) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(
            transport::TransportCapability::kRemoteAtomicAddRelease) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier);

    static constexpr auto kHybridCapabilities =
        transport::capability_bit(transport::TransportCapability::kScaleUpTeam) |
        transport::capability_bit(transport::TransportCapability::kScaleOutTeam);

    void require_transport(
        const char* operation, transport::TransportCapabilities required) const {
        const auto status = transport_->require_capabilities(required, operation);
        if (!status.ok())
            raise_transport_status(status);
    }

public:
    using cpu_comm_t = std::vector<std::pair<int, int>>;

    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& comm_handle, const cpu_comm_t& cpu_comm,
                  const int64_t& num_buffer_bytes,
                  const int64_t& num_cpu_buffer_bytes,
                  const bool& allow_hybrid_mode, const bool&, const bool&,
                  const int& sl_idx, const int& num_allocated_qps,
                  const int&, const int&, const bool&)
        : rank_idx_(rank_idx), num_ranks_(num_ranks),
          num_buffer_bytes_(num_buffer_bytes),
          allow_hybrid_mode_(allow_hybrid_mode) {
        transport::TransportConfig config{
            rank_idx, num_ranks, comm_handle, cpu_comm.empty(), num_buffer_bytes,
            num_cpu_buffer_bytes, allow_hybrid_mode, sl_idx, num_allocated_qps};
        auto result = transport::make_stub_transport(config);
        TORCH_CHECK(result.status.ok(), "DeepEP Ascend backend: ",
                    result.status.operation, " ", result.status.message);
        transport_ = std::move(result.transport);
    }

    void destroy() {
        if (destroyed_)
            return;
        destroyed_ = true;
        const auto status = transport_->destroy();
        if (!status.ok())
            raise_transport_status(status);
    }

    pybind11::object get_comm_stream() const {
        raise_unsupported(
            "get_comm_stream",
            "is unavailable until the Ascend device transport is implemented");
    }

    std::tuple<int, int> get_physical_domain_size() const {
        transport::TransportTopology topology;
        auto status = transport_->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_physical_domain_size";
            raise_transport_status(status);
        }
        return {topology.scale_out_size, topology.scale_up_size};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        transport::TransportTopology topology;
        auto status = transport_->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_logical_domain_size";
            raise_transport_status(status);
        }
        return {topology.scale_out_size, topology.scale_up_size};
    }

    void barrier(const bool&, const bool&, const bool&) const {
        auto required = transport::capability_bit(
            transport::TransportCapability::kDeviceBarrier);
        if (allow_hybrid_mode_)
            required |= kHybridCapabilities;
        require_transport("barrier", required);
    }

    static int64_t calculate_buffer_size(
        const int64_t&, const int&, const int&, int, const bool&,
        const bool&, const bool&) {
        raise_unsupported(
            "calculate_elastic_buffer_size",
            "is unavailable until the Ascend device transport is implemented");
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, int, int, std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor&, const std::optional<torch::Tensor>&,
             const torch::Tensor&, const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&, const std::optional<int>&,
             const std::optional<int>&,
             const std::optional<std::vector<int>>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const std::optional<torch::Tensor>&,
             const int&, const int&, const int&, const int&, const int&,
             const std::optional<EventHandle>&,
             const std::optional<EventHandle>&,
             const bool&, const bool&, const bool&, const bool&, const bool&,
             const bool&, const bool&) const {
        auto required = kDispatchCapabilities;
        if (allow_hybrid_mode_)
            required |= kHybridCapabilities;
        require_transport("dispatch", required);
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    combine(const torch::Tensor&, const std::optional<torch::Tensor>&,
            const std::optional<torch::Tensor>&,
            const std::optional<torch::Tensor>&, const torch::Tensor&,
            const torch::Tensor&, const torch::Tensor&,
            const std::optional<torch::Tensor>&,
            const std::optional<torch::Tensor>&,
            const int&, const int&, const int&, const int&,
            const std::optional<EventHandle>&,
            const std::optional<EventHandle>&,
            const bool&, const bool&, const bool&) const {
        auto required = kCombineCapabilities;
        if (allow_hybrid_mode_) {
            required |= kHybridCapabilities;
            required |= transport::capability_bit(
                transport::TransportCapability::kAsyncCompletion);
        }
        require_transport("combine", required);
    }
};

}  // namespace deep_ep::ascend

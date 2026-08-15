#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "elastic/layout.hpp"
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
    const transport::TransportStatus& status, int rank) {
    if (status.code == transport::TransportStatusCode::kUnsupportedCapability)
        raise_unsupported(status.operation.c_str(), status.message);
    if (status.code == transport::TransportStatusCode::kRuntimeFailure) {
        TORCH_CHECK(false, "DeepEP Ascend backend: ", status.operation,
                    " failed on rank ", rank, " with backend error ",
                    status.backend_code, ": ", status.message);
    }
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

    static constexpr auto kBarrierCapabilities =
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier);

    void require_transport(
        const char* operation, transport::TransportCapabilities required) const {
        const auto status = transport_->require_capabilities(required, operation);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
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
            raise_transport_status(status, rank_idx_);
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
            raise_transport_status(status, rank_idx_);
        }
        return {topology.scale_out_size, topology.scale_up_size};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        transport::TransportTopology topology;
        auto status = transport_->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_logical_domain_size";
            raise_transport_status(status, rank_idx_);
        }
        return {topology.scale_out_size, topology.scale_up_size};
    }

    void barrier(const bool&, const bool&, const bool&) const {
        auto required = kBarrierCapabilities;
        if (allow_hybrid_mode_)
            required |= kHybridCapabilities;
        require_transport("barrier", required);
        raise_unsupported(
            "barrier",
            "is unavailable until the Ascend device transport is implemented");
    }

    static int64_t calculate_buffer_size(
        const int64_t& comm_handle, const int& num_max_tokens_per_rank,
        const int& hidden, int num_topk, const bool& use_fp8_dispatch,
        const bool& allow_hybrid_mode, const bool&) {
        TORCH_CHECK(
            comm_handle != 0,
            "DeepEP Ascend backend: calculate_elastic_buffer_size "
            "communicator_handle must be nonzero");
        if (use_fp8_dispatch)
            raise_unsupported(
                "calculate_elastic_buffer_size", "does not support FP8");
        if (allow_hybrid_mode)
            raise_unsupported(
                "calculate_elastic_buffer_size",
                "does not support hybrid mode");
        TORCH_CHECK(
            num_max_tokens_per_rank > 0 && hidden > 0 && num_topk >= 0,
            "DeepEP Ascend backend: calculate_elastic_buffer_size requires "
            "positive token capacity and hidden size and nonnegative top-k");

        elastic::SymmetricWindowInput input{};
        input.world_size = 2;
        input.num_max_tokens_per_rank =
            static_cast<std::uint64_t>(num_max_tokens_per_rank);
        input.hidden = static_cast<std::uint64_t>(hidden);
        input.num_topk = static_cast<std::uint64_t>(num_topk);
        input.element_bytes = 2;
        elastic::SymmetricWindowLayout layout{};
        const auto status =
            elastic::build_symmetric_window_layout(input, &layout);
        TORCH_CHECK(
            status.ok(), "DeepEP Ascend backend: calculate_elastic_buffer_size ",
            status.message);
        TORCH_CHECK(
            layout.total_bytes <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()),
            "DeepEP Ascend backend: calculate_elastic_buffer_size overflow");
        return static_cast<std::int64_t>(layout.total_bytes);
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
        raise_unsupported(
            "dispatch",
            "is unavailable until the Ascend device transport is implemented");
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
        raise_unsupported(
            "combine",
            "is unavailable until the Ascend device transport is implemented");
    }
};

}  // namespace deep_ep::ascend

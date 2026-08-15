#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstddef>
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
#include "elastic/runtime.hpp"
#include "runtime/cann_runtime.hpp"

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
    std::unique_ptr<runtime::CannRuntimeResources> resources_;
    std::uint64_t barrier_generation_ = 0;
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
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(transport::TransportCapability::kScaleUpTeam);

    transport::HostTransport* host_transport() const {
        TORCH_CHECK(resources_ != nullptr && resources_->transport() != nullptr,
                    "DeepEP Ascend backend: runtime is destroyed");
        return resources_->transport();
    }

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
        const auto status =
            host_transport()->require_capabilities(required, operation);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
    }

    static const char* diagnostic_name(
        transport::DeviceTransportError error) {
        switch (error) {
            case transport::DeviceTransportError::kNone: return "none";
            case transport::DeviceTransportError::kInvalidAbi:
                return "invalid_abi";
            case transport::DeviceTransportError::kInvalidRank:
                return "invalid_rank";
            case transport::DeviceTransportError::kInvalidChannel:
                return "invalid_channel";
            case transport::DeviceTransportError::kInvalidAddress:
                return "invalid_address";
            case transport::DeviceTransportError::kInvalidProtocol:
                return "invalid_protocol";
            case transport::DeviceTransportError::kInvalidQueue:
                return "invalid_queue";
            case transport::DeviceTransportError::kUnsupportedOperation:
                return "unsupported_operation";
            case transport::DeviceTransportError::kCommandOverflow:
                return "command_overflow";
            case transport::DeviceTransportError::kCompletionTimeout:
                return "completion_timeout";
            case transport::DeviceTransportError::kCompletionFailure:
                return "completion_failure";
        }
        return "unknown";
    }

    [[noreturn]] void raise_barrier_diagnostic(
        const transport::DeviceTransportDiagnostic& diagnostic,
        const char* detail) const {
        auto message = std::string("device diagnostic ") + detail +
            " error=" + diagnostic_name(diagnostic.error) +
            " command_index=" + std::to_string(diagnostic.command_index) +
            " opcode=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.opcode)) +
            " peer=" + std::to_string(diagnostic.peer) +
            " channel=" + std::to_string(diagnostic.channel) +
            " generation=" + std::to_string(diagnostic.generation);
        raise_transport_status(
            transport::TransportStatus::runtime_failure(
                "barrier", static_cast<int>(diagnostic.backend_status),
                std::move(message)),
            rank_idx_);
    }

    elastic::CoreTiling build_barrier_tiling() const {
        const auto& context = resources_->device_context();
        elastic::CoreTilingInput input{};
        input.operation = elastic::OperationKind::kBarrier;
        input.topology.world_rank = context.topology.world_rank;
        input.topology.world_size = context.topology.world_size;
        input.topology.scale_up_rank = context.topology.scale_up_rank;
        input.topology.scale_up_size = context.topology.scale_up_size;
        input.topology.scale_out_rank = context.topology.scale_out_rank;
        input.topology.scale_out_size = context.topology.scale_out_size;
        elastic::CoreTiling tiling{};
        const auto status = elastic::build_core_tiling(input, &tiling);
        TORCH_CHECK(status.ok(), "DeepEP Ascend backend: barrier ",
                    status.message);
        tiling.transport_context = context;
        return tiling;
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
        TORCH_CHECK(num_ranks == 2,
                    "DeepEP Ascend backend: exactly two ranks are required");
        TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks,
                    "DeepEP Ascend backend: rank must be in [0, world_size)");
        TORCH_CHECK(comm_handle != 0,
                    "DeepEP Ascend backend: communicator_handle must be nonzero");
        TORCH_CHECK(cpu_comm.empty(),
                    "DeepEP Ascend backend: cpu communicator must be empty");
        TORCH_CHECK(num_cpu_buffer_bytes == 0,
                    "DeepEP Ascend backend: cpu_buffer_bytes must be zero");
        TORCH_CHECK(!allow_hybrid_mode,
                    "DeepEP Ascend backend: hybrid mode is unsupported");
        TORCH_CHECK(num_allocated_qps == 0,
                    "DeepEP Ascend backend: CUDA QP count must be zero");
        TORCH_CHECK(
            num_buffer_bytes >=
                    static_cast<std::int64_t>(
                        elastic::kPublicElasticBufferAlignment) &&
                num_buffer_bytes % static_cast<std::int64_t>(
                    elastic::kPublicElasticBufferAlignment) == 0,
            "DeepEP Ascend backend: device buffer must be positive and "
            "2 MiB-aligned");

        transport::TransportConfig config{
            rank_idx, num_ranks, comm_handle, cpu_comm.empty(), num_buffer_bytes,
            num_cpu_buffer_bytes, allow_hybrid_mode, sl_idx, 1};
        auto resources = std::make_unique<runtime::CannRuntimeResources>();
        const auto status = resources->initialize(
            config, 2 * elastic::kPublicElasticBufferAlignment);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        resources_ = std::move(resources);
    }

    void destroy() {
        if (destroyed_)
            return;
        destroyed_ = true;
        const auto status = resources_->destroy();
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        resources_.reset();
    }

    pybind11::object get_comm_stream() const {
        raise_unsupported(
            "get_comm_stream",
            "is unavailable until the Ascend device transport is implemented");
    }

    std::tuple<int, int> get_physical_domain_size() const {
        transport::TransportTopology topology;
        auto status = host_transport()->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_physical_domain_size";
            raise_transport_status(status, rank_idx_);
        }
        return {topology.scale_out_size, topology.scale_up_size};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        transport::TransportTopology topology;
        auto status = host_transport()->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_logical_domain_size";
            raise_transport_status(status, rank_idx_);
        }
        return {topology.scale_out_size, topology.scale_up_size};
    }

    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync,
                 const bool& sequential) {
        TORCH_CHECK(sequential,
                    "DeepEP Ascend backend: barrier requires sequential=True");
        (void)use_comm_stream;
        require_transport("barrier", elastic::kBarrierTransportCapabilities);

        if (with_cpu_sync) {
            const auto status = resources_->synchronize_device();
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }

        ++barrier_generation_;
        if (barrier_generation_ == 0)
            ++barrier_generation_;
        auto tiling = build_barrier_tiling();
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_),
            resources_->workspace_bytes()};
        const elastic::BarrierArguments arguments{
            resources_->workspace(), barrier_generation_};
        const auto launch_status = elastic::launch_internal_barrier(
            arguments, tiling, storage, resources_->stream());
        if (!launch_status.ok()) {
            const auto status = launch_status.code ==
                    elastic::CoreRuntimeStatusCode::kLaunchFailure ?
                transport::TransportStatus::runtime_failure(
                    "barrier", launch_status.backend_code,
                    launch_status.message) :
                transport::TransportStatus::invalid(
                    "barrier", launch_status.message);
            raise_transport_status(status, rank_idx_);
        }

        auto status = resources_->synchronize_stream();
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        transport::DeviceTransportDiagnostic diagnostic{};
        status = host_transport()->read_diagnostic(&diagnostic);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
#if DEEP_EP_ASCEND_TESTING
        if (std::getenv("DEEP_EP_ASCEND_TEST_DIAGNOSTIC") != nullptr) {
            diagnostic.abi_version = transport::kTransportCommandAbiVersion;
            diagnostic.error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic.command_index = 0;
            diagnostic.opcode = transport::TransportCommandOpcode::kBarrier;
            diagnostic.peer = static_cast<std::uint32_t>(rank_idx_);
            diagnostic.channel = 0;
            diagnostic.backend_status = 0;
            diagnostic.generation = barrier_generation_;
        }
#endif

        std::uint64_t completion = 0;
        const auto completion_offset =
            tiling.symmetric_window_layout.control_offset +
            offsetof(elastic::SymmetricControlHeader, barrier_completion) +
            static_cast<std::uint64_t>(rank_idx_) * sizeof(std::uint64_t);
        const auto* completion_address = reinterpret_cast<const void*>(
            reinterpret_cast<std::uintptr_t>(resources_->window_base()) +
            completion_offset);
        status = resources_->copy_to_host(
            &completion, completion_address, sizeof(completion));
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        if (with_cpu_sync) {
            status = resources_->synchronize_device();
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }

        if (diagnostic.abi_version !=
                transport::kTransportCommandAbiVersion ||
            diagnostic.error != transport::DeviceTransportError::kNone)
            raise_barrier_diagnostic(diagnostic, "reported failure");
        if (diagnostic.generation != barrier_generation_)
            raise_barrier_diagnostic(diagnostic, "generation mismatch");
        if (completion != barrier_generation_)
            raise_barrier_diagnostic(diagnostic, "completion mismatch");
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

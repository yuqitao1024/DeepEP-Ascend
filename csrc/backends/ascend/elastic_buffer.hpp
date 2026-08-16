#pragma once

#include <cstdlib>
#include <atomic>
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
#include "elastic/barrier_state.hpp"
#include "elastic/dispatch_state.hpp"
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
    elastic::BarrierSequence barrier_sequence_;
    mutable elastic::DispatchSequence dispatch_sequence_;
    std::uint64_t dispatch_family_ = 0;
    std::uint64_t barrier_timeout_cycles_ = 0;
    bool destroyed_ = false;

    inline static std::atomic_uint64_t next_dispatch_family_{0};

    static constexpr auto kDispatchCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(transport::TransportCapability::kScaleUpTeam);

    transport::HostTransport* host_transport() const {
        TORCH_CHECK(!destroyed_ && resources_ != nullptr &&
                        resources_->transport() != nullptr,
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

    [[noreturn]] void raise_dispatch_diagnostic(
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
                "dispatch", static_cast<int>(diagnostic.backend_status),
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

    elastic::CoreTiling build_dispatch_tiling(
        std::uint64_t num_tokens, std::uint64_t hidden,
        std::uint64_t num_experts, std::uint64_t num_topk,
        std::uint64_t expert_alignment,
        std::uint64_t num_max_tokens_per_rank,
        elastic::CoreModeFlags mode_flags) const {
        const auto& context = resources_->device_context();
        elastic::CoreTilingInput input{};
        input.operation = elastic::OperationKind::kDispatch;
        input.element_kind = elastic::ElementKind::kBFloat16;
        input.mode_flags = mode_flags;
        input.num_tokens = num_tokens;
        input.hidden = hidden;
        input.num_experts = num_experts;
        input.num_topk = num_topk;
        input.expert_alignment = expert_alignment;
        input.num_max_tokens_per_rank = num_max_tokens_per_rank;
        input.has_reusable_slots =
            elastic::has_mode(mode_flags, elastic::CoreMode::kCached);
        input.topology.world_rank = context.topology.world_rank;
        input.topology.world_size = context.topology.world_size;
        input.topology.scale_up_rank = context.topology.scale_up_rank;
        input.topology.scale_up_size = context.topology.scale_up_size;
        input.topology.scale_out_rank = context.topology.scale_out_rank;
        input.topology.scale_out_size = context.topology.scale_out_size;
        elastic::CoreTiling tiling{};
        const auto status = elastic::build_core_tiling(input, &tiling);
        TORCH_CHECK(status.ok(), "DeepEP Ascend backend: dispatch ",
                    status.message);
        tiling.transport_context = context;
        return tiling;
    }

    static void validate_npu_tensor(
        const torch::Tensor& tensor, int64_t dimensions,
        torch::ScalarType type, const torch::Device& device,
        const char* name) {
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                        tensor.device() == device &&
                        tensor.is_contiguous() &&
                        tensor.dim() == dimensions &&
                        tensor.scalar_type() == type,
                    "DeepEP Ascend backend: dispatch requires contiguous NPU ",
                    name, " with the expected rank and dtype");
    }

    static bool align_without_overflow(
        std::uint64_t value, std::uint64_t alignment,
        std::uint64_t* output) {
        if (alignment == 0 ||
            value > std::numeric_limits<std::uint64_t>::max() -
                        (alignment - 1))
            return false;
        *output = ((value + alignment - 1) / alignment) * alignment;
        return true;
    }

    static void raise_launch_status(
        const elastic::CoreRuntimeStatus& status, int rank) {
        const auto transport_status = status.code ==
                elastic::CoreRuntimeStatusCode::kLaunchFailure ?
            transport::TransportStatus::runtime_failure(
                "dispatch", status.backend_code, status.message) :
            transport::TransportStatus::invalid("dispatch", status.message);
        raise_transport_status(transport_status, rank);
    }

#if DEEP_EP_ASCEND_TESTING
    struct TestingTag {};
    ElasticBuffer(TestingTag, int rank, std::unique_ptr<runtime::CannRuntimeResources> resources,
                  std::int64_t buffer_bytes, std::uint64_t timeout_cycles)
        : rank_idx_(rank), num_ranks_(2), num_buffer_bytes_(buffer_bytes),
          allow_hybrid_mode_(false), resources_(std::move(resources)),
          barrier_timeout_cycles_(timeout_cycles) {
        dispatch_family_ = 7;
    }
#endif

public:
    using cpu_comm_t = std::vector<std::pair<int, int>>;

    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& comm_handle, const cpu_comm_t& cpu_comm,
                  const int64_t& num_buffer_bytes,
                  const int64_t& num_cpu_buffer_bytes,
                  const bool& allow_hybrid_mode, const bool&, const bool&,
                  const int& sl_idx, const int& num_allocated_qps,
                  const int&, const int& num_gpu_timeout_secs, const bool&)
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
            elastic::timeout_cycles_from_seconds(
                num_gpu_timeout_secs, &barrier_timeout_cycles_),
            "DeepEP Ascend backend: num_gpu_timeout_secs must be positive");
        TORCH_CHECK(
            num_buffer_bytes >=
                    static_cast<std::int64_t>(
                        elastic::kPublicElasticBufferAlignment) &&
                num_buffer_bytes % static_cast<std::int64_t>(
                    elastic::kPublicElasticBufferAlignment) == 0,
            "DeepEP Ascend backend: device buffer must be positive and "
            "2 MiB-aligned");
        const auto previous_family = next_dispatch_family_.fetch_add(
            1, std::memory_order_relaxed);
        TORCH_CHECK(previous_family != std::numeric_limits<std::uint64_t>::max(),
                    "DeepEP Ascend backend: dispatch family space is exhausted");
        dispatch_family_ = previous_family + 1;

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

#if DEEP_EP_ASCEND_TESTING
    static std::unique_ptr<ElasticBuffer> make_testing_buffer(
        int rank, std::unique_ptr<runtime::CannRuntimeResources> resources,
        std::int64_t buffer_bytes, std::uint64_t timeout_cycles) {
        return std::unique_ptr<ElasticBuffer>(new ElasticBuffer(
            TestingTag{}, rank, std::move(resources), buffer_bytes, timeout_cycles));
    }
#endif

    void destroy() {
        if (resources_ == nullptr)
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

        elastic::BarrierAttempt attempt(barrier_sequence_);
        TORCH_CHECK(
            attempt.valid(),
            "DeepEP Ascend backend: barrier cannot continue after a failed "
            "or concurrent attempt");
        auto tiling = build_barrier_tiling();
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_),
            resources_->workspace_bytes()};
        const elastic::BarrierArguments arguments{
            resources_->workspace(), attempt.generation(),
            barrier_timeout_cycles_};
        void* stream = nullptr;
        auto status = resources_->current_stream(&stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        const auto launch_status = elastic::launch_internal_barrier(
            arguments, tiling, storage, stream);
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

        status = resources_->synchronize_stream(stream);
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
            diagnostic.generation = attempt.generation();
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
        if (diagnostic.generation != attempt.generation())
            raise_barrier_diagnostic(diagnostic, "generation mismatch");
        if (completion != attempt.generation())
            raise_barrier_diagnostic(diagnostic, "completion mismatch");
        attempt.complete();
    }

    static int64_t calculate_buffer_size(
        const int64_t& comm_handle, const int& num_max_tokens_per_rank,
        const int& hidden, int num_topk, const bool& use_fp8_dispatch,
        const bool& allow_hybrid_mode,
        const bool& allow_multiple_reduction) {
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
        input.expanded = true;
        input.allow_multiple_reduction = allow_multiple_reduction;
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
    dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank, const int& num_experts,
             const int& expert_alignment, const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream, const bool& do_handle_copy,
             const bool& do_cpu_sync, const bool& do_expand,
             const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf) const {
        require_transport("dispatch", kDispatchCapabilities);
        TORCH_CHECK(!destroyed_ && !allow_hybrid_mode_,
                    "DeepEP Ascend backend: dispatch does not support hybrid mode");
        TORCH_CHECK(!sf.has_value() && !cumulative_local_expert_recv_stats.has_value(),
                    "DeepEP Ascend backend: dispatch does not support scale factors "
                    "or cumulative expert stats");
        TORCH_CHECK(!previous_event.has_value() &&
                        !previous_event_before_epilogue.has_value() &&
                        !async_with_compute_stream && !allocate_on_comm_stream,
                    "DeepEP Ascend backend: dispatch is synchronous");
        TORCH_CHECK(!use_tma_aligned_col_major_sf,
                    "DeepEP Ascend backend: dispatch does not support TMA mode");
        const bool cached_mode = cached_num_recv_tokens.has_value();
        TORCH_CHECK(do_cpu_sync || cached_mode,
                    "DeepEP Ascend backend: dispatch requires do_cpu_sync unless cached");
        TORCH_CHECK(num_sms == 1 && num_qps == 0,
                    "DeepEP Ascend backend: dispatch requires num_sms=1 and num_qps=0");
        TORCH_CHECK(!do_zero_padding || do_expand,
                    "DeepEP Ascend backend: dispatch zero padding requires expansion");
        TORCH_CHECK(num_max_tokens_per_rank > 0 && num_experts > 0 &&
                        expert_alignment > 0 && num_experts % 2 == 0,
                    "DeepEP Ascend backend: dispatch requires positive, two-rank "
                    "expert-aligned capacity");
        const auto device = x.device();
        int current_device = -1;
        auto device_status = resources_->current_device(&current_device);
        if (!device_status.ok())
            raise_transport_status(device_status, rank_idx_);
        TORCH_CHECK(device.index() == resources_->owning_device() &&
                        current_device == resources_->owning_device(),
                    "DeepEP Ascend backend: dispatch tensors and current NPU "
                    "must match the buffer device");
        validate_npu_tensor(x, 2, torch::kBFloat16, device, "BF16 x");
        validate_npu_tensor(topk_idx, 2, torch::kLong, device, "int64 topk_idx");
        TORCH_CHECK(x.size(0) == topk_idx.size(0) && x.size(0) >= 0 &&
                        x.size(0) <= num_max_tokens_per_rank && x.size(1) > 0 &&
                        topk_idx.size(1) > 0 && topk_idx.size(1) <= num_experts,
                    "DeepEP Ascend backend: dispatch input shapes exceed capacity");
        if (topk_weights.has_value()) {
            validate_npu_tensor(
                *topk_weights, 2, torch::kFloat, device, "float32 topk_weights");
            TORCH_CHECK(topk_weights->sizes() == topk_idx.sizes(),
                        "DeepEP Ascend backend: dispatch topk_weights shape mismatch");
        }

        const bool any_cached_tensor = cached_num_expanded_tokens.has_value() ||
            cached_num_recv_tokens_per_expert_list.has_value() ||
            cached_psum_num_recv_tokens_per_scaleup_rank.has_value() ||
            cached_psum_num_recv_tokens_per_expert.has_value() ||
            cached_num_unaligned_recv_tokens_per_expert.has_value() ||
            cached_dst_buffer_slot_idx.has_value() ||
            cached_token_metadata_at_forward.has_value() ||
            cached_recv_src_metadata.has_value();
        TORCH_CHECK(cached_mode || !any_cached_tensor,
                    "DeepEP Ascend backend: dispatch cached handles require cached counts");
        TORCH_CHECK(!cached_channel_linked_list.has_value(),
                    "DeepEP Ascend backend: dispatch does not support channel handles");

        elastic::CoreModeFlags mode_flags = 0;
        if (cached_mode)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kCached);
        if (do_expand)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kExpanded);
        if (do_zero_padding)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kZeroPadding);
        const auto num_tokens = static_cast<std::uint64_t>(x.size(0));
        const auto hidden = static_cast<std::uint64_t>(x.size(1));
        const auto num_topk = static_cast<std::uint64_t>(topk_idx.size(1));
        const auto capacity = static_cast<std::uint64_t>(num_max_tokens_per_rank);
        const auto experts = static_cast<std::uint64_t>(num_experts);
        const auto alignment = static_cast<std::uint64_t>(expert_alignment);
        const auto tiling = build_dispatch_tiling(
            num_tokens, hidden, experts, num_topk, alignment, capacity, mode_flags);
        TORCH_CHECK(tiling.communication_buffer_bytes <=
                        static_cast<std::uint64_t>(num_buffer_bytes_) &&
                        tiling.workspace_bytes <= resources_->workspace_bytes(),
                    "DeepEP Ascend backend: dispatch capacity exceeds runtime storage");

        const auto descriptor_mode_flags = mode_flags & ~elastic::mode_bit(
            elastic::CoreMode::kCached);
        const auto expected_descriptor = elastic::make_dispatch_handle_descriptor(
            dispatch_family_, tiling.topology, num_tokens, hidden, experts,
            num_topk, alignment, capacity, descriptor_mode_flags);
        const auto int_options = x.options().dtype(torch::kInt);
        const auto metadata_options = x.options().dtype(torch::kByte);
        const auto max_recv_tokens = capacity * static_cast<std::uint64_t>(num_ranks_);
        const auto local_experts = experts / static_cast<std::uint64_t>(num_ranks_);
        const auto expanded_records = tiling.dispatch_output_capacity;
        TORCH_CHECK(max_recv_tokens <= static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max()) &&
                        expanded_records <= static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()),
                    "DeepEP Ascend backend: dispatch output count overflow");

        torch::Tensor rank_prefix;
        torch::Tensor expert_prefix;
        torch::Tensor unaligned;
        torch::Tensor destination_slots;
        torch::Tensor source_metadata;
        auto kernel_expert_prefix = torch::empty(
            {num_experts + 1}, int_options);
        auto kernel_unaligned = torch::empty(
            {num_experts}, int_options);
        std::vector<std::int32_t> host_rank_prefix(num_ranks_);
        std::vector<std::int32_t> host_kernel_expert_prefix(num_experts + 1);
        std::vector<std::int32_t> host_kernel_unaligned(num_experts);
        std::vector<std::int32_t> host_expert_prefix(local_experts);
        std::vector<std::int32_t> host_unaligned(local_experts);
        std::vector<int> per_expert_list;
        int num_expanded_tokens = 0;
        const int first_local_expert =
            rank_idx_ * static_cast<int>(local_experts);
        if (cached_mode) {
            TORCH_CHECK(cached_num_expanded_tokens.has_value() &&
                            cached_num_recv_tokens_per_expert_list.has_value() &&
                            cached_psum_num_recv_tokens_per_scaleup_rank.has_value() &&
                            cached_psum_num_recv_tokens_per_expert.has_value() &&
                            cached_num_unaligned_recv_tokens_per_expert.has_value() &&
                            cached_dst_buffer_slot_idx.has_value() &&
                            cached_token_metadata_at_forward.has_value() &&
                            cached_recv_src_metadata.has_value(),
                        "DeepEP Ascend backend: dispatch cached mode requires all handles");
            TORCH_CHECK(*cached_num_recv_tokens >= 0 &&
                            *cached_num_expanded_tokens >= 0 &&
                            *cached_num_recv_tokens <= static_cast<int>(max_recv_tokens) &&
                            *cached_num_expanded_tokens <= static_cast<int>(expanded_records) &&
                            cached_num_recv_tokens_per_expert_list->size() == local_experts,
                        "DeepEP Ascend backend: dispatch cached counts are invalid");
            validate_npu_tensor(*cached_psum_num_recv_tokens_per_scaleup_rank,
                                1, torch::kInt, device, "cached rank prefix");
            validate_npu_tensor(*cached_psum_num_recv_tokens_per_expert,
                                1, torch::kInt, device, "cached expert prefix");
            validate_npu_tensor(*cached_num_unaligned_recv_tokens_per_expert,
                                1, torch::kInt, device, "cached unaligned counts");
            validate_npu_tensor(*cached_dst_buffer_slot_idx,
                                2, torch::kInt, device, "cached destination slots");
            validate_npu_tensor(*cached_recv_src_metadata,
                                2, torch::kInt, device, "cached source metadata");
            validate_npu_tensor(*cached_token_metadata_at_forward,
                                1, torch::kByte, device, "cached descriptor");
            TORCH_CHECK(cached_psum_num_recv_tokens_per_scaleup_rank->size(0) ==
                                static_cast<int64_t>(num_ranks_) &&
                            cached_psum_num_recv_tokens_per_expert->size(0) ==
                                static_cast<int64_t>(local_experts) &&
                            cached_num_unaligned_recv_tokens_per_expert->size(0) ==
                                static_cast<int64_t>(local_experts) &&
                            cached_dst_buffer_slot_idx->sizes() == topk_idx.sizes() &&
                            cached_recv_src_metadata->size(0) == *cached_num_recv_tokens &&
                            cached_recv_src_metadata->size(1) ==
                                static_cast<int64_t>(num_topk + 2) &&
                            cached_token_metadata_at_forward->numel() ==
                                static_cast<int64_t>(sizeof(elastic::DispatchHandleDescriptor)),
                        "DeepEP Ascend backend: dispatch cached handle shape mismatch");
            elastic::DispatchHandleDescriptor descriptor{};
            auto status = resources_->copy_to_host(
                &descriptor, cached_token_metadata_at_forward->data_ptr(),
                sizeof(descriptor));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            const auto descriptor_status = elastic::validate_dispatch_handle(
                expected_descriptor, descriptor);
            TORCH_CHECK(descriptor_status.ok(), "DeepEP Ascend backend: dispatch ",
                        descriptor_status.message);
            rank_prefix = *cached_psum_num_recv_tokens_per_scaleup_rank;
            expert_prefix = *cached_psum_num_recv_tokens_per_expert;
            unaligned = *cached_num_unaligned_recv_tokens_per_expert;
            destination_slots = *cached_dst_buffer_slot_idx;
            source_metadata = *cached_recv_src_metadata;
            status = resources_->copy_to_host(
                host_rank_prefix.data(), rank_prefix.data_ptr(),
                host_rank_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_expert_prefix.data(), expert_prefix.data_ptr(),
                host_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_unaligned.data(), unaligned.data_ptr(),
                host_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            TORCH_CHECK(host_rank_prefix.back() == *cached_num_recv_tokens,
                        "DeepEP Ascend backend: dispatch cached prefix tail mismatch");
            for (std::size_t index = 0; index < host_rank_prefix.size(); ++index)
                TORCH_CHECK(host_rank_prefix[index] >= 0 &&
                                (index == 0 || host_rank_prefix[index] >=
                                                    host_rank_prefix[index - 1]),
                            "DeepEP Ascend backend: dispatch cached rank prefix is invalid");
            std::int32_t expanded_tail = 0;
            for (int expert = 0; expert < num_experts; ++expert) {
                host_kernel_expert_prefix[expert] = expanded_tail;
                if (expert < first_local_expert ||
                    expert >= first_local_expert +
                                  static_cast<int>(local_experts))
                    continue;
                const int local_expert = expert - first_local_expert;
                const std::int32_t actual = host_unaligned[local_expert];
                std::uint64_t aligned_actual = 0;
                TORCH_CHECK(actual >= 0 && align_without_overflow(
                                static_cast<std::uint64_t>(actual), alignment,
                                &aligned_actual) &&
                                aligned_actual <= static_cast<std::uint64_t>(
                                    std::numeric_limits<int>::max()) &&
                                (*cached_num_recv_tokens_per_expert_list)[local_expert] ==
                                    static_cast<int>(aligned_actual),
                            "DeepEP Ascend backend: dispatch cached expert counts mismatch");
                TORCH_CHECK(aligned_actual <= static_cast<std::uint64_t>(
                                std::numeric_limits<int>::max() - expanded_tail),
                            "DeepEP Ascend backend: dispatch cached expert counts mismatch");
                const std::int32_t expected_public_prefix = do_expand ?
                    expanded_tail + actual :
                    expanded_tail + static_cast<std::int32_t>(aligned_actual);
                TORCH_CHECK(host_expert_prefix[local_expert] ==
                                expected_public_prefix,
                            "DeepEP Ascend backend: dispatch cached expert prefix is invalid");
                host_kernel_unaligned[expert] = actual;
                expanded_tail += static_cast<std::int32_t>(aligned_actual);
            }
            host_kernel_expert_prefix[num_experts] = expanded_tail;
            TORCH_CHECK(expanded_tail == *cached_num_expanded_tokens,
                        "DeepEP Ascend backend: dispatch cached prefix tail mismatch");
            status = resources_->copy_from_host(
                kernel_expert_prefix.data_ptr(), host_kernel_expert_prefix.data(),
                host_kernel_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_from_host(
                kernel_unaligned.data_ptr(), host_kernel_unaligned.data(),
                host_kernel_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            per_expert_list = *cached_num_recv_tokens_per_expert_list;
            num_expanded_tokens = *cached_num_expanded_tokens;
        } else {
            rank_prefix = torch::empty({num_ranks_}, int_options);
            expert_prefix = torch::empty(
                {static_cast<int64_t>(local_experts)}, int_options);
            unaligned = torch::empty(
                {static_cast<int64_t>(local_experts)}, int_options);
            destination_slots = torch::empty({x.size(0), topk_idx.size(1)}, int_options);
            source_metadata = torch::empty(
                {static_cast<int64_t>(max_recv_tokens), topk_idx.size(1) + 2},
                int_options);
        }

        auto recv_x = torch::empty(
            {static_cast<int64_t>(do_expand ? expanded_records : max_recv_tokens),
             x.size(1)}, x.options());
        auto recv_topk_indices = torch::empty(
            {static_cast<int64_t>(max_recv_tokens), topk_idx.size(1)},
            topk_idx.options());
        auto recv_topk_weights = std::optional<torch::Tensor>();
        if (topk_weights.has_value()) {
            recv_topk_weights = do_expand ?
                torch::empty(
                    {static_cast<int64_t>(expanded_records)},
                    topk_weights->options()) :
                torch::empty(
                    {static_cast<int64_t>(max_recv_tokens), topk_idx.size(1)},
                    topk_weights->options());
        }
        auto copied_topk_idx = !cached_mode && do_handle_copy ?
            std::optional<torch::Tensor>(topk_idx.clone()) :
            std::optional<torch::Tensor>();
        auto descriptor_tensor = cached_mode ?
            *cached_token_metadata_at_forward :
            torch::empty(
                {static_cast<int64_t>(sizeof(elastic::DispatchHandleDescriptor))},
                metadata_options);
        auto status = transport::TransportStatus{};
        if (!cached_mode) {
            status = resources_->copy_from_host(
                descriptor_tensor.data_ptr(), &expected_descriptor,
                sizeof(expected_descriptor));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }

        elastic::DispatchArguments arguments{};
        arguments.x = x.data_ptr();
        arguments.topk_indices = topk_idx.data_ptr<std::int64_t>();
        arguments.topk_weights = topk_weights.has_value() ?
            topk_weights->data_ptr<float>() : nullptr;
        arguments.communication_buffer = resources_->window_base();
        arguments.workspace = resources_->workspace();
        arguments.recv_x = recv_x.data_ptr();
        arguments.recv_topk_indices = recv_topk_indices.data_ptr<std::int64_t>();
        arguments.recv_topk_weights = recv_topk_weights.has_value() ?
            recv_topk_weights->data_ptr<float>() : nullptr;
        arguments.prefix_per_rank = rank_prefix.data_ptr<std::int32_t>();
        arguments.prefix_per_expert =
            kernel_expert_prefix.data_ptr<std::int32_t>();
        arguments.unaligned_per_expert =
            kernel_unaligned.data_ptr<std::int32_t>();
        arguments.destination_slots = destination_slots.data_ptr<std::int32_t>();
        arguments.source_metadata = source_metadata.data_ptr<std::int32_t>();
        arguments.timeout_cycles = barrier_timeout_cycles_;
        void* stream = nullptr;
        status = resources_->current_stream(&stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_), resources_->workspace_bytes()};
        elastic::DispatchAttempt attempt(dispatch_sequence_);
        TORCH_CHECK(attempt.valid(), "DeepEP Ascend backend: dispatch cannot continue "
                    "after a failed or concurrent attempt");
        arguments.generation = attempt.generation();
        const auto launch_status = elastic::launch_internal_dispatch(
            arguments, tiling, storage, stream);
        if (!launch_status.ok())
            raise_launch_status(launch_status, rank_idx_);
        status = resources_->synchronize_stream(stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        transport::DeviceTransportDiagnostic diagnostic{};
        status = host_transport()->read_diagnostic(&diagnostic);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
#if DEEP_EP_ASCEND_TESTING
        if (std::getenv("DEEP_EP_ASCEND_TEST_DISPATCH_DIAGNOSTIC") != nullptr) {
            diagnostic.abi_version = transport::kTransportCommandAbiVersion;
            diagnostic.error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic.generation = attempt.generation();
        }
#endif
        if (diagnostic.abi_version != transport::kTransportCommandAbiVersion ||
            diagnostic.error != transport::DeviceTransportError::kNone ||
            diagnostic.generation != attempt.generation())
            raise_dispatch_diagnostic(diagnostic, "reported failure");
        if (!cached_mode) {
            status = resources_->copy_to_host(
                host_rank_prefix.data(), rank_prefix.data_ptr(),
                host_rank_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_kernel_expert_prefix.data(),
                kernel_expert_prefix.data_ptr(),
                host_kernel_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_kernel_unaligned.data(), kernel_unaligned.data_ptr(),
                host_kernel_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            std::int32_t expanded_tail = 0;
            for (int expert = 0; expert < num_experts; ++expert) {
                const std::int32_t actual = host_kernel_unaligned[expert];
                const bool local = expert >= first_local_expert &&
                    expert < first_local_expert +
                                 static_cast<int>(local_experts);
                std::uint64_t aligned_actual = 0;
                TORCH_CHECK(actual >= 0 &&
                                (local || actual == 0) &&
                                host_kernel_expert_prefix[expert] ==
                                    expanded_tail &&
                                align_without_overflow(
                                    static_cast<std::uint64_t>(actual),
                                    alignment, &aligned_actual) &&
                                aligned_actual <= static_cast<std::uint64_t>(
                                    std::numeric_limits<int>::max() -
                                    expanded_tail),
                            "DeepEP Ascend backend: dispatch returned invalid expert counts");
                if (local) {
                    const int local_expert = expert - first_local_expert;
                    host_unaligned[local_expert] = actual;
                    host_expert_prefix[local_expert] = do_expand ?
                        expanded_tail + actual :
                        expanded_tail +
                            static_cast<std::int32_t>(aligned_actual);
                    per_expert_list.push_back(
                        static_cast<int>(aligned_actual));
                }
                expanded_tail += static_cast<std::int32_t>(aligned_actual);
            }
            TORCH_CHECK(host_kernel_expert_prefix[num_experts] ==
                            expanded_tail,
                        "DeepEP Ascend backend: dispatch returned invalid expert counts");
            num_expanded_tokens = expanded_tail;
            status = resources_->copy_from_host(
                expert_prefix.data_ptr(), host_expert_prefix.data(),
                host_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_from_host(
                unaligned.data_ptr(), host_unaligned.data(),
                host_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        const int num_recv_tokens = host_rank_prefix.back();
        TORCH_CHECK(num_recv_tokens >= 0 &&
                        num_recv_tokens <= static_cast<int>(max_recv_tokens) &&
                        num_expanded_tokens >= 0 &&
                        num_expanded_tokens <= static_cast<int>(expanded_records),
                    "DeepEP Ascend backend: dispatch returned invalid output counts");
        attempt.complete();
        const int output_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto narrowed_x = recv_x.narrow(0, 0, output_tokens);
        auto narrowed_topk_idx = do_expand ? std::optional<torch::Tensor>() :
            std::optional<torch::Tensor>(recv_topk_indices.narrow(0, 0, output_tokens));
        auto narrowed_topk_weights = recv_topk_weights.has_value() ?
            std::optional<torch::Tensor>(
                recv_topk_weights->narrow(0, 0, output_tokens)) :
            std::optional<torch::Tensor>();
        auto narrowed_metadata = source_metadata.narrow(0, 0, num_recv_tokens);
        return {narrowed_x, std::nullopt, narrowed_topk_idx,
                narrowed_topk_weights, copied_topk_idx, num_recv_tokens,
                num_expanded_tokens, per_expert_list, rank_prefix,
                expert_prefix, unaligned, narrowed_metadata,
                destination_slots, descriptor_tensor, std::nullopt, std::nullopt};
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

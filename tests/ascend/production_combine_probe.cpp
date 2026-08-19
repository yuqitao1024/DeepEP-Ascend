#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "csrc/backends/ascend/elastic_buffer.hpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;
namespace elastic = deep_ep::ascend::elastic;

enum class LifecycleFailure {
    kNone,
    kMaskedProducerProtocol,
    kConsumedGenerationMismatch,
};

enum class LifecycleSnapshotMutation {
    kNone,
    kStagedHeader,
    kQueueHeader,
    kQueueCount,
    kRegistrationCookie,
    kServiceHeader,
};

struct Trace {
    int launches = 0;
    int copies = 0;
    int device = 0;
    int fail_copy_at = -1;
    int fail_create_event_on = 0;
    bool fail_launch = false;
    bool fail_stream = false;
    bool fail_sync = false;
    bool fail_completion_record = false;
    bool event_ready = true;
    bool bad_diagnostic = false;
    bool bad_completion = false;
    LifecycleFailure lifecycle_failure = LifecycleFailure::kNone;
    LifecycleSnapshotMutation lifecycle_snapshot_mutation =
        LifecycleSnapshotMutation::kNone;
    std::uint64_t generation = 0;
    std::vector<std::uint64_t> generations;
    elastic::CoreModeFlags mode_flags = 0;
    const void* bias_0 = nullptr;
    const void* bias_1 = nullptr;
    int create_event_calls = 0;
    int record_event_calls = 0;
    int query_event_calls = 0;
    int destroy_event_calls = 0;
    int synchronize_stream_calls = 0;
    int compute_stream_token = 0;
    int comm_stream_token = 0;
    bool all_launches_on_comm_stream = true;
    std::vector<std::string> order;
    std::uint32_t communicator_rank = 0;
    std::uint32_t communicator_size = 2;
    std::vector<void*> transport_allocations;
} trace;

int alloc(void*, std::uint64_t bytes, void** pointer) {
    *pointer = std::malloc(bytes);
    return *pointer == nullptr ? 1 : 0;
}
int zero(void*, void* pointer, std::uint64_t bytes) {
    std::memset(pointer, 0, bytes);
    return 0;
}
int free_(void*, void* pointer) { std::free(pointer); return 0; }
int current_device(void*, int* device) { *device = trace.device; return 0; }
int stream(void*, runtime::StreamIdentity* value) {
    *value = {trace.fail_stream ? nullptr : &trace.compute_stream_token,
              7, trace.device, 20};
    return 0;
}
int pool_stream(
    void*, int device, bool high_priority, runtime::StreamIdentity* value) {
    if (!high_priority)
        return 86;
    *value = {&trace.comm_stream_token, 11, device, 20};
    return 0;
}
int create_event(void*, void** event) {
    const int call = ++trace.create_event_calls;
    if (call == trace.fail_create_event_on)
        return 86;
    *event = new int(1);
    trace.order.emplace_back("create event");
    return 0;
}
int record_event(void*, void*, void* stream_value) {
    ++trace.record_event_calls;
    const bool communication = stream_value == &trace.comm_stream_token;
    trace.order.emplace_back(
        communication ? "record completion event" : "record dependency event");
    return communication && trace.fail_completion_record ? 87 : 0;
}
int query_event(void*, void*, bool* complete) {
    ++trace.query_event_calls;
    *complete = trace.event_ready;
    trace.order.emplace_back(
        trace.event_ready ? "finish event" : "event pending");
    return 0;
}
int wait_event(void*, void* stream_value, void*) {
    if (stream_value != &trace.comm_stream_token)
        return 88;
    trace.order.emplace_back("comm waits dependency");
    return 0;
}
int synchronize_event(void*, void*, std::uint64_t) { return 0; }
int destroy_event(void*, void* event) {
    ++trace.destroy_event_calls;
    delete static_cast<int*>(event);
    return 0;
}
int sync(void*, void*) {
    ++trace.synchronize_stream_calls;
    return trace.fail_sync ? 83 : 0;
}
int sync_device(void*) { return 0; }
int h2d(void*, void* destination, const void* source, std::uint64_t bytes) {
    std::memcpy(destination, source, bytes);
    return 0;
}
int d2h(void*, void* destination, const void* source, std::uint64_t bytes) {
    const int copy = trace.copies++;
    if (copy == trace.fail_copy_at)
        return 84;
    bool diagnostic_source = false;
    for (std::size_t index = 3; index < trace.transport_allocations.size();
         index += 5)
        diagnostic_source = diagnostic_source ||
            source == trace.transport_allocations[index];
    if (diagnostic_source) {
        if (bytes != sizeof(transport::DeviceTransportDiagnostic))
            return 85;
        auto* diagnostic =
            static_cast<transport::DeviceTransportDiagnostic*>(destination);
        *diagnostic = {};
        diagnostic->abi_version = transport::kTransportCommandAbiVersion;
        diagnostic->generation = trace.generation;
        if (trace.bad_diagnostic)
            diagnostic->error =
                transport::DeviceTransportError::kCompletionFailure;
        return 0;
    }
    std::memcpy(destination, source, bytes);
    return 0;
}
int rank_(void*, std::int64_t, std::uint32_t* rank) {
    *rank = trace.communicator_rank;
    return 0;
}
int size_(void*, std::int64_t, std::uint32_t* size) {
    *size = trace.communicator_size;
    return 0;
}
int team(void*, std::int64_t, std::uint32_t, std::uint32_t,
         const std::uint32_t*, std::uint32_t, std::uint32_t,
         std::uintptr_t* value) { *value = 2; return 0; }
int window(void*, std::int64_t, std::uintptr_t, void*, std::uint64_t,
           std::uintptr_t* value) { *value = 3; return 0; }
int channels(void*, std::int64_t, std::uintptr_t, std::uint32_t) { return 0; }
int ha(void*, std::uint64_t bytes, void** pointer) {
    constexpr std::uint64_t alignment = 64;
    const auto aligned_bytes =
        ((bytes + alignment - 1) / alignment) * alignment;
    *pointer = std::aligned_alloc(
        static_cast<std::size_t>(alignment),
        static_cast<std::size_t>(aligned_bytes));
    if (*pointer == nullptr)
        return 1;
    trace.transport_allocations.push_back(*pointer);
    return 0;
}
int hz(void*, void* pointer, std::uint64_t bytes) {
    return zero(nullptr, pointer, bytes);
}
int hd(void*, void* destination, const void* source, std::uint64_t bytes) {
    return h2d(nullptr, destination, source, bytes);
}
int dh(void*, void* destination, const void* source, std::uint64_t bytes) {
    return d2h(nullptr, destination, source, bytes);
}
int hf(void*, void* pointer) { return free_(nullptr, pointer); }
int noop2(void*, std::uintptr_t, std::uintptr_t) { return 0; }
int noop1(void*, std::uintptr_t) { return 0; }

std::unique_ptr<runtime::CannRuntimeResources> resources(
    int world_size = 2, int rank = 0, bool hybrid = false) {
    trace.communicator_rank = static_cast<std::uint32_t>(rank);
    trace.communicator_size = static_cast<std::uint32_t>(world_size);
    auto result = std::make_unique<runtime::CannRuntimeResources>();
    runtime::CannRuntimeApi runtime_api{
        nullptr, alloc, zero, free_, sync, sync_device, h2d, d2h};
    runtime::StreamEventApi stream_api{
        nullptr, current_device, stream, pool_stream, create_event, record_event,
        query_event, wait_event, synchronize_event, destroy_event};
    transport::CannHostApi host_api{
        nullptr, rank_, size_, team, window, channels, ha, hz, hd, dh, hf,
        noop2, noop1};
    transport::TransportConfig config{};
    config.rank = rank;
    config.world_size = world_size;
    config.communicator_handle = 1;
    config.device_buffer_bytes = 2 * 1024 * 1024;
    config.requested_channels = 1;
    if (hybrid) {
        config.scale_up_size = 2;
        config.topology_kind =
            transport::TransportTopologyKind::kLogicalSimulation;
        config.allow_hybrid_mode = true;
    }
    if (!result->initialize(
            config, 4096, runtime_api, host_api, stream_api).ok())
        return {};
    return result;
}

extern "C" int deep_ep_ascend_launch_barrier(
    elastic::BarrierArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch(
    elastic::DispatchArguments arguments, elastic::CoreTiling tiling, void*) {
    trace.generation = arguments.generation;
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        tiling.transport_context.local_window_base +
        tiling.symmetric_window_layout.control_offset);
    control->dispatch_generation = arguments.generation;
    arguments.prefix_per_rank[0] = 1;
    arguments.prefix_per_rank[1] = 2;
    const auto aligned = static_cast<std::int32_t>(
        ((2 + tiling.expert_alignment - 1) / tiling.expert_alignment) *
        tiling.expert_alignment);
    arguments.prefix_per_expert[0] = 0;
    arguments.prefix_per_expert[1] = aligned;
    arguments.prefix_per_expert[2] = aligned;
    arguments.unaligned_per_expert[0] = 2;
    arguments.unaligned_per_expert[1] = 0;
    arguments.destination_slots[0] = 0;
    arguments.destination_slots[1] = 0;
    const std::array<std::int32_t, 8> metadata{
        0, 0, -1, -1,
        static_cast<std::int32_t>(tiling.num_max_tokens_per_rank),
        3, -1, -1};
    std::memcpy(arguments.source_metadata, metadata.data(), sizeof(metadata));
    return 0;
}
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    elastic::DispatchArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine(
    elastic::CombineArguments arguments, elastic::CoreTiling tiling,
    void* stream_value) {
    ++trace.launches;
    trace.all_launches_on_comm_stream = trace.all_launches_on_comm_stream &&
        stream_value == &trace.comm_stream_token;
    trace.order.emplace_back("launch combine on comm");
    trace.generation = arguments.generation;
    trace.generations.push_back(arguments.generation);
    trace.mode_flags = tiling.mode_flags;
    trace.bias_0 = arguments.bias_0;
    trace.bias_1 = arguments.bias_1;
    if (trace.fail_launch)
        return 73;
    auto* output = static_cast<std::uint16_t*>(arguments.combined_x);
    for (std::uint64_t index = 0; index < tiling.num_tokens * tiling.hidden;
         ++index)
        output[index] = static_cast<std::uint16_t>(0x3000 + index);
    if (arguments.combined_topk_weights != nullptr) {
        for (std::uint64_t index = 0;
             index < tiling.num_tokens * tiling.num_topk; ++index)
            arguments.combined_topk_weights[index] = 0.25F * (index + 1);
    }
    if (trace.lifecycle_failure != LifecycleFailure::kNone) {
        auto* scratch_status = reinterpret_cast<std::uint64_t*>(
            static_cast<std::uint8_t*>(arguments.workspace) +
            tiling.workspace_layout.scratch_status_offset);
        auto* staged = reinterpret_cast<transport::StagedTransportContext*>(
            tiling.transport_context.backend_context);
        auto* queue = reinterpret_cast<transport::TransportCommandQueue*>(
            staged->command_queue);
        auto* service = reinterpret_cast<transport::TransportServiceState*>(
            queue->service_state);
        queue->generation = arguments.generation;
        if (trace.lifecycle_failure ==
            LifecycleFailure::kMaskedProducerProtocol) {
            *scratch_status = (std::uint64_t{2} << 32U) | 4U;
            queue->count = 0;
            service->consumed_generation = arguments.generation;
        } else {
            *scratch_status = 0;
            queue->count = 4;
            service->consumed_generation = 0;
        }
        switch (trace.lifecycle_snapshot_mutation) {
            case LifecycleSnapshotMutation::kNone:
                break;
            case LifecycleSnapshotMutation::kStagedHeader:
                staged->abi_version = 0;
                break;
            case LifecycleSnapshotMutation::kQueueHeader:
                queue->abi_version = 0;
                break;
            case LifecycleSnapshotMutation::kQueueCount:
                queue->count = queue->capacity + 1;
                break;
            case LifecycleSnapshotMutation::kRegistrationCookie:
                staged->reserved = 0;
                break;
            case LifecycleSnapshotMutation::kServiceHeader:
                service->abi_version = 0;
                break;
        }
    }
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        arguments.local_window_base +
        tiling.symmetric_window_layout.control_offset);
    control->combine_generation = arguments.generation +
        (trace.bad_completion ||
         trace.lifecycle_failure != LifecycleFailure::kNone ? 1 : 0);
    return 0;
}
extern "C" int deep_ep_ascend_launch_combine_epilogue(
    elastic::CombineArguments, elastic::CoreTiling, void*) { return 0; }

using Buffer = deep_ep::ascend::ElasticBuffer;
using Event = deep_ep::ascend::EventHandle;
using Tensor = torch::Tensor;

struct Inputs {
    Tensor x = torch::empty(
        {2, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    Tensor weights = torch::empty(
        {2, 2}, torch::TensorOptions().dtype(torch::kFloat));
    Tensor source = torch::empty(
        {2, 4}, torch::TensorOptions().dtype(torch::kInt));
    Tensor indices = torch::empty(
        {1, 2}, torch::TensorOptions().dtype(torch::kLong));
    Tensor prefix = torch::empty(
        {2}, torch::TensorOptions().dtype(torch::kInt));
    Tensor descriptor = torch::empty(
        {static_cast<std::int64_t>(sizeof(elastic::DispatchHandleDescriptor))},
        torch::TensorOptions().dtype(torch::kByte));
    Tensor bias0 = torch::empty(
        {1, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    Tensor bias1 = torch::empty(
        {1, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    elastic::DispatchHandleDescriptor descriptor_value{};

    explicit Inputs(bool expanded = false) {
        if (expanded)
            weights = torch::empty(
                {2}, torch::TensorOptions().dtype(torch::kFloat));
        indices.data_ptr<std::int64_t>()[0] = 0;
        indices.data_ptr<std::int64_t>()[1] = 1;
        prefix.data_ptr<std::int32_t>()[0] = 1;
        prefix.data_ptr<std::int32_t>()[1] = 2;
        const std::array<std::int32_t, 8> normal{
            0, 0, -1, -1, 4, 3, -1, -1};
        const std::array<std::int32_t, 8> expanded_metadata{
            0, 0, 0, -1, 4, 3, -1, 1};
        const auto& metadata = expanded ? expanded_metadata : normal;
        std::memcpy(source.data_ptr(), metadata.data(), sizeof(metadata));
        descriptor_value = elastic::make_attested_dispatch_handle_descriptor(
            7, {0, 2, 0, 2, 0, 1}, 1, 1, 8, 2, 2, 4, 4,
            expanded ? elastic::mode_bit(elastic::CoreMode::kExpanded) : 0);
        write_descriptor();
    }

    void write_descriptor() {
        std::memcpy(descriptor.data_ptr(), &descriptor_value,
                    sizeof(descriptor_value));
    }

    void attest_descriptor_mode(elastic::CoreModeFlags mode_flags) {
        descriptor_value = elastic::make_attested_dispatch_handle_descriptor(
            7, {0, 2, 0, 2, 0, 1}, 1, 1, 8, 2, 2, 4, 4, mode_flags);
        write_descriptor();
    }
};

using Result = std::tuple<Tensor, std::optional<Tensor>, std::optional<Event>>;

Result call(Buffer& buffer, Inputs& inputs,
            const std::optional<Tensor>& weights = std::nullopt,
            const std::optional<Tensor>& bias0 = std::nullopt,
            const std::optional<Tensor>& bias1 = std::nullopt,
            const std::optional<Tensor>& channel = std::nullopt,
            int num_sms = 1, int num_qps = 0,
            const std::optional<Event>& event = std::nullopt,
            bool async = false, bool allocate_on_stream = false,
            bool expanded = false, int num_experts = 2,
            int capacity = 4,
            const std::optional<Event>& event_before_epilogue = std::nullopt) {
    return buffer.combine(
        inputs.x, weights, bias0, bias1, inputs.source, inputs.indices,
        inputs.prefix, inputs.descriptor, channel, num_experts, capacity,
        num_sms, num_qps,
        event, event_before_epilogue, async, allocate_on_stream, expanded);
}

std::unique_ptr<Buffer> buffer(
    bool allow_multiple_reduction = true, bool allow_hybrid_mode = false) {
    const int world_size = allow_hybrid_mode ? 4 : 2;
    auto owned = resources(world_size, 0, allow_hybrid_mode);
    if (!owned)
        return {};
    return Buffer::make_testing_buffer(
        0, std::move(owned), 2 * 1024 * 1024, 1,
        allow_multiple_reduction, 7, world_size, 1, allow_hybrid_mode);
}

bool error_contains(
    const std::function<void()>& operation, const char* expected);

std::string error_message(const std::function<void()>& operation);

bool rank_parameterized_capacity() {
    trace = {};
    auto owned = resources(3);
    if (!owned)
        return false;
    auto target = Buffer::make_testing_buffer(
        0, std::move(owned), 2 * 1024 * 1024, 1, true, 7, 3, 1);
    Inputs inputs;
    inputs.x = torch::empty(
        {3, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    inputs.weights = torch::empty(
        {3, 2}, torch::TensorOptions().dtype(torch::kFloat));
    inputs.source = torch::empty(
        {3, 4}, torch::TensorOptions().dtype(torch::kInt));
    inputs.prefix = torch::empty(
        {3}, torch::TensorOptions().dtype(torch::kInt));
    const std::array<std::int32_t, 12> metadata{
        0, 0, -1, -1,
        1, 2, -1, -1,
        2, 4, -1, -1};
    std::memcpy(inputs.source.data_ptr(), metadata.data(), sizeof(metadata));
    const std::array<std::int32_t, 3> prefix{1, 2, 3};
    std::memcpy(inputs.prefix.data_ptr(), prefix.data(), sizeof(prefix));
    inputs.descriptor_value = elastic::make_attested_dispatch_handle_descriptor(
        7, {0, 3, 0, 3, 0, 1}, 1, 1, 8, 3, 2, 4, 1, 0);
    inputs.write_descriptor();

    if (std::get<2>(call(
            *target, inputs, inputs.weights, std::nullopt, std::nullopt,
            std::nullopt, 1, 0, std::nullopt, false, false, false, 3, 1))
            .has_value() ||
        trace.launches != 1)
        return false;
    if (!error_contains(
            [&] { (void)call(
                *target, inputs, inputs.weights, std::nullopt, std::nullopt,
                std::nullopt, 1, 0, std::nullopt, false, false, false, 2, 1); },
            "rank-partitioned expert capacity"))
        return false;
    return error_contains(
        [&] { (void)call(
            *target, inputs, inputs.weights, std::nullopt, std::nullopt,
            std::nullopt, 1, 0, std::nullopt, false, false, false, 3,
            std::numeric_limits<std::int32_t>::max() / 2); },
        "rank-partitioned expert capacity");
}

std::unique_ptr<Buffer> stateless_buffer(std::uint64_t dispatch_family = 7) {
    auto owned = resources();
    if (!owned)
        return {};
    return Buffer::make_testing_buffer(
        0, std::move(owned), 2 * 1024 * 1024, 1, true,
        dispatch_family, 2, 1);
}

auto run_uncached_dispatch(
    Buffer& target, Inputs& inputs, int capacity = 4, int alignment = 4) {
    const std::optional<Tensor> no_tensor;
    const std::optional<int> no_int;
    const std::optional<std::vector<int>> no_list;
    const std::optional<Event> no_event;
    auto dispatch_x = inputs.x.narrow(0, 0, 1);
    return target.dispatch(
        dispatch_x, no_tensor, inputs.indices, no_tensor, no_tensor,
        no_int, no_int, no_list, no_tensor, no_tensor, no_tensor, no_tensor,
        no_tensor, no_tensor, no_tensor, capacity, 2, alignment, 1, 0,
        no_event, no_event,
        false, false, true, true, false, false, false);
}

template <typename DispatchResult>
auto run_cached_dispatch(
    Buffer& target, Inputs& inputs, const DispatchResult& cached,
    int capacity, int alignment) {
    const std::optional<Tensor> no_tensor;
    const std::optional<Event> no_event;
    auto dispatch_x = inputs.x.narrow(0, 0, 1);
    return target.dispatch(
        dispatch_x, no_tensor, inputs.indices, no_tensor, no_tensor,
        std::get<5>(cached), std::get<6>(cached), std::get<7>(cached),
        std::get<8>(cached), std::get<9>(cached), std::get<10>(cached),
        std::get<12>(cached), std::get<13>(cached), std::get<11>(cached),
        no_tensor, capacity, 2, alignment, 1, 0, no_event, no_event,
        false, false, true, true, false, false, false);
}

bool error_contains(const std::function<void()>& operation,
                    const char* expected) {
    return error_message(operation).find(expected) != std::string::npos;
}

std::string error_message(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::runtime_error& error) {
        return error.what();
    } catch (const pybind11::error_already_set&) {
        return python_error;
    }
    return {};
}

template <typename Value, std::size_t Size>
bool has_values(const Tensor& tensor, const std::array<Value, Size>& expected) {
    if (tensor.numel() != static_cast<std::int64_t>(Size))
        return false;
    const auto* actual = tensor.data_ptr<Value>();
    for (std::size_t index = 0; index < Size; ++index)
        if (actual[index] != expected[index])
            return false;
    return true;
}

bool normal_and_expanded_success() {
    trace = {};
    auto normal_buffer = buffer();
    if (!normal_buffer)
        return false;
    Inputs normal;
    auto result = call(*normal_buffer, normal, normal.weights, normal.bias0,
                       normal.bias1);
    const std::array<std::uint16_t, 8> output{
        0x3000, 0x3001, 0x3002, 0x3003,
        0x3004, 0x3005, 0x3006, 0x3007};
    const std::array<float, 2> output_weights{0.25F, 0.5F};
    if (std::get<0>(result).sizes() != std::vector<std::int64_t>{1, 8} ||
        std::get<0>(result).scalar_type() != torch::kBFloat16 ||
        !has_values(std::get<0>(result), output) ||
        !std::get<1>(result).has_value() ||
        std::get<1>(result)->sizes() != std::vector<std::int64_t>{1, 2} ||
        std::get<1>(result)->scalar_type() != torch::kFloat ||
        !has_values(*std::get<1>(result), output_weights) ||
        std::get<2>(result).has_value() || trace.launches != 1 ||
        trace.copies != 5 ||
        trace.generations != std::vector<std::uint64_t>{1} ||
        trace.bias_0 != normal.bias0.data_ptr() ||
        trace.bias_1 != normal.bias1.data_ptr())
        return false;

    trace = {};
    auto expanded_buffer = buffer();
    Inputs expanded(true);
    auto expanded_result = call(
        *expanded_buffer, expanded, expanded.weights, expanded.bias0,
        std::nullopt, std::nullopt, 1, 0, std::nullopt, false, false, true);
    return std::get<0>(expanded_result).sizes() ==
            std::vector<std::int64_t>{1, 8} &&
        std::get<1>(expanded_result).has_value() &&
        !std::get<2>(expanded_result).has_value() && trace.launches == 1 &&
        trace.copies == 5 &&
        elastic::has_mode(trace.mode_flags, elastic::CoreMode::kExpanded) &&
        elastic::has_mode(
            trace.mode_flags, elastic::CoreMode::kAllowMultipleReduction) &&
        trace.bias_0 == expanded.bias0.data_ptr() && trace.bias_1 == nullptr;
}

bool async_weight_and_layout_matrix_uses_native_completion() {
    trace = {};
    auto target = buffer();
    if (!target)
        return false;
    Inputs normal;
    auto weighted = call(*target, normal, normal.weights);
    if (std::get<2>(weighted).has_value())
        return false;

    auto unweighted_async = call(
        *target, normal, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(unweighted_async).has_value() || trace.copies != 8)
        return false;
    std::get<2>(unweighted_async)->current_stream_wait();
    if (trace.copies != 10)
        return false;

    Inputs expanded(true);
    auto expanded_weighted = call(
        *target, expanded, expanded.weights, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, true, false, true);
    if (!std::get<1>(expanded_weighted).has_value() ||
        !std::get<2>(expanded_weighted).has_value() || trace.copies != 13)
        return false;
    std::get<2>(expanded_weighted)->current_stream_wait();
    auto expanded_unweighted = call(
        *target, expanded, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, true, true);
    return !std::get<1>(expanded_unweighted).has_value() &&
        !std::get<2>(expanded_unweighted).has_value() &&
        trace.generations == std::vector<std::uint64_t>({1, 2, 3, 4}) &&
        trace.copies == 20 && trace.all_launches_on_comm_stream &&
        trace.synchronize_stream_calls == 0;
}

bool async_empty_and_asymmetric_inputs_complete() {
    trace = {};
    auto target = buffer();
    if (!target)
        return false;
    Inputs empty;
    empty.x = torch::empty(
        {0, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    empty.source = torch::empty(
        {0, 4}, torch::TensorOptions().dtype(torch::kInt));
    empty.prefix.data_ptr<std::int32_t>()[0] = 0;
    empty.prefix.data_ptr<std::int32_t>()[1] = 0;
    auto empty_result = call(
        *target, empty, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(empty_result).has_value())
        return false;
    std::get<2>(empty_result)->current_stream_wait();

    Inputs asymmetric;
    asymmetric.prefix.data_ptr<std::int32_t>()[0] = 0;
    asymmetric.prefix.data_ptr<std::int32_t>()[1] = 2;
    const std::array<std::int32_t, 8> metadata{
        4, 3, -1, -1,
        5, 3, -1, -1};
    std::memcpy(asymmetric.source.data_ptr(), metadata.data(), sizeof(metadata));
    const auto asymmetric_result = call(
        *target, asymmetric, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, true);
    return !std::get<2>(asymmetric_result).has_value() &&
        trace.generations == std::vector<std::uint64_t>({1, 2}) &&
        trace.all_launches_on_comm_stream;
}

bool async_predecessor_and_deferred_mode_contract() {
    trace = {};
    auto source = buffer();
    auto target = buffer();
    if (!source || !target)
        return false;
    Inputs inputs;
    auto seed = call(
        *source, inputs, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(seed).has_value())
        return false;
    const std::optional<Event> predecessor = std::get<2>(seed);
    const auto records_after_capture = trace.record_event_calls;
    auto async_result = call(
        *target, inputs, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, predecessor, true, true);
    if (!std::get<2>(async_result).has_value() ||
        trace.record_event_calls != records_after_capture + 1)
        return false;
    std::get<2>(async_result)->current_stream_wait();
    if (!error_contains(
            [&] { (void)call(
                *target, inputs, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, 1, 0, predecessor, false, false); },
            "allocate_on_comm_stream=True") || trace.launches != 2)
        return false;
    if (!error_contains(
            [&] { (void)call(
                *target, inputs, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, 1, 0, std::nullopt, false, true, false, 2, 4,
                predecessor); },
            "previous_event_before_epilogue") || trace.launches != 2)
        return false;
    std::get<2>(seed)->current_stream_wait();
    auto allocated_sync = call(
        *target, inputs, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, true);
    return !std::get<2>(allocated_sync).has_value() && trace.launches == 3 &&
        trace.generations == std::vector<std::uint64_t>({1, 1, 2});
}

bool async_multiflight_and_stable_failure_are_generation_bound() {
    trace = {};
    auto target = buffer();
    if (!target)
        return false;
    Inputs inputs;
    auto first = call(
        *target, inputs, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(first).has_value() ||
        !error_contains([&] { (void)call(*target, inputs); }, "busy") ||
        trace.launches != 1)
        return false;
    std::get<2>(first)->current_stream_wait();
    if (!error_contains(
            [&] { (void)call(*target, inputs); }, "cannot continue") ||
        trace.launches != 1)
        return false;

    trace = {};
    auto sequential_target = buffer();
    Inputs sequential_inputs;
    auto sequential_first = call(
        *sequential_target, sequential_inputs, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(sequential_first).has_value())
        return false;
    std::get<2>(sequential_first)->current_stream_wait();
    auto second = call(
        *sequential_target, sequential_inputs, std::nullopt, std::nullopt,
        std::nullopt,
        std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(second).has_value())
        return false;
    std::get<2>(second)->current_stream_wait();
    if (trace.generations != std::vector<std::uint64_t>({1, 2}))
        return false;

    trace = {};
    auto failing_target = buffer();
    Inputs failing_inputs;
    trace.bad_completion = true;
    auto failed = call(
        *failing_target, failing_inputs, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(failed).has_value())
        return false;
    const auto first_failure = error_message(
        [&] { std::get<2>(failed)->current_stream_wait(); });
    trace.bad_completion = false;
    const auto repeated_failure = error_message(
        [&] { std::get<2>(failed)->current_stream_wait(); });
    return first_failure.find("completion mismatch") !=
            std::string::npos &&
        repeated_failure == first_failure &&
        error_contains(
            [&] { (void)call(*failing_target, failing_inputs); },
            "cannot continue") &&
        trace.launches == 1;
}

bool async_prelaunch_and_record_failures_are_bounded() {
    trace = {};
    auto create_target = buffer();
    Inputs create_inputs;
    trace.fail_create_event_on = 2;
    if (!error_contains(
            [&] { (void)call(
                *create_target, create_inputs, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, 1, 0, std::nullopt, true); },
            "create_event") || trace.launches != 0)
        return false;
    trace.fail_create_event_on = 0;
    auto recovered = call(
        *create_target, create_inputs, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, 1, 0, std::nullopt, true);
    if (!std::get<2>(recovered).has_value())
        return false;
    std::get<2>(recovered)->current_stream_wait();
    if (trace.generations != std::vector<std::uint64_t>{1})
        return false;

    trace = {};
    auto record_target = buffer();
    Inputs record_inputs;
    trace.fail_completion_record = true;
    const auto message = error_message([&] { (void)call(
        *record_target, record_inputs, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, 1, 0, std::nullopt, true); });
    return message.find("record_event") != std::string::npos &&
        trace.launches == 1 &&
        error_contains(
            [&] { (void)call(*record_target, record_inputs); },
            "cannot continue") &&
        trace.launches == 1;
}

bool hybrid_async_rejects_before_launch() {
    trace = {};
    auto target = buffer(true, true);
    Inputs inputs;
    return error_contains(
            [&] { (void)call(
                *target, inputs, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, 1, 0, std::nullopt, true); },
            "pure-scale-up") &&
        trace.launches == 0;
}

bool successful_dispatch_handle_is_statelessly_valid() {
    trace = {};
    auto target = stateless_buffer();
    if (!target)
        return false;
    Inputs inputs;
    auto dispatched = run_uncached_dispatch(*target, inputs);
    inputs.descriptor = *std::get<13>(dispatched);
    const auto result = call(*target, inputs);
    return !std::get<2>(result).has_value() && trace.launches == 1 &&
        trace.generations == std::vector<std::uint64_t>{2};
}

bool dispatch_descriptor_modes_are_exactly_compatible() {
    const auto expanded = elastic::mode_bit(elastic::CoreMode::kExpanded);
    const auto zero_padding =
        elastic::mode_bit(elastic::CoreMode::kZeroPadding);
    const auto combine_mode =
        elastic::mode_bit(elastic::CoreMode::kAllowMultipleReduction);

    trace = {};
    auto aligned_target = buffer();
    Inputs aligned(true);
    aligned.attest_descriptor_mode(expanded | zero_padding);
    const auto aligned_result = call(
        *aligned_target, aligned, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, false, true);
    if (std::get<2>(aligned_result).has_value() || trace.launches != 1 ||
        !elastic::has_mode(trace.mode_flags, elastic::CoreMode::kExpanded) ||
        elastic::has_mode(trace.mode_flags, elastic::CoreMode::kZeroPadding))
        return false;

    trace = {};
    auto normal_target = buffer();
    Inputs normal;
    normal.attest_descriptor_mode(zero_padding);
    if (!error_contains(
            [&] { (void)call(*normal_target, normal); }, "dispatch handle") ||
        trace.launches != 0)
        return false;

    trace = {};
    auto unsupported_target = buffer();
    Inputs unsupported(true);
    unsupported.attest_descriptor_mode(expanded | combine_mode);
    if (!error_contains(
            [&] { (void)call(
                *unsupported_target, unsupported, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, 1, 0, std::nullopt, false, false,
                true); },
            "dispatch handle") || trace.launches != 0)
        return false;

    trace = {};
    auto padded_target = buffer();
    Inputs padded(true);
    padded.descriptor_value.mode_flags |= zero_padding;
    padded.write_descriptor();
    const auto padded_result = call(
        *padded_target, padded, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, false, true);
    return !std::get<2>(padded_result).has_value() && trace.launches == 1 &&
        elastic::has_mode(trace.mode_flags, elastic::CoreMode::kExpanded) &&
        !elastic::has_mode(trace.mode_flags, elastic::CoreMode::kZeroPadding);
}

bool padding_expanded_extent_is_accepted() {
    const auto expanded = elastic::mode_bit(elastic::CoreMode::kExpanded);
    const auto zero_padding =
        elastic::mode_bit(elastic::CoreMode::kZeroPadding);

    trace = {};
    auto target = buffer();
    Inputs inputs(true);
    inputs.x = torch::empty(
        {24, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    inputs.weights = torch::empty(
        {24}, torch::TensorOptions().dtype(torch::kFloat));
    inputs.source.data_ptr<std::int32_t>()[6] = -1;
    inputs.source.data_ptr<std::int32_t>()[7] = 23;
    inputs.descriptor_value =
        elastic::make_attested_dispatch_handle_descriptor(
            7, {0, 2, 0, 2, 0, 1}, 1, 1, 8, 2, 2, 8, 4,
            expanded | zero_padding);
    inputs.write_descriptor();

    const auto result = call(
        *target, inputs, inputs.weights, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, false, true);
    return !std::get<2>(result).has_value() && trace.launches == 1 &&
        elastic::has_mode(trace.mode_flags, elastic::CoreMode::kExpanded) &&
        !elastic::has_mode(trace.mode_flags, elastic::CoreMode::kZeroPadding);
}

bool long_lived_dispatch_validation_state_is_constant() {
    trace = {};
    auto target = stateless_buffer();
    if (!target)
        return false;
    Inputs inputs;
    const auto validation_state_bytes =
        target->testing_dispatch_validation_state_bytes();
    if (validation_state_bytes != 2 * sizeof(std::uint64_t))
        return false;
    auto oldest = run_uncached_dispatch(*target, inputs, 4, 1);
    auto current = oldest;
    for (int iteration = 0; iteration < 64; ++iteration)
        current = run_uncached_dispatch(
            *target, inputs, 4 + (iteration + 1) % 2,
            (iteration + 1) % 2 == 0 ? 1 : 4);
    if (target->testing_dispatch_validation_state_bytes() !=
            validation_state_bytes)
        return false;
    if (!error_contains(
            [&] { (void)run_cached_dispatch(*target, inputs, oldest, 4, 1); },
            "dispatch handle") || trace.launches != 0)
        return false;
    inputs.descriptor = *std::get<13>(oldest);
    if (!error_contains([&] { (void)call(*target, inputs); },
                        "dispatch handle") || trace.launches != 0)
        return false;
    auto cached = run_cached_dispatch(*target, inputs, current, 4, 1);
    if (target->testing_dispatch_validation_state_bytes() !=
            validation_state_bytes)
        return false;
    elastic::DispatchHandleDescriptor refreshed{};
    std::memcpy(&refreshed, std::get<13>(cached)->data_ptr(), sizeof(refreshed));
    if (refreshed.generation != 66)
        return false;
    inputs.descriptor = *std::get<13>(cached);
    const auto result = call(*target, inputs);
    return !std::get<2>(result).has_value() && trace.launches == 1 &&
        trace.generations == std::vector<std::uint64_t>{67};
}

bool cross_buffer_dispatch_handle_is_retryable() {
    trace = {};
    auto source = stateless_buffer(7);
    auto target = stateless_buffer(8);
    if (!source || !target)
        return false;
    Inputs inputs;
    auto foreign = run_uncached_dispatch(*source, inputs);
    inputs.descriptor = *std::get<13>(foreign);
    if (!error_contains(
            [&] { (void)call(*target, inputs); }, "dispatch handle") ||
        trace.launches != 0)
        return false;
    auto local = run_uncached_dispatch(*target, inputs);
    inputs.descriptor = *std::get<13>(local);
    const auto result = call(*target, inputs);
    return !std::get<2>(result).has_value() && trace.launches == 1 &&
        trace.generations == std::vector<std::uint64_t>{2};
}

bool preflight_is_retryable() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    const auto reject_then_restore = [&](const char* expected,
                                         const std::function<void()>& mutate,
                                         const std::function<void()>& restore) {
        mutate();
        const bool rejected = error_contains(
            [&] { (void)call(*target, inputs); }, expected);
        restore();
        return rejected && trace.launches == 0;
    };

    const auto original = inputs.descriptor_value;
    if (!reject_then_restore("dispatch handle", [&] {
            ++inputs.descriptor_value.family; inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }) ||
        !reject_then_restore("dispatch handle", [&] {
            inputs.descriptor_value.topology.scale_up_size = 1;
            inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }) ||
        !reject_then_restore("dispatch handle", [&] {
            inputs.descriptor_value.hidden = 16; inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }) ||
        !reject_then_restore("dispatch handle", [&] {
            inputs.descriptor_value.num_experts = 4; inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }) ||
        !reject_then_restore("dispatch handle", [&] {
            inputs.descriptor_value.expert_alignment = 8;
            inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }) ||
        !reject_then_restore("dispatch handle", [&] {
            inputs.descriptor_value.num_max_tokens_per_rank = 3;
            inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }) ||
        !reject_then_restore("dispatch handle", [&] {
            inputs.descriptor_value.mode_flags =
                elastic::mode_bit(elastic::CoreMode::kExpanded);
            inputs.write_descriptor();
        }, [&] { inputs.descriptor_value = original; inputs.write_descriptor(); }))
        return false;

    auto saved_prefix = inputs.prefix.data_ptr<std::int32_t>()[1];
    if (!reject_then_restore("prefix", [&] {
            inputs.prefix.data_ptr<std::int32_t>()[1] = 1;
        }, [&] { inputs.prefix.data_ptr<std::int32_t>()[1] = saved_prefix; }))
        return false;
    auto saved_metadata = inputs.source.data_ptr<std::int32_t>()[0];
    if (!reject_then_restore("source metadata", [&] {
            inputs.source.data_ptr<std::int32_t>()[0] = 7;
        }, [&] { inputs.source.data_ptr<std::int32_t>()[0] = saved_metadata; }))
        return false;

    trace.fail_copy_at = trace.copies;
    if (!error_contains([&] { (void)call(*target, inputs); }, "copy_to_host") ||
        trace.launches != 0)
        return false;
    trace.fail_copy_at = -1;
    trace.fail_stream = true;
    if (!error_contains([&] { (void)call(*target, inputs); }, "current_stream") ||
        trace.launches != 0)
        return false;
    trace.fail_stream = false;
    return !std::get<2>(call(*target, inputs)).has_value() &&
        trace.launches == 1 &&
        trace.generations == std::vector<std::uint64_t>{1};
}

bool interior_origin_extent_is_retryable() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    inputs.source.data_ptr<std::int32_t>()[0] = 1;
    if (!error_contains(
            [&] { (void)call(*target, inputs); }, "source metadata") ||
        trace.launches != 0)
        return false;
    inputs.source.data_ptr<std::int32_t>()[0] = 0;
    const auto result = call(*target, inputs);
    return !std::get<2>(result).has_value() && trace.launches == 1 &&
        trace.generations == std::vector<std::uint64_t>{1};
}

bool positive_alignment_mutation_is_retryable() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    inputs.descriptor_value.expert_alignment = 8;
    inputs.write_descriptor();
    if (!error_contains(
            [&] { (void)call(*target, inputs); }, "dispatch handle") ||
        trace.launches != 0)
        return false;
    inputs.descriptor_value.expert_alignment = 4;
    inputs.write_descriptor();
    const auto result = call(*target, inputs);
    return !std::get<2>(result).has_value() && trace.launches == 1 &&
        trace.generations == std::vector<std::uint64_t>{1};
}

bool expanded_slot_validation_is_retryable() {
    trace = {};
    auto target = buffer();
    Inputs inputs(true);
    inputs.source.data_ptr<std::int32_t>()[2] = 2;
    if (!error_contains(
            [&] { (void)call(*target, inputs, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, 1, 0, std::nullopt,
                            false, false, true); },
            "expanded slot") || trace.launches != 0)
        return false;
    inputs.source.data_ptr<std::int32_t>()[2] = 0;
    return !std::get<2>(call(
        *target, inputs, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, false, true)).has_value() &&
        trace.launches == 1;
}

bool single_reduction_constructor_flag() {
    trace = {};
    auto target = buffer(false);
    Inputs inputs(true);
    if (!error_contains(
            [&] { (void)call(*target, inputs, inputs.weights, std::nullopt,
                            std::nullopt, std::nullopt, 1, 0, std::nullopt,
                            false, false, true); },
            "allow_multiple_reduction") || trace.launches != 0)
        return false;
    const auto result = call(
        *target, inputs, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, 1, 0, std::nullopt, false, false, true);
    return !std::get<1>(result).has_value() &&
        !std::get<2>(result).has_value() && trace.launches == 1 &&
        elastic::has_mode(trace.mode_flags, elastic::CoreMode::kExpanded) &&
        !elastic::has_mode(
            trace.mode_flags, elastic::CoreMode::kAllowMultipleReduction);
}

bool tensor_and_flag_validation() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    Tensor wrong_device = torch::empty(
        {2, 8}, inputs.x.options().device(1));
    auto saved_x = inputs.x;
    inputs.x = wrong_device;
    if (!error_contains([&] { (void)call(*target, inputs); }, "buffer device"))
        return false;
    inputs.x = saved_x;
    const std::optional<Tensor> channel = Tensor{};
    if (!error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, channel); }, "channel") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 2); }, "num_sms=1") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 1, 1); }, "num_qps=0"))
        return false;
    return trace.launches == 0;
}

bool runtime_failure_poisons(const std::function<void()>& inject,
                             const char* failure) {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    inject();
    if (!error_contains([&] { (void)call(*target, inputs); }, failure) ||
        trace.launches != 1)
        return false;
    trace.fail_launch = false;
    trace.fail_sync = false;
    trace.bad_diagnostic = false;
    trace.bad_completion = false;
    trace.fail_copy_at = -1;
    return error_contains(
        [&] { (void)call(*target, inputs); }, "cannot continue") &&
        trace.launches == 1;
}

bool runtime_failures_poison() {
    return runtime_failure_poisons([] { trace.fail_launch = true; },
                                   "backend error 73") &&
        runtime_failure_poisons([] { trace.bad_diagnostic = true; },
                                "device diagnostic") &&
        runtime_failure_poisons([] { trace.bad_completion = true; },
                                "completion mismatch") &&
        runtime_failure_poisons([] { trace.fail_copy_at = 4; },
                                "copy_to_host");
}

bool masked_producer_failure_reports_lifecycle() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    trace.lifecycle_failure = LifecycleFailure::kMaskedProducerProtocol;
    const auto message = error_message([&] { (void)call(*target, inputs); });
    return message.find("completion mismatch") != std::string::npos &&
        message.find("lifecycle_snapshot=available") != std::string::npos &&
        message.find("scratch_status=8589934596") != std::string::npos &&
        message.find("scratch_peer=1") != std::string::npos &&
        message.find("scratch_error=4") != std::string::npos &&
        message.find("queue_generation=1") != std::string::npos &&
        message.find("queue_count=0") != std::string::npos &&
        message.find("consumed_generation=1") != std::string::npos &&
        message.find("diagnostic_generation=1") != std::string::npos &&
        message.find("diagnostic_error=none") != std::string::npos;
}

bool consumed_generation_mismatch_reports_lifecycle() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    trace.lifecycle_failure = LifecycleFailure::kConsumedGenerationMismatch;
    const auto message = error_message([&] { (void)call(*target, inputs); });
    return message.find("completion mismatch") != std::string::npos &&
        message.find("lifecycle_snapshot=available") != std::string::npos &&
        message.find("scratch_status=0") != std::string::npos &&
        message.find("scratch_peer=-1") != std::string::npos &&
        message.find("scratch_error=0") != std::string::npos &&
        message.find("queue_generation=1") != std::string::npos &&
        message.find("queue_count=4") != std::string::npos &&
        message.find("consumed_generation=0") != std::string::npos &&
        message.find("diagnostic_generation=1") != std::string::npos &&
        message.find("diagnostic_error=none") != std::string::npos;
}

bool snapshot_failure_preserves_combine_failure() {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    trace.lifecycle_failure = LifecycleFailure::kConsumedGenerationMismatch;
    trace.fail_copy_at = 5;
    const auto message = error_message([&] { (void)call(*target, inputs); });
    return message.find("completion mismatch") != std::string::npos &&
        message.find("lifecycle_snapshot=unavailable") != std::string::npos &&
        message.find("snapshot_operation=copy_to_host") != std::string::npos &&
        message.find("snapshot_backend_code=84") != std::string::npos;
}

bool malformed_lifecycle_snapshot_reports_unavailable(
    LifecycleSnapshotMutation mutation, const char* expected_error) {
    trace = {};
    auto target = buffer();
    Inputs inputs;
    trace.lifecycle_failure = LifecycleFailure::kConsumedGenerationMismatch;
    trace.lifecycle_snapshot_mutation = mutation;
    const auto message = error_message([&] { (void)call(*target, inputs); });
    return message.find("completion mismatch") != std::string::npos &&
        message.find("lifecycle_snapshot=unavailable") != std::string::npos &&
        message.find("snapshot_operation=combine_lifecycle_snapshot") !=
            std::string::npos &&
        message.find(expected_error) != std::string::npos;
}

int main() {
    int failures = 0;
    const auto check = [&failures](bool passed, const char* name) {
        if (!passed) {
            ++failures;
            std::cerr << "failed: " << name << '\n';
        }
    };
    check(normal_and_expanded_success(), "normal and expanded outputs");
    check(async_weight_and_layout_matrix_uses_native_completion(),
          "async weighted and expanded completion matrix");
    check(async_empty_and_asymmetric_inputs_complete(),
          "async empty and asymmetric inputs");
    check(async_predecessor_and_deferred_mode_contract(),
          "async predecessor and deferred mode contract");
    check(async_multiflight_and_stable_failure_are_generation_bound(),
          "async generation and stable failure contract");
    check(async_prelaunch_and_record_failures_are_bounded(),
          "async prelaunch and record failure contract");
    check(hybrid_async_rejects_before_launch(),
          "hybrid async reject before launch");
    check(rank_parameterized_capacity(), "rank-parameterized capacity");
    check(successful_dispatch_handle_is_statelessly_valid(),
          "successful dispatch handle is statelessly valid");
    check(dispatch_descriptor_modes_are_exactly_compatible(),
          "dispatch descriptor modes are exactly compatible");
    check(padding_expanded_extent_is_accepted(),
          "padding-expanded extent is accepted");
    check(long_lived_dispatch_validation_state_is_constant(),
          "long-lived dispatch validation state is constant");
    check(cross_buffer_dispatch_handle_is_retryable(),
          "cross-buffer dispatch handle retry");
    check(preflight_is_retryable(), "descriptor and metadata preflight retry");
    check(interior_origin_extent_is_retryable(), "interior origin extent retry");
    check(positive_alignment_mutation_is_retryable(),
          "positive alignment mutation retry");
    check(expanded_slot_validation_is_retryable(), "expanded slot preflight");
    check(single_reduction_constructor_flag(), "single reduction constructor flag");
    check(tensor_and_flag_validation(), "tensor and deferred flags");
    check(runtime_failures_poison(), "post-launch failure poisoning");
    check(masked_producer_failure_reports_lifecycle(),
          "masked producer failure lifecycle diagnostic");
    check(consumed_generation_mismatch_reports_lifecycle(),
          "consumed generation mismatch lifecycle diagnostic");
    check(snapshot_failure_preserves_combine_failure(),
          "snapshot failure preserves combine failure");
    check(malformed_lifecycle_snapshot_reports_unavailable(
              LifecycleSnapshotMutation::kStagedHeader,
              "malformed staged transport context"),
          "malformed staged header snapshot");
    check(malformed_lifecycle_snapshot_reports_unavailable(
              LifecycleSnapshotMutation::kQueueHeader,
              "malformed transport command queue"),
          "malformed queue header snapshot");
    check(malformed_lifecycle_snapshot_reports_unavailable(
              LifecycleSnapshotMutation::kQueueCount,
              "malformed transport command queue"),
          "malformed queue count snapshot");
    check(malformed_lifecycle_snapshot_reports_unavailable(
              LifecycleSnapshotMutation::kRegistrationCookie,
              "malformed transport registration cookie"),
          "malformed registration cookie snapshot");
    check(malformed_lifecycle_snapshot_reports_unavailable(
              LifecycleSnapshotMutation::kServiceHeader,
              "malformed transport service state"),
          "malformed service header snapshot");
    return failures == 0 ? 0 : 1;
}

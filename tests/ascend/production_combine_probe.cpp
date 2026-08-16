#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "csrc/backends/ascend/elastic_buffer.hpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;
namespace elastic = deep_ep::ascend::elastic;

struct Trace {
    int launches = 0;
    int copies = 0;
    int device = 0;
    int fail_copy_at = -1;
    bool fail_launch = false;
    bool fail_stream = false;
    bool fail_sync = false;
    bool bad_diagnostic = false;
    bool bad_completion = false;
    std::uint64_t generation = 0;
    std::vector<std::uint64_t> generations;
    elastic::CoreModeFlags mode_flags = 0;
    const void* bias_0 = nullptr;
    const void* bias_1 = nullptr;
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
void* stream(void*) { return trace.fail_stream ? nullptr : &trace; }
int sync(void*, void*) { return trace.fail_sync ? 83 : 0; }
int sync_device(void*) { return 0; }
int h2d(void*, void* destination, const void* source, std::uint64_t bytes) {
    std::memcpy(destination, source, bytes);
    return 0;
}
int d2h(void*, void* destination, const void* source, std::uint64_t bytes) {
    const int copy = trace.copies++;
    if (copy == trace.fail_copy_at)
        return 84;
    if (bytes == sizeof(transport::DeviceTransportDiagnostic)) {
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
int rank_(void*, std::int64_t, std::uint32_t* rank) { *rank = 0; return 0; }
int size_(void*, std::int64_t, std::uint32_t* size) { *size = 2; return 0; }
int team(void*, std::int64_t, std::uint32_t, std::uint32_t,
         const std::uint32_t*, std::uint32_t, std::uint32_t,
         std::uintptr_t* value) { *value = 2; return 0; }
int window(void*, std::int64_t, std::uintptr_t, void*, std::uint64_t,
           std::uintptr_t* value) { *value = 3; return 0; }
int channels(void*, std::int64_t, std::uintptr_t, std::uint32_t) { return 0; }
int ha(void*, std::uint64_t bytes, void** pointer) {
    return alloc(nullptr, bytes, pointer);
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

std::unique_ptr<runtime::CannRuntimeResources> resources() {
    auto result = std::make_unique<runtime::CannRuntimeResources>();
    runtime::CannRuntimeApi runtime_api{
        nullptr, alloc, zero, free_, current_device, stream, sync, sync_device,
        h2d, d2h};
    transport::CannHostApi host_api{
        nullptr, rank_, size_, team, window, channels, ha, hz, hd, dh, hf,
        noop2, noop1};
    transport::TransportConfig config{};
    config.rank = 0;
    config.world_size = 2;
    config.communicator_handle = 1;
    config.device_buffer_bytes = 2 * 1024 * 1024;
    config.requested_channels = 1;
    if (!result->initialize(config, 4096, runtime_api, host_api).ok())
        return {};
    return result;
}

extern "C" int deep_ep_ascend_launch_barrier(
    elastic::BarrierArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch(
    elastic::DispatchArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    elastic::DispatchArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine(
    elastic::CombineArguments arguments, elastic::CoreTiling tiling, void*) {
    ++trace.launches;
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
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        arguments.local_window_base +
        tiling.symmetric_window_layout.control_offset);
    control->combine_generation = arguments.generation +
        (trace.bad_completion ? 1 : 0);
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
        descriptor_value = elastic::make_dispatch_handle_descriptor(
            7, {0, 2, 0, 2, 0, 1}, 1, 8, 2, 2, 4, 4,
            expanded ? elastic::mode_bit(elastic::CoreMode::kExpanded) : 0);
        write_descriptor();
    }

    void write_descriptor() {
        std::memcpy(descriptor.data_ptr(), &descriptor_value,
                    sizeof(descriptor_value));
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
            bool expanded = false) {
    return buffer.combine(
        inputs.x, weights, bias0, bias1, inputs.source, inputs.indices,
        inputs.prefix, inputs.descriptor, channel, 2, 4, num_sms, num_qps,
        event, event, async, allocate_on_stream, expanded);
}

std::unique_ptr<Buffer> buffer(bool allow_multiple_reduction = true) {
    auto owned = resources();
    if (!owned)
        return {};
    return Buffer::make_testing_buffer(
        0, std::move(owned), 2 * 1024 * 1024, 1,
        allow_multiple_reduction);
}

bool error_contains(const std::function<void()>& operation,
                    const char* expected) {
    try {
        operation();
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    } catch (const pybind11::error_already_set&) {
        return python_error.find(expected) != std::string::npos;
    }
    return false;
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
        elastic::has_mode(trace.mode_flags, elastic::CoreMode::kExpanded) &&
        elastic::has_mode(
            trace.mode_flags, elastic::CoreMode::kAllowMultipleReduction) &&
        trace.bias_0 == expanded.bias0.data_ptr() && trace.bias_1 == nullptr;
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
            inputs.descriptor_value.expert_alignment = 0;
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
    const std::optional<Event> event = Event{};
    const std::optional<Tensor> channel = Tensor{};
    if (!error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, channel); }, "channel") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 2); }, "num_sms=1") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 1, 1); }, "num_qps=0") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 1, 0, event); },
            "synchronous") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 1, 0, std::nullopt,
            true); }, "synchronous") ||
        !error_contains([&] { (void)call(*target, inputs, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, 1, 0, std::nullopt,
            false, true); }, "synchronous"))
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
        runtime_failure_poisons([] { trace.fail_sync = true; },
                                "synchronize_stream") &&
        runtime_failure_poisons([] { trace.bad_diagnostic = true; },
                                "device diagnostic") &&
        runtime_failure_poisons([] { trace.bad_completion = true; },
                                "completion mismatch") &&
        runtime_failure_poisons([] { trace.fail_copy_at = 4; },
                                "copy_to_host");
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
    check(preflight_is_retryable(), "descriptor and metadata preflight retry");
    check(expanded_slot_validation_is_retryable(), "expanded slot preflight");
    check(single_reduction_constructor_flag(), "single reduction constructor flag");
    check(tensor_and_flag_validation(), "tensor and deferred flags");
    check(runtime_failures_poison(), "post-launch failure poisoning");
    return failures == 0 ? 0 : 1;
}

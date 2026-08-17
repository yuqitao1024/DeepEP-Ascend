#include <array>
#include <cstdlib>
#include <cstring>
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
    int d2h_copies = 0;
    int h2d_copies = 0;
    int device = 0;
    std::uint64_t generation = 0;
    std::vector<std::uint64_t> generations;
    std::vector<const void*> kernel_expert_prefixes;
    std::vector<const void*> kernel_unaligned_counts;
    bool cached_private_contract = true;
    bool bad_diagnostic = false;
    bool fail_d2h = false;
    bool fail_h2d = false;
    bool fail_launch = false;
    bool fail_stream = false;
} trace;

int alloc(void*, std::uint64_t n, void** p) { *p = std::malloc(n); return *p ? 0 : 1; }
int zero(void*, void* p, std::uint64_t n) { std::memset(p, 0, n); return 0; }
int free_(void*, void* p) { std::free(p); return 0; }
int current_device(void*, int* device) { *device = trace.device; return 0; }
void* stream(void*) { return trace.fail_stream ? nullptr : &trace; }
int sync(void*, void*) { return 0; }
int sync_device(void*) { return 0; }
int h2d(void*, void* d, const void* s, std::uint64_t n) {
    ++trace.h2d_copies;
    if (trace.fail_h2d) return 92;
    std::memcpy(d, s, n);
    return 0;
}
int d2h(void*, void* d, const void* s, std::uint64_t n) {
    ++trace.d2h_copies;
    if (n == sizeof(transport::DeviceTransportDiagnostic)) {
        auto* diagnostic = static_cast<transport::DeviceTransportDiagnostic*>(d);
        *diagnostic = {};
        diagnostic->abi_version = transport::kTransportCommandAbiVersion;
        diagnostic->generation = trace.generation;
        if (trace.bad_diagnostic) diagnostic->error = transport::DeviceTransportError::kCompletionFailure;
        return 0;
    }
    if (trace.fail_d2h) return 91;
    std::memcpy(d, s, n); return 0;
}
int rank_(void*, std::int64_t, std::uint32_t* v) { *v = 0; return 0; }
int size_(void*, std::int64_t, std::uint32_t* v) { *v = 2; return 0; }
int team(void*, std::int64_t, std::uint32_t, std::uint32_t, const std::uint32_t*, std::uint32_t, std::uint32_t, std::uintptr_t* v) { *v = 2; return 0; }
int window(void*, std::int64_t, std::uintptr_t, void*, std::uint64_t, std::uintptr_t* v) { *v = 3; return 0; }
int channels(void*, std::int64_t, std::uintptr_t, std::uint32_t) { return 0; }
int ha(void*, std::uint64_t n, void** p) { return alloc(nullptr,n,p); }
int hz(void*, void* p, std::uint64_t n) { return zero(nullptr,p,n); }
int hd(void*, void* d, const void* s, std::uint64_t n) { return h2d(nullptr,d,s,n); }
int dh(void*, void* d, const void* s, std::uint64_t n) { return d2h(nullptr,d,s,n); }
int hf(void*, void* p) { return free_(nullptr,p); }
int noop2(void*, std::uintptr_t, std::uintptr_t) { return 0; }
int noop1(void*, std::uintptr_t) { return 0; }

std::unique_ptr<runtime::CannRuntimeResources> resources() {
    auto result = std::make_unique<runtime::CannRuntimeResources>();
    runtime::CannRuntimeApi r{nullptr,alloc,zero,free_,current_device,stream,sync,sync_device,h2d,d2h};
    transport::CannHostApi h{nullptr,rank_,size_,team,window,channels,ha,hz,hd,dh,hf,noop2,noop1};
    transport::TransportConfig c{}; c.rank=0; c.world_size=2; c.communicator_handle=1;
    c.device_buffer_bytes=2*1024*1024; c.requested_channels=1;
    if (!result->initialize(c, 4096, r, h).ok()) return {};
    return result;
}

extern "C" int deep_ep_ascend_launch_barrier(elastic::BarrierArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch(elastic::DispatchArguments a, elastic::CoreTiling t, void*) {
    ++trace.launches;
    trace.generation = a.generation;
    trace.generations.push_back(a.generation);
    trace.kernel_expert_prefixes.push_back(a.prefix_per_expert);
    trace.kernel_unaligned_counts.push_back(a.unaligned_per_expert);
    if (trace.fail_launch) return 73;
    const bool cached = elastic::has_mode(t.mode_flags, elastic::CoreMode::kCached);
    const bool expanded = elastic::has_mode(t.mode_flags, elastic::CoreMode::kExpanded);
    const std::array<std::int32_t, 3> private_expert_prefix{0, 4, 4};
    const std::array<std::int32_t, 2> private_unaligned{2, 0};
    if (t.num_tokens == 0) {
        a.prefix_per_rank[0] = 0;
        a.prefix_per_rank[1] = 0;
        std::memset(a.prefix_per_expert, 0,
                    private_expert_prefix.size() * sizeof(std::int32_t));
        std::memset(a.unaligned_per_expert, 0,
                    private_unaligned.size() * sizeof(std::int32_t));
        return 0;
    }
    if (cached &&
        (std::memcmp(a.prefix_per_expert, private_expert_prefix.data(),
                     sizeof(private_expert_prefix)) != 0 ||
         std::memcmp(a.unaligned_per_expert, private_unaligned.data(),
                     sizeof(private_unaligned)) != 0))
        trace.cached_private_contract = false;
    const std::array<std::uint16_t, 16> payload{
        0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 0x0108,
        0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208};
    std::memset(a.recv_x, 0, (expanded ? 4 : 2) * 8 * sizeof(std::uint16_t));
    std::memcpy(a.recv_x, payload.data(), payload.size() * sizeof(payload[0]));
    a.recv_topk_indices[0] = 0;
    a.recv_topk_indices[1] = -1;
    a.recv_topk_indices[2] = 0;
    a.recv_topk_indices[3] = -1;
    if (a.recv_topk_weights != nullptr) {
        a.recv_topk_weights[0] = 0.25F;
        a.recv_topk_weights[1] = expanded ? 0.5F : 0.75F;
        a.recv_topk_weights[2] = expanded ? 0.0F : 0.5F;
        a.recv_topk_weights[3] = expanded ? 0.0F : 1.0F;
    }
    a.prefix_per_rank[0] = 1;
    a.prefix_per_rank[1] = 2;
    std::memcpy(a.prefix_per_expert, private_expert_prefix.data(),
                sizeof(private_expert_prefix));
    std::memcpy(a.unaligned_per_expert, private_unaligned.data(),
                sizeof(private_unaligned));
    a.destination_slots[0] = 0;
    a.destination_slots[1] = 0;
    const std::array<std::int32_t, 8> normal_metadata{
        1, 0, -1, -1, 7, 2, -1, -1};
    const std::array<std::int32_t, 8> expanded_metadata{
        1, 0, 0, -1, 7, 2, 1, -1};
    const auto& metadata = expanded ? expanded_metadata : normal_metadata;
    std::memcpy(
        a.source_metadata, metadata.data(), metadata.size() * sizeof(metadata[0]));
    return 0;
}
extern "C" int deep_ep_ascend_launch_combine(elastic::CombineArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine_epilogue(elastic::CombineArguments, elastic::CoreTiling, void*) { return 0; }

using Buffer = deep_ep::ascend::ElasticBuffer;
using Tensor = torch::Tensor;

struct Inputs {
    Tensor x = torch::empty(
        {1, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    Tensor idx = torch::empty(
        {1, 2}, torch::TensorOptions().dtype(torch::kLong));
    Tensor weights = torch::empty(
        {1, 2}, torch::TensorOptions().dtype(torch::kFloat));

    Inputs() {
        const std::array<std::uint16_t, 8> payload{
            0x1001, 0x1002, 0x1003, 0x1004,
            0x1005, 0x1006, 0x1007, 0x1008};
        std::memcpy(x.data_ptr(), payload.data(), payload.size() * sizeof(payload[0]));
        idx.data_ptr<std::int64_t>()[0] = 0;
        idx.data_ptr<std::int64_t>()[1] = 1;
        weights.data_ptr<float>()[0] = 0.5F;
        weights.data_ptr<float>()[1] = 0.75F;
    }
};

auto uncached_dispatch(
    Buffer& buffer, const Inputs& inputs,
    const std::optional<Tensor>& weights, bool copy_handle = true,
    bool expanded = false) {
    const std::optional<torch::Tensor> none;
    const std::optional<int> no_int;
    const std::optional<std::vector<int>> no_list;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    return buffer.dispatch(
        inputs.x, none, inputs.idx, weights, none, no_int, no_int, no_list,
        none, none, none, none, none, none, none, 4, 2, 4, 1, 0, no_event, no_event,
        false, false, copy_handle, true, expanded, false, false);
}

template <typename Result>
auto cached_dispatch(
    Buffer& buffer, const Inputs& inputs, const Result& handle,
    const Tensor& rank_prefix, bool expanded = false,
    const std::optional<std::vector<int>>& per_expert = std::nullopt) {
    const std::optional<Tensor> none;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    return buffer.dispatch(
        inputs.x, none, inputs.idx, inputs.weights, none,
        std::get<5>(handle), std::get<6>(handle),
        per_expert.has_value() ? *per_expert : std::get<7>(handle),
        rank_prefix, std::get<9>(handle), std::get<10>(handle),
        std::get<12>(handle), std::get<13>(handle), std::get<11>(handle), none,
        4, 2, 4, 1, 0, no_event, no_event,
        false, false, false, false, expanded, false, false);
}

bool has_shape(const Tensor& tensor, std::initializer_list<std::int64_t> shape) {
    return tensor.sizes() == std::vector<std::int64_t>(shape);
}

template <typename Value, std::size_t Size>
bool has_values(const Tensor& tensor, const std::array<Value, Size>& expected) {
    if (tensor.numel() != static_cast<std::int64_t>(Size)) return false;
    const auto* actual = tensor.data_ptr<Value>();
    for (std::size_t index = 0; index < Size; ++index)
        if (actual[index] != expected[index]) return false;
    return true;
}

template <typename Result>
bool has_exact_result(
    const Result& result, bool expect_weights, bool expect_copied_index,
    bool expanded = false) {
    const std::array<std::uint16_t, 16> payload{
        0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 0x0108,
        0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208};
    const std::array<std::uint16_t, 32> expanded_payload{
        0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 0x0108,
        0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const std::array<std::int64_t, 4> indices{0, -1, 0, -1};
    const std::array<float, 4> weights{0.25F, 0.75F, 0.5F, 1.0F};
    const std::array<float, 4> expanded_weights{0.25F, 0.5F, 0.0F, 0.0F};
    const std::array<std::int64_t, 2> copied_index{0, 1};
    const std::array<std::int32_t, 2> rank_prefix{1, 2};
    const std::array<std::int32_t, 1> normal_prefix{4};
    const std::array<std::int32_t, 1> expanded_prefix{2};
    const std::array<std::int32_t, 1> unaligned{2};
    const std::array<std::int32_t, 8> normal_metadata{
        1, 0, -1, -1, 7, 2, -1, -1};
    const std::array<std::int32_t, 8> expanded_metadata{
        1, 0, 0, -1, 7, 2, 1, -1};
    const std::array<std::int32_t, 2> slots{0, 0};
    if (!has_shape(std::get<0>(result), {expanded ? 4 : 2, 8}) ||
        std::get<0>(result).scalar_type() != torch::kBFloat16 ||
        (expanded ? !has_values(std::get<0>(result), expanded_payload) :
                    !has_values(std::get<0>(result), payload)) ||
        std::get<1>(result).has_value() ||
        std::get<2>(result).has_value() == expanded ||
        (!expanded && (!has_shape(*std::get<2>(result), {2, 2}) ||
                       !has_values(*std::get<2>(result), indices))) ||
        std::get<3>(result).has_value() != expect_weights ||
        (expect_weights &&
         ((expanded ? !has_shape(*std::get<3>(result), {4}) :
                      !has_shape(*std::get<3>(result), {2, 2})) ||
          (expanded ? !has_values(*std::get<3>(result), expanded_weights) :
                      !has_values(*std::get<3>(result), weights)))) ||
        std::get<4>(result).has_value() != expect_copied_index ||
        (expect_copied_index &&
         (!has_shape(*std::get<4>(result), {1, 2}) ||
          !has_values(*std::get<4>(result), copied_index))) ||
        std::get<5>(result) != 2 || std::get<6>(result) != 4 ||
        std::get<7>(result) != std::vector<int>{4} ||
        !has_shape(std::get<8>(result), {2}) ||
        !has_values(std::get<8>(result), rank_prefix) ||
        !has_shape(std::get<9>(result), {1}) ||
        (expanded ? !has_values(std::get<9>(result), expanded_prefix) :
                    !has_values(std::get<9>(result), normal_prefix)) ||
        !has_shape(std::get<10>(result), {1}) ||
        !has_values(std::get<10>(result), unaligned) ||
        !has_shape(std::get<11>(result), {2, 4}) ||
        (expanded ? !has_values(std::get<11>(result), expanded_metadata) :
                    !has_values(std::get<11>(result), normal_metadata)) ||
        !has_shape(std::get<12>(result), {1, 2}) ||
        !has_values(std::get<12>(result), slots) ||
        !std::get<13>(result).has_value() ||
        !has_shape(*std::get<13>(result), {
            static_cast<std::int64_t>(sizeof(elastic::DispatchHandleDescriptor))}) ||
        std::get<13>(result)->scalar_type() != torch::kByte ||
        std::get<14>(result).has_value() || std::get<15>(result).has_value())
        return false;

    elastic::DispatchHandleDescriptor descriptor{};
    std::memcpy(
        &descriptor, std::get<13>(result)->data_ptr(), sizeof(descriptor));
    const std::uint64_t expected_family = expanded ?
        0x34c68e658004e440ULL : 0xe20f7c80618f7da3ULL;
    return descriptor.abi_version ==
               elastic::kDispatchHandleDescriptorAbiVersion &&
        descriptor.struct_size == sizeof(elastic::DispatchHandleDescriptor) &&
        descriptor.family == expected_family &&
        descriptor.topology.world_rank == 0 &&
        descriptor.topology.world_size == 2 &&
        descriptor.topology.scale_up_rank == 0 &&
        descriptor.topology.scale_up_size == 2 &&
        descriptor.topology.scale_out_rank == 0 &&
        descriptor.topology.scale_out_size == 1 &&
        descriptor.topology.kind ==
            transport::TransportTopologyKind::kFlatScaleUp &&
        descriptor.topology.epoch == 1 && descriptor.num_tokens == 1 &&
        descriptor.hidden == 8 && descriptor.num_experts == 2 &&
        descriptor.num_topk == 2 && descriptor.expert_alignment == 4 &&
        descriptor.num_max_tokens_per_rank == 4 &&
        descriptor.mode_flags == (expanded ?
            elastic::mode_bit(elastic::CoreMode::kExpanded) : 0);
}

bool exact_and_cached_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    auto result = uncached_dispatch(*buffer, inputs, weights);
    if (!has_exact_result(result, true, true) || trace.launches != 1 ||
        trace.generations != std::vector<std::uint64_t>{1} ||
        trace.kernel_expert_prefixes[0] == std::get<9>(result).data_ptr() ||
        trace.kernel_unaligned_counts[0] == std::get<10>(result).data_ptr())
        return false;

    auto malformed_rank_prefix = std::get<8>(result).clone();
    malformed_rank_prefix.data_ptr<std::int32_t>()[1] = 1;
    try {
        (void)cached_dispatch(*buffer, inputs, result, malformed_rank_prefix);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("cached prefix tail mismatch") ==
                std::string::npos ||
            trace.launches != 1)
            return false;
    }

    auto cached = cached_dispatch(*buffer, inputs, result, std::get<8>(result));
    if (!has_exact_result(cached, true, false) || trace.launches != 2 ||
        trace.generations != std::vector<std::uint64_t>({1, 2}) ||
        !trace.cached_private_contract ||
        trace.kernel_expert_prefixes[1] == std::get<9>(result).data_ptr() ||
        trace.kernel_unaligned_counts[1] == std::get<10>(result).data_ptr())
        return false;

    try {
        (void)cached_dispatch(
            *buffer, inputs, result, std::get<8>(result), false,
            std::vector<int>{2});
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("cached expert counts mismatch") ==
                std::string::npos ||
            trace.launches != 2)
            return false;
    }

    auto wrong_device_rank_prefix = torch::empty(
        {2}, torch::TensorOptions().dtype(torch::kInt).device(1));
    std::memcpy(
        wrong_device_rank_prefix.data_ptr(), std::get<8>(result).data_ptr(),
        2 * sizeof(std::int32_t));
    try {
        (void)cached_dispatch(*buffer, inputs, result, wrong_device_rank_prefix);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("cached rank prefix") ==
                std::string::npos ||
            trace.launches != 2)
            return false;
    }

    const std::optional<Tensor> no_weights;
    auto no_weight_result = uncached_dispatch(*buffer, inputs, no_weights);
    return has_exact_result(no_weight_result, false, true) &&
        trace.launches == 3 &&
        trace.generations == std::vector<std::uint64_t>({1, 2, 3});
}

bool expanded_public_contract_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    auto result = uncached_dispatch(*buffer, inputs, weights, true, true);
    if (!has_exact_result(result, true, true, true) || trace.launches != 1 ||
        trace.kernel_expert_prefixes[0] == std::get<9>(result).data_ptr() ||
        trace.kernel_unaligned_counts[0] == std::get<10>(result).data_ptr())
        return false;
    auto cached = cached_dispatch(
        *buffer, inputs, result, std::get<8>(result), true);
    return has_exact_result(cached, true, false, true) &&
        trace.launches == 2 && trace.cached_private_contract &&
        trace.kernel_expert_prefixes[1] != std::get<9>(result).data_ptr() &&
        trace.kernel_unaligned_counts[1] != std::get<10>(result).data_ptr();
}

bool descriptor_copy_retry_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> no_weights;
    trace.fail_h2d = true;
    try {
        (void)uncached_dispatch(*buffer, inputs, no_weights);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("copy_from_host failed") ==
                std::string::npos ||
            trace.launches != 0)
            return false;
    }
    trace.fail_h2d = false;
    try {
        const auto result = uncached_dispatch(*buffer, inputs, no_weights);
        return has_exact_result(result, false, true) && trace.launches == 1 &&
            trace.generations == std::vector<std::uint64_t>{1};
    } catch (const std::runtime_error&) {
        return false;
    }
}

bool stream_retry_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> no_weights;
    trace.fail_stream = true;
    try {
        (void)uncached_dispatch(*buffer, inputs, no_weights);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("current_stream failed") ==
                std::string::npos ||
            trace.launches != 0)
            return false;
    }
    trace.fail_stream = false;
    try {
        const auto result = uncached_dispatch(*buffer, inputs, no_weights);
        return has_exact_result(result, false, true) && trace.launches == 1 &&
            trace.generations == std::vector<std::uint64_t>{1};
    } catch (const std::runtime_error&) {
        return false;
    }
}

bool launch_poison_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> no_weights;
    trace.fail_launch = true;
    try {
        (void)uncached_dispatch(*buffer, inputs, no_weights);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("backend error 73") ==
                std::string::npos ||
            trace.launches != 1 || trace.generation != 1 ||
            trace.generations != std::vector<std::uint64_t>{1})
            return false;
    }
    trace.fail_launch = false;
    try {
        (void)uncached_dispatch(*buffer, inputs, no_weights);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("cannot continue") !=
                std::string::npos &&
            trace.launches == 1 && trace.generation == 1 &&
            trace.generations == std::vector<std::uint64_t>{1};
    }
}

bool diagnostic_order_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> no_weights;
    trace.bad_diagnostic = true;
    trace.fail_d2h = true;
    const int copies_before_diagnostic = trace.d2h_copies;
    try {
        (void)uncached_dispatch(*buffer, inputs, no_weights);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("device diagnostic reported failure") !=
                std::string::npos &&
            trace.d2h_copies == copies_before_diagnostic + 1;
    }
}

bool empty_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources) return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    inputs.x = torch::empty({0, 8}, inputs.x.options());
    inputs.idx = torch::empty({0, 1}, inputs.idx.options());
    const std::optional<Tensor> no_weights;
    const auto result = uncached_dispatch(*buffer, inputs, no_weights);
    return has_shape(std::get<0>(result), {0, 8}) &&
        std::get<5>(result) == 0 && std::get<6>(result) == 0;
}

bool testing_topology_mismatch_probe() {
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    try {
        (void)Buffer::make_testing_buffer(
            0, std::move(runtime_resources), 2 * 1024 * 1024, 1,
            true, 7, 3);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(
                   "testing topology must match runtime resources") !=
            std::string::npos;
    }
    return false;
}

int main() {
    int failures = 0;
    const auto check = [&failures](bool passed, const char* name) {
        if (!passed) {
            ++failures;
            std::cerr << "failed: " << name << '\n';
        }
    };
    check(exact_and_cached_probe(), "exact outputs and cached preflight retry");
    check(expanded_public_contract_probe(), "expanded public expert prefix");
    check(descriptor_copy_retry_probe(), "descriptor copy retry");
    check(stream_retry_probe(), "stream acquisition retry");
    check(launch_poison_probe(), "launch failure poisoning");
    check(diagnostic_order_probe(), "diagnostic ordering");
    check(empty_probe(), "empty dispatch");
    check(testing_topology_mismatch_probe(), "testing topology mismatch");
    return failures == 0 ? 0 : 1;
}

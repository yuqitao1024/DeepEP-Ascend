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
    bool corrupt_fresh_route = false;
    bool masked_producer_failure = false;
    int masked_producer_world_rank = 0;
    bool pending_diagnostic = false;
    std::uint64_t preserved_scratch_status = 0;
    int world_size = 2;
    int event_creates = 0;
    int event_destroys = 0;
    int frees = 0;
    int fail_event_create_on = 0;
    bool fail_completion_record = false;
    std::vector<std::string> order;
} trace;

int compute_stream_token = 0;
int comm_stream_token = 0;

int alloc(void*, std::uint64_t n, void** p) { *p = std::malloc(n); return *p ? 0 : 1; }
int zero(void*, void* p, std::uint64_t n) { std::memset(p, 0, n); return 0; }
int free_(void*, void* p) { ++trace.frees; std::free(p); return 0; }
int current_device(void*, int* device) { *device = trace.device; return 0; }
int stream(void*, runtime::StreamIdentity* value) {
    trace.order.emplace_back("capture compute dependency");
    *value = {trace.fail_stream ? nullptr : &compute_stream_token,
              7, trace.device, 20};
    return 0;
}
int pool_stream(
    void*, int device, bool high_priority, runtime::StreamIdentity* value) {
    if (!high_priority)
        return 93;
    *value = {&comm_stream_token, 11, device, 20};
    return 0;
}
int create_event(void*, void** event) {
    ++trace.event_creates;
    if (trace.event_creates == trace.fail_event_create_on)
        return 95;
    *event = new int(trace.event_creates);
    trace.order.emplace_back("create event");
    return 0;
}
int record_event(void*, void*, void* stream_value) {
    trace.order.emplace_back(stream_value == &comm_stream_token ?
        "record completion event" : "record compute dependency");
    if (stream_value == &comm_stream_token && trace.fail_completion_record)
        return 96;
    return 0;
}
int query_event(void*, void*, bool* completed) {
    *completed = true;
    trace.order.emplace_back("finish completion event");
    return 0;
}
int wait_event(void*, void* stream_value, void*) {
    if (stream_value != &comm_stream_token)
        return 94;
    trace.order.emplace_back("comm waits dependency/previous event");
    return 0;
}
int synchronize_event(void*, void*, std::uint64_t) { return 0; }
int destroy_event(void*, void* event) {
    ++trace.event_destroys;
    delete static_cast<int*>(event);
    return 0;
}
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
    if (trace.pending_diagnostic &&
        n == sizeof(transport::DeviceTransportDiagnostic)) {
        trace.pending_diagnostic = false;
        auto* diagnostic = static_cast<transport::DeviceTransportDiagnostic*>(d);
        *diagnostic = {};
        diagnostic->abi_version = transport::kTransportCommandAbiVersion;
        diagnostic->generation = trace.generation;
        if (trace.masked_producer_failure) {
            const auto decoded = elastic::decode_dispatch_protocol_scratch(
                trace.preserved_scratch_status);
            const bool valid_producer_failure =
                decoded.valid && decoded.world_rank >= 0 &&
                decoded.world_rank < trace.world_size;
            const int reported_world_rank =
                valid_producer_failure ? decoded.world_rank : 0;
            const auto reported_error = valid_producer_failure ?
                decoded.error : elastic::DispatchProtocolError::kInvalidControl;
            const auto restored = elastic::make_dispatch_protocol_failure(
                reported_world_rank,
                elastic::DispatchProtocolStage::kProducer,
                trace.generation, reported_error);
            diagnostic->error =
                transport::DeviceTransportError::kInvalidProtocol;
            diagnostic->command_index = 0;
            diagnostic->opcode = transport::TransportCommandOpcode::kNone;
            diagnostic->peer =
                static_cast<std::uint32_t>(reported_world_rank);
            diagnostic->world_peer = reported_world_rank;
            diagnostic->team = transport::TransportTeam::kWorld;
            diagnostic->channel = 0;
            diagnostic->backend_status = restored.backend_status;
        } else if (trace.bad_diagnostic) {
            diagnostic->error =
                transport::DeviceTransportError::kCompletionFailure;
            diagnostic->world_peer = 1;
            diagnostic->team = transport::TransportTeam::kScaleOut;
            diagnostic->backend_status = 37;
            diagnostic->reserved = 0x1234;
        }
        return 0;
    }
    if (trace.fail_d2h) return 91;
    std::memcpy(d, s, n); return 0;
}
int rank_(void*, std::int64_t, std::uint32_t* v) { *v = 0; return 0; }
int size_(void*, std::int64_t, std::uint32_t* v) {
    *v = static_cast<std::uint32_t>(trace.world_size);
    return 0;
}
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

std::unique_ptr<runtime::CannRuntimeResources> resources(
    int world_size = 2, bool hybrid = false) {
    auto result = std::make_unique<runtime::CannRuntimeResources>();
    runtime::CannRuntimeApi r{nullptr,alloc,zero,free_,sync,sync_device,h2d,d2h};
    runtime::StreamEventApi s{
        nullptr, current_device, stream, pool_stream, create_event, record_event,
        query_event, wait_event, synchronize_event, destroy_event};
    transport::CannHostApi h{nullptr,rank_,size_,team,window,channels,ha,hz,hd,dh,hf,noop2,noop1};
    transport::TransportConfig c{}; c.rank=0; c.world_size=world_size; c.communicator_handle=1;
    c.device_buffer_bytes=2*1024*1024; c.requested_channels=1;
    if (hybrid) {
        c.scale_up_size = 2;
        c.topology_kind =
            transport::TransportTopologyKind::kLogicalSimulation;
    }
    if (!result->initialize(c, 4096, r, h, s).ok()) return {};
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
    trace.pending_diagnostic = true;
    trace.order.emplace_back("activate lease and launch cached dispatch on comm");
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        t.transport_context.local_window_base +
        t.symmetric_window_layout.control_offset);
    control->dispatch_generation = a.generation;
    const bool cached = elastic::has_mode(t.mode_flags, elastic::CoreMode::kCached);
    const bool expanded = elastic::has_mode(t.mode_flags, elastic::CoreMode::kExpanded);
    const bool hybrid = elastic::has_mode(t.mode_flags, elastic::CoreMode::kHybrid);
    if (hybrid) {
        if (trace.masked_producer_failure) {
            trace.preserved_scratch_status =
                elastic::make_dispatch_protocol_failure(
                    trace.masked_producer_world_rank,
                    elastic::DispatchProtocolStage::kProducer, a.generation,
                    elastic::DispatchProtocolError::kInvalidTopk)
                    .scratch_status;
            return 0;
        }
        if (a.route_records == nullptr || a.route_record_capacity < 1)
            return 74;
        const std::array<std::uint16_t, 8> payload{
            0x0101, 0x0102, 0x0103, 0x0104,
            0x0105, 0x0106, 0x0107, 0x0108};
        std::memcpy(a.recv_x, payload.data(), payload.size() * sizeof(payload[0]));
        a.recv_topk_indices[0] = 0;
        a.recv_topk_indices[1] = -1;
        if (a.recv_topk_weights != nullptr) {
            a.recv_topk_weights[0] = 0.25F;
            a.recv_topk_weights[1] = 0.75F;
        }
        for (int rank = 0; rank < t.topology.world_size; ++rank)
            a.prefix_per_rank[rank] = 1;
        const std::array<std::int32_t, 9> private_expert_prefix{
            0, 4, 4, 4, 4, 4, 4, 4, 4};
        const std::array<std::int32_t, 8> private_unaligned{
            1, 0, 0, 0, 0, 0, 0, 0};
        std::memcpy(a.prefix_per_expert, private_expert_prefix.data(),
                    sizeof(private_expert_prefix));
        std::memcpy(a.unaligned_per_expert, private_unaligned.data(),
                    sizeof(private_unaligned));
        a.destination_slots[0] = 0;
        a.destination_slots[1] = 0;
        const std::array<std::int32_t, 4> metadata{0, 0, -1, -1};
        std::memcpy(a.source_metadata, metadata.data(), sizeof(metadata));
        a.route_records[0] = {
            0, 0, 0, 0,
            trace.corrupt_fresh_route ? 4U : 0U,
            elastic::kInvalidHybridRouteSlot,
            elastic::kInvalidHybridRouteSlot,
            a.generation, t.topology.epoch,
            elastic::kHybridRouteCompleteStageFlags, 0};
        return 0;
    }
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

auto uncached_hybrid_dispatch(
    Buffer& buffer, const Inputs& inputs,
    const std::optional<Tensor>& weights) {
    const std::optional<torch::Tensor> none;
    const std::optional<int> no_int;
    const std::optional<std::vector<int>> no_list;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    return buffer.dispatch(
        inputs.x, none, inputs.idx, weights, none, no_int, no_int, no_list,
        none, none, none, none, none, none, none, 4, 8, 4, 1, 0,
        no_event, no_event, false, false, true, true, false, false, false);
}

template <typename Result>
auto cached_hybrid_dispatch(
    Buffer& buffer, const Inputs& inputs, const Result& handle) {
    const std::optional<Tensor> none;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    return buffer.dispatch(
        inputs.x, none, inputs.idx, inputs.weights, none,
        std::get<5>(handle), std::get<6>(handle), std::get<7>(handle),
        std::get<8>(handle), std::get<9>(handle), std::get<10>(handle),
        std::get<12>(handle), std::get<13>(handle), std::get<11>(handle), none,
        4, 8, 4, 1, 0, no_event, no_event, false, false, false, false,
        false, false, false);
}

template <typename Result>
auto cached_dispatch(
    Buffer& buffer, const Inputs& inputs, const Result& handle,
    const Tensor& rank_prefix, bool expanded = false,
    const std::optional<std::vector<int>>& per_expert = std::nullopt,
    bool async = false, bool allocate_on_comm_stream = false,
    const std::optional<deep_ep::ascend::EventHandle>& previous_event =
        std::nullopt,
    const std::optional<deep_ep::ascend::EventHandle>&
        previous_event_before_epilogue = std::nullopt) {
    const std::optional<Tensor> none;
    return buffer.dispatch(
        inputs.x, none, inputs.idx, inputs.weights, none,
        std::get<5>(handle), std::get<6>(handle),
        per_expert.has_value() ? *per_expert : std::get<7>(handle),
        rank_prefix, std::get<9>(handle), std::get<10>(handle),
        std::get<12>(handle), std::get<13>(handle), std::get<11>(handle), none,
        4, 2, 4, 1, 0, previous_event, previous_event_before_epilogue,
        async, allocate_on_comm_stream, false, false, expanded, false, false);
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
    const std::uint64_t expected_family = elastic::attest_dispatch_handle_family(
        7, descriptor.topology, descriptor.generation, descriptor.num_tokens,
        descriptor.hidden, descriptor.num_experts, descriptor.num_topk,
        descriptor.expert_alignment, descriptor.num_max_tokens_per_rank,
        descriptor.mode_flags);
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
        descriptor.topology.epoch == 1 && descriptor.generation ==
            trace.generation && descriptor.num_tokens == 1 &&
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
        std::get<13>(cached)->data_ptr() != std::get<13>(result)->data_ptr() ||
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

bool cached_hybrid_route_validation_probe() {
    trace = {};
    trace.world_size = 4;
    auto runtime_resources = resources(4, true);
    if (!runtime_resources) return false;
    if (transport::has_capability(
            runtime_resources->transport()->capabilities(),
            transport::TransportCapability::kScaleOutTeam))
        return false;
    std::unique_ptr<Buffer> buffer;
    try {
        buffer = Buffer::make_testing_buffer(
            0, std::move(runtime_resources), 2 * 1024 * 1024, 1,
            true, 7, 4, 0, true);
    } catch (const std::runtime_error&) {
        return false;
    }
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    decltype(uncached_hybrid_dispatch(*buffer, inputs, weights)) first;
    try {
        first = uncached_hybrid_dispatch(*buffer, inputs, weights);
    } catch (const std::exception& error) {
        std::cerr << "initial hybrid dispatch: " << error.what() << '\n';
        return false;
    }
    if (trace.launches != 1 || !std::get<13>(first).has_value())
        return false;

    auto corrupted = first;
    std::get<13>(corrupted) = std::get<13>(first)->clone();
    auto* record = reinterpret_cast<elastic::HybridRouteRecord*>(
        static_cast<std::uint8_t*>(std::get<13>(corrupted)->data_ptr()) +
        sizeof(elastic::DispatchHandleDescriptor));
    record->origin_source_row = 1;
    try {
        (void)cached_hybrid_dispatch(*buffer, inputs, corrupted);
        std::cerr << "corrupted cached hybrid dispatch was accepted\n";
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("hybrid route record") ==
                std::string::npos ||
            trace.launches != 1) {
            std::cerr << "corrupted cached hybrid dispatch: "
                      << error.what() << '\n';
            return false;
        }
    }

    corrupted = first;
    std::get<13>(corrupted) = std::get<13>(first)->clone();
    record = reinterpret_cast<elastic::HybridRouteRecord*>(
        static_cast<std::uint8_t*>(std::get<13>(corrupted)->data_ptr()) +
        sizeof(elastic::DispatchHandleDescriptor));
    record->destination_local_expert = 1;
    try {
        (void)cached_hybrid_dispatch(*buffer, inputs, corrupted);
        std::cerr << "corrupted cached hybrid expert was accepted\n";
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("hybrid route record") ==
                std::string::npos ||
            trace.launches != 1) {
            std::cerr << "corrupted cached hybrid expert: "
                      << error.what() << '\n';
            return false;
        }
    }

    try {
        (void)cached_hybrid_dispatch(*buffer, inputs, first);
    } catch (const std::runtime_error& error) {
        std::cerr << "valid cached hybrid dispatch: " << error.what() << '\n';
        return false;
    }
    return trace.launches == 2;
}

bool failed_fresh_hybrid_route_poisons_probe() {
    trace = {};
    trace.world_size = 4;
    auto runtime_resources = resources(4, true);
    if (!runtime_resources) return false;
    std::unique_ptr<Buffer> buffer;
    try {
        buffer = Buffer::make_testing_buffer(
            0, std::move(runtime_resources), 2 * 1024 * 1024, 1,
            true, 7, 4, 0, true);
    } catch (const std::runtime_error&) {
        return false;
    }
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    trace.corrupt_fresh_route = true;
    try {
        (void)uncached_hybrid_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("hybrid route record") ==
                std::string::npos ||
            trace.launches != 1)
            return false;
    }
    trace.corrupt_fresh_route = false;
    try {
        (void)uncached_hybrid_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("cannot continue") !=
                std::string::npos && trace.launches == 1;
    }
}

bool post_activation_copy_failure_poisons_probe() {
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
            trace.launches != 1 ||
            trace.generations != std::vector<std::uint64_t>{1})
            return false;
    }
    trace.fail_h2d = false;
    try {
        (void)uncached_dispatch(*buffer, inputs, no_weights);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("cannot continue") !=
                std::string::npos && trace.launches == 1 &&
            trace.generations == std::vector<std::uint64_t>{1};
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
        const std::string message(error.what());
        return message.find("device diagnostic reported failure") !=
                std::string::npos &&
            message.find("world_peer=1") != std::string::npos &&
            message.find("team=2") != std::string::npos &&
            message.find("backend_status=37") != std::string::npos &&
            message.find("reserved=4660") != std::string::npos &&
            trace.d2h_copies == copies_before_diagnostic + 1;
    }
}

bool masked_hybrid_producer_failure_probe(
    int encoded_world_rank, std::uint32_t expected_backend_status) {
    trace = {};
    trace.world_size = 4;
    trace.masked_producer_failure = true;
    trace.masked_producer_world_rank = encoded_world_rank;
    auto runtime_resources = resources(4, true);
    if (!runtime_resources) return false;
    std::unique_ptr<Buffer> buffer;
    try {
        buffer = Buffer::make_testing_buffer(
            0, std::move(runtime_resources), 2 * 1024 * 1024, 1,
            true, 7, 4, 0, true);
    } catch (const std::runtime_error&) {
        return false;
    }
    Inputs inputs;
    inputs.idx.data_ptr<std::int64_t>()[0] = 8;
    const std::optional<Tensor> weights = inputs.weights;
    try {
        (void)uncached_hybrid_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        const std::string message(error.what());
        const std::string expected_backend =
            std::to_string(expected_backend_status);
        return message.find("dispatch failed on rank 0 with backend error " +
                            expected_backend) !=
                std::string::npos &&
            message.find(
                "device diagnostic reported failure error=invalid_protocol") !=
                std::string::npos &&
            message.find("command_index=0") != std::string::npos &&
            message.find("opcode=0") != std::string::npos &&
            message.find("peer=0") != std::string::npos &&
            message.find("world_peer=0") != std::string::npos &&
            message.find("team=0") != std::string::npos &&
            message.find("channel=0") != std::string::npos &&
            message.find("backend_status=" + expected_backend) !=
                std::string::npos &&
            message.find("generation=1") != std::string::npos &&
            message.find("dispatch returned invalid expert counts") ==
                std::string::npos &&
            trace.launches == 1 && trace.preserved_scratch_status ==
                ((static_cast<std::uint64_t>(encoded_world_rank + 1) << 32U) |
                 static_cast<std::uint32_t>(
                     elastic::DispatchProtocolError::kInvalidTopk));
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

bool cached_async_order_and_commit_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    auto initial = uncached_dispatch(*buffer, inputs, weights);
    auto* descriptor = reinterpret_cast<elastic::DispatchHandleDescriptor*>(
        std::get<13>(initial)->data_ptr());
    if (descriptor->generation != 1)
        return false;

    trace.order.clear();
    decltype(cached_dispatch(
        *buffer, inputs, initial, std::get<8>(initial))) pending;
    try {
        pending = cached_dispatch(
            *buffer, inputs, initial, std::get<8>(initial), false,
            std::nullopt, true, false);
    } catch (const std::exception&) {
        return false;
    }
    const std::vector<std::string> expected_order{
        "capture compute dependency",
        "create event",
        "record compute dependency",
        "comm waits dependency/previous event",
        "create event",
        "activate lease and launch cached dispatch on comm",
        "record completion event",
    };
    if (trace.order != expected_order || !std::get<15>(pending).has_value() ||
        descriptor->generation != 1 || trace.generations.back() != 2)
        return false;

    try {
        (void)cached_dispatch(
            *buffer, inputs, initial, std::get<8>(initial), false,
            std::nullopt, true, false);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("busy") == std::string::npos ||
            trace.launches != 2)
            return false;
    }

    std::get<15>(pending)->current_stream_wait();
    std::get<15>(pending)->current_stream_wait();
    return descriptor->generation == 2 && trace.event_destroys == 1 &&
        buffer->testing_operation_generation() == 2;
}

bool completion_create_failure_precedes_cached_launch_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    auto initial = uncached_dispatch(*buffer, inputs, weights);
    const int launches_before = trace.launches;
    const auto generation_before = buffer->testing_operation_generation();
    trace.fail_event_create_on = trace.event_creates + 2;
    try {
        (void)cached_dispatch(
            *buffer, inputs, initial, std::get<8>(initial), false,
            std::nullopt, true, false);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("backend error 95") !=
                std::string::npos &&
            trace.launches == launches_before &&
            buffer->testing_operation_generation() == generation_before;
    }
}

bool completion_record_failure_retains_launched_dispatch_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    auto initial = uncached_dispatch(*buffer, inputs, weights);
    const int launches_before = trace.launches;
    trace.fail_completion_record = true;
    try {
        (void)cached_dispatch(
            *buffer, inputs, initial, std::get<8>(initial), false,
            std::nullopt, true, false);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("backend error 96") ==
                std::string::npos ||
            trace.launches != launches_before + 1)
            return false;
    }

    bool first_destroy_failed = false;
    bool second_destroy_failed = false;
    try {
        buffer->destroy();
    } catch (const std::runtime_error&) {
        first_destroy_failed = true;
    }
    try {
        buffer->destroy();
    } catch (const std::runtime_error&) {
        second_destroy_failed = true;
    }
    if (!first_destroy_failed || !second_destroy_failed || trace.frees != 0)
        return false;
    buffer.reset();
    return trace.frees == 0;
}

bool completion_mismatch_fault_does_not_publish() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    auto initial = uncached_dispatch(*buffer, inputs, weights);
    auto* descriptor = reinterpret_cast<elastic::DispatchHandleDescriptor*>(
        std::get<13>(initial)->data_ptr());

    auto unaffected = cached_dispatch(
        *buffer, inputs, initial, std::get<8>(initial), false,
        std::nullopt, true, false);
    setenv("DEEP_EP_ASCEND_TEST_COMPLETION_FAULT",
           "completion_mismatch_typo", 1);
    try {
        std::get<15>(unaffected)->current_stream_wait();
    } catch (const std::runtime_error&) {
        unsetenv("DEEP_EP_ASCEND_TEST_COMPLETION_FAULT");
        return false;
    }
    unsetenv("DEEP_EP_ASCEND_TEST_COMPLETION_FAULT");
    if (descriptor->generation != 2)
        return false;

    auto injected = cached_dispatch(
        *buffer, inputs, initial, std::get<8>(initial), false,
        std::nullopt, true, false);
    setenv("DEEP_EP_ASCEND_TEST_COMPLETION_FAULT",
           "completion_mismatch", 1);
    std::string failure;
    try {
        std::get<15>(injected)->current_stream_wait();
    } catch (const std::runtime_error& error) {
        failure = error.what();
    }
    unsetenv("DEEP_EP_ASCEND_TEST_COMPLETION_FAULT");
    if (failure.find("device completion generation mismatch") ==
            std::string::npos || descriptor->generation != 2)
        return false;
    std::string destroy_failure;
    try {
        buffer->destroy();
    } catch (const std::runtime_error& error) {
        destroy_failure = error.what();
    }
    return destroy_failure.find("device completion generation mismatch") !=
            std::string::npos && trace.event_destroys == 3 &&
        buffer->is_destroyed();
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
    check(cached_hybrid_route_validation_probe(),
          "cached hybrid route validation before activation");
    check(failed_fresh_hybrid_route_poisons_probe(),
          "failed fresh hybrid route poisoning");
    check(post_activation_copy_failure_poisons_probe(),
          "post-activation copy failure poisoning");
    check(stream_retry_probe(), "stream acquisition retry");
    check(launch_poison_probe(), "launch failure poisoning");
    check(diagnostic_order_probe(), "diagnostic ordering");
    check(masked_hybrid_producer_failure_probe(0, 65537),
          "masked hybrid producer failure diagnostic");
    check(masked_hybrid_producer_failure_probe(4, 65542),
          "out-of-world hybrid producer failure diagnostic");
    check(empty_probe(), "empty dispatch");
    check(testing_topology_mismatch_probe(), "testing topology mismatch");
    check(cached_async_order_and_commit_probe(),
          "cached async order and deferred descriptor commit");
    check(completion_create_failure_precedes_cached_launch_probe(),
          "completion create failure precedes cached launch");
    check(completion_record_failure_retains_launched_dispatch_probe(),
          "completion record failure retains launched dispatch");
    check(completion_mismatch_fault_does_not_publish(),
          "completion mismatch fault does not publish");
    return failures == 0 ? 0 : 1;
}

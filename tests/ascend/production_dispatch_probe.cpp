#include <algorithm>
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

struct SfObservation {
    bool epilogue = false;
    bool cached = false;
    bool cpu_sync = false;
    void* output = nullptr;
    std::uint64_t token_stride = 0;
    std::uint64_t pack_stride = 0;
    std::uint64_t output_tokens = 0;
};

struct EventRecordObservation {
    int event_id = 0;
    void* stream = nullptr;
    std::uint64_t sequence = 0;
};

struct TensorStreamObservation {
    const void* storage = nullptr;
    void* stream = nullptr;
    std::uint64_t sequence = 0;
};

struct Trace {
    int launches = 0;
    int epilogue_launches = 0;
    int stream_syncs = 0;
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
    bool fail_epilogue_launch = false;
    bool fail_stream = false;
    bool fail_comm_records = false;
    bool event_ready = true;
    bool corrupt_fresh_route = false;
    bool masked_producer_failure = false;
    int masked_producer_world_rank = 0;
    int pending_diagnostic_reads = 0;
    std::uint64_t preserved_scratch_status = 0;
    int world_size = 2;
    int event_creates = 0;
    int event_records = 0;
    int event_destroys = 0;
    int frees = 0;
    int fail_event_create_on = 0;
    int fail_event_record_on = 0;
    bool count_stage_outputs_null = true;
    bool epilogue_outputs_ready = true;
    bool zero_epilogue_outputs_null = true;
    std::vector<std::string> order;
    std::vector<elastic::CoreModeFlags> modes;
    std::vector<SfObservation> sf_observations;
    std::vector<EventRecordObservation> event_record_observations;
    std::vector<TensorStreamObservation> tensor_stream_observations;
    std::uint64_t sequence = 0;
    std::uint64_t count_launch_sequence = 0;
    std::uint64_t cached_launch_sequence = 0;
    std::uint64_t epilogue_launch_sequence = 0;
} trace;

int compute_stream_token = 0;
int comm_stream_token = 0;

void record_tensor_stream(const torch::Tensor& tensor, void* stream_value) {
    trace.tensor_stream_observations.push_back({
        tensor.storage_identity(), stream_value, ++trace.sequence});
}

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
int record_event(void*, void* event, void* stream_value) {
    ++trace.event_records;
    trace.event_record_observations.push_back({
        *static_cast<int*>(event), stream_value, ++trace.sequence});
    trace.order.emplace_back(stream_value == &comm_stream_token ?
        "record completion event" : "record compute dependency");
    if ((stream_value == &comm_stream_token && trace.fail_comm_records) ||
        trace.event_records == trace.fail_event_record_on)
        return 96;
    return 0;
}
int query_event(void*, void*, bool* completed) {
    *completed = trace.event_ready;
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
int sync(void*, void*) {
    ++trace.stream_syncs;
    trace.order.emplace_back("synchronize count stage");
    return 0;
}
int sync_device(void*) { return 0; }
int h2d(void*, void* d, const void* s, std::uint64_t n) {
    ++trace.h2d_copies;
    if (trace.fail_h2d) return 92;
    std::memcpy(d, s, n);
    return 0;
}
int d2h(void*, void* d, const void* s, std::uint64_t n) {
    ++trace.d2h_copies;
    if (trace.pending_diagnostic_reads > 0 &&
        n == sizeof(transport::DeviceTransportDiagnostic)) {
        --trace.pending_diagnostic_reads;
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
    trace.world_size = world_size;
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
    trace.modes.push_back(t.mode_flags);
    trace.generation = a.generation;
    trace.generations.push_back(a.generation);
    trace.kernel_expert_prefixes.push_back(a.prefix_per_expert);
    trace.kernel_unaligned_counts.push_back(a.unaligned_per_expert);
    if (trace.fail_launch) return 73;
    trace.pending_diagnostic_reads = elastic::has_mode(
        t.mode_flags, elastic::CoreMode::kCpuSync) ? 2 : 1;
    trace.order.emplace_back(
        elastic::has_mode(t.mode_flags, elastic::CoreMode::kCpuSync) ?
            "launch uncached count stage" :
            "activate lease and launch cached dispatch on comm");
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        t.transport_context.local_window_base +
        t.symmetric_window_layout.control_offset);
    const bool cpu_sync = elastic::has_mode(
        t.mode_flags, elastic::CoreMode::kCpuSync);
    if (!cpu_sync)
        control->dispatch_generation = a.generation;
    const bool cached = elastic::has_mode(t.mode_flags, elastic::CoreMode::kCached);
    const bool expanded = elastic::has_mode(t.mode_flags, elastic::CoreMode::kExpanded);
    const bool hybrid = elastic::has_mode(t.mode_flags, elastic::CoreMode::kHybrid);
    const bool fp8 = t.element_kind == elastic::ElementKind::kFloat8E4M3;
    if (cached)
        trace.cached_launch_sequence = ++trace.sequence;
    else if (cpu_sync)
        trace.count_launch_sequence = ++trace.sequence;
    if (fp8) {
        trace.sf_observations.push_back({
            false, cached, cpu_sync, a.recv_scale_factors,
            a.recv_scale_factor_token_stride,
            a.recv_scale_factor_pack_stride, a.num_output_tokens});
    }
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
        for (int rank = 0; rank < t.topology.world_size; ++rank)
            a.prefix_per_rank[rank] = 0;
        std::memset(a.prefix_per_expert, 0,
                    private_expert_prefix.size() * sizeof(std::int32_t));
        std::memset(a.unaligned_per_expert, 0,
                    private_unaligned.size() * sizeof(std::int32_t));
        if (cpu_sync) {
            trace.count_stage_outputs_null =
                a.recv_x == nullptr && a.recv_topk_indices == nullptr &&
                a.recv_topk_weights == nullptr && a.source_metadata == nullptr;
        }
        return 0;
    }
    if (cached &&
        (std::memcmp(a.prefix_per_expert, private_expert_prefix.data(),
                     sizeof(private_expert_prefix)) != 0 ||
         std::memcmp(a.unaligned_per_expert, private_unaligned.data(),
                     sizeof(private_unaligned)) != 0))
        trace.cached_private_contract = false;
    if (cpu_sync) {
        for (int rank = 0; rank < t.topology.world_size; ++rank)
            a.prefix_per_rank[rank] = static_cast<std::int32_t>(
                (rank + 1) * 2 / t.topology.world_size);
        std::memcpy(a.prefix_per_expert, private_expert_prefix.data(),
                    sizeof(private_expert_prefix));
        std::memcpy(a.unaligned_per_expert, private_unaligned.data(),
                    sizeof(private_unaligned));
        trace.count_stage_outputs_null =
            a.recv_x == nullptr && a.recv_topk_indices == nullptr &&
            a.recv_topk_weights == nullptr && a.source_metadata == nullptr;
        return 0;
    }
    const std::array<std::uint16_t, 16> payload{
        0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 0x0108,
        0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208};
    if (fp8) {
        std::memset(a.recv_x, 0,
                    static_cast<std::size_t>(a.num_output_tokens * t.hidden));
    } else {
        std::memset(a.recv_x, 0,
                    (expanded ? 4 : 2) * 8 * sizeof(std::uint16_t));
        std::memcpy(a.recv_x, payload.data(), payload.size() * sizeof(payload[0]));
    }
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
    for (int rank = 0; rank < t.topology.world_size; ++rank)
        a.prefix_per_rank[rank] = static_cast<std::int32_t>(
            (rank + 1) * 2 / t.topology.world_size);
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
    if (fp8 && a.recv_scale_factors != nullptr) {
        auto* output = static_cast<std::uint8_t*>(a.recv_scale_factors);
        for (std::uint64_t token = 0; token < a.num_output_tokens; ++token) {
            for (std::uint64_t pack = 0; pack < t.num_scale_factor_packs; ++pack) {
                const std::uint32_t value =
                    static_cast<std::uint32_t>(100 + token * 10 + pack);
                std::memcpy(
                    output + (token * a.recv_scale_factor_token_stride +
                              pack * a.recv_scale_factor_pack_stride) *
                                 sizeof(value),
                    &value, sizeof(value));
            }
        }
    }
    return 0;
}
extern "C" int deep_ep_ascend_launch_dispatch_pipeline(
    elastic::DispatchArguments arguments, elastic::CoreTiling tiling,
    void* producer_stream, void*) {
    return deep_ep_ascend_launch_dispatch(arguments, tiling, producer_stream);
}
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    elastic::DispatchArguments a, elastic::CoreTiling t, void*) {
    ++trace.epilogue_launches;
    trace.epilogue_launch_sequence = ++trace.sequence;
    trace.order.emplace_back("launch uncached copy epilogue");
    if (trace.fail_epilogue_launch)
        return 79;
    const bool expanded = elastic::has_mode(
        t.mode_flags, elastic::CoreMode::kExpanded);
    const bool fp8 = t.element_kind == elastic::ElementKind::kFloat8E4M3;
    if (fp8) {
        trace.sf_observations.push_back({
            true, false, true, a.recv_scale_factors,
            a.recv_scale_factor_token_stride,
            a.recv_scale_factor_pack_stride, a.num_output_tokens});
    }
    if (t.num_tokens == 0) {
        trace.zero_epilogue_outputs_null =
            a.recv_x == nullptr && a.recv_topk_indices == nullptr &&
            a.recv_topk_weights == nullptr && a.source_metadata == nullptr;
    } else {
        trace.epilogue_outputs_ready =
            a.recv_x != nullptr && a.recv_topk_indices != nullptr &&
            a.source_metadata != nullptr;
        const std::array<std::uint16_t, 16> payload{
            0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 0x0108,
            0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208};
        if (fp8) {
            std::memset(a.recv_x, 0,
                        static_cast<std::size_t>(a.num_output_tokens * t.hidden));
        } else {
            std::memset(a.recv_x, 0,
                        static_cast<std::size_t>(expanded ? 4 : 2) * 8 *
                            sizeof(std::uint16_t));
            std::memcpy(
                a.recv_x, payload.data(), payload.size() * sizeof(payload[0]));
        }
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
        const std::array<std::int32_t, 8> normal_metadata{
            1, 0, -1, -1, 7, 2, -1, -1};
        const std::array<std::int32_t, 8> expanded_metadata{
            1, 0, 0, -1, 7, 2, 1, -1};
        const auto& metadata = expanded ? expanded_metadata : normal_metadata;
        std::memcpy(a.source_metadata, metadata.data(),
                    metadata.size() * sizeof(metadata[0]));
        if (fp8 && a.recv_scale_factors != nullptr) {
            auto* output = static_cast<std::uint8_t*>(a.recv_scale_factors);
            for (std::uint64_t token = 0; token < a.num_output_tokens; ++token) {
                for (std::uint64_t pack = 0;
                     pack < t.num_scale_factor_packs; ++pack) {
                    const std::uint32_t value =
                        static_cast<std::uint32_t>(100 + token * 10 + pack);
                    std::memcpy(
                        output +
                            (token * a.recv_scale_factor_token_stride +
                             pack * a.recv_scale_factor_pack_stride) *
                                sizeof(value),
                        &value, sizeof(value));
                }
            }
        }
    }
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        t.transport_context.local_window_base +
        t.symmetric_window_layout.control_offset);
    control->dispatch_generation = a.generation;
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
    std::optional<Tensor> sf;

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
    bool expanded = false, bool async = false,
    bool allocate_on_comm_stream = false,
    const std::optional<deep_ep::ascend::EventHandle>& previous_event =
        std::nullopt,
    bool column_major_sf = false) {
    const std::optional<torch::Tensor> none;
    const std::optional<int> no_int;
    const std::optional<std::vector<int>> no_list;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    return buffer.dispatch(
        inputs.x, inputs.sf, inputs.idx, weights, none, no_int, no_int, no_list,
        none, none, none, none, none, none, none, 4, 2, 4, 1, 0,
        previous_event, no_event, async, allocate_on_comm_stream, copy_handle,
        true, expanded, false, column_major_sf);
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
        previous_event_before_epilogue = std::nullopt,
    bool column_major_sf = false) {
    const std::optional<Tensor> none;
    return buffer.dispatch(
        inputs.x, inputs.sf, inputs.idx, inputs.weights, none,
        std::get<5>(handle), std::get<6>(handle),
        per_expert.has_value() ? *per_expert : std::get<7>(handle),
        rank_prefix, std::get<9>(handle), std::get<10>(handle),
        std::get<12>(handle), std::get<13>(handle), std::get<11>(handle), none,
        4, 2, 4, 1, 0, previous_event, previous_event_before_epilogue,
        async, allocate_on_comm_stream, false, false, expanded, false,
        column_major_sf);
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
    bool expanded = false, bool expect_event = false) {
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
    const auto exact_storage = [](const Tensor& tensor, std::size_t bytes) {
        return tensor.storage_nbytes() ==
            static_cast<std::size_t>(tensor.numel()) * bytes;
    };
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
        std::get<14>(result).has_value() ||
        std::get<15>(result).has_value() != expect_event ||
        (expect_copied_index &&
         (!exact_storage(std::get<0>(result), sizeof(std::uint16_t)) ||
          (!expanded && !exact_storage(*std::get<2>(result), sizeof(std::int64_t))) ||
          (expect_weights && !exact_storage(*std::get<3>(result), sizeof(float))) ||
          !exact_storage(std::get<11>(result), sizeof(std::int32_t)))))
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
        !elastic::has_mode(
            trace.modes[0], elastic::CoreMode::kAllowMultipleReduction) ||
        trace.kernel_expert_prefixes[0] == std::get<9>(result).data_ptr() ||
        trace.kernel_unaligned_counts[0] == std::get<10>(result).data_ptr())
        return false;
    auto cached = cached_dispatch(
        *buffer, inputs, result, std::get<8>(result), true);
    return has_exact_result(cached, true, false, true) &&
        trace.launches == 2 &&
        elastic::has_mode(
            trace.modes[1], elastic::CoreMode::kAllowMultipleReduction) &&
        trace.cached_private_contract &&
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

    auto& committed_tensor = *std::get<13>(first);
    if (buffer->get_dispatch_handle_generation(committed_tensor) != 1)
        return false;
    auto* committed_record = reinterpret_cast<elastic::HybridRouteRecord*>(
        static_cast<std::uint8_t*>(committed_tensor.data_ptr()) +
        sizeof(elastic::DispatchHandleDescriptor));
    const auto saved_reserved = committed_record->reserved;
    committed_record->reserved = saved_reserved + 1;
    const auto launches_before_identity_query = trace.launches;
    const auto mutated_generation =
        buffer->get_dispatch_handle_generation(committed_tensor);
    committed_record->reserved = saved_reserved;
    if (mutated_generation != 0 ||
        trace.launches != launches_before_identity_query ||
        buffer->get_dispatch_handle_generation(committed_tensor) != 1)
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
    const std::optional<Tensor> none;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    const auto cached = buffer->dispatch(
        inputs.x, none, inputs.idx, no_weights, none,
        std::get<5>(result), std::get<6>(result), std::get<7>(result),
        std::get<8>(result), std::get<9>(result), std::get<10>(result),
        std::get<12>(result), std::get<13>(result), std::get<11>(result), none,
        4, 2, 4, 1, 0, no_event, no_event, false, false, false, false,
        false, false, false);
    return has_shape(std::get<0>(result), {0, 8}) &&
        std::get<0>(result).storage_nbytes() == 0 &&
        std::get<2>(result)->storage_nbytes() == 0 &&
        std::get<11>(result).storage_nbytes() == 0 &&
        std::get<5>(result) == 0 && std::get<6>(result) == 0 &&
        has_shape(std::get<0>(cached), {0, 8}) &&
        has_shape(*std::get<2>(cached), {0, 1}) &&
        has_shape(std::get<11>(cached), {0, 3}) &&
        std::get<5>(cached) == 0 && std::get<6>(cached) == 0 &&
        trace.launches == 2 && trace.epilogue_launches == 1 &&
        trace.count_stage_outputs_null && trace.zero_epilogue_outputs_null;
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
        "create event",
        "comm waits dependency/previous event",
        "record completion event",
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
    return descriptor->generation == 2 && trace.event_destroys == 3 &&
        buffer->testing_operation_generation() == 2;
}

enum class PostWaitFailure {
    kAllocation,
    kCompletionEvent,
    kActivation,
    kStaging,
    kLaunch,
};

bool failed_cached_dispatch_retains_enqueued_predecessor_wait(
    PostWaitFailure failure) {
    trace = {};
    auto source_resources = resources();
    auto target_resources = resources();
    if (!source_resources || !target_resources)
        return false;
    auto source = Buffer::make_testing_buffer(
        0, std::move(source_resources), 2 * 1024 * 1024, 1);
    auto target = Buffer::make_testing_buffer(
        0, std::move(target_resources), 2 * 1024 * 1024, 1);
    Inputs source_inputs;
    Inputs target_inputs;
    const std::optional<Tensor> source_weights = source_inputs.weights;
    const std::optional<Tensor> target_weights = target_inputs.weights;
    auto source_handle = uncached_dispatch(
        *source, source_inputs, source_weights);
    auto target_handle = uncached_dispatch(
        *target, target_inputs, target_weights);
    auto source_pending = cached_dispatch(
        *source, source_inputs, source_handle, std::get<8>(source_handle),
        false, std::nullopt, true, false);
    if (!std::get<15>(source_pending).has_value())
        return false;
    const std::optional<deep_ep::ascend::EventHandle> predecessor =
        std::get<15>(source_pending);

    const char* injected_fault = nullptr;
    const char* expected_failure = nullptr;
    switch (failure) {
        case PostWaitFailure::kAllocation:
            injected_fault = "allocation";
            expected_failure = "injected post-wait allocation failure";
            break;
        case PostWaitFailure::kCompletionEvent:
            trace.fail_event_create_on = trace.event_creates + 2;
            expected_failure = "backend error 95";
            break;
        case PostWaitFailure::kActivation:
            injected_fault = "activation";
            expected_failure = "injected post-wait activation failure";
            break;
        case PostWaitFailure::kStaging:
            injected_fault = "dispatch_staging";
            expected_failure = "injected post-wait dispatch_staging failure";
            break;
        case PostWaitFailure::kLaunch:
            trace.fail_launch = true;
            expected_failure = "backend error 73";
            break;
    }
    if (injected_fault != nullptr)
        setenv("DEEP_EP_ASCEND_TEST_POST_WAIT_FAULT", injected_fault, 1);
    std::string failure_message;
    try {
        (void)cached_dispatch(
            *target, target_inputs, target_handle, std::get<8>(target_handle),
            false, std::nullopt, true, true, predecessor);
    } catch (const std::runtime_error& error) {
        failure_message = error.what();
    }
    unsetenv("DEEP_EP_ASCEND_TEST_POST_WAIT_FAULT");
    trace.fail_event_create_on = 0;
    trace.fail_launch = false;
    if (failure_message.find(expected_failure) == std::string::npos)
        return false;

    try {
        source->destroy();
    } catch (...) {
        return false;
    }
    try {
        target->destroy();
    } catch (...) {
        return false;
    }
    return true;
}

bool all_failed_cached_dispatch_paths_retain_enqueued_predecessor_waits() {
    return failed_cached_dispatch_retains_enqueued_predecessor_wait(
               PostWaitFailure::kAllocation) &&
        failed_cached_dispatch_retains_enqueued_predecessor_wait(
            PostWaitFailure::kCompletionEvent) &&
        failed_cached_dispatch_retains_enqueued_predecessor_wait(
            PostWaitFailure::kActivation) &&
        failed_cached_dispatch_retains_enqueued_predecessor_wait(
            PostWaitFailure::kStaging) &&
        failed_cached_dispatch_retains_enqueued_predecessor_wait(
            PostWaitFailure::kLaunch);
}

enum class RetirementFailure {
    kRecord,
    kFinish,
};

bool failed_retirement_marker_is_quarantined_and_retryable(
    RetirementFailure failure) {
    trace = {};
    auto source_resources = resources();
    auto target_resources = resources();
    if (!source_resources || !target_resources)
        return false;
    auto source = Buffer::make_testing_buffer(
        0, std::move(source_resources), 2 * 1024 * 1024, 1);
    auto target = Buffer::make_testing_buffer(
        0, std::move(target_resources), 2 * 1024 * 1024, 1,
        true, 7, 2, 0, false, 0);
    Inputs source_inputs;
    Inputs target_inputs;
    const std::optional<Tensor> source_weights = source_inputs.weights;
    const std::optional<Tensor> target_weights = target_inputs.weights;
    auto source_handle = uncached_dispatch(
        *source, source_inputs, source_weights);
    auto target_handle = uncached_dispatch(
        *target, target_inputs, target_weights);
    auto source_pending = cached_dispatch(
        *source, source_inputs, source_handle, std::get<8>(source_handle),
        false, std::nullopt, true, false);
    if (!std::get<15>(source_pending).has_value())
        return false;
    const std::optional<deep_ep::ascend::EventHandle> predecessor =
        std::get<15>(source_pending);

    trace.fail_comm_records = failure == RetirementFailure::kRecord;
    trace.event_ready = failure != RetirementFailure::kFinish;
    setenv("DEEP_EP_ASCEND_TEST_POST_WAIT_FAULT", "allocation", 1);
    std::string failure_message;
    try {
        (void)cached_dispatch(
            *target, target_inputs, target_handle, std::get<8>(target_handle),
            false, std::nullopt, true, true, predecessor);
    } catch (const std::runtime_error& error) {
        failure_message = error.what();
    }
    unsetenv("DEEP_EP_ASCEND_TEST_POST_WAIT_FAULT");
    trace.fail_comm_records = false;
    trace.event_ready = true;
    if (failure_message.empty())
        return false;

    bool poisoned = false;
    try {
        (void)target->get_logical_domain_size();
    } catch (const std::runtime_error& error) {
        poisoned = std::string(error.what()).find("poisoned") !=
            std::string::npos;
    }
    if (!poisoned)
        return false;
    bool source_blocked = false;
    try {
        source->destroy();
    } catch (const std::runtime_error& error) {
        source_blocked = std::string(error.what()).find(
            "outstanding stream waits") != std::string::npos;
    }
    if (!source_blocked)
        return false;
    try {
        target->destroy();
        source->destroy();
    } catch (...) {
        return false;
    }
    return true;
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
    trace.fail_event_record_on = trace.event_records + 3;
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
            std::string::npos && trace.event_destroys == 6 &&
        buffer->is_destroyed();
}

bool uncached_split_mode_matrix_probe() {
    for (const bool expanded : {false, true}) {
        for (const bool async : {false, true}) {
            for (const bool allocate_on_comm_stream : {false, true}) {
                trace = {};
                auto runtime_resources = resources();
                if (!runtime_resources)
                    return false;
                auto buffer = Buffer::make_testing_buffer(
                    0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
                Inputs inputs;
                const std::optional<Tensor> weights = inputs.weights;
                auto result = uncached_dispatch(
                    *buffer, inputs, weights, true, expanded, async,
                    allocate_on_comm_stream);
                if (trace.launches != 1 || trace.epilogue_launches != 1 ||
                    trace.stream_syncs != 1 || trace.modes.size() != 1 ||
                    !elastic::has_mode(
                        trace.modes.front(), elastic::CoreMode::kCpuSync) ||
                    elastic::has_mode(
                        trace.modes.front(), elastic::CoreMode::kAsyncEvent) !=
                        async ||
                    std::get<15>(result).has_value() != async)
                    return false;

                const auto count_launch = std::find(
                    trace.order.begin(), trace.order.end(),
                    "launch uncached count stage");
                const auto count_sync = std::find(
                    trace.order.begin(), trace.order.end(),
                    "synchronize count stage");
                const auto copy_launch = std::find(
                    trace.order.begin(), trace.order.end(),
                    "launch uncached copy epilogue");
                const auto completion_create = std::find(
                    count_sync, trace.order.end(), "create event");
                if (count_launch == trace.order.end() ||
                    count_sync == trace.order.end() ||
                    copy_launch == trace.order.end() ||
                    completion_create == trace.order.end() ||
                    !(count_launch < count_sync &&
                      count_sync < completion_create &&
                      completion_create < copy_launch) ||
                    !trace.count_stage_outputs_null ||
                    !trace.epilogue_outputs_ready)
                    return false;

                const auto& handle = *std::get<13>(result);
                if (async && buffer->get_dispatch_handle_generation(handle) != 0)
                    return false;
                if (async)
                    std::get<15>(result)->current_stream_wait();
                if (!has_exact_result(
                        result, true, true, expanded, async) ||
                    buffer->get_dispatch_handle_generation(handle) != 1)
                    return false;
            }
        }
    }
    return true;
}

bool uncached_completion_create_failure_follows_count_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    trace.fail_event_create_on = 1;
    try {
        (void)uncached_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("backend error 95") ==
                std::string::npos ||
            trace.launches != 1 || trace.stream_syncs != 1 ||
            trace.epilogue_launches != 0 ||
            buffer->testing_operation_generation() != 1)
            return false;
    }
    try {
        (void)uncached_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("cannot continue") !=
            std::string::npos;
    }
}

bool uncached_epilogue_failure_poisons_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    Inputs inputs;
    const std::optional<Tensor> weights = inputs.weights;
    trace.fail_epilogue_launch = true;
    try {
        (void)uncached_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("backend error 79") ==
                std::string::npos ||
            trace.launches != 1 || trace.epilogue_launches != 1)
            return false;
    }
    trace.fail_epilogue_launch = false;
    try {
        (void)uncached_dispatch(*buffer, inputs, weights);
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("cannot continue") !=
                std::string::npos && trace.launches == 1 &&
            trace.epilogue_launches == 1;
    }
}

bool uncached_previous_event_orders_epilogue_probe() {
    trace = {};
    auto source_resources = resources();
    auto target_resources = resources();
    if (!source_resources || !target_resources)
        return false;
    auto source = Buffer::make_testing_buffer(
        0, std::move(source_resources), 2 * 1024 * 1024, 1);
    auto target = Buffer::make_testing_buffer(
        0, std::move(target_resources), 2 * 1024 * 1024, 1);
    Inputs source_inputs;
    Inputs target_inputs;
    const std::optional<Tensor> source_weights = source_inputs.weights;
    const std::optional<Tensor> target_weights = target_inputs.weights;
    auto source_result = uncached_dispatch(
        *source, source_inputs, source_weights, true, false, true, false);
    if (!std::get<15>(source_result).has_value())
        return false;
    std::get<15>(source_result)->current_stream_wait();
    const std::optional<deep_ep::ascend::EventHandle> predecessor =
        std::get<15>(source_result);

    trace.order.clear();
    auto target_result = uncached_dispatch(
        *target, target_inputs, target_weights, true, false, true, true,
        predecessor);
    const auto wait = std::find(
        trace.order.begin(), trace.order.end(),
        "comm waits dependency/previous event");
    const auto count_launch = std::find(
        trace.order.begin(), trace.order.end(),
        "launch uncached count stage");
    if (wait == trace.order.end() || count_launch == trace.order.end() ||
        !(wait < count_launch) || !std::get<15>(target_result).has_value() ||
        target->get_dispatch_handle_generation(*std::get<13>(target_result)) != 0)
        return false;
    std::get<15>(target_result)->current_stream_wait();
    return target->get_dispatch_handle_generation(
               *std::get<13>(target_result)) == 1 &&
        trace.launches == 2 && trace.epilogue_launches == 2;
}

Inputs fp8_inputs(torch::ScalarType scale_type, std::int64_t tokens = 1) {
    Inputs inputs;
    inputs.x = torch::empty(
        {tokens, 64}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn));
    inputs.sf = torch::empty(
        {tokens, 2}, torch::TensorOptions().dtype(scale_type));
    inputs.idx = torch::empty(
        {tokens, 2}, torch::TensorOptions().dtype(torch::kLong));
    inputs.weights = torch::empty(
        {tokens, 2}, torch::TensorOptions().dtype(torch::kFloat));
    if (tokens != 0) {
        inputs.idx.data_ptr<std::int64_t>()[0] = 0;
        inputs.idx.data_ptr<std::int64_t>()[1] = 1;
        inputs.weights.data_ptr<float>()[0] = 0.5F;
        inputs.weights.data_ptr<float>()[1] = 0.75F;
    }
    return inputs;
}

bool has_exact_sf(
    const Tensor& sf, std::int64_t tokens, torch::ScalarType type,
    bool column_major) {
    constexpr std::int64_t packs = 2;
    const auto aligned_tokens = tokens == 0 ? 0 : ((tokens + 3) / 4) * 4;
    const auto expected_token_stride = column_major ? 1 : packs;
    const auto expected_pack_stride = column_major ? aligned_tokens : 1;
    const auto storage_elements = tokens == 0 ? 0 :
        (column_major ? aligned_tokens * (packs - 1) + tokens : tokens * packs);
    if (!has_shape(sf, {tokens, packs}) || sf.scalar_type() != type ||
        sf.stride(0) != expected_token_stride ||
        sf.stride(1) != expected_pack_stride ||
        sf.storage_nbytes() != static_cast<std::size_t>(storage_elements * 4))
        return false;
    for (std::int64_t token = 0; token < tokens; ++token) {
        for (std::int64_t pack = 0; pack < packs; ++pack) {
            std::uint32_t value = 0;
            const auto* storage = static_cast<const std::uint8_t*>(sf.data_ptr());
            std::memcpy(
                &value,
                storage + (token * sf.stride(0) + pack * sf.stride(1)) *
                    static_cast<std::int64_t>(sizeof(value)),
                sizeof(value));
            if (value != static_cast<std::uint32_t>(100 + token * 10 + pack))
                return false;
        }
    }
    return true;
}

bool fp8_exact_sf_case(
    torch::ScalarType scale_type, bool column_major, bool expanded,
    bool cached, bool empty) {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    auto inputs = fp8_inputs(scale_type, empty ? 0 : 1);
    const std::optional<Tensor> weights = inputs.weights;
    auto fresh = uncached_dispatch(
        *buffer, inputs, weights, true, expanded, false, false, std::nullopt,
        column_major);
    auto result = cached ?
        cached_dispatch(
            *buffer, inputs, fresh, std::get<8>(fresh), expanded,
            std::nullopt, false, false, std::nullopt, std::nullopt,
            column_major) :
        std::move(fresh);
    const std::int64_t output_tokens = empty ? 0 : (expanded ? 4 : 2);
    if (!std::get<1>(result).has_value() ||
        !has_exact_sf(
            *std::get<1>(result), output_tokens, scale_type, column_major) ||
        std::get<15>(result).has_value() || trace.sf_observations.empty())
        return false;
    const auto& observed = trace.sf_observations.back();
    return observed.epilogue == !cached && observed.cached == cached &&
        observed.output == std::get<1>(result)->data_ptr() &&
        observed.token_stride == static_cast<std::uint64_t>(
            std::get<1>(result)->stride(0)) &&
        observed.pack_stride == static_cast<std::uint64_t>(
            std::get<1>(result)->stride(1)) &&
        observed.output_tokens == static_cast<std::uint64_t>(output_tokens);
}

bool fp8_exact_sf_matrix_probe() {
    for (const auto scale_type : {torch::kFloat, torch::kInt}) {
        for (const bool column_major : {false, true}) {
            for (const bool expanded : {false, true}) {
                for (const bool cached : {false, true}) {
                    if (!fp8_exact_sf_case(
                            scale_type, column_major, expanded, cached, false))
                        return false;
                }
            }
        }
        if (!fp8_exact_sf_case(
                scale_type, true, false, false, true))
            return false;
    }
    return true;
}

bool fp8_async_sf_lifetime_and_busy_probe(bool cached) {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    std::optional<deep_ep::ascend::EventHandle> completion;
    std::weak_ptr<std::vector<std::uint8_t>> input_x_storage;
    std::weak_ptr<std::vector<std::uint8_t>> input_sf_storage;
    std::weak_ptr<std::vector<std::uint8_t>> output_sf_storage;
    {
        auto inputs = fp8_inputs(torch::kInt);
        const std::optional<Tensor> weights = inputs.weights;
        auto handle = uncached_dispatch(
            *buffer, inputs, weights, true, true, false, false, std::nullopt,
            true);
        trace.order.clear();
        trace.event_record_observations.clear();
        trace.tensor_stream_observations.clear();
        trace.event_ready = false;
        auto pending = cached ?
            cached_dispatch(
                *buffer, inputs, handle, std::get<8>(handle), true,
                std::nullopt, true, false, std::nullopt, std::nullopt, true) :
            uncached_dispatch(
                *buffer, inputs, weights, true, true, true, false,
                std::nullopt, true);
        if (!std::get<1>(pending).has_value() ||
            !has_exact_sf(*std::get<1>(pending), 4, torch::kInt, true) ||
            !std::get<15>(pending).has_value())
            return false;
        const auto find_tensor_record = [](const void* storage) {
            return std::find_if(
                trace.tensor_stream_observations.begin(),
                trace.tensor_stream_observations.end(),
                [storage](const TensorStreamObservation& observation) {
                    return observation.storage == storage;
                });
        };
        const auto input_sf_record = find_tensor_record(
            inputs.sf->storage_identity());
        const auto output_sf_record = find_tensor_record(
            std::get<1>(pending)->storage_identity());
        const auto input_launch_sequence = cached ?
            trace.cached_launch_sequence : trace.count_launch_sequence;
        const auto output_launch_sequence = cached ?
            trace.cached_launch_sequence : trace.epilogue_launch_sequence;
        if (input_sf_record == trace.tensor_stream_observations.end() ||
            output_sf_record == trace.tensor_stream_observations.end() ||
            input_sf_record->stream != &comm_stream_token ||
            output_sf_record->stream != &comm_stream_token ||
            input_launch_sequence == 0 || output_launch_sequence == 0 ||
            input_sf_record->sequence >= input_launch_sequence ||
            output_sf_record->sequence >= output_launch_sequence)
            return false;
        if (!cached) {
            const auto completion_record = std::find_if(
                trace.event_record_observations.begin(),
                trace.event_record_observations.end(),
                [](const EventRecordObservation& observation) {
                    return observation.event_id == trace.event_creates;
                });
            if (completion_record == trace.event_record_observations.end() ||
                completion_record->stream != &comm_stream_token ||
                trace.epilogue_launch_sequence >= completion_record->sequence)
                return false;
        }
        const auto launches_before_retry = trace.launches;
        bool busy = false;
        try {
            if (cached) {
                (void)cached_dispatch(
                    *buffer, inputs, handle, std::get<8>(handle), true,
                    std::nullopt, true, false, std::nullopt, std::nullopt,
                    true);
            } else {
                (void)uncached_dispatch(
                    *buffer, inputs, weights, true, true, true, false,
                    std::nullopt, true);
            }
        } catch (const std::runtime_error& error) {
            busy = std::string(error.what()).find("busy on this buffer") !=
                std::string::npos;
        }
        if (!busy || trace.launches != launches_before_retry)
            return false;
        completion = std::get<15>(pending);
        input_x_storage = inputs.x.weak_storage();
        input_sf_storage = inputs.sf->weak_storage();
        output_sf_storage = std::get<1>(pending)->weak_storage();
    }
    if (input_x_storage.expired() || input_sf_storage.expired() ||
        output_sf_storage.expired() || !completion.has_value())
        return false;
    trace.event_ready = true;
    completion->current_stream_wait();
    return input_x_storage.expired() && input_sf_storage.expired() &&
        output_sf_storage.expired();
}

bool fp8_epilogue_failure_does_not_record_completion_probe() {
    trace = {};
    auto runtime_resources = resources();
    if (!runtime_resources)
        return false;
    auto buffer = Buffer::make_testing_buffer(
        0, std::move(runtime_resources), 2 * 1024 * 1024, 1);
    auto inputs = fp8_inputs(torch::kFloat);
    const std::optional<Tensor> weights = inputs.weights;
    trace.fail_epilogue_launch = true;
    bool failed = false;
    try {
        (void)uncached_dispatch(
            *buffer, inputs, weights, true, false, true, false, std::nullopt,
            false);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("backend error 79") !=
            std::string::npos;
    }
    const auto completion_record = std::find_if(
        trace.event_record_observations.begin(),
        trace.event_record_observations.end(),
        [](const EventRecordObservation& observation) {
            return observation.event_id == trace.event_creates;
        });
    return failed && trace.epilogue_launches == 1 &&
        completion_record == trace.event_record_observations.end();
}

int main() {
    // Keep this protocol probe focused on its explicit legacy-mode contracts;
    // the selector default is covered by core_operator_contract_probe.
    setenv("DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX", "0", 1);
    setenv("DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES", "512", 1);
    setenv("DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX", "0", 1);
    setenv("DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT", "0", 1);
    torch::set_deep_ep_tensor_stream_record_hook(record_tensor_stream);
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
    check(all_failed_cached_dispatch_paths_retain_enqueued_predecessor_waits(),
          "failed cached dispatch retains enqueued predecessor waits");
    check(failed_retirement_marker_is_quarantined_and_retryable(
              RetirementFailure::kRecord) &&
              failed_retirement_marker_is_quarantined_and_retryable(
                  RetirementFailure::kFinish),
          "failed retirement marker is quarantined and retryable");
    check(completion_create_failure_precedes_cached_launch_probe(),
          "completion create failure precedes cached launch");
    check(completion_record_failure_retains_launched_dispatch_probe(),
          "completion record failure retains launched dispatch");
    check(completion_mismatch_fault_does_not_publish(),
          "completion mismatch fault does not publish");
    check(uncached_split_mode_matrix_probe(),
          "uncached count and event epilogue mode matrix");
    check(uncached_completion_create_failure_follows_count_probe(),
          "uncached completion event creation follows count stage");
    check(uncached_epilogue_failure_poisons_probe(),
          "uncached epilogue failure poisoning");
    check(uncached_previous_event_orders_epilogue_probe(),
          "uncached previous event orders count and epilogue stages");
    check(fp8_exact_sf_matrix_probe(),
          "FP8 exact scale-factor allocation and binding matrix");
    check(fp8_async_sf_lifetime_and_busy_probe(false),
          "FP8 uncached async scale-factor lifetime and busy rejection");
    check(fp8_async_sf_lifetime_and_busy_probe(true),
          "FP8 cached async scale-factor lifetime and busy rejection");
    check(fp8_epilogue_failure_does_not_record_completion_probe(),
          "FP8 epilogue failure does not record completion event");
    return failures == 0 ? 0 : 1;
}

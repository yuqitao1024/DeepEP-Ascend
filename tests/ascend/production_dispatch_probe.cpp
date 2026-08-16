#include <cstdlib>
#include <cstring>
#include <memory>

#include "csrc/backends/ascend/elastic_buffer.hpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;
namespace elastic = deep_ep::ascend::elastic;

struct Trace { int launches = 0, copies = 0; std::uint64_t generation = 0; bool bad_diagnostic = false, fail_copy = false, fail_launch = false; } trace;
int alloc(void*, std::uint64_t n, void** p) { *p = std::malloc(n); return *p ? 0 : 1; }
int zero(void*, void* p, std::uint64_t n) { std::memset(p, 0, n); return 0; }
int free_(void*, void* p) { std::free(p); return 0; }
void* stream(void*) { return &trace; }
int sync(void*, void*) { return 0; }
int sync_device(void*) { return 0; }
int h2d(void*, void* d, const void* s, std::uint64_t n) { std::memcpy(d, s, n); return 0; }
int d2h(void*, void* d, const void* s, std::uint64_t n) {
    ++trace.copies;
    if (n == sizeof(transport::DeviceTransportDiagnostic)) {
        auto* diagnostic = static_cast<transport::DeviceTransportDiagnostic*>(d);
        *diagnostic = {};
        diagnostic->abi_version = transport::kTransportCommandAbiVersion;
        diagnostic->generation = trace.generation;
        if (trace.bad_diagnostic) diagnostic->error = transport::DeviceTransportError::kCompletionFailure;
        return 0;
    }
    if (trace.fail_copy) return 91;
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
    runtime::CannRuntimeApi r{nullptr,alloc,zero,free_,stream,sync,sync_device,h2d,d2h};
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
    if (trace.fail_launch) return 73;
    if (t.num_tokens == 0) return 0;
    a.prefix_per_rank[0]=1; a.prefix_per_rank[1]=2;
    a.prefix_per_expert[0]=0; a.prefix_per_expert[1]=1; a.prefix_per_expert[2]=1;
    a.unaligned_per_expert[0]=1; a.unaligned_per_expert[1]=0;
    return 0;
}
extern "C" int deep_ep_ascend_launch_combine(elastic::CombineArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine_epilogue(elastic::CombineArguments, elastic::CoreTiling, void*) { return 0; }

int main() {
    auto buffer = deep_ep::ascend::ElasticBuffer::make_testing_buffer(0, resources(), 2*1024*1024, 1);
    auto x = torch::empty({1, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    auto idx = torch::empty({1, 1}, torch::TensorOptions().dtype(torch::kLong));
    auto weights = torch::empty({1, 1}, torch::TensorOptions().dtype(torch::kFloat));
    const std::optional<torch::Tensor> none;
    const std::optional<int> no_int;
    const std::optional<std::vector<int>> no_list;
    const std::optional<deep_ep::ascend::EventHandle> no_event;
    auto result = buffer->dispatch(x, none, idx, weights, none, no_int, no_int, no_list,
        none, none, none, none, none, none, none, 4, 2, 1, 1, 0, no_event, no_event,
        false, false, true, true, false, false, false);
    if (trace.launches != 1 || std::get<5>(result) != 2 || std::get<0>(result).size(0) != 2 ||
        !std::get<3>(result).has_value() || !std::get<13>(result).has_value()) return 1;
    auto cached = buffer->dispatch(x, none, idx, weights, none,
        std::get<5>(result), std::get<6>(result), std::get<7>(result),
        std::get<8>(result), std::get<9>(result), std::get<10>(result),
        std::get<12>(result), std::get<13>(result), std::get<11>(result), none,
        4, 2, 1, 1, 0, no_event, no_event,
        false, false, false, false, false, false, false);
    if (trace.launches != 2 || std::get<5>(cached) != 2) return 3;
    try {
        (void)buffer->dispatch(x, none, idx, weights, none,
            1, std::get<6>(result), std::get<7>(result), std::get<8>(result),
            std::get<9>(result), std::get<10>(result), std::get<12>(result),
            std::get<13>(result), std::get<11>(result), none,
            4, 2, 1, 1, 0, no_event, no_event,
            false, false, false, false, false, false, false);
        return 4;
    } catch (const std::runtime_error&) {
        if (trace.launches != 2) return 5;
    }
    auto wrong_device_idx = torch::empty(
        {1, 1}, torch::TensorOptions().dtype(torch::kLong).device(1));
    try {
        (void)buffer->dispatch(x, none, wrong_device_idx, none, none,
            no_int, no_int, no_list, none, none, none, none, none, none, none,
            4, 2, 1, 1, 0, no_event, no_event,
            false, false, false, true, false, false, false);
        return 6;
    } catch (const std::runtime_error&) {
        if (trace.launches != 2) return 7;
    }
    trace.fail_launch = true;
    try {
        (void)buffer->dispatch(x, none, idx, none, none, no_int, no_int, no_list,
            none, none, none, none, none, none, none, 4, 2, 1, 1, 0,
            no_event, no_event, false, false, false, true, false, false, false);
        return 8;
    } catch (const std::runtime_error&) {}
    const int launches_after_failure = trace.launches;
    try {
        (void)buffer->dispatch(x, none, idx, none, none, no_int, no_int, no_list,
            none, none, none, none, none, none, none, 4, 2, 1, 1, 0,
            no_event, no_event, false, false, false, true, false, false, false);
        return 9;
    } catch (const std::runtime_error&) {
        if (trace.launches != launches_after_failure) return 10;
    }
    trace.fail_launch = false;
    auto diagnostic_buffer = deep_ep::ascend::ElasticBuffer::make_testing_buffer(
        0, resources(), 2*1024*1024, 1);
    trace.bad_diagnostic = true;
    trace.fail_copy = true;
    const int copies_before_diagnostic = trace.copies;
    try {
        (void)diagnostic_buffer->dispatch(x, none, idx, none, none,
            no_int, no_int, no_list, none, none, none, none, none, none, none,
            4, 2, 1, 1, 0, no_event, no_event,
            false, false, false, true, false, false, false);
        return 11;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("device diagnostic reported failure") ==
                std::string::npos || trace.copies != copies_before_diagnostic + 1)
            return 12;
    }
    trace.bad_diagnostic = false;
    trace.fail_copy = false;
    auto empty_buffer = deep_ep::ascend::ElasticBuffer::make_testing_buffer(
        0, resources(), 2*1024*1024, 1);
    auto empty = torch::empty({0, 8}, x.options());
    auto empty_idx = torch::empty({0, 1}, idx.options());
    auto empty_result = empty_buffer->dispatch(empty, none, empty_idx, none, none, no_int, no_int, no_list,
        none, none, none, none, none, none, none, 4, 2, 1, 1, 0, no_event, no_event,
        false, false, false, true, false, false, false);
    return std::get<0>(empty_result).size(0) == 0 ? 0 : 2;
}

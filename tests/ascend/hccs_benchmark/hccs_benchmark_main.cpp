#include "hccs_benchmark.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"

#include "csrc/backends/ascend/transport/cann_transport.hpp"
#include "csrc/backends/ascend/transport/topology_config.hpp"
#include "csrc/backends/ascend/elastic/tiling.hpp"

namespace transport = deep_ep::ascend::transport;
namespace benchmark = deep_ep::ascend::hccs_benchmark;
namespace elastic = deep_ep::ascend::elastic;

namespace {

void write_error(char* output, std::size_t capacity, const char* format, ...) {
    if (output == nullptr || capacity == 0)
        return;
    va_list arguments;
    va_start(arguments, format);
    (void)std::vsnprintf(output, capacity, format, arguments);
    va_end(arguments);
}

bool check_acl(
    int status, const char* operation, char* error,
    std::size_t error_capacity) {
    if (status == ACL_SUCCESS)
        return true;
    write_error(
        error, error_capacity, "%s failed with status %d", operation,
        status);
    return false;
}

bool checked_window_bytes(
    std::uint32_t world_size, std::uint64_t peer_stride,
    std::uint64_t* bytes) {
    if (bytes == nullptr || world_size < 2 || peer_stride == 0)
        return false;
    const auto slots = static_cast<std::uint64_t>(world_size) * 2;
    if (peer_stride > std::numeric_limits<std::uint64_t>::max() / slots)
        return false;
    *bytes = slots * peer_stride;
    return true;
}

class BenchmarkResources {
public:
    ~BenchmarkResources() {
        (void)shutdown(nullptr, 0);
    }

    bool initialize(
        std::int64_t communicator, std::uint32_t rank,
        std::uint32_t world_size, std::uint64_t peer_stride,
        char* error, std::size_t error_capacity) {
        std::uint64_t window_bytes = 0;
        if (!checked_window_bytes(world_size, peer_stride, &window_bytes)) {
            write_error(error, error_capacity, "invalid benchmark window");
            return false;
        }
        communicator_ = communicator;
        rank_ = rank;
        world_size_ = world_size;
        peer_stride_ = peer_stride;
        window_bytes_ = window_bytes;
        destination_bytes_ = static_cast<std::uint64_t>(world_size) *
            peer_stride;

        if (!check_acl(
                aclrtMalloc(
                    &window_, static_cast<std::size_t>(window_bytes_),
                    ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate symmetric window", error, error_capacity) ||
            !check_acl(
                aclrtMemset(
                    window_, static_cast<std::size_t>(window_bytes_), 0,
                    static_cast<std::size_t>(window_bytes_)),
                "zero symmetric window", error, error_capacity) ||
            !check_acl(
                aclrtMemset(
                    static_cast<std::uint8_t*>(window_) + destination_bytes_,
                    static_cast<std::size_t>(destination_bytes_),
                    static_cast<int>(rank + 1),
                    static_cast<std::size_t>(destination_bytes_)),
                "initialize source payload", error, error_capacity) ||
            !check_acl(
                aclrtMalloc(
                    reinterpret_cast<void**>(&peer_bytes_),
                    world_size * sizeof(std::uint64_t),
                    ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate peer byte table", error, error_capacity) ||
            !check_acl(
                aclrtMalloc(
                    reinterpret_cast<void**>(&control_), sizeof(*control_),
                    ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate benchmark control", error, error_capacity) ||
            !check_acl(
                aclrtCreateStream(&stream_), "create benchmark stream",
                error, error_capacity))
            return false;

        transport::TransportConfig config;
        config.rank = static_cast<int>(rank);
        config.world_size = static_cast<int>(world_size);
        config.communicator_handle = communicator;
        config.device_buffer_bytes = window_bytes_;
        config.requested_channels = 1;
        config.stage_profile_enabled = false;
        auto status =
            transport::configure_transport_topology_from_environment(&config);
        if (!status.ok()) {
            write_error(
                error, error_capacity, "%s failed: backend=%d %s",
                status.operation.c_str(), status.backend_code,
                status.message.c_str());
            return false;
        }
        auto created = transport::make_cann_transport(config);
        if (!created.status.ok()) {
            write_error(
                error, error_capacity, "%s failed: backend=%d %s",
                created.status.operation.c_str(),
                created.status.backend_code,
                created.status.message.c_str());
            return false;
        }
        transport_ = std::move(created.transport);
        status = transport_->register_symmetric_window(window_, window_bytes_);
        if (status.ok())
            status = transport_->acquire_channels(
                1, transport::CooperationScope::kParticipant);
        if (status.ok())
            status = transport_->export_device_context(&context_);
        if (!status.ok()) {
            write_error(
                error, error_capacity, "%s failed: backend=%d %s",
                status.operation.c_str(), status.backend_code,
                status.message.c_str());
            return false;
        }
        return true;
    }

    bool reset(char* error, std::size_t error_capacity) {
        return check_acl(
            aclrtMemset(
                window_, static_cast<std::size_t>(destination_bytes_), 0,
                static_cast<std::size_t>(destination_bytes_)),
            "reset destination payload", error, error_capacity);
    }

    bool run(
        const std::uint64_t* peer_bytes, std::uint64_t generation,
        benchmark::BenchmarkCycles* cycles, char* error,
        std::size_t error_capacity) {
        if (peer_bytes == nullptr || cycles == nullptr || generation == 0) {
            write_error(error, error_capacity, "invalid run arguments");
            return false;
        }
        const auto table_bytes = world_size_ * sizeof(std::uint64_t);
        if (!check_acl(
                aclrtMemcpy(
                    peer_bytes_, table_bytes, peer_bytes, table_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE),
                "copy peer byte table", error, error_capacity) ||
            !check_acl(
                aclrtMemset(
                    control_, sizeof(*control_), 0, sizeof(*control_)),
                "reset benchmark control", error, error_capacity))
            return false;

        const auto barrier_status = HcclBarrier(
            reinterpret_cast<HcclComm>(
                static_cast<std::uintptr_t>(communicator_)),
            stream_);
        if (barrier_status != HCCL_SUCCESS) {
            write_error(
                error, error_capacity, "HcclBarrier failed with status %d",
                static_cast<int>(barrier_status));
            return false;
        }
        const int launch_status = deep_ep_hccs_benchmark_launch(
            window_, peer_stride_, peer_bytes_, context_, generation,
            control_, stream_);
        if (launch_status != 0) {
            write_error(
                error, error_capacity, "kernel launch failed with status %d",
                launch_status);
            return false;
        }
        benchmark::BenchmarkControl control;
        if (!check_acl(
                aclrtSynchronizeStream(stream_), "synchronize benchmark",
                error, error_capacity) ||
            !check_acl(
                aclrtMemcpy(
                    &control, sizeof(control), control_, sizeof(control),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "copy benchmark control", error, error_capacity))
            return false;
        if (control.diagnostic.error != transport::DeviceTransportError::kNone) {
            write_error(
                error, error_capacity,
                "transport diagnostic error=%u opcode=%u peer=%u world=%d",
                static_cast<unsigned>(control.diagnostic.error),
                static_cast<unsigned>(control.diagnostic.opcode),
                control.diagnostic.peer, control.diagnostic.world_peer);
            return false;
        }
        if (control.total_cycles == 0 ||
            control.producer_cycles + control.service_cycles !=
                control.total_cycles) {
            write_error(error, error_capacity, "invalid device cycle result");
            return false;
        }
        cycles->producer_cycles = control.producer_cycles;
        cycles->service_cycles = control.service_cycles;
        cycles->total_cycles = control.total_cycles;
        return true;
    }

    bool verify(
        const std::uint64_t* expected_bytes_by_sender,
        char* error, std::size_t error_capacity) {
        if (expected_bytes_by_sender == nullptr) {
            write_error(error, error_capacity, "missing verification table");
            return false;
        }
        constexpr std::size_t kChunkBytes = 4 * 1024 * 1024;
        std::vector<std::uint8_t> host(kChunkBytes);
        for (std::uint32_t sender = 0; sender < world_size_; ++sender) {
            const auto expected_bytes = expected_bytes_by_sender[sender];
            if (expected_bytes == 0)
                continue;
            if (expected_bytes > peer_stride_) {
                write_error(
                    error, error_capacity,
                    "sender %u payload exceeds peer stride", sender);
                return false;
            }
            const auto* source = static_cast<const std::uint8_t*>(window_) +
                static_cast<std::uint64_t>(sender) * peer_stride_;
            std::uint64_t offset = 0;
            while (offset < expected_bytes) {
                const auto bytes = static_cast<std::size_t>(
                    std::min<std::uint64_t>(
                        kChunkBytes, expected_bytes - offset));
                if (!check_acl(
                        aclrtMemcpy(
                            host.data(), bytes, source + offset, bytes,
                            ACL_MEMCPY_DEVICE_TO_HOST),
                        "copy verification payload", error,
                        error_capacity))
                    return false;
                const auto expected = static_cast<std::uint8_t>(sender + 1);
                const auto mismatch = std::find_if(
                    host.begin(), host.begin() + bytes,
                    [expected](std::uint8_t value) {
                        return value != expected;
                    });
                if (mismatch != host.begin() + bytes) {
                    write_error(
                        error, error_capacity,
                        "payload mismatch sender=%u offset=%llu expected=%u "
                        "actual=%u",
                        sender,
                        static_cast<unsigned long long>(
                            offset + (mismatch - host.begin())),
                        static_cast<unsigned>(expected),
                        static_cast<unsigned>(*mismatch));
                    return false;
                }
                offset += bytes;
            }
        }
        return true;
    }

    bool shutdown(char* error, std::size_t error_capacity) {
        bool success = true;
        if (transport_ != nullptr) {
            const auto status = transport_->destroy();
            if (!status.ok()) {
                write_error(
                    error, error_capacity, "%s failed: backend=%d %s",
                    status.operation.c_str(), status.backend_code,
                    status.message.c_str());
                success = false;
            }
            transport_.reset();
        }
        if (stream_ != nullptr) {
            success = check_acl(
                aclrtDestroyStream(stream_), "destroy benchmark stream",
                error, error_capacity) && success;
            stream_ = nullptr;
        }
        if (control_ != nullptr) {
            success = check_acl(
                aclrtFree(control_), "free benchmark control", error,
                error_capacity) && success;
            control_ = nullptr;
        }
        if (peer_bytes_ != nullptr) {
            success = check_acl(
                aclrtFree(peer_bytes_), "free peer byte table", error,
                error_capacity) && success;
            peer_bytes_ = nullptr;
        }
        if (window_ != nullptr) {
            success = check_acl(
                aclrtFree(window_), "free symmetric window", error,
                error_capacity) && success;
            window_ = nullptr;
        }
        return success;
    }

private:
    std::int64_t communicator_ = 0;
    std::uint32_t rank_ = 0;
    std::uint32_t world_size_ = 0;
    std::uint64_t peer_stride_ = 0;
    std::uint64_t window_bytes_ = 0;
    std::uint64_t destination_bytes_ = 0;
    void* window_ = nullptr;
    std::uint64_t* peer_bytes_ = nullptr;
    benchmark::BenchmarkControl* control_ = nullptr;
    aclrtStream stream_ = nullptr;
    transport::DeviceTransportContext context_{};
    std::unique_ptr<transport::HostTransport> transport_;
};

BenchmarkResources* resource(void* handle) {
    return static_cast<BenchmarkResources*>(handle);
}

}  // namespace

extern "C" std::uint64_t
deep_ep_hccs_benchmark_representative_record_bytes() {
    elastic::CoreTilingInput input{};
    input.operation = elastic::OperationKind::kDispatch;
    input.element_kind = elastic::ElementKind::kFloat8E4M3;
    input.num_tokens = 8192;
    input.hidden = 7168;
    input.num_experts = 256;
    input.num_topk = 8;
    input.expert_alignment = 128;
    input.num_max_tokens_per_rank = 8192;
    input.num_scale_factor_packs = 7168 / 128;
    input.scale_factor_pack_bytes = sizeof(float);
    input.data_num_blocks = 72;
    input.topology.world_rank = 0;
    input.topology.world_size = 8;
    input.topology.scale_up_rank = 0;
    input.topology.scale_up_size = 8;
    input.topology.scale_out_rank = 0;
    input.topology.scale_out_size = 1;
    input.topology.kind = transport::TransportTopologyKind::kFlatScaleUp;
    input.topology.epoch = 1;

    elastic::CoreTiling tiling{};
    const auto status = elastic::build_core_tiling(input, &tiling);
    return status.ok() ? tiling.token_layout.stride_bytes : 0;
}

extern "C" void* deep_ep_hccs_benchmark_create(
    std::int64_t communicator, std::uint32_t rank,
    std::uint32_t world_size, std::uint64_t peer_stride,
    char* error, std::size_t error_capacity) {
    auto result = std::make_unique<BenchmarkResources>();
    if (!result->initialize(
            communicator, rank, world_size, peer_stride,
            error, error_capacity))
        return nullptr;
    return result.release();
}

extern "C" int deep_ep_hccs_benchmark_reset(
    void* handle, char* error, std::size_t error_capacity) {
    return handle != nullptr && resource(handle)->reset(error, error_capacity)
        ? 0 : 1;
}

extern "C" int deep_ep_hccs_benchmark_run(
    void* handle, const std::uint64_t* peer_bytes,
    std::uint64_t generation, benchmark::BenchmarkCycles* cycles,
    char* error, std::size_t error_capacity) {
    return handle != nullptr && resource(handle)->run(
        peer_bytes, generation, cycles, error, error_capacity) ? 0 : 1;
}

extern "C" int deep_ep_hccs_benchmark_verify(
    void* handle, const std::uint64_t* expected_bytes_by_sender,
    char* error, std::size_t error_capacity) {
    return handle != nullptr && resource(handle)->verify(
        expected_bytes_by_sender, error, error_capacity) ? 0 : 1;
}

extern "C" int deep_ep_hccs_benchmark_destroy(
    void* handle, char* error, std::size_t error_capacity) {
    if (handle == nullptr)
        return 0;
    std::unique_ptr<BenchmarkResources> owned(resource(handle));
    return owned->shutdown(error, error_capacity) ? 0 : 1;
}

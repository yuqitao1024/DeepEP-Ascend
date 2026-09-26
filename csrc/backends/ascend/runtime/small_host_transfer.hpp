#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>

#include "../transport/types.hpp"

namespace deep_ep::ascend::runtime {

struct SmallHostTransferApi {
    void* user_data = nullptr;
    int (*allocate)(void*, void**, std::uint64_t) = nullptr;
    int (*free)(void*, void*) = nullptr;
    int (*copy_async)(void*, void*, const void*, std::uint64_t, bool, void*) = nullptr;
    int (*synchronize)(void*, void*) = nullptr;
};

// Reuses pinned staging memory, but returns only after the stream copy finishes.
// Callers must order source writes before the copy on the supplied stream.
class SmallHostTransfer {
public:
    static constexpr std::uint64_t kCapacity = 4096;

    void configure(SmallHostTransferApi api) { api_ = api; }
    bool supported() const noexcept {
        return api_.allocate && api_.free && api_.copy_async && api_.synchronize;
    }

    transport::TransportStatus copy(
        void* destination, const void* source, std::uint64_t bytes,
        bool to_host, void* stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!supported() || !destination || !source || !stream ||
            bytes == 0 || bytes > kCapacity)
            return transport::TransportStatus::invalid(
                "small_host_transfer", "invalid stream copy request");
        if (pending_stream_)
            return transport::TransportStatus::invalid(
                "small_host_transfer", "a failed copy must be drained before reuse");
        if (!host_) {
            const int result = api_.allocate(api_.user_data, &host_, kCapacity);
            if (result != 0)
                return failure("allocate_pinned_host", result);
            if (!host_)
                return failure("allocate_pinned_host", -1);
        }
        if (!to_host) std::memcpy(host_, source, bytes);
        pending_stream_ = stream;
        const int submitted = api_.copy_async(
            api_.user_data, to_host ? host_ : destination,
            to_host ? source : host_, bytes, to_host, stream);
        // Drain even after an enqueue failure: staging memory cannot be reused
        // or released while the backend might still reference it.
        const int completed = api_.synchronize(api_.user_data, stream);
        if (completed == 0) pending_stream_ = nullptr;
        if (submitted != 0) return failure("copy_async", submitted);
        if (completed != 0) return failure("copy_stream_wait", completed);
        if (to_host) std::memcpy(destination, host_, bytes);
        return transport::TransportStatus::success();
    }

    transport::TransportStatus destroy() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_stream_) {
            const int result = api_.synchronize(api_.user_data, pending_stream_);
            if (result != 0) return failure("drain_pinned_host", result);
            pending_stream_ = nullptr;
        }
        if (host_) {
            const int result = api_.free(api_.user_data, host_);
            if (result != 0) return failure("free_pinned_host", result);
            host_ = nullptr;
        }
        return transport::TransportStatus::success();
    }

private:
    static transport::TransportStatus failure(const char* operation, int code) {
        return transport::TransportStatus::runtime_failure(
            operation, code, "pinned host transfer failed");
    }
    SmallHostTransferApi api_{};
    void* host_ = nullptr;
    void* pending_stream_ = nullptr;
    std::mutex mutex_;
};

}  // namespace deep_ep::ascend::runtime

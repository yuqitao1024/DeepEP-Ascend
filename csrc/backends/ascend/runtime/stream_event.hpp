#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "../transport/types.hpp"

namespace deep_ep::ascend::runtime {

struct StreamIdentity {
    void* raw = nullptr;
    std::int64_t stream_id = 0;
    int device_index = -1;
    int device_type = 0;
};

struct StreamEventApi {
    void* user_data = nullptr;
    int (*current_device)(void*, int*) = nullptr;
    int (*current_stream)(void*, StreamIdentity*) = nullptr;
    int (*pool_stream)(void*, int, bool, StreamIdentity*) = nullptr;
    int (*create_event)(void*, void**) = nullptr;
    int (*record_event)(void*, void*, void*) = nullptr;
    int (*query_event)(void*, void*, bool*) = nullptr;
    int (*wait_event)(void*, void*, void*) = nullptr;
    int (*synchronize_event)(void*, void*, std::uint64_t) = nullptr;
    int (*destroy_event)(void*, void*) = nullptr;
};

StreamEventApi make_stream_event_api();

struct NativeEventCreateResult;
class NativeEventWaitLease;

class NativeEventState : public std::enable_shared_from_this<NativeEventState> {
public:
    ~NativeEventState();

    NativeEventState(const NativeEventState&) = delete;
    NativeEventState& operator=(const NativeEventState&) = delete;

    transport::TransportStatus record(StreamIdentity);
    transport::TransportStatus wait(
        StreamIdentity, std::shared_ptr<NativeEventWaitLease>* wait_lease);
    transport::TransportStatus current_stream(StreamIdentity*) const;
    transport::TransportStatus finish(std::uint64_t timeout_ms);
    transport::TransportStatus destroy();
    int device_index() const noexcept;

private:
    enum class State : std::uint8_t {
        Created,
        Recorded,
        Completed,
        Destroyed,
    };

    NativeEventState(StreamEventApi api, void* native_event, int device_index)
        : api_(api), native_event_(native_event), device_index_(device_index) {}

    StreamEventApi api_{};
    void* native_event_ = nullptr;
    int device_index_ = -1;
    mutable std::mutex mutex_;
    State state_ = State::Created;
    std::uint64_t active_wait_leases_ = 0;

    void release_wait_lease() noexcept;

    friend NativeEventCreateResult create_native_event(
        StreamEventApi api, int device_index);
    friend class NativeEventWaitLease;
};

class NativeEventWaitLease {
public:
    ~NativeEventWaitLease();

    NativeEventWaitLease(const NativeEventWaitLease&) = delete;
    NativeEventWaitLease& operator=(const NativeEventWaitLease&) = delete;

private:
    explicit NativeEventWaitLease(std::shared_ptr<NativeEventState> event)
        : event_(std::move(event)) {}

    void arm() noexcept { armed_ = true; }

    std::shared_ptr<NativeEventState> event_;
    bool armed_ = false;

    friend class NativeEventState;
};

struct NativeEventCreateResult {
    transport::TransportStatus status;
    std::shared_ptr<NativeEventState> event;
};

NativeEventCreateResult create_native_event(
    StreamEventApi api, int device_index);

}  // namespace deep_ep::ascend::runtime

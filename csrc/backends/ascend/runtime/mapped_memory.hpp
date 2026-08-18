#pragma once

#include <cstdint>

#include "../transport/types.hpp"

namespace deep_ep::ascend::runtime {

struct MappedSegmentDescriptor {
    void* host_address = nullptr;
    std::uint64_t device_address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t registration = 0;
    std::uint64_t topology_epoch = 0;
};

class MappedMemoryProvider {
public:
    virtual ~MappedMemoryProvider() = default;

    virtual transport::TransportStatus allocate(
        std::uint64_t bytes, std::uint64_t alignment,
        void** host_address) = 0;
    virtual transport::TransportStatus map(
        void* host_address, std::uint64_t bytes,
        std::uint64_t* device_address) = 0;
    virtual transport::TransportStatus register_segment(
        void* host_address, std::uint64_t device_address,
        std::uint64_t bytes, std::uint64_t* registration) = 0;
    virtual transport::TransportStatus publish(
        const MappedSegmentDescriptor& descriptor,
        std::uint64_t* release_generation) = 0;
    virtual transport::TransportStatus acquire(
        std::uint64_t release_generation) = 0;
    virtual transport::TransportStatus invalidate(
        std::uint64_t topology_epoch) = 0;
    virtual transport::TransportStatus deregister_segment(
        std::uint64_t registration) = 0;
    virtual transport::TransportStatus unmap(
        std::uint64_t device_address) = 0;
    virtual transport::TransportStatus free(void* host_address) = 0;
};

class MappedSegmentOwner {
public:
    explicit MappedSegmentOwner(MappedMemoryProvider& provider) noexcept;
    ~MappedSegmentOwner();

    MappedSegmentOwner(const MappedSegmentOwner&) = delete;
    MappedSegmentOwner& operator=(const MappedSegmentOwner&) = delete;

    transport::TransportStatus initialize(
        std::uint64_t bytes, std::uint64_t alignment,
        std::uint64_t topology_epoch);
    transport::TransportStatus acquire();
    transport::TransportStatus invalidate_for_timeout();
    transport::TransportStatus teardown();

    const MappedSegmentDescriptor& descriptor() const noexcept {
        return descriptor_;
    }
    std::uint64_t release_generation() const noexcept {
        return release_generation_;
    }
    bool is_published() const noexcept { return published_; }
    bool valid_for_epoch(std::uint64_t topology_epoch) const noexcept;
    bool contains(std::uint64_t offset, std::uint64_t bytes) const noexcept;

private:
    transport::TransportStatus invalidate_publication();
    bool has_resources() const noexcept;

    MappedMemoryProvider& provider_;
    MappedSegmentDescriptor descriptor_{};
    std::uint64_t release_generation_ = 0;
    bool published_ = false;
};

inline bool mapped_cpu_memory_supported() noexcept {
    // CANN 9.2 exposes pinned allocation but not every required mapped-memory edge.
    return false;
}

}  // namespace deep_ep::ascend::runtime

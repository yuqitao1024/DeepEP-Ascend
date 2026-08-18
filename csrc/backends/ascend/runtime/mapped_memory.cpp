#include "mapped_memory.hpp"

namespace deep_ep::ascend::runtime {
namespace {

using transport::TransportStatus;

bool is_power_of_two(std::uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

MappedSegmentOwner::MappedSegmentOwner(MappedMemoryProvider& provider) noexcept
    : provider_(provider) {}

MappedSegmentOwner::~MappedSegmentOwner() {
    (void)teardown();
}

bool MappedSegmentOwner::has_resources() const noexcept {
    return descriptor_.host_address != nullptr || descriptor_.device_address != 0 ||
        descriptor_.registration != 0 || published_;
}

TransportStatus MappedSegmentOwner::initialize(
    std::uint64_t bytes, std::uint64_t alignment,
    std::uint64_t topology_epoch) {
    if (has_resources())
        return TransportStatus::invalid(
            "initialize_mapped_segment", "mapped segment is already initialized");
    if (bytes == 0)
        return TransportStatus::success();
    if (!is_power_of_two(alignment) || topology_epoch == 0)
        return TransportStatus::invalid(
            "initialize_mapped_segment", "invalid alignment or topology epoch");

    descriptor_.bytes = bytes;
    descriptor_.topology_epoch = topology_epoch;
    auto status = provider_.allocate(bytes, alignment, &descriptor_.host_address);
    if (!status.ok()) {
        descriptor_ = {};
        return status;
    }
    status = provider_.map(
        descriptor_.host_address, bytes, &descriptor_.device_address);
    if (!status.ok()) {
        const auto failure = status;
        (void)teardown();
        return failure;
    }
    status = provider_.register_segment(
        descriptor_.host_address, descriptor_.device_address, bytes,
        &descriptor_.registration);
    if (!status.ok()) {
        const auto failure = status;
        (void)teardown();
        return failure;
    }
    status = provider_.publish(descriptor_, &release_generation_);
    if (status.ok())
        published_ = true;
    if (!status.ok() || release_generation_ == 0) {
        const auto failure = status.ok() ? TransportStatus::runtime_failure(
            "publish_mapped_segment", 0,
            "provider returned a zero release generation") : status;
        release_generation_ = 0;
        (void)teardown();
        return failure;
    }
    return TransportStatus::success();
}

TransportStatus MappedSegmentOwner::acquire() {
    if (!published_ || release_generation_ == 0)
        return TransportStatus::invalid(
            "acquire_mapped_segment", "mapped segment is not published");
    return provider_.acquire(release_generation_);
}

TransportStatus MappedSegmentOwner::invalidate_publication() {
    if (!published_)
        return TransportStatus::success();
    const auto status = provider_.invalidate(descriptor_.topology_epoch);
    if (!status.ok())
        return status;
    published_ = false;
    release_generation_ = 0;
    descriptor_.topology_epoch = 0;
    return TransportStatus::success();
}

TransportStatus MappedSegmentOwner::invalidate_for_timeout() {
    return invalidate_publication();
}

TransportStatus MappedSegmentOwner::teardown() {
    auto status = invalidate_publication();
    if (!status.ok())
        return status;
    if (descriptor_.registration != 0) {
        status = provider_.deregister_segment(descriptor_.registration);
        if (!status.ok())
            return status;
        descriptor_.registration = 0;
    }
    if (descriptor_.device_address != 0) {
        status = provider_.unmap(descriptor_.device_address);
        if (!status.ok())
            return status;
        descriptor_.device_address = 0;
    }
    if (descriptor_.host_address != nullptr) {
        status = provider_.free(descriptor_.host_address);
        if (!status.ok())
            return status;
        descriptor_ = {};
    }
    return TransportStatus::success();
}

bool MappedSegmentOwner::valid_for_epoch(
    std::uint64_t topology_epoch) const noexcept {
    return published_ && topology_epoch != 0 &&
        descriptor_.topology_epoch == topology_epoch;
}

bool MappedSegmentOwner::contains(
    std::uint64_t offset, std::uint64_t bytes) const noexcept {
    return offset <= descriptor_.bytes && bytes <= descriptor_.bytes - offset;
}

}  // namespace deep_ep::ascend::runtime

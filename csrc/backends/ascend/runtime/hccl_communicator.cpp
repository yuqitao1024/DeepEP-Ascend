#include "hccl_communicator.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include <torch/python.h>

#if __has_include(<hccl/hccl_comm.h>)
#define DEEP_EP_ASCEND_HAS_HCCL 1
#include <hccl/hccl_comm.h>
#else
#define DEEP_EP_ASCEND_HAS_HCCL 0
#endif

namespace deep_ep::ascend::runtime {
namespace {

constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kDefaultSymmetricMemoryStrideGiB = 16;

void destroy_handle(std::int64_t handle) {
#if DEEP_EP_ASCEND_HAS_HCCL
    if (handle != 0)
        (void)HcclCommDestroy(reinterpret_cast<HcclComm>(handle));
#else
    (void)handle;
#endif
}

}  // namespace

HcclCommunicator::HcclCommunicator(HcclCommunicator&& other) noexcept
    : handle(other.handle) {
    other.handle = 0;
}

HcclCommunicator& HcclCommunicator::operator=(HcclCommunicator&& other) noexcept {
    if (this != &other) {
        destroy_handle(handle);
        handle = other.handle;
        other.handle = 0;
    }
    return *this;
}

HcclCommunicator::~HcclCommunicator() {
    destroy_handle(handle);
}

pybind11::bytes get_hccl_root_info() {
    std::string bytes(kHcclRootInfoBytes, '\0');
#if DEEP_EP_ASCEND_HAS_HCCL
    HcclRootInfo root_info{};
    const HcclResult result = HcclGetRootInfo(&root_info);
    TORCH_CHECK(result == HCCL_SUCCESS,
                "DeepEP Ascend backend: HcclGetRootInfo failed with backend error ",
                static_cast<int>(result));
    static_assert(sizeof(root_info) == kHcclRootInfoBytes,
                  "CANN 9.3 HcclRootInfo size changed");
    std::memcpy(bytes.data(), &root_info, kHcclRootInfoBytes);
#else
    TORCH_CHECK(false,
                "DeepEP Ascend backend: HCCL communicator APIs are unavailable");
#endif
    return pybind11::bytes(bytes.data(), static_cast<pybind11::ssize_t>(
                                             bytes.size()));
}

std::unique_ptr<HcclCommunicator> create_hccl_communicator(
    const HcclRootInfoBuffer& root_info, int rank, int world_size,
    std::uint64_t symmetric_bytes) {
    TORCH_CHECK(rank >= 0 && world_size > 1 && rank < world_size,
                "DeepEP Ascend backend: invalid symmetric HCCL communicator "
                "topology");
    TORCH_CHECK(symmetric_bytes > 0,
                "DeepEP Ascend backend: symmetric window bytes must be positive");
#if DEEP_EP_ASCEND_HAS_HCCL
    HcclRootInfo hccl_root_info{};
    std::memcpy(&hccl_root_info, root_info.data(), root_info.size());
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    const auto requested_gib =
        (symmetric_bytes + kGiB - 1) / kGiB;
    config.hcclSymWinMaxMemSizePerRank = std::max(
        kDefaultSymmetricMemoryStrideGiB, requested_gib);

    HcclComm communicator = nullptr;
    const HcclResult result = HcclCommInitRootInfoConfig(
        world_size, &hccl_root_info, rank, &config, &communicator);
    TORCH_CHECK(result == HCCL_SUCCESS && communicator != nullptr,
                "DeepEP Ascend backend: HcclCommInitRootInfoConfig failed "
                "with backend error ",
                static_cast<int>(result));
    auto created = std::make_unique<HcclCommunicator>();
    created->handle = reinterpret_cast<std::int64_t>(communicator);
    return created;
#else
    (void)root_info;
    (void)rank;
    (void)world_size;
    (void)symmetric_bytes;
    TORCH_CHECK(false,
                "DeepEP Ascend backend: HCCL communicator APIs are unavailable");
    return nullptr;
#endif
}

}  // namespace deep_ep::ascend::runtime

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <pybind11/pytypes.h>

namespace deep_ep::ascend::runtime {

inline constexpr std::size_t kHcclRootInfoBytes = 4108;
using HcclRootInfoBuffer = std::array<std::uint8_t, kHcclRootInfoBytes>;

struct HcclCommunicator {
    std::int64_t handle = 0;

    HcclCommunicator() = default;
    HcclCommunicator(const HcclCommunicator&) = delete;
    HcclCommunicator& operator=(const HcclCommunicator&) = delete;
    HcclCommunicator(HcclCommunicator&& other) noexcept;
    HcclCommunicator& operator=(HcclCommunicator&& other) noexcept;
    ~HcclCommunicator();
};

pybind11::bytes get_hccl_root_info();
std::unique_ptr<HcclCommunicator> create_hccl_communicator(
    const HcclRootInfoBuffer& root_info, int rank, int world_size,
    std::uint64_t symmetric_bytes);

}  // namespace deep_ep::ascend::runtime

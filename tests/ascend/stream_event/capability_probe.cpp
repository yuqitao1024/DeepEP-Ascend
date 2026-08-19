#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <acl/acl_rt.h>

namespace {

constexpr std::uint32_t kTimeoutMs = 5000;

void check_acl(aclError result, const char* operation) {
    if (result != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with ACL " +
                                 std::to_string(result));
    }
}

void wait_for_event(aclrtEvent event, std::uint32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    while (true) {
        check_acl(aclrtQueryEventStatus(event, &status),
                  "aclrtQueryEventStatus");
        if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            return;
        }
        if (status != ACL_EVENT_RECORDED_STATUS_NOT_READY) {
            throw std::runtime_error("aclrtQueryEventStatus returned an "
                                     "unknown recorded status");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("event did not complete within timeout");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

pybind11::tuple probe_stream_event_capability() {
    const auto device_index = c10_npu::current_device();
    const auto compute_stream = c10_npu::getCurrentNPUStream(device_index);
    const auto comm_stream = c10_npu::getStreamFromPool(true, device_index);
    const auto options = torch::TensorOptions()
        .device(c10::Device(c10::DeviceType::PrivateUse1, device_index))
        .dtype(torch::kFloat);
    auto source = torch::ones({16}, options);
    auto destination = torch::zeros({16}, options);

    aclrtEvent event = nullptr;
    aclrtEvent completion_event = nullptr;
    check_acl(aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC),
              "aclrtCreateEventWithFlag dependency");
    source.fill_(1.0);
    check_acl(aclrtRecordEvent(event, compute_stream.stream(false)),
              "aclrtRecordEvent dependency");
    check_acl(aclrtStreamWaitEvent(comm_stream.stream(false), event),
              "aclrtStreamWaitEvent");
    {
        c10_npu::NPUStreamGuard guard(comm_stream);
        destination.fill_(0.0);
        destination.copy_(source);
        c10_npu::NPUCachingAllocator::recordStream(
            source.storage().data_ptr(), comm_stream);
        c10_npu::NPUCachingAllocator::recordStream(
            destination.storage().data_ptr(), comm_stream);
        check_acl(aclrtCreateEventWithFlag(&completion_event, ACL_EVENT_SYNC),
                  "aclrtCreateEventWithFlag completion");
        check_acl(aclrtRecordEvent(completion_event, comm_stream.stream(false)),
                  "aclrtRecordEvent completion");
    }

    const auto timeout_ms = kTimeoutMs;
    wait_for_event(event, timeout_ms);
    wait_for_event(completion_event, timeout_ms);
    check_acl(aclrtDestroyEvent(completion_event), "aclrtDestroyEvent completion");
    check_acl(aclrtDestroyEvent(event), "aclrtDestroyEvent dependency");

    return pybind11::make_tuple(
        static_cast<std::int64_t>(comm_stream.id()), device_index,
        static_cast<std::int64_t>(comm_stream.device_type()));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("probe_stream_event_capability", &probe_stream_event_capability);
}

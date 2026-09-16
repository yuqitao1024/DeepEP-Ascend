#include <cstdint>
#include <cstdio>

#include "acl/acl.h"

extern "C" int deep_ep_argument_list_330_repro_launch(
    std::uint32_t* output, const std::uint32_t* input, void* stream);

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS ||
        aclrtSetDevice(0) != ACL_SUCCESS)
        return 1;

    aclrtStream stream = nullptr;
    void* input = nullptr;
    void* output = nullptr;
    std::uint32_t host_input = 32;
    std::uint32_t host_output = 0;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS ||
        aclrtMalloc(&input, sizeof(host_input), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&output, sizeof(host_output), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(input, sizeof(host_input), &host_input, sizeof(host_input),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        deep_ep_argument_list_330_repro_launch(
            static_cast<std::uint32_t*>(output),
            static_cast<const std::uint32_t*>(input), stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        aclrtMemcpy(&host_output, sizeof(host_output), output, sizeof(host_output),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::fprintf(stderr, "ACL failure: %s\n", aclGetRecentErrMsg());
        return 2;
    }

#if defined(DEEP_EP_REPRO_LEGACY_ARGS)
    constexpr const char* kernel_form = "legacy";
#else
    constexpr const char* kernel_form = "packed";
#endif
    std::printf("%s kernel output=%u (expected 42)\n", kernel_form, host_output);
    aclrtFree(input);
    aclrtFree(output);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return host_output == 42 ? 0 : 3;
}

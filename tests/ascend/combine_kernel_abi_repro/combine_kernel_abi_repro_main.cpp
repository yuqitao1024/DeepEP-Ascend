#include <cstdint>
#include <cstdio>

#include "acl/acl.h"

extern "C" int deep_ep_combine_kernel_abi_repro_launch(
    std::uint32_t* output, void* stream);

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS)
        return 1;
    aclrtStream stream = nullptr;
    void* output = nullptr;
    std::uint32_t host_output = 0;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS ||
        aclrtMalloc(&output, sizeof(host_output), ACL_MEM_MALLOC_HUGE_FIRST) !=
            ACL_SUCCESS ||
        deep_ep_combine_kernel_abi_repro_launch(
            static_cast<std::uint32_t*>(output), stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        aclrtMemcpy(&host_output, sizeof(host_output), output,
                    sizeof(host_output), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
        std::fprintf(stderr, "ACL failure: %s\n", aclGetRecentErrMsg());
        return 2;
    }
    std::printf("combine kernel ABI probe output=%u (expected 42)\n", host_output);
    aclrtFree(output);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return host_output == 42 ? 0 : 3;
}

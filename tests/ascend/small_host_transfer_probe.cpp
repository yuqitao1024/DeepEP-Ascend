#include <cassert>
#include <cstring>
#include "csrc/backends/ascend/runtime/small_host_transfer.hpp"

using namespace deep_ep::ascend::runtime;
struct Fake {
    unsigned char host[SmallHostTransfer::kCapacity]{};
    void* destination = nullptr;
    const void* source = nullptr;
    std::uint64_t bytes = 0;
    int allocations = 0, frees = 0, waits = 0;
    bool fail_wait = false, fail_submit = false;
};
int allocate(void* data, void** pointer, std::uint64_t bytes) {
    auto& f = *static_cast<Fake*>(data);
    assert(bytes == sizeof(f.host)); ++f.allocations; *pointer = f.host; return 0;
}
int release(void* data, void* pointer) {
    auto& f = *static_cast<Fake*>(data);
    assert(pointer == f.host && f.destination == nullptr); ++f.frees; return 0;
}
int submit(void* data, void* dst, const void* src, std::uint64_t bytes, bool, void*) {
    auto& f = *static_cast<Fake*>(data);
    f.destination = dst; f.source = src; f.bytes = bytes;
    return f.fail_submit ? 17 : 0;
}
int wait(void* data, void*) {
    auto& f = *static_cast<Fake*>(data); ++f.waits;
    if (f.fail_wait) return 18;
    if (f.destination) std::memcpy(f.destination, f.source, f.bytes);
    f.destination = nullptr; return 0;
}
int main() {
    Fake fake;
    SmallHostTransfer transfer;
    assert(!transfer.supported());
    transfer.configure({&fake, allocate, release, submit, wait});
    unsigned char source[160], output[160];
    std::memset(source, 37, sizeof(source));
    std::memset(output, 0, sizeof(output));
    void* stream = &fake;
    assert(transfer.copy(output, source, sizeof(source), true, stream).ok());
    assert(std::memcmp(output, source, sizeof(source)) == 0 && fake.waits == 1);
    std::memset(source, 19, sizeof(source));
    assert(transfer.copy(output, source, sizeof(source), false, stream).ok());
    assert(std::memcmp(output, source, sizeof(source)) == 0 && fake.allocations == 1);
    fake.fail_wait = true;
    std::memset(source, 71, sizeof(source));
    assert(!transfer.copy(output, source, sizeof(source), true, stream).ok());
    assert(output[0] == 19);  // Failed D2H must not publish staging bytes.
    assert(!transfer.copy(output, source, sizeof(source), true, stream).ok());
    assert(!transfer.destroy().ok() && fake.frees == 0);
    fake.fail_wait = false;
    assert(transfer.destroy().ok() && fake.frees == 1);
    fake.fail_submit = true;
    assert(!transfer.copy(output, source, sizeof(source), true, stream).ok());
    assert(fake.destination == nullptr && output[0] == 19);
    assert(transfer.destroy().ok() && fake.frees == 2);
    assert(transfer.destroy().ok() && fake.frees == 2);
    assert(!transfer.copy(output, source, SmallHostTransfer::kCapacity + 1, true, stream).ok());
}

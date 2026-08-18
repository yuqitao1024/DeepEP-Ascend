#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/backends/ascend/runtime/mapped_memory.hpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;

namespace {

int failures = 0;

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": "                \
                      << #expression << '\n';                                \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

class TraceProvider final : public runtime::MappedMemoryProvider {
public:
    std::string fail_operation;
    std::vector<std::string> calls;

    transport::TransportStatus allocate(
        std::uint64_t, std::uint64_t, void** host_address) override {
        calls.emplace_back("allocate");
        if (fails("allocate"))
            return failure("allocate");
        allocation_ = std::malloc(512);
        *host_address = allocation_;
        return allocation_ == nullptr ? failure("allocate") : success();
    }

    transport::TransportStatus map(
        void*, std::uint64_t, std::uint64_t* device_address) override {
        calls.emplace_back("map");
        if (fails("map"))
            return failure("map");
        *device_address = 0x4000;
        return success();
    }

    transport::TransportStatus register_segment(
        void*, std::uint64_t, std::uint64_t,
        std::uint64_t* registration) override {
        calls.emplace_back("register");
        if (fails("register"))
            return failure("register");
        *registration = 0x5000;
        return success();
    }

    transport::TransportStatus publish(
        const runtime::MappedSegmentDescriptor&, std::uint64_t* generation) override {
        calls.emplace_back("publish");
        if (fails("publish"))
            return failure("publish");
        *generation = 17;
        return success();
    }

    transport::TransportStatus acquire(std::uint64_t generation) override {
        calls.emplace_back("acquire");
        return generation == 17 && !fails("acquire") ? success() : failure("acquire");
    }

    transport::TransportStatus invalidate(std::uint64_t epoch) override {
        calls.emplace_back("invalidate");
        return epoch == 9 && !fails("invalidate") ? success() : failure("invalidate");
    }

    transport::TransportStatus deregister_segment(std::uint64_t registration) override {
        calls.emplace_back("deregister");
        return registration == 0x5000 && !fails("deregister") ?
            success() : failure("deregister");
    }

    transport::TransportStatus unmap(std::uint64_t device_address) override {
        calls.emplace_back("unmap");
        return device_address == 0x4000 && !fails("unmap") ? success() : failure("unmap");
    }

    transport::TransportStatus free(void* host_address) override {
        calls.emplace_back("free");
        if (fails("free"))
            return failure("free");
        std::free(host_address);
        allocation_ = nullptr;
        return success();
    }

private:
    bool fails(const char* operation) const {
        return fail_operation == operation;
    }

    static transport::TransportStatus success() {
        return transport::TransportStatus::success();
    }

    static transport::TransportStatus failure(const char* operation) {
        return transport::TransportStatus::runtime_failure(operation, 1, "injected");
    }

    void* allocation_ = nullptr;
};

bool has_calls(const TraceProvider& provider,
               const std::vector<std::string>& expected) {
    return provider.calls == expected;
}

void check_success_and_reverse_teardown() {
    TraceProvider provider;
    runtime::MappedSegmentOwner owner(provider);
    CHECK(owner.initialize(256, 64, 9).ok());
    CHECK(owner.is_published());
    CHECK(owner.descriptor().host_address != nullptr);
    CHECK(owner.descriptor().device_address == 0x4000);
    CHECK(owner.descriptor().bytes == 256);
    CHECK(owner.descriptor().registration == 0x5000);
    CHECK(owner.descriptor().topology_epoch == 9);
    CHECK(owner.release_generation() == 17);
    CHECK(owner.acquire().ok());
    CHECK(owner.teardown().ok());
    CHECK(has_calls(provider, {"allocate", "map", "register", "publish", "acquire",
                               "invalidate", "deregister", "unmap", "free"}));
}

void check_construction_failures_unwind() {
    for (const std::string& operation : {
             std::string("allocate"), std::string("map"),
             std::string("register"), std::string("publish")}) {
        TraceProvider provider;
        provider.fail_operation = operation;
        runtime::MappedSegmentOwner owner(provider);
        CHECK(!owner.initialize(256, 64, 9).ok());
        if (operation == "allocate")
            CHECK(has_calls(provider, {"allocate"}));
        if (operation == "map")
            CHECK(has_calls(provider, {"allocate", "map", "free"}));
        if (operation == "register")
            CHECK(has_calls(provider, {"allocate", "map", "register", "unmap", "free"}));
        if (operation == "publish")
            CHECK(has_calls(provider, {"allocate", "map", "register", "publish",
                                       "deregister", "unmap", "free"}));
        CHECK(owner.teardown().ok());
    }
}

void check_retryable_teardown() {
    TraceProvider provider;
    runtime::MappedSegmentOwner owner(provider);
    CHECK(owner.initialize(256, 64, 9).ok());
    provider.fail_operation = "deregister";
    CHECK(!owner.teardown().ok());
    CHECK(has_calls(provider, {"allocate", "map", "register", "publish", "invalidate",
                               "deregister"}));
    provider.fail_operation.clear();
    CHECK(owner.teardown().ok());
    CHECK(has_calls(provider, {"allocate", "map", "register", "publish", "invalidate",
                               "deregister", "deregister", "unmap", "free"}));
}

void check_epoch_bounds_and_zero_byte_behavior() {
    TraceProvider provider;
    runtime::MappedSegmentOwner owner(provider);
    CHECK(owner.initialize(256, 64, 9).ok());
    CHECK(owner.valid_for_epoch(9));
    CHECK(owner.contains(0, 64));
    CHECK(owner.contains(192, 64));
    CHECK(!owner.contains(256, 1));
    CHECK(!owner.contains(UINT64_MAX, 1));
    CHECK(owner.invalidate_for_timeout().ok());
    CHECK(!owner.valid_for_epoch(9));
    CHECK(owner.descriptor().topology_epoch == 0);
    CHECK(!owner.acquire().ok());
    CHECK(owner.teardown().ok());
    CHECK(has_calls(provider, {"allocate", "map", "register", "publish", "invalidate",
                               "deregister", "unmap", "free"}));

    TraceProvider zero_provider;
    runtime::MappedSegmentOwner zero_owner(zero_provider);
    CHECK(zero_owner.initialize(0, 64, 9).ok());
    CHECK(zero_owner.descriptor().host_address == nullptr);
    CHECK(!zero_owner.is_published());
    CHECK(zero_owner.teardown().ok());
    CHECK(zero_provider.calls.empty());
}

}  // namespace

int main() {
    CHECK(!runtime::mapped_cpu_memory_supported());
    check_success_and_reverse_teardown();
    check_construction_failures_unwind();
    check_retryable_teardown();
    check_epoch_bounds_and_zero_byte_behavior();
    return failures == 0 ? 0 : 1;
}

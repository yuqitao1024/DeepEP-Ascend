#include <cstdlib>
#include <cstdint>
#include <vector>

#include "tests/ascend/simt_urma/runtime_probe.hpp"

namespace probe = deep_ep::ascend::transport::runtime_probe;

#define CHECK(condition)           \
    do {                           \
        if (!(condition))          \
            std::abort();          \
    } while (false)

int main() {
    void* const expected_stream = reinterpret_cast<void*>(
        std::uintptr_t{0x1234});
    void* synchronized_stream = nullptr;
    void* launched_stream = nullptr;
    std::vector<int> events;
    auto reset = [&] {
        events.push_back(1);
        return true;
    };
    auto synchronize = [&](void* stream) {
        events.push_back(2);
        synchronized_stream = stream;
        return true;
    };
    auto launch = [&](void* stream) {
        events.push_back(3);
        launched_stream = stream;
        return true;
    };

    CHECK(probe::reset_synchronize_and_launch(
        true, expected_stream, reset, synchronize, launch));
    CHECK(events == std::vector<int>({1, 2, 3}));
    CHECK(synchronized_stream == expected_stream);
    CHECK(launched_stream == expected_stream);

    events.clear();
    synchronized_stream = nullptr;
    launched_stream = nullptr;
    CHECK(probe::reset_synchronize_and_launch(
        false, expected_stream, reset, synchronize, launch));
    CHECK(events == std::vector<int>({1, 3}));
    CHECK(synchronized_stream == nullptr);
    CHECK(launched_stream == expected_stream);

    events.clear();
    launched_stream = nullptr;
    auto failed_synchronize = [&](void* stream) {
        events.push_back(2);
        synchronized_stream = stream;
        return false;
    };
    CHECK(!probe::reset_synchronize_and_launch(
        true, expected_stream, reset, failed_synchronize, launch));
    CHECK(events == std::vector<int>({1, 2}));
    CHECK(synchronized_stream == expected_stream);
    CHECK(launched_stream == nullptr);

    events.clear();
    auto failed_reset = [&] {
        events.push_back(1);
        return false;
    };
    CHECK(!probe::reset_synchronize_and_launch(
        true, expected_stream, failed_reset, synchronize, launch));
    CHECK(events == std::vector<int>({1}));
    return 0;
}

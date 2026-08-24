#include <cstdlib>
#include <vector>

#include "tests/ascend/simt_urma/runtime_probe.hpp"

namespace probe = deep_ep::ascend::transport::runtime_probe;

#define CHECK(condition)           \
    do {                           \
        if (!(condition))          \
            std::abort();          \
    } while (false)

int main() {
    std::vector<int> events;
    auto reset = [&] {
        events.push_back(1);
        return true;
    };
    auto barrier = [&] {
        events.push_back(2);
        return true;
    };
    auto launch = [&] {
        events.push_back(3);
        return true;
    };

    CHECK(probe::reset_synchronize_and_launch(
        true, reset, barrier, launch));
    CHECK(events == std::vector<int>({1, 2, 3}));

    events.clear();
    CHECK(probe::reset_synchronize_and_launch(
        false, reset, barrier, launch));
    CHECK(events == std::vector<int>({1, 3}));

    events.clear();
    auto failed_barrier = [&] {
        events.push_back(2);
        return false;
    };
    CHECK(!probe::reset_synchronize_and_launch(
        true, reset, failed_barrier, launch));
    CHECK(events == std::vector<int>({1, 2}));

    events.clear();
    auto failed_reset = [&] {
        events.push_back(1);
        return false;
    };
    CHECK(!probe::reset_synchronize_and_launch(
        true, failed_reset, barrier, launch));
    CHECK(events == std::vector<int>({1}));
    return 0;
}

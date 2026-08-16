#include <cstdint>

#include "csrc/backends/ascend/elastic/barrier_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::uint64_t timeout_cycles = 0;
    CHECK(timeout_cycles_from_seconds(1, &timeout_cycles));
    CHECK(timeout_cycles == 1000000000ULL);
    CHECK(timeout_cycles_from_seconds(100, &timeout_cycles));
    CHECK(timeout_cycles == 100000000000ULL);
    CHECK(!timeout_cycles_from_seconds(0, &timeout_cycles));
    CHECK(!timeout_cycles_from_seconds(-1, &timeout_cycles));
    CHECK(!timeout_cycles_from_seconds(1, nullptr));

    BarrierSequence sequence;
    {
        BarrierAttempt first(sequence);
        CHECK(first.valid());
        CHECK(first.generation() == 1);
        first.complete();
    }
    CHECK(!sequence.poisoned());

    {
        BarrierAttempt second(sequence);
        CHECK(second.valid());
        CHECK(second.generation() == 2);
    }
    CHECK(sequence.poisoned());

    BarrierAttempt rejected(sequence);
    CHECK(!rejected.valid());
    CHECK(rejected.generation() == 0);
    return 0;
}

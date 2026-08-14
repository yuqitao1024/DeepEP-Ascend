#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "csrc/backends/ascend/transport/urma_wqe.hpp"

namespace cann_abi = deep_ep::ascend::transport::cann_abi;
namespace urma = deep_ep::ascend::transport::urma;

namespace {

int failures = 0;

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": "                \
                      << #expression << '\n';                                 \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

template <std::size_t WordCount, typename Request>
void check_words(const Request& request,
                 const std::array<std::uint32_t, WordCount>& expected) {
    static_assert(sizeof(request) == WordCount * sizeof(std::uint32_t));
    std::array<std::uint32_t, WordCount> actual{};
    std::memcpy(actual.data(), &request, sizeof(request));
    for (std::size_t index = 0; index < WordCount; ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << "word " << index << ": actual=0x" << std::hex
                      << actual[index] << " expected=0x" << expected[index]
                      << std::dec << '\n';
            ++failures;
        }
    }
}

cann_abi::SqContext make_sq() {
    cann_abi::SqContext sq{};
    sq.depth = 8;
    sq.entry_bytes = 64;
    sq.transport_path_id = 0x00abc123;
    const std::uint64_t eid[2] = {
        0x0123456789abcdefULL,
        0xfedcba9876543210ULL,
    };
    std::memcpy(sq.remote_eid, eid, sizeof(eid));
    return sq;
}

cann_abi::RegisteredBuffer make_remote_memory() {
    cann_abi::RegisteredBuffer remote{};
    remote.type = 1;
    remote.address = 0x1111000000000000ULL;
    remote.bytes = 0x100000;
    remote.protection_type = 1;
    remote.token_id = 0x54321;
    remote.token_value = 0x89abcdef;
    return remote;
}

void check_write() {
    const auto request = urma::make_write(
        make_sq(), make_remote_memory(), 3,
        0x1111222233334444ULL, 0xaaaabbbbccccddddULL, 0x80,
        0x13572468);

    const std::array<std::uint32_t, 16> expected = {
        0xb02d0003, 0x00000300, 0x01abc123, 0x00054321,
        0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98,
        0x89abcdef, 0x00000000, 0x33334444, 0x11112222,
        0x00000080, 0x13572468, 0xccccdddd, 0xaaaabbbb,
    };
    check_words(request, expected);
}

void check_inline_write64() {
    const auto request = urma::make_inline_write64(
        make_sq(), make_remote_memory(), 7,
        0x1111222233334444ULL, 0x0badf00dcafebeefULL);

    const std::array<std::uint32_t, 16> expected = {
        0xb06d0007, 0x02000300, 0x00abc123, 0x00054321,
        0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98,
        0x89abcdef, 0x00000000, 0x33334444, 0x11112222,
        0xcafebeef, 0x0badf00d, 0x00000000, 0x00000000,
    };
    check_words(request, expected);
}

void check_faa64_and_wrap() {
    const auto request = urma::make_faa64(
        make_sq(), make_remote_memory(), 7,
        0x1111222233334444ULL, 0x5555666677778888ULL,
        0xfffffffffffffff7ULL, 0x2468ace0);

    const std::array<std::uint32_t, 32> expected = {
        0xb02d0007, 0x00000b00, 0x01abc123, 0x00054321,
        0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98,
        0x89abcdef, 0x00000000, 0x33334444, 0x11112222,
        0x00000008, 0x2468ace0, 0x77778888, 0x55556666,
        0xfffffff7, 0xffffffff, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    };
    check_words(request, expected);

    CHECK(urma::sq_slot(7, 8) == 7);
    CHECK(urma::sq_slot(8, 8) == 0);
    CHECK(urma::sq_slot(9, 8) == 1);
    CHECK(urma::owner_for(7, 8) == 1);
    CHECK(urma::owner_for(8, 8) == 0);
    CHECK(urma::owner_for(16, 8) == 1);
    CHECK(urma::cqe_owner_valid(1, 7, 8));
    CHECK(urma::cqe_owner_valid(0, 8, 8));
    CHECK(!urma::cqe_owner_valid(1, 8, 8));
}

}  // namespace

int main() {
    check_write();
    check_inline_write64();
    check_faa64_and_wrap();
    return failures == 0 ? 0 : 1;
}

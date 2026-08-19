#include <array>
#include <cstdint>
#include <cstring>

#include "csrc/backends/ascend/elastic/kernels.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

struct MarkerRecord {
    std::uint64_t payload = 0;
    std::int64_t destination = 0;
    std::int32_t master_lane = 0;
    std::int32_t origin_row = 0;
};

struct Address {
    int physical_rank = 0;
    std::uint64_t offset = 0;

    bool operator==(const Address& other) const {
        return physical_rank == other.physical_rank && offset == other.offset;
    }
};

Address local_stage_address(
    const SymmetricWindowLayout& layout, int origin, int destination) {
    const auto route = classify_world_route(origin, destination, 2);
    return {origin,
            layout.hybrid_dispatch_ingress_staging_offset +
                static_cast<std::uint64_t>(route.ingress_world_rank) *
                    layout.hybrid_dispatch_ingress_staging_shard_bytes};
}

Address remote_receive_address(
    const SymmetricWindowLayout& layout, int origin, int destination) {
    const auto route = classify_world_route(origin, destination, 2);
    return {route.ingress_world_rank,
            layout.hybrid_dispatch_ingress_shard_offset +
                static_cast<std::uint64_t>(origin) *
                    layout.hybrid_dispatch_ingress_shard_bytes};
}

std::uint64_t forwarded_count(
    const MarkerRecord& record, int ingress_rank) {
    const int domain_begin = ingress_rank / 2 * 2;
    return record.destination >= domain_begin &&
                   record.destination < domain_begin + 2 ? 1 : 0;
}

}  // namespace

int main() {
    SymmetricWindowInput input{};
    input.world_size = 4;
    input.num_max_tokens_per_rank = 8;
    input.hidden = 128;
    input.num_topk = 2;
    input.element_bytes = 2;
    input.hybrid = true;
    input.hybrid_route_capacity = 40;
    SymmetricWindowLayout layout{};
    CHECK(build_symmetric_window_layout(input, &layout).ok());

    CHECK(layout.hybrid_dispatch_ingress_staging_shard_count == 4);
    CHECK(layout.hybrid_dispatch_ingress_staging_shard_bytes ==
          layout.hybrid_dispatch_ingress_shard_bytes);
    CHECK(layout.hybrid_dispatch_ingress_staging_bytes ==
          layout.hybrid_dispatch_ingress_bytes);

    constexpr int diagonal_destinations[4] = {3, 2, 1, 0};
    constexpr int ingress_ranks[4] = {2, 3, 0, 1};
    constexpr int reciprocal_origins[4] = {2, 3, 0, 1};
    constexpr int reciprocal_destinations[4] = {1, 0, 3, 2};
    constexpr int old_collision_slots[4] = {2, 3, 0, 1};
    for (int origin = 0; origin < 4; ++origin) {
        const int destination = diagonal_destinations[origin];
        const auto route = classify_world_route(origin, destination, 2);
        CHECK(route.kind == WorldRouteKind::kDiagonal);
        CHECK(route.ingress_world_rank == ingress_ranks[origin]);
        CHECK(old_collision_slots[origin] == route.ingress_world_rank);

        const auto local = local_stage_address(layout, origin, destination);
        const auto remote = remote_receive_address(
            layout, reciprocal_origins[origin],
            reciprocal_destinations[origin]);
        CHECK(local.physical_rank == origin);
        CHECK(remote.physical_rank == origin);
        CHECK(!(local == remote));
    }

    constexpr MarkerRecord local_marker{0xaaaa, 3, 0, 11};
    constexpr MarkerRecord remote_marker{0xbbbb, 1, 0, 22};
    for (const bool local_stage_last : {false, true}) {
        std::array<std::uint8_t, 32> staging{};
        std::array<std::uint8_t, 32> receive{};
        if (local_stage_last) {
            std::memcpy(receive.data(), &remote_marker, sizeof(remote_marker));
            std::memcpy(staging.data(), &local_marker, sizeof(local_marker));
        } else {
            std::memcpy(staging.data(), &local_marker, sizeof(local_marker));
            std::memcpy(receive.data(), &remote_marker, sizeof(remote_marker));
        }
        MarkerRecord staged{};
        MarkerRecord received{};
        std::memcpy(&staged, staging.data(), sizeof(staged));
        std::memcpy(&received, receive.data(), sizeof(received));
        CHECK(staged.payload == local_marker.payload);
        CHECK(staged.master_lane == local_marker.master_lane);
        CHECK(staged.origin_row == local_marker.origin_row);
        CHECK(received.payload == remote_marker.payload);
        CHECK(received.master_lane == remote_marker.master_lane);
        CHECK(received.origin_row == remote_marker.origin_row);
        CHECK(forwarded_count(received, 0) == 1);
        CHECK(forwarded_count(staged, 0) == 0);
    }

    for (const bool cached : {false, true}) {
        for (std::uint64_t generation = 1; generation <= 3; ++generation) {
            MarkerRecord staged{
                1000 + generation + (cached ? 100 : 0), 3, 0,
                static_cast<std::int32_t>(generation)};
            MarkerRecord received{
                2000 + generation + (cached ? 100 : 0), 1, 0,
                static_cast<std::int32_t>(generation + 10)};
            CHECK(staged.payload != received.payload);
            CHECK(staged.origin_row == static_cast<std::int32_t>(generation));
            CHECK(received.origin_row ==
                  static_cast<std::int32_t>(generation + 10));
            CHECK(forwarded_count(received, 0) == 1);
        }
    }

    return 0;
}

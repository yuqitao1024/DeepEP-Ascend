#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "csrc/backends/ascend/elastic/kernels.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

#ifndef DEEP_EP_TEST_MUTATE_INGRESS_SOURCE
#define DEEP_EP_TEST_MUTATE_INGRESS_SOURCE 0
#endif

#ifndef DEEP_EP_TEST_MUTATE_CACHED_STALE
#define DEEP_EP_TEST_MUTATE_CACHED_STALE 0
#endif

namespace {

struct MarkerRecord {
    std::uint64_t payload = 0;
    std::int64_t destination = 0;
    std::int32_t master_lane = 0;
    std::int32_t origin_row = 0;
    std::uint64_t generation = 0;
};

struct PublishedControl {
    std::uint64_t generation = 0;
    std::uint64_t count = 0;
};

struct Address {
    int physical_rank = 0;
    std::uint64_t offset = 0;

    bool operator==(const Address& other) const {
        return physical_rank == other.physical_rank && offset == other.offset;
    }
};

Address local_stage_address(
    const SymmetricWindowLayout& layout, int origin, int destination,
    std::uint64_t slot = 0) {
    const auto route = classify_world_route(origin, destination, 2);
    return {origin,
            layout.hybrid_dispatch_ingress_staging_offset +
                static_cast<std::uint64_t>(route.ingress_world_rank) *
                    layout.hybrid_dispatch_ingress_staging_shard_bytes +
                slot * layout.dispatch_record_bytes};
}

Address modeled_stage_address(
    const SymmetricWindowLayout& layout, int origin, int destination,
    std::uint64_t slot) {
#if DEEP_EP_TEST_MUTATE_INGRESS_SOURCE
    const auto route = classify_world_route(origin, destination, 2);
    return {origin,
            layout.hybrid_dispatch_ingress_shard_offset +
                static_cast<std::uint64_t>(route.ingress_world_rank) *
                    layout.hybrid_dispatch_ingress_shard_bytes +
                slot * layout.dispatch_record_bytes};
#else
    return local_stage_address(layout, origin, destination, slot);
#endif
}

Address remote_receive_address(
    const SymmetricWindowLayout& layout, int origin, int destination,
    std::uint64_t slot = 0) {
    const auto route = classify_world_route(origin, destination, 2);
    return {route.ingress_world_rank,
            layout.hybrid_dispatch_ingress_shard_offset +
                static_cast<std::uint64_t>(origin) *
                    layout.hybrid_dispatch_ingress_shard_bytes +
                slot * layout.dispatch_record_bytes};
}

std::uint64_t forwarded_count(
    const MarkerRecord& record, int ingress_rank) {
    const int domain_begin = ingress_rank / 2 * 2;
    return record.destination >= domain_begin &&
                   record.destination < domain_begin + 2 ? 1 : 0;
}

using SymmetricWindows = std::array<std::vector<std::uint8_t>, 4>;

void write_record(
    SymmetricWindows* windows, const Address& address,
    const MarkerRecord& record) {
    std::memcpy((*windows)[address.physical_rank].data() + address.offset,
                &record, sizeof(record));
}

MarkerRecord read_record(
    const SymmetricWindows& windows, const Address& address) {
    MarkerRecord record{};
    std::memcpy(&record,
                windows[address.physical_rank].data() + address.offset,
                sizeof(record));
    return record;
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

    for (int interleaving = 0; interleaving < 2; ++interleaving) {
        SymmetricWindows windows{};
        for (auto& window : windows)
            window.resize(layout.total_bytes);
        std::array<std::array<PublishedControl, 4>, 4> controls{};
        std::array<std::int64_t, 4> cached_slots{-1, -1, -1, -1};

        for (std::uint64_t generation = 1; generation <= 3; ++generation) {
            const bool cached = generation > 1;
            std::array<std::uint64_t, 4> outbound_counts{};
            std::array<std::uint64_t, 4> slots{};
            std::array<MarkerRecord, 4> expected{};
            for (int origin = 0; origin < 4; ++origin) {
                const int ingress_rank = ingress_ranks[origin];
                if (!cached) {
                    slots[origin] = outbound_counts[ingress_rank]++;
                    cached_slots[origin] =
                        static_cast<std::int64_t>(slots[origin]);
                } else {
                    CHECK(cached_slots[origin] >= 0);
                    slots[origin] =
                        static_cast<std::uint64_t>(cached_slots[origin]);
                }
                CHECK(slots[origin] == 0);
                expected[origin] = {
                    generation * 1000 +
                        static_cast<std::uint64_t>(interleaving * 100 + origin),
                    diagonal_destinations[origin], origin % 2, origin,
                    generation};
            }

            const auto write_origin = [&](int origin) {
#if DEEP_EP_TEST_MUTATE_CACHED_STALE
                if (cached)
                    return true;
#endif
                const auto address = modeled_stage_address(
                    layout, origin, diagonal_destinations[origin],
                    slots[origin]);
                if (address.offset + sizeof(MarkerRecord) >
                    layout.total_bytes)
                    return false;
                write_record(&windows, address, expected[origin]);
                return true;
            };
            const auto stage_one = [&](int origin) {
                const auto source = modeled_stage_address(
                    layout, origin, diagonal_destinations[origin],
                    slots[origin]);
                const auto destination = remote_receive_address(
                    layout, origin, diagonal_destinations[origin],
                    slots[origin]);
                if (source.offset + sizeof(MarkerRecord) >
                        layout.total_bytes ||
                    destination.offset + sizeof(MarkerRecord) >
                        layout.total_bytes)
                    return false;
                write_record(
                    &windows, destination, read_record(windows, source));
                controls[destination.physical_rank][origin].count = 1;
                controls[destination.physical_rank][origin].generation =
                    generation;
                return true;
            };

            if (interleaving == 0) {
                for (int origin = 0; origin < 4; ++origin)
                    CHECK(write_origin(origin));
                for (const int origin : {0, 2, 1, 3})
                    CHECK(stage_one(origin));
            } else {
                for (const int origin : {2, 0, 3, 1}) {
                    CHECK(write_origin(origin));
                    CHECK(stage_one(origin));
                }
            }

            std::uint64_t total_forwarded = 0;
            for (int origin = 0; origin < 4; ++origin) {
                const int ingress_rank = ingress_ranks[origin];
                const auto& control = controls[ingress_rank][origin];
                CHECK(control.generation == generation);
                CHECK(control.count == 1);
                const auto received = read_record(
                    windows,
                    remote_receive_address(
                        layout, origin, diagonal_destinations[origin],
                        slots[origin]));
                const auto count = forwarded_count(received, ingress_rank);
                CHECK(count == 1);
                total_forwarded += count;
                CHECK(received.payload == expected[origin].payload);
                CHECK(received.destination == expected[origin].destination);
                CHECK(received.master_lane == expected[origin].master_lane);
                CHECK(received.origin_row == origin);
                CHECK(received.generation == generation);
            }
            CHECK(total_forwarded == 4);
        }
    }

    return 0;
}

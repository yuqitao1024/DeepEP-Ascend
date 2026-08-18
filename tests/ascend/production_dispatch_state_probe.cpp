#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/dispatch_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    static_assert(std::is_standard_layout_v<DispatchControlSlot>);
    static_assert(std::is_trivially_copyable_v<DispatchControlSlot>);
    static_assert(sizeof(DispatchControlSlot) == 16);
    static_assert(std::is_standard_layout_v<DispatchHandleDescriptor>);
    static_assert(std::is_trivially_copyable_v<DispatchHandleDescriptor>);
    static_assert(kDispatchHandleDescriptorAbiVersion == 4);
    static_assert(std::is_standard_layout_v<HybridRouteRecord>);
    static_assert(std::is_trivially_copyable_v<HybridRouteRecord>);
    static_assert(sizeof(HybridRouteRecord) == 64);

    DispatchSequence sequence;
    {
        DispatchAttempt first(sequence);
        CHECK(first.valid());
        CHECK(first.generation() == 1);
        first.complete();
    }
    CHECK(!sequence.poisoned());

    {
        DispatchAttempt incomplete(sequence);
        CHECK(incomplete.valid());
        CHECK(incomplete.generation() == 2);
    }
    CHECK(sequence.poisoned());
    DispatchAttempt rejected(sequence);
    CHECK(!rejected.valid());
    CHECK(rejected.generation() == 0);

    DispatchSequence exhausted(std::numeric_limits<std::uint64_t>::max());
    DispatchAttempt exhausted_attempt(exhausted);
    CHECK(!exhausted_attempt.valid());
    CHECK(exhausted_attempt.generation() == 0);
    CHECK(exhausted.poisoned());

    CoreTopology topology{};
    topology.world_size = 4;
    topology.scale_up_size = 2;
    topology.scale_out_size = 2;
    topology.epoch = 9;
    DispatchHandleDescriptor expected = make_dispatch_handle_descriptor(
        17, topology, 11, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    CHECK(expected.abi_version == kDispatchHandleDescriptorAbiVersion);
    CHECK(expected.struct_size == sizeof(DispatchHandleDescriptor));
    CHECK(expected.generation == 11);
    CHECK(expected.routing_mode == DispatchRoutingMode::kDirect);
    CHECK(expected.route_record_count == 0);
    CHECK(expected.route_record_stride == 0);
    CHECK(expected.dispatch_generation == 11);
    CHECK(expected.route_stage_flags == 0);
    CHECK(validate_dispatch_handle(expected, expected).ok());

    auto zero_padded = expected;
    zero_padded.mode_flags |= mode_bit(CoreMode::kZeroPadding);
    CHECK(validate_dispatch_handle(expected, zero_padded).ok());
    CHECK(validate_dispatch_handle(zero_padded, expected).ok());

    const auto attested_expanded = make_attested_dispatch_handle_descriptor(
        17, topology, 11, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    const auto attested_zero_padded =
        make_attested_dispatch_handle_descriptor(
            17, topology, 11, 3, 128, 8, 2, 4, 256,
            mode_bit(CoreMode::kExpanded) |
                mode_bit(CoreMode::kZeroPadding));
    CHECK(attested_expanded.family == attested_zero_padded.family);
    CHECK(validate_dispatch_handle(
        attested_expanded, attested_zero_padded).ok());

    auto mismatch = expected;
    mismatch.family += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.topology.scale_up_rank = 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.generation = 10;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_tokens += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.hidden += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_experts += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_topk += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.expert_alignment += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.num_max_tokens_per_rank += 1;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());
    mismatch = expected;
    mismatch.mode_flags = 0;
    CHECK(!validate_dispatch_handle(expected, mismatch).ok());

    const auto attested = make_attested_dispatch_handle_descriptor(
        99, topology, 11, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    const auto stale = make_attested_dispatch_handle_descriptor(
        99, topology, 10, 3, 128, 8, 2, 4, 256,
        mode_bit(CoreMode::kExpanded));
    CHECK(attested.family != stale.family);

    const auto hybrid_mode = mode_bit(CoreMode::kExpanded) |
        mode_bit(CoreMode::kHybrid);
    const auto complete_stages =
        hybrid_stage_bit(HybridRouteStage::kIngressComplete) |
        hybrid_stage_bit(HybridRouteStage::kForwardComplete);
    const auto hybrid = make_attested_dispatch_handle_descriptor(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion, 1,
        sizeof(HybridRouteRecord), 11, complete_stages);
    CHECK(hybrid.routing_mode == DispatchRoutingMode::kHybrid);
    CHECK(hybrid.route_layout_version == kHybridRouteLayoutVersion);
    CHECK(hybrid.route_record_count == 1);
    CHECK(hybrid.route_record_stride == sizeof(HybridRouteRecord));
    CHECK(hybrid.dispatch_generation == 11);
    CHECK(hybrid.route_stage_flags == complete_stages);
    CHECK(!validate_dispatch_handle(expected, hybrid).ok());

    HybridRouteRecord diagonal{};
    diagonal.origin_world_rank = 0;
    diagonal.origin_source_row = 7;
    diagonal.ingress_world_rank = 2;
    diagonal.destination_world_rank = 3;
    diagonal.destination_local_expert = 1;
    diagonal.ingress_slot = 5;
    diagonal.forwarded_slot = 4;
    diagonal.generation = 11;
    diagonal.topology_epoch = 9;
    diagonal.stage_flags = hybrid_stage_bit(HybridRouteStage::kIngressComplete) |
                          hybrid_stage_bit(HybridRouteStage::kForwardComplete);
    HybridRouteTableView route_table{&diagonal, 1};
    CHECK(validate_hybrid_route_table(hybrid, route_table, 8, 8, 2).ok());
    CHECK(!validate_hybrid_route_table(expected, route_table, 8, 8, 2).ok());

    auto invalid_hybrid = hybrid;
    invalid_hybrid.dispatch_generation = 10;
    CHECK(!validate_hybrid_route_table(invalid_hybrid, route_table, 8, 8, 2).ok());
    invalid_hybrid = hybrid;
    invalid_hybrid.topology.epoch = 8;
    CHECK(!validate_hybrid_route_table(invalid_hybrid, route_table, 8, 8, 2).ok());
    invalid_hybrid = hybrid;
    invalid_hybrid.route_stage_flags =
        hybrid_stage_bit(HybridRouteStage::kIngressComplete);
    CHECK(!validate_hybrid_route_table(invalid_hybrid, route_table, 8, 8, 2).ok());
    invalid_hybrid = hybrid;
    invalid_hybrid.route_record_count = 2;
    CHECK(!validate_hybrid_route_table(invalid_hybrid, route_table, 8, 8, 2).ok());
    invalid_hybrid = hybrid;
    invalid_hybrid.route_record_stride = sizeof(HybridRouteRecord) - 1;
    CHECK(!validate_hybrid_route_table(invalid_hybrid, route_table, 8, 8, 2).ok());

    auto invalid_record = diagonal;
    invalid_record.generation = 10;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.topology_epoch = 8;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.stage_flags = hybrid_stage_bit(HybridRouteStage::kIngressComplete);
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.origin_world_rank = 4;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.origin_source_row = 8;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.destination_local_expert = 2;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.ingress_slot = 8;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());

    HybridRouteRecord duplicate_records[2]{diagonal, diagonal};
    duplicate_records[1].origin_world_rank = 1;
    duplicate_records[1].origin_source_row = 6;
    auto two_routes = hybrid;
    two_routes.route_record_count = 2;
    CHECK(!validate_hybrid_route_table(
        two_routes, {duplicate_records, 2}, 8, 8, 2).ok());
    duplicate_records[1].ingress_slot = 6;
    CHECK(!validate_hybrid_route_table(
        two_routes, {duplicate_records, 2}, 8, 8, 2).ok());
    duplicate_records[1].forwarded_slot = 7;
    CHECK(validate_hybrid_route_table(
        two_routes, {duplicate_records, 2}, 8, 8, 2).ok());

    duplicate_records[0].ingress_slot = kInvalidHybridRouteSlot;
    duplicate_records[1].ingress_slot = kInvalidHybridRouteSlot;
    duplicate_records[0].forwarded_slot = kInvalidHybridRouteSlot;
    duplicate_records[1].forwarded_slot = kInvalidHybridRouteSlot;
    CHECK(validate_hybrid_route_table(
        two_routes, {duplicate_records, 2}, 8, 8, 2).ok());
    return 0;
}

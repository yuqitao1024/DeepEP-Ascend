#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

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
    static_assert(offsetof(HybridRouteRecord, origin_world_rank) == 0);
    static_assert(offsetof(HybridRouteRecord, destination_world_rank) == 4);
    static_assert(offsetof(HybridRouteRecord, ingress_world_rank) == 8);
    static_assert(offsetof(HybridRouteRecord, destination_local_expert) == 12);
    static_assert(offsetof(HybridRouteRecord, origin_source_row) == 16);
    static_assert(offsetof(HybridRouteRecord, ingress_slot) == 24);
    static_assert(offsetof(HybridRouteRecord, forwarded_slot) == 32);
    static_assert(offsetof(HybridRouteRecord, generation) == 40);
    static_assert(offsetof(HybridRouteRecord, topology_epoch) == 48);
    static_assert(offsetof(HybridRouteRecord, stage_flags) == 56);
    static_assert(offsetof(HybridRouteRecord, reserved) == 60);

    CHECK(validate_hybrid_route_control(29, 29, 8, 8) ==
          DispatchProtocolError::kNone);
    CHECK(validate_hybrid_route_control(29, 28, 8, 8) ==
          DispatchProtocolError::kInvalidControl);
    CHECK(validate_hybrid_route_control(29, 29, 9, 8) ==
          DispatchProtocolError::kCapacityOverflow);
    const auto prepare_failure = make_dispatch_protocol_failure(
        3, DispatchProtocolStage::kPrepareEpilogue, 29,
        DispatchProtocolError::kInvalidControl);
    CHECK(prepare_failure.world_rank == 3);
    CHECK(prepare_failure.generation == 29);
    CHECK(prepare_failure.scratch_status ==
          ((std::uint64_t{4} << 32U) |
           static_cast<std::uint32_t>(
               DispatchProtocolError::kInvalidControl)));
    CHECK(prepare_failure.backend_status ==
          ((static_cast<std::uint32_t>(
                DispatchProtocolStage::kPrepareEpilogue) << 16U) |
           static_cast<std::uint32_t>(
               DispatchProtocolError::kInvalidControl)));

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
    topology.world_rank = 3;
    topology.world_size = 4;
    topology.scale_up_rank = 1;
    topology.scale_up_size = 2;
    topology.scale_out_rank = 1;
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
    mismatch.topology.scale_up_rank = 0;
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
    const auto hybrid_family = attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion, 1,
        sizeof(HybridRouteRecord), 11, complete_stages);
    CHECK(hybrid.family == hybrid_family);
    CHECK(hybrid_family != attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kDirect, kHybridRouteLayoutVersion, 1,
        sizeof(HybridRouteRecord), 11, complete_stages));
    CHECK(hybrid_family != attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion + 1, 1,
        sizeof(HybridRouteRecord), 11, complete_stages));
    CHECK(hybrid_family != attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion, 2,
        sizeof(HybridRouteRecord), 11, complete_stages));
    CHECK(hybrid_family != attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion, 1,
        sizeof(HybridRouteRecord) + 8, 11, complete_stages));
    CHECK(hybrid_family != attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion, 1,
        sizeof(HybridRouteRecord), 10, complete_stages));
    CHECK(hybrid_family != attest_dispatch_handle_family(
        99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
        DispatchRoutingMode::kHybrid, kHybridRouteLayoutVersion, 1,
        sizeof(HybridRouteRecord), 11,
        hybrid_stage_bit(HybridRouteStage::kIngressComplete)));

    HybridRouteRecord diagonal{};
    diagonal.origin_world_rank = 0;
    diagonal.origin_source_row = 7;
    diagonal.ingress_world_rank = 2;
    diagonal.destination_world_rank = 3;
    diagonal.destination_local_expert = 1;
    diagonal.ingress_slot = 0;
    diagonal.forwarded_slot = 0;
    diagonal.generation = 11;
    diagonal.topology_epoch = 9;
    diagonal.stage_flags = hybrid_stage_bit(HybridRouteStage::kIngressComplete) |
                          hybrid_stage_bit(HybridRouteStage::kForwardComplete);
    HybridRouteTableView route_table{&diagonal, 1};
    CHECK(validate_hybrid_route_table(hybrid, route_table, 8, 8, 2).ok());
    CHECK(!validate_hybrid_route_table(expected, route_table, 8, 8, 2).ok());

    const auto route_family = [&](HybridRouteTableView table) {
        return attest_hybrid_dispatch_handle_family(
            99, topology, 11, 3, 128, 8, 2, 4, 256, hybrid_mode,
            kHybridRouteLayoutVersion, table.count,
            sizeof(HybridRouteRecord), complete_stages, table);
    };
    const auto attested_route_family = route_family(route_table);
    CHECK(attested_route_family == route_family(route_table));
    CHECK(attested_route_family != hybrid.family);
    const auto route_mutation_changes_family =
        [&](const HybridRouteRecord& record) {
        return attested_route_family != route_family({&record, 1});
    };
    auto attestation_mutation = diagonal;
    ++attestation_mutation.origin_world_rank;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    --attestation_mutation.destination_world_rank;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    --attestation_mutation.ingress_world_rank;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    --attestation_mutation.destination_local_expert;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    --attestation_mutation.origin_source_row;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    ++attestation_mutation.ingress_slot;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    ++attestation_mutation.forwarded_slot;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    --attestation_mutation.generation;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    --attestation_mutation.topology_epoch;
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    attestation_mutation.stage_flags =
        hybrid_stage_bit(HybridRouteStage::kIngressComplete);
    CHECK(route_mutation_changes_family(attestation_mutation));
    attestation_mutation = diagonal;
    ++attestation_mutation.reserved;
    CHECK(route_mutation_changes_family(attestation_mutation));
    std::array<HybridRouteRecord, 2> ordered_routes{diagonal, diagonal};
    ordered_routes[1].origin_source_row = 6;
    const auto ordered_family = route_family(
        {ordered_routes.data(), ordered_routes.size()});
    std::swap(ordered_routes[0], ordered_routes[1]);
    CHECK(ordered_family != route_family(
        {ordered_routes.data(), ordered_routes.size()}));

    std::int32_t source_metadata[] = {
        encode_dispatch_source_index(0, 8, 7),
        encode_dispatch_source_index(0, 2, 1), -1, -1};
    std::int64_t received_topk[] = {-1, 1};
    HybridRouteBindingView binding{};
    binding.source_metadata = source_metadata;
    binding.received_topk_indices = received_topk;
    binding.row_count = 1;
    binding.num_topk = 2;
    binding.shard_capacity = 8;
    CHECK(validate_hybrid_route_bindings(hybrid, route_table, binding).ok());

    auto wrong_binding_record = diagonal;
    wrong_binding_record.destination_world_rank = 2;
    CHECK(!validate_hybrid_route_bindings(
        hybrid, {&wrong_binding_record, 1}, binding).ok());
    wrong_binding_record = diagonal;
    wrong_binding_record.ingress_world_rank = 1;
    CHECK(!validate_hybrid_route_bindings(
        hybrid, {&wrong_binding_record, 1}, binding).ok());
    wrong_binding_record = diagonal;
    wrong_binding_record.ingress_slot = 1;
    CHECK(!validate_hybrid_route_bindings(
        hybrid, {&wrong_binding_record, 1}, binding).ok());
    wrong_binding_record = diagonal;
    wrong_binding_record.forwarded_slot = 1;
    CHECK(!validate_hybrid_route_bindings(
        hybrid, {&wrong_binding_record, 1}, binding).ok());
    wrong_binding_record = diagonal;
    wrong_binding_record.destination_local_expert = 0;
    CHECK(!validate_hybrid_route_bindings(
        hybrid, {&wrong_binding_record, 1}, binding).ok());
    source_metadata[0] = encode_dispatch_source_index(1, 8, 7);
    CHECK(!validate_hybrid_route_bindings(hybrid, route_table, binding).ok());
    source_metadata[0] = encode_dispatch_source_index(0, 8, 7);
    received_topk[1] = 0;
    CHECK(!validate_hybrid_route_bindings(hybrid, route_table, binding).ok());
    received_topk[1] = 1;
    auto invalid_binding_descriptor = hybrid;
    invalid_binding_descriptor.topology.scale_up_size = 0;
    CHECK(!validate_hybrid_route_bindings(
        invalid_binding_descriptor, route_table, binding).ok());

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
    invalid_record.origin_world_rank = -1;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.destination_world_rank = -1;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.destination_world_rank = 4;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.ingress_world_rank = -1;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.ingress_world_rank = 4;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.origin_source_row = 8;
    CHECK(!validate_hybrid_route_table(
        hybrid, {&invalid_record, 1}, 8, 8, 2).ok());
    invalid_record = diagonal;
    invalid_record.destination_local_expert = -1;
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
    invalid_record = diagonal;
    invalid_record.forwarded_slot = 8;
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

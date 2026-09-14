#pragma once

struct CombineVfScalars {
    std::uint64_t route_record_count;
    std::uint64_t generation;
    std::uint64_t timeout_cycles;
    std::uint64_t num_source_rows;
    std::uint64_t num_input_rows;
    std::uintptr_t local_window_base;
    std::uint32_t expanded_vector_reduce;
    std::uint32_t local_copy_datacopy;
    std::uint32_t direct_local_placement;
    std::uint32_t vector_reduce_tile_elements;
    DirectCombineStage stage;
    bool profile_enabled;
    CoreTiling tiling;
};
// Producer payload is MTE-bound.  A 2 KiB tile reduces the per-record
// MTE2/MTE3 event pairs for the common hidden=7168 case from fourteen to
// seven while preserving the 16-element DataCopy alignment.
constexpr std::uint32_t kCombineProducerVectorTileElements = 1024;
constexpr std::uint32_t kCombineVectorReduceDefaultTileElements = 512;
constexpr std::uint32_t kCombineDataCopyAlignmentElements = 16;
constexpr std::uint32_t kCombineLocalCopyDefaultTileBytes = 32768;
constexpr std::uint32_t kCombineDataCopyAlignmentBytes = 32;
constexpr std::uint64_t kCombineCommonTopk = 8;
constexpr std::uint64_t kCombineCommonHidden = 7168;

enum class CombineProtocolError : std::uint32_t {
    kNone = 0,
    kInvalidLayout,
    kCapacityOverflow,
    kInvalidPrefix,
    kInvalidMetadata,
    kInvalidControl,
    kInvalidHeader,
    kDuplicateRecord,
};

DEEP_EP_ASCEND_SIMT_CALLEE void record_combine_protocol_error(
    const transport::DeviceTransportContext& context,
    transport::DeviceAddress status_address, int peer_rank,
    CombineProtocolError error) {
    transport::DeviceTransportFacade transport(context, 0);
    const std::uint64_t status =
        (static_cast<std::uint64_t>(peer_rank + 1) << 32U) |
        static_cast<std::uint32_t>(error);
    transport.store_release(status_address, status);
#if defined(DEEP_EP_ASCEND_STAGED_URMA) && DEEP_EP_ASCEND_STAGED_URMA
    auto* queue = transport::device::detail::command_queue(context);
    auto* diagnostic = transport::device::detail::diagnostic(queue);
    if (diagnostic != nullptr) {
        const auto observed_transport_error =
            transport::simt::load_observed(
                reinterpret_cast<__gm__ std::uint32_t*>(
                    &diagnostic->error));
        if (observed_transport_error == static_cast<std::uint32_t>(
                transport::DeviceTransportError::kNone)) {
            transport::simt::store_published(
                &diagnostic->backend_status,
                static_cast<std::uint32_t>(error));
            transport::simt::system_fence();
        }
    }
    transport::device::detail::record_error(
        queue, transport::DeviceTransportError::kInvalidProtocol,
        transport::TransportCommandOpcode::kNone, peer_rank, 0);
#endif
}

DEEP_EP_ASCEND_SIMT_CALLEE CombineRecordHeader
load_observed_combine_record_header(
    __gm__ const CombineRecordHeader* header) {
    CombineRecordHeader result{};
    result.abi_version = transport::simt::load_observed(
        &header->abi_version);
    result.struct_size = transport::simt::load_observed(
        &header->struct_size);
    result.origin_token = static_cast<std::int32_t>(
        transport::simt::load_observed(
            reinterpret_cast<__gm__ const std::uint32_t*>(
                &header->origin_token)));
    result.contributor_rank = static_cast<std::int32_t>(
        transport::simt::load_observed(
            reinterpret_cast<__gm__ const std::uint32_t*>(
                &header->contributor_rank)));
    result.master_lane = static_cast<std::int32_t>(
        transport::simt::load_observed(
            reinterpret_cast<__gm__ const std::uint32_t*>(
                &header->master_lane)));
    result.contribution_lane = static_cast<std::int32_t>(
        transport::simt::load_observed(
            reinterpret_cast<__gm__ const std::uint32_t*>(
                &header->contribution_lane)));
    return result;
}

DEEP_EP_ASCEND_SIMT_CALLEE HybridCombineRouteMetadata
load_observed_hybrid_combine_route_metadata(
    __gm__ const HybridCombineRouteMetadata* metadata) {
    HybridCombineRouteMetadata result{};
    result.ingress_slot = transport::simt::load_observed(
        reinterpret_cast<__gm__ const std::uint64_t*>(
            &metadata->ingress_slot));
    result.forwarded_slot = transport::simt::load_observed(
        reinterpret_cast<__gm__ const std::uint64_t*>(
            &metadata->forwarded_slot));
    return result;
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t hybrid_combine_records_for_row(
    __gm__ const std::int32_t* metadata, std::uint64_t num_topk,
    bool expanded, bool allow_multiple_reduction) {
    if (!expanded)
        return 1;
    const std::uint64_t lane_count = combine_expanded_record_count(
        metadata + 2, num_topk, false);
    return allow_multiple_reduction ? (lane_count == 0 ? 0 : 1) : lane_count;
}











__aicore__ inline void direct_combine_producer_expanded_vector_reduce_impl(
    __gm__ const bfloat16_t* x,
    __gm__ const std::int32_t* source_metadata,
    __gm__ const std::uint8_t* workspace,
    std::uintptr_t local_window_base,
    std::uint64_t num_source_rows, std::uint64_t num_input_rows,
    std::uint64_t num_topk, std::uint64_t combine_record_bytes,
    std::uint64_t combine_receive_offset,
    std::uint64_t combine_receive_shard_bytes,
    std::uint64_t combine_staging_offset,
    std::uint64_t combine_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t workspace_rank_values_offset,
    std::uint64_t workspace_slot_offset,
    std::uint64_t combine_producer_tile_rank_count_offset,
    std::uint64_t combine_producer_tile_count,
    int world_rank, int world_size, std::uint64_t hidden_elements,
    std::uint32_t direct_local_placement) {
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const auto payload_plan =
        aicore_combine_expanded_producer_payload_plan(
            hidden_elements, kCombineProducerVectorTileElements,
            kCombineDataCopyAlignmentElements, true);
    if (!payload_plan.valid || payload_plan.vector_elements == 0)
        return;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratch_buffer;
    (void)pipe.InitBuffer(
        input_queue, 1,
        kCombineProducerVectorTileElements * sizeof(bfloat16_t));
    (void)pipe.InitBuffer(
        output_queue, 1,
        kCombineProducerVectorTileElements * sizeof(bfloat16_t));
    (void)pipe.InitBuffer(
        scratch_buffer,
        2 * kCombineProducerVectorTileElements * sizeof(float));

    AscendC::GlobalTensor<std::uint64_t> begins_global;
    AscendC::GlobalTensor<std::int32_t> slots_global;
    AscendC::GlobalTensor<std::uint64_t> tile_prefixes_global;
    AscendC::GlobalTensor<std::int32_t> metadata_global;
    AscendC::GlobalTensor<bfloat16_t> input_global;
    AscendC::GlobalTensor<bfloat16_t> output_global;
    const std::uint32_t row_count =
        static_cast<std::uint32_t>(num_source_rows);
    const std::uint32_t input_row_count =
        static_cast<std::uint32_t>(num_input_rows);
    const std::uint32_t topk = static_cast<std::uint32_t>(num_topk);
    const std::uint32_t rank_count = static_cast<std::uint32_t>(world_size);
    const std::uint32_t tile_count =
        static_cast<std::uint32_t>(combine_producer_tile_count);
    const std::uint64_t metadata_stride = 2 + topk;
    begins_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint64_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            workspace_rank_values_offset),
        rank_count);
    slots_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            workspace_slot_offset),
        row_count);
    tile_prefixes_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint64_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            combine_producer_tile_rank_count_offset),
        static_cast<std::uint64_t>(tile_count) * rank_count);
    metadata_global.SetGlobalBuffer(
        const_cast<__gm__ std::int32_t*>(source_metadata),
        static_cast<std::uint64_t>(row_count) * metadata_stride);

    const std::uint32_t block_index =
        static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_count =
        static_cast<std::uint32_t>(AscendC::GetBlockNum());
    auto scratch = scratch_buffer.Get<float>();
    auto accumulation = scratch;
    auto contribution = scratch[kCombineProducerVectorTileElements];
    for (std::uint32_t row = block_index; row < row_count;
         row += block_count) {
        const std::int32_t local_occurrence = slots_global.GetValue(row);
        if (local_occurrence < 0)
            continue;
        std::uint32_t destination_rank = rank_count - 1;
        for (std::uint32_t rank = 1; rank < rank_count; ++rank) {
            if (row < begins_global.GetValue(rank)) {
                destination_rank = rank - 1;
                break;
            }
        }
        const std::uint32_t tile = row /
            static_cast<std::uint32_t>(kCombineRecordsPerTile);
        if (tile >= tile_count)
            continue;
        const std::uint64_t tile_prefix = tile_prefixes_global.GetValue(
            static_cast<std::uint64_t>(tile) * rank_count + destination_rank);
        const std::uint64_t output_slot = tile_prefix +
            static_cast<std::uint32_t>(local_occurrence);
        if (output_slot >=
            combine_staging_shard_bytes / combine_record_bytes)
            continue;
        const bool direct_local = direct_local_placement != 0 &&
            destination_rank == static_cast<std::uint32_t>(world_rank);
        const std::uint64_t record_region_offset = direct_local ?
            combine_receive_offset : combine_staging_offset;
        const std::uint64_t record_region_shard_bytes = direct_local ?
            combine_receive_shard_bytes : combine_staging_shard_bytes;
        auto* record = reinterpret_cast<__gm__ bfloat16_t*>(
            local_window_base + record_region_offset +
            static_cast<std::uint64_t>(destination_rank) *
                record_region_shard_bytes + output_slot * combine_record_bytes);

        std::int32_t input_rows[32];
        std::uint32_t input_count = 0;
        const std::uint64_t metadata_base =
            static_cast<std::uint64_t>(row) * metadata_stride + 2;
        for (std::uint32_t lane = 0; lane < topk; ++lane) {
            const std::int32_t input_row =
                metadata_global.GetValue(metadata_base + lane);
            if (input_row >= 0 &&
                static_cast<std::uint32_t>(input_row) < input_row_count)
                input_rows[input_count++] = input_row;
        }

        for (std::uint32_t hidden = 0;
             hidden < payload_plan.vector_elements;
             hidden += kCombineProducerVectorTileElements) {
            AscendC::Duplicate(
                accumulation, 0.0F,
                static_cast<std::int32_t>(kCombineProducerVectorTileElements));
            for (std::uint32_t input_index = 0;
                 input_index < input_count; ++input_index) {
                input_global.SetGlobalBuffer(
                    const_cast<__gm__ bfloat16_t*>(x) +
                        static_cast<std::uint64_t>(input_rows[input_index]) *
                            hidden_elements + hidden,
                    kCombineProducerVectorTileElements);
                auto input_local = input_queue.AllocTensor<bfloat16_t>();
                AscendC::DataCopy(
                    input_local, input_global, kCombineProducerVectorTileElements);
                input_queue.EnQue(input_local);
                input_local = input_queue.DeQue<bfloat16_t>();
                AscendC::Cast(
                    contribution, input_local,
                    AscendC::RoundMode::CAST_NONE,
                    kCombineProducerVectorTileElements);
                AscendC::Add(
                    accumulation, accumulation, contribution,
                    static_cast<std::int32_t>(
                        kCombineProducerVectorTileElements));
                input_queue.FreeTensor(input_local);
            }

            auto output_local = output_queue.AllocTensor<bfloat16_t>();
            AscendC::Cast(
                output_local, accumulation,
                AscendC::RoundMode::CAST_RINT,
                kCombineProducerVectorTileElements);
            output_queue.EnQue(output_local);
            output_local = output_queue.DeQue<bfloat16_t>();
            output_global.SetGlobalBuffer(
                record + hidden, kCombineProducerVectorTileElements);
            AscendC::DataCopy(
                output_global, output_local, kCombineProducerVectorTileElements);
            output_queue.FreeTensor(output_local);
        }
    }
}

__aicore__ inline void direct_combine_producer_vector_payload_impl(
    __gm__ const bfloat16_t* x,
    __gm__ const std::uint8_t* workspace,
    std::uintptr_t local_window_base,
    std::uint64_t num_source_rows, std::uint64_t combine_record_bytes,
    std::uint64_t combine_receive_offset,
    std::uint64_t combine_receive_shard_bytes,
    std::uint64_t combine_staging_offset,
    std::uint64_t combine_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t workspace_rank_values_offset,
    std::uint64_t workspace_slot_offset,
    std::uint64_t combine_producer_tile_rank_count_offset,
    std::uint64_t combine_producer_tile_count,
    int world_rank, int world_size, std::uint64_t hidden_elements,
    std::uint32_t direct_local_placement) {
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const auto payload_plan = aicore_combine_producer_payload_copy_plan(
        hidden_elements, kCombineProducerVectorTileElements,
        kCombineDataCopyAlignmentElements, false);
    if (!payload_plan.valid || payload_plan.vector_elements == 0)
        return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> payload_buffer;
    (void)pipe.InitBuffer(
        payload_buffer,
        kCombineProducerVectorTileElements * sizeof(bfloat16_t));
    auto payload_local = payload_buffer.Get<bfloat16_t>();
    AscendC::GlobalTensor<std::uint64_t> begins_global;
    AscendC::GlobalTensor<std::int32_t> slots_global;
    AscendC::GlobalTensor<std::uint64_t> tile_prefixes_global;
    AscendC::GlobalTensor<bfloat16_t> input_global;
    AscendC::GlobalTensor<bfloat16_t> output_global;
    const std::uint32_t row_count =
        static_cast<std::uint32_t>(num_source_rows);
    const std::uint32_t rank_count = static_cast<std::uint32_t>(world_size);
    const std::uint32_t tile_count =
        static_cast<std::uint32_t>(combine_producer_tile_count);
    begins_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint64_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            workspace_rank_values_offset),
        rank_count);
    slots_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            workspace_slot_offset),
        row_count);
    tile_prefixes_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint64_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            combine_producer_tile_rank_count_offset),
        static_cast<std::uint64_t>(tile_count) * rank_count);

    const std::uint32_t block_index =
        static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_count =
        static_cast<std::uint32_t>(AscendC::GetBlockNum());
    for (std::uint32_t row = block_index; row < row_count;
         row += block_count) {
        const std::int32_t local_occurrence = slots_global.GetValue(row);
        if (local_occurrence < 0)
            continue;
        std::uint32_t destination_rank = rank_count - 1;
        for (std::uint32_t rank = 1; rank < rank_count; ++rank) {
            if (row < begins_global.GetValue(rank)) {
                destination_rank = rank - 1;
                break;
            }
        }
        const std::uint32_t tile = row /
            static_cast<std::uint32_t>(kCombineRecordsPerTile);
        if (tile >= tile_count)
            continue;
        const std::uint64_t tile_prefix = tile_prefixes_global.GetValue(
            static_cast<std::uint64_t>(tile) * rank_count + destination_rank);
        const std::uint64_t output_slot = tile_prefix +
            static_cast<std::uint32_t>(local_occurrence);
        if (output_slot >=
            combine_staging_shard_bytes / combine_record_bytes)
            continue;
        const bool direct_local = direct_local_placement != 0 &&
            destination_rank == static_cast<std::uint32_t>(world_rank);
        const std::uint64_t record_region_offset = direct_local ?
            combine_receive_offset : combine_staging_offset;
        const std::uint64_t record_region_shard_bytes = direct_local ?
            combine_receive_shard_bytes : combine_staging_shard_bytes;
        auto* record = reinterpret_cast<__gm__ bfloat16_t*>(
            local_window_base + record_region_offset +
            static_cast<std::uint64_t>(destination_rank) *
                record_region_shard_bytes + output_slot * combine_record_bytes);
        for (std::uint64_t hidden = 0;
             hidden < payload_plan.vector_elements;
             hidden += kCombineProducerVectorTileElements) {
            input_global.SetGlobalBuffer(
                const_cast<__gm__ bfloat16_t*>(x) +
                    static_cast<std::uint64_t>(row) * hidden_elements + hidden,
                kCombineProducerVectorTileElements);
            AscendC::DataCopy(
                payload_local, input_global, kCombineProducerVectorTileElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            output_global.SetGlobalBuffer(
                record + hidden, kCombineProducerVectorTileElements);
            AscendC::DataCopy(
                output_global, payload_local, kCombineProducerVectorTileElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

template <std::uint32_t TileBytes>
__aicore__ inline void direct_combine_producer_local_copy_impl(
    __gm__ std::uint8_t* workspace,
    std::uintptr_t local_window_base,
    std::uint64_t combine_record_bytes,
    std::uint64_t combine_receive_offset,
    std::uint64_t combine_receive_shard_bytes,
    std::uint64_t combine_staging_offset,
    std::uint64_t combine_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t workspace_rank_counts_offset,
    int world_rank) {
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const auto* counts = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_rank_counts_offset);
    const std::uint64_t bytes =
        transport::aicore::load_device(counts + world_rank) *
        combine_record_bytes;
    const auto copy_plan = aicore_combine_local_copy_plan(
        bytes, TileBytes, kCombineDataCopyAlignmentBytes, true);
    if (!copy_plan.valid || copy_plan.vector_bytes == 0)
        return;

    auto* source = reinterpret_cast<__gm__ const std::uint8_t*>(
        combine_staging_shard_address(
            local_window_base, combine_staging_offset,
            static_cast<std::uint64_t>(world_rank),
            combine_staging_shard_bytes));
    auto* destination = reinterpret_cast<__gm__ std::uint8_t*>(
        combine_receive_shard_address(
            local_window_base, combine_receive_offset,
            static_cast<std::uint64_t>(world_rank),
            combine_receive_shard_bytes));
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> payload_buffer;
    (void)pipe.InitBuffer(payload_buffer, TileBytes);
    auto payload_local = payload_buffer.Get<std::uint8_t>();
    AscendC::GlobalTensor<std::uint8_t> input_global;
    AscendC::GlobalTensor<std::uint8_t> output_global;
    const std::uint64_t block_index =
        static_cast<std::uint64_t>(AscendC::GetBlockIdx());
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(AscendC::GetBlockNum());
    for (std::uint64_t byte = block_index * TileBytes;
         byte < copy_plan.vector_bytes;
         byte += block_count * TileBytes) {
        input_global.SetGlobalBuffer(
            const_cast<__gm__ std::uint8_t*>(source) + byte,
            TileBytes);
        output_global.SetGlobalBuffer(
            destination + byte, TileBytes);
        AscendC::DataCopy(
            payload_local, input_global, TileBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopy(
            output_global, payload_local, TileBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}









struct CombineOriginDeviceRecordSource {
    __gm__ const std::uint8_t* receive_base;
    std::uint64_t receive_shard_bytes;
    std::uint64_t record_bytes;
    std::uint64_t header_offset;
    std::uint64_t weight_offset;
    std::uint64_t hidden_elements;
    __gm__ const std::uint64_t* contributor_counts;
    std::uint64_t contributor_count;

    DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t record_count(
        std::uint64_t contributor_rank) const {
        return contributor_rank < contributor_count ?
            contributor_counts[contributor_rank] : 0;
    }

    DEEP_EP_ASCEND_SIMT_CALLEE CombineOriginRecordView record_at(
        std::uint64_t contributor_rank, std::uint64_t slot,
        std::uint64_t hidden) const {
        auto* record = receive_base + contributor_rank * receive_shard_bytes +
            slot * record_bytes;
        auto* header = reinterpret_cast<__gm__ const CombineRecordHeader*>(
            record + header_offset);
        auto* payload = reinterpret_cast<__gm__ const bfloat16_t*>(record);
        auto* weights = reinterpret_cast<__gm__ const float*>(
            record + weight_offset);
        return {load_observed_combine_record_header(header),
                hidden < hidden_elements ?
                    static_cast<float>(payload[hidden]) : 0.0F,
                weights};
    }
};











DEEP_EP_ASCEND_SIMT_CALLEE std::uint32_t topk_compact_owner_ordinal(
    TopkSubgroupMask owner_mask, std::int32_t key) {
    std::uint32_t ordinal = 0;
    TopkSubgroupMask remaining = owner_mask;
    while (remaining != 0) {
        const std::int32_t owner_lane = topk_first_set_lane(remaining);
        const std::int32_t owner_key = asc_shfl(
            key, static_cast<std::uint32_t>(owner_lane),
            kTopkSubgroupWidth);
        if (owner_key < key)
            ++ordinal;
        remaining &= ~(TopkSubgroupMask{1} <<
                       static_cast<std::uint32_t>(owner_lane));
    }
    return ordinal;
}



template <std::uint64_t StaticNumTopk,
          std::uint64_t StaticHiddenElements,
          std::uint32_t TileElements = kCombineVectorTileElements>
__aicore__ inline void direct_combine_epilogue_vector_reduce_impl(
    __gm__ const std::int64_t* combined_topk_indices,
    __gm__ const bfloat16_t* bias_0,
    __gm__ const bfloat16_t* bias_1,
    __gm__ std::uint8_t* workspace,
    __gm__ bfloat16_t* combined_x,
    std::uintptr_t local_window_base,
    int world_size, std::uint64_t num_tokens,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t combine_record_bytes,
    std::uint64_t combine_receive_offset,
    std::uint64_t combine_receive_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t slot_offset,
    std::uint64_t hidden_elements) {
    static_assert(
        (StaticNumTopk == 0) == (StaticHiddenElements == 0),
        "static combine extents must be both zero or both nonzero");
    const std::uint64_t effective_num_topk =
        StaticNumTopk == 0 ? num_topk : StaticNumTopk;
    const std::uint64_t effective_hidden_elements =
        StaticHiddenElements == 0 ? hidden_elements : StaticHiddenElements;
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const std::uint32_t num_tokens_u32 =
        static_cast<std::uint32_t>(num_tokens);
    const std::uint32_t num_experts_u32 =
        static_cast<std::uint32_t>(num_experts);
    const std::uint32_t num_topk_u32 =
        static_cast<std::uint32_t>(effective_num_topk);
    const std::uint32_t hidden_elements_u32 =
        static_cast<std::uint32_t>(effective_hidden_elements);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratch_buffer;
    (void)pipe.InitBuffer(
        input_queue, 2,
        TileElements * sizeof(bfloat16_t));
    (void)pipe.InitBuffer(
        output_queue, 1,
        TileElements * sizeof(bfloat16_t));
    (void)pipe.InitBuffer(
        scratch_buffer,
        2 * TileElements * sizeof(float));

    AscendC::GlobalTensor<std::int64_t> topk_global;
    AscendC::GlobalTensor<std::int32_t> slots_global;
    AscendC::GlobalTensor<bfloat16_t> input_global;
    topk_global.SetGlobalBuffer(
        const_cast<__gm__ std::int64_t*>(combined_topk_indices),
        num_tokens * effective_num_topk);
    slots_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(workspace + slot_offset),
        num_tokens * effective_num_topk);

    const std::uint32_t local_experts =
        num_experts_u32 / static_cast<std::uint32_t>(world_size);
    const std::uint32_t vector_end = hidden_elements_u32 -
        hidden_elements_u32 %
            static_cast<std::uint32_t>(TileElements);
    const std::uint32_t block_index =
        static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_count =
        static_cast<std::uint32_t>(AscendC::GetBlockNum());
    auto scratch = scratch_buffer.Get<float>();
    auto accumulation = scratch;
    auto contribution = scratch[TileElements];

    for (std::uint32_t token = block_index; token < num_tokens_u32;
         token += block_count) {
        std::int32_t contributor_ranks[kTopkSubgroupWidth];
        std::int32_t receive_slots[kTopkSubgroupWidth];
        std::uint32_t contributor_count = 0;
        const std::uint64_t token_base =
            static_cast<std::uint64_t>(token) * num_topk_u32;
        for (int contributor_rank = 0;
             contributor_rank < world_size; ++contributor_rank) {
            std::int32_t receive_slot = -1;
            for (std::uint32_t lane = 0;
                 lane < num_topk_u32; ++lane) {
                const std::int64_t expert =
                    topk_global.GetValue(token_base + lane);
                if (expert >= 0 &&
                    static_cast<std::uint64_t>(expert) < num_experts_u32 &&
                    static_cast<std::uint64_t>(expert) / local_experts ==
                        static_cast<std::uint64_t>(contributor_rank)) {
                    receive_slot = slots_global.GetValue(token_base + lane);
                    break;
                }
            }
            if (receive_slot >= 0) {
                contributor_ranks[contributor_count] = contributor_rank;
                receive_slots[contributor_count] = receive_slot;
                ++contributor_count;
            }
        }

        for (std::uint32_t hidden = 0; hidden < vector_end;
             hidden += TileElements) {
            AscendC::Duplicate(
                accumulation, 0.0F,
                static_cast<std::int32_t>(TileElements));
            if (contributor_count != 0) {
                const std::uintptr_t shard = local_window_base +
                    combine_receive_offset +
                    static_cast<std::uint64_t>(contributor_ranks[0]) *
                        combine_receive_shard_bytes;
                auto* payload = reinterpret_cast<__gm__ bfloat16_t*>(
                    shard + static_cast<std::uint64_t>(receive_slots[0]) *
                        combine_record_bytes);
                input_global.SetGlobalBuffer(
                    payload + hidden, TileElements);
                auto input_local = input_queue.AllocTensor<bfloat16_t>();
                AscendC::DataCopy(input_local, input_global, TileElements);
                input_queue.EnQue(input_local);
            }
            for (std::uint32_t contributor_index = 0;
                 contributor_index < contributor_count;
                 ++contributor_index) {
                if (contributor_index + 1 < contributor_count) {
                    const std::uint32_t next = contributor_index + 1;
                    const std::uintptr_t shard = local_window_base +
                        combine_receive_offset +
                        static_cast<std::uint64_t>(contributor_ranks[next]) *
                            combine_receive_shard_bytes;
                    auto* payload = reinterpret_cast<__gm__ bfloat16_t*>(
                        shard + static_cast<std::uint64_t>(
                            receive_slots[next]) * combine_record_bytes);
                    input_global.SetGlobalBuffer(
                        payload + hidden, TileElements);
                    auto input_local = input_queue.AllocTensor<bfloat16_t>();
                    AscendC::DataCopy(
                        input_local, input_global, TileElements);
                    input_queue.EnQue(input_local);
                }
                auto input_local = input_queue.DeQue<bfloat16_t>();
                AscendC::Cast(
                    contribution, input_local,
                    AscendC::RoundMode::CAST_NONE,
                    TileElements);
                AscendC::Add(
                    accumulation, accumulation, contribution,
                    static_cast<std::int32_t>(
                        TileElements));
                input_queue.FreeTensor(input_local);
            }

            const std::uint64_t logical =
                static_cast<std::uint64_t>(token) * hidden_elements_u32 +
                hidden;
            if (bias_0 != nullptr) {
                input_global.SetGlobalBuffer(
                    const_cast<__gm__ bfloat16_t*>(bias_0) + logical,
                    TileElements);
                auto input_local =
                    input_queue.AllocTensor<bfloat16_t>();
                AscendC::DataCopy(
                    input_local, input_global,
                    TileElements);
                input_queue.EnQue(input_local);
                input_local = input_queue.DeQue<bfloat16_t>();
                AscendC::Cast(
                    contribution, input_local,
                    AscendC::RoundMode::CAST_NONE,
                    TileElements);
                AscendC::Add(
                    accumulation, accumulation, contribution,
                    static_cast<std::int32_t>(
                        TileElements));
                input_queue.FreeTensor(input_local);
            }
            if (bias_1 != nullptr) {
                input_global.SetGlobalBuffer(
                    const_cast<__gm__ bfloat16_t*>(bias_1) + logical,
                    TileElements);
                auto input_local =
                    input_queue.AllocTensor<bfloat16_t>();
                AscendC::DataCopy(
                    input_local, input_global,
                    TileElements);
                input_queue.EnQue(input_local);
                input_local = input_queue.DeQue<bfloat16_t>();
                AscendC::Cast(
                    contribution, input_local,
                    AscendC::RoundMode::CAST_NONE,
                    TileElements);
                AscendC::Add(
                    accumulation, accumulation, contribution,
                    static_cast<std::int32_t>(
                        TileElements));
                input_queue.FreeTensor(input_local);
            }

            auto output_local =
                output_queue.AllocTensor<bfloat16_t>();
            AscendC::Cast(
                output_local, accumulation,
                AscendC::RoundMode::CAST_RINT,
                TileElements);
            output_queue.EnQue(output_local);
            output_local = output_queue.DeQue<bfloat16_t>();
            AscendC::GlobalTensor<bfloat16_t> output_tile;
            output_tile.SetGlobalBuffer(
                combined_x + logical, TileElements);
            AscendC::DataCopy(
                output_tile, output_local,
                TileElements);
            output_queue.FreeTensor(output_local);
        }

    }
}

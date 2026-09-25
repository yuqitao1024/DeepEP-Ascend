#pragma once

#include "dispatch_device_helpers.hpp"

constexpr std::uint32_t kDispatchVectorTileBytes = 512;
// The producer payload path is MTE-bound.  A 2 KiB tile cuts the
// per-record MTE2/MTE3 event pairs from fourteen to four for hidden=7168,
// while leaving the final 1024-byte suffix on the scalar/SIMT path.
constexpr std::uint32_t kDispatchProducerVectorTileBytes = 2048;
constexpr std::uint32_t kDispatchUbPayloadOffset = 0;

__aicore__ inline __ubuf__ std::uint8_t* dispatch_ub_payload() {
    return reinterpret_cast<__ubuf__ std::uint8_t*>(
        kDispatchUbPayloadOffset);
}
__aicore__ inline void dispatch_copy_gm_to_ub(
    __ubuf__ std::uint8_t* destination,
    __gm__ const std::uint8_t* source,
    std::uint32_t bytes) {
    copy_gm_to_ubuf_align_v2(
        destination, const_cast<__gm__ std::uint8_t*>(source), 0, 1,
        bytes, 0, 0, false, 0, bytes, bytes);
}

__aicore__ inline void dispatch_copy_ub_to_gm(
    __gm__ std::uint8_t* destination,
    __ubuf__ std::uint8_t* source,
    std::uint32_t bytes) {
    copy_ubuf_to_gm_align_v2(
        destination, source, 0, 1, bytes, 0, bytes, bytes);
}

DEEP_EP_ASCEND_SIMT_CALLEE void dispatch_simt_poll_nop() {
    transport::simt::poll_nop();
}

DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_store_scale_factor_pack(
    __gm__ std::uint8_t* record, std::uint64_t record_offset,
    __gm__ const std::uint8_t* scale_factors,
    std::uint64_t scale_factor_offset) {
    auto* destination = reinterpret_cast<__gm__ std::uint32_t*>(
        record + record_offset);
    const std::uint32_t value = scale_factors == nullptr ? 0U :
        *reinterpret_cast<__gm__ const std::uint32_t*>(
            scale_factors + scale_factor_offset);
    *destination = value;
}

DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_load_scale_factor_pack(
    __gm__ const std::uint8_t* record, std::uint64_t record_offset,
    __gm__ std::uint8_t* scale_factors,
    std::uint64_t scale_factor_offset) {
    const std::uint32_t value =
        *reinterpret_cast<__gm__ const std::uint32_t*>(
            record + record_offset);
    *reinterpret_cast<__gm__ std::uint32_t*>(
        scale_factors + scale_factor_offset) = value;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool
direct_dispatch_compact_record_coordinates(
    std::uint32_t compact_record,
    __gm__ const std::uint64_t* source_bases,
    __gm__ const std::uint64_t* source_counts,
    std::uint32_t world_size, std::uint32_t* source_rank,
    std::uint32_t* source_slot) {
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        const std::uint64_t base = source_bases[rank];
        if (compact_record >= base &&
            static_cast<std::uint64_t>(compact_record) - base <
                source_counts[rank]) {
            *source_rank = rank;
            *source_slot = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(compact_record) - base);
            return true;
        }
    }
    return false;
}

DEEP_EP_ASCEND_SIMT_CALLEE void record_dispatch_protocol_error(
    const transport::DeviceTransportContext& context,
    transport::DeviceAddress status_address, int world_rank,
    DispatchProtocolStage stage, std::uint64_t generation,
    DispatchProtocolError error, std::uint32_t diagnostic_detail = 0) {
    transport::DeviceTransportFacade transport(context, 0);
    const auto failure = make_dispatch_protocol_failure(
        world_rank, stage, generation, error);
    transport.store_release(status_address, failure.scratch_status);
#if defined(DEEP_EP_ASCEND_STAGED_URMA) && DEEP_EP_ASCEND_STAGED_URMA
    auto* queue = transport::device::detail::command_queue(context);
    transport::device::detail::record_error(
        queue, transport::DeviceTransportError::kInvalidProtocol,
        transport::TransportCommandOpcode::kNone, world_rank, 0,
        failure.backend_status,
        failure.generation |
            (static_cast<std::uint64_t>(diagnostic_detail) << 32U));
#endif
}

DEEP_EP_ASCEND_SIMT_CALLEE void record_dispatch_protocol_error(
    const transport::DeviceTransportContext& context,
    transport::DeviceAddress status_address, int world_rank,
    DispatchProtocolError error) {
    std::uint64_t generation = 0;
#if defined(DEEP_EP_ASCEND_STAGED_URMA) && DEEP_EP_ASCEND_STAGED_URMA
    auto* queue = transport::device::detail::command_queue(context);
    if (queue != nullptr)
        generation = transport::simt::load_observed(&queue->generation);
#endif
    record_dispatch_protocol_error(
        context, status_address, world_rank,
        DispatchProtocolStage::kEpilogue, generation, error);
}

DEEP_EP_ASCEND_SIMT_CALLEE DispatchProtocolError
validate_cached_dispatch_state(
    __gm__ const std::int64_t* topk_indices,
    __gm__ const std::int32_t* destination_slots,
    bool cached, int world_rank, int world_size, std::uint64_t num_tokens,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t shard_capacity,
    __gm__ std::uint64_t* destination_counts,
    __gm__ std::uint64_t* maximum_slots,
    __gm__ std::uint8_t* selected_destinations,
    __gm__ std::int32_t* token_slots) {
    if (num_tokens > shard_capacity || shard_capacity == 0 ||
        shard_capacity >
            static_cast<std::uint64_t>(0x7fffffff) /
                static_cast<std::uint64_t>(world_size))
        return DispatchProtocolError::kCapacityOverflow;
    const std::uint64_t num_local_experts =
        num_experts / static_cast<std::uint64_t>(world_size);
    for (int destination_rank = 0;
         destination_rank < world_size; ++destination_rank) {
        destination_counts[destination_rank] = 0;
        maximum_slots[destination_rank] = 0;
    }

    for (std::uint64_t token = 0; token < num_tokens; ++token) {
        for (int destination_rank = 0;
             destination_rank < world_size; ++destination_rank) {
            selected_destinations[destination_rank] = 0;
            token_slots[destination_rank] = -1;
        }
        for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
            const std::int64_t expert =
                topk_indices[token * num_topk + lane];
            const std::int32_t encoded =
                destination_slots[token * num_topk + lane];
            if (expert == -1) {
                if (cached && encoded != -1)
                    return DispatchProtocolError::kInvalidCachedSlot;
                continue;
            }
            if (expert < 0 ||
                static_cast<std::uint64_t>(expert) >= num_experts)
                return DispatchProtocolError::kInvalidTopk;
            const int destination_rank = static_cast<int>(
                static_cast<std::uint64_t>(expert) / num_local_experts);
            selected_destinations[destination_rank] = 1;
            if (!cached)
                continue;
            if (encoded < 0)
                return DispatchProtocolError::kInvalidCachedSlot;
            const int encoded_rank = decode_dispatch_source_rank(
                encoded, shard_capacity);
            const std::int32_t shard_slot = decode_dispatch_local_index(
                encoded, shard_capacity);
            if (encoded_rank != world_rank || shard_slot < 0 ||
                (token_slots[destination_rank] >= 0 &&
                 token_slots[destination_rank] != shard_slot))
                return DispatchProtocolError::kInvalidCachedSlot;
            token_slots[destination_rank] = shard_slot;
        }

        for (int destination_rank = 0;
             destination_rank < world_size; ++destination_rank) {
            if (!selected_destinations[destination_rank])
                continue;
            if (!cached) {
                ++destination_counts[destination_rank];
                continue;
            }
            const std::int32_t shard_slot = token_slots[destination_rank];
            if (shard_slot < 0)
                return DispatchProtocolError::kInvalidCachedSlot;
            for (std::uint64_t previous = 0; previous < token; ++previous) {
                std::int32_t previous_slot = -1;
                for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
                    const std::int64_t expert =
                        topk_indices[previous * num_topk + lane];
                    if (expert < 0)
                        continue;
                    const int previous_destination = static_cast<int>(
                        static_cast<std::uint64_t>(expert) /
                        num_local_experts);
                    if (previous_destination == destination_rank) {
                        previous_slot = decode_dispatch_local_index(
                            destination_slots[previous * num_topk + lane],
                            shard_capacity);
                        break;
                    }
                }
                if (previous_slot == shard_slot)
                    return DispatchProtocolError::kInvalidCachedSlot;
            }
            ++destination_counts[destination_rank];
            const std::uint64_t slot =
                static_cast<std::uint64_t>(shard_slot);
            if (maximum_slots[destination_rank] < slot + 1)
                maximum_slots[destination_rank] = slot + 1;
        }
    }

    for (int destination_rank = 0;
         destination_rank < world_size; ++destination_rank) {
        if (destination_counts[destination_rank] > shard_capacity ||
            (cached && maximum_slots[destination_rank] !=
                           destination_counts[destination_rank]))
            return cached ? DispatchProtocolError::kInvalidCachedSlot :
                            DispatchProtocolError::kCapacityOverflow;
    }
    return DispatchProtocolError::kNone;
}








DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_write_record(
    __gm__ const std::uint8_t* x,
    __gm__ const std::uint8_t* scale_factors,
    __gm__ const std::int64_t* topk_indices,
    __gm__ const float* topk_weights,
    __gm__ std::uint8_t* communication_buffer,
    std::uintptr_t transport_local_window_base,
    int transport_world_rank, int transport_world_size,
    std::uint32_t token, int destination_rank,
    std::int32_t master_lane, std::int32_t encoded_slot,
    std::uint64_t num_topk, std::uint64_t shard_capacity,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t dispatch_staging_offset,
    std::uint64_t dispatch_staging_shard_bytes,
    std::uint64_t token_stride_bytes,
    std::uint64_t token_hidden_offset,
    std::uint64_t token_hidden_bytes,
    std::uint64_t hidden_copy_begin,
    std::uint64_t token_scale_factor_offset,
    std::uint64_t token_scale_factor_bytes,
    std::uint64_t scale_factor_token_stride,
    std::uint64_t scale_factor_pack_stride,
    std::uint64_t token_topk_index_offset,
    std::uint64_t token_topk_weight_offset,
    std::uint64_t token_source_metadata_offset) {
    const std::int32_t local_slot = decode_dispatch_local_index(
        encoded_slot, shard_capacity);
    __gm__ std::uint8_t* shard = nullptr;
    if (transport_world_size == 1) {
        shard = communication_buffer;
    } else if (destination_rank == transport_world_rank) {
        shard = reinterpret_cast<__gm__ std::uint8_t*>(
            transport_local_window_base + dispatch_receive_offset +
            static_cast<std::uint64_t>(transport_world_rank) *
                dispatch_receive_shard_bytes);
    } else {
        shard = reinterpret_cast<__gm__ std::uint8_t*>(
            transport_local_window_base + dispatch_staging_offset +
            static_cast<std::uint64_t>(destination_rank) *
                dispatch_staging_shard_bytes);
    }
    __gm__ std::uint8_t* record = shard +
        static_cast<std::uint64_t>(local_slot) * token_stride_bytes;
    for (std::uint64_t byte = hidden_copy_begin;
         byte < token_hidden_bytes; ++byte)
        record[token_hidden_offset + byte] =
            x[token * token_hidden_bytes + byte];
    for (std::uint64_t pack = 0;
         pack < token_scale_factor_bytes / 4; ++pack) {
        const std::uint64_t source_offset = scale_factor_byte_offset(
            token, pack, scale_factor_token_stride,
            scale_factor_pack_stride, 4);
        direct_dispatch_store_scale_factor_pack(
            record, token_scale_factor_offset + pack * 4,
            scale_factors, source_offset);
    }
    auto* record_topk = reinterpret_cast<__gm__ std::int64_t*>(
        record + token_topk_index_offset);
    auto* record_weights = reinterpret_cast<__gm__ float*>(
        record + token_topk_weight_offset);
    auto* record_metadata = reinterpret_cast<__gm__ std::int32_t*>(
        record + token_source_metadata_offset);
    const std::uint32_t num_topk_u32 =
        static_cast<std::uint32_t>(num_topk);
    for (std::uint32_t lane = 0; lane < num_topk_u32; ++lane) {
        record_topk[lane] = topk_indices[token * num_topk + lane];
        record_weights[lane] = topk_weights == nullptr ? 0.0F :
            topk_weights[token * num_topk + lane];
        record_metadata[2 + lane] = -1;
    }
    record_metadata[0] = static_cast<std::int32_t>(token);
    record_metadata[1] = master_lane;
}

__aicore__ inline void direct_dispatch_producer_vector_payload_impl(
    __gm__ const std::uint8_t* x,
    __gm__ std::uint8_t* communication_buffer,
    __gm__ const std::uint8_t* workspace,
    __gm__ const std::int32_t* destination_slots,
    std::uintptr_t transport_local_window_base,
    int transport_world_rank, int transport_world_size,
    std::uint64_t num_tokens, std::uint64_t num_topk,
    std::uint64_t shard_capacity,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t dispatch_staging_offset,
    std::uint64_t dispatch_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t dispatch_group_owner_offset,
    std::uint64_t dispatch_group_tile_count,
    std::uint32_t launch_num_threads,
    std::uint64_t pipeline_chunk_begin,
    std::uint64_t pipeline_chunk_end,
    std::uint32_t pipeline_source_chunk,
    std::uint64_t token_stride_bytes,
    std::uint64_t token_hidden_offset,
    std::uint64_t token_hidden_bytes) {
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const std::uint64_t vector_bytes = token_hidden_bytes -
        token_hidden_bytes % kDispatchProducerVectorTileBytes;
    if (vector_bytes == 0)
        return;
    const std::uint32_t num_tokens_u32 =
        static_cast<std::uint32_t>(num_tokens);
    const std::uint32_t num_topk_u32 =
        static_cast<std::uint32_t>(num_topk);
    const std::uint32_t tile_count_u32 =
        static_cast<std::uint32_t>(dispatch_group_tile_count);

    auto* payload_ub = dispatch_ub_payload();
    AscendC::GlobalTensor<std::int32_t> owners_global;
    AscendC::GlobalTensor<std::int32_t> slots_global;
    const std::uint32_t world_size =
        static_cast<std::uint32_t>(transport_world_size);
    owners_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            dispatch_group_owner_offset),
        num_tokens_u32 * world_size);
    slots_global.SetGlobalBuffer(
        const_cast<__gm__ std::int32_t*>(destination_slots),
        num_tokens_u32 * num_topk_u32);

    const std::uint32_t block_index =
        static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_count =
        static_cast<std::uint32_t>(AscendC::GetBlockNum());
    const std::uint32_t subgroups_per_block =
        launch_num_threads / kTopkSubgroupWidth;
    const std::uint32_t tile_stride =
        block_count * subgroups_per_block;
    const std::uint32_t source_tile_begin = pipeline_source_chunk != 0 ?
        static_cast<std::uint32_t>(pipeline_chunk_begin) : 0;
    const std::uint32_t source_tile_end = pipeline_source_chunk != 0 ?
        static_cast<std::uint32_t>(pipeline_chunk_end) : tile_count_u32;
    for (std::uint32_t subgroup = 0; subgroup < subgroups_per_block;
         ++subgroup) {
        for (std::uint32_t tile =
                 source_tile_begin +
                 block_index * subgroups_per_block + subgroup;
             tile < source_tile_end; tile += tile_stride) {
            const std::uint32_t token_begin =
                tile * static_cast<std::uint32_t>(
                    kDispatchGroupingTokensPerTile);
            for (std::uint32_t local_token = 0;
                 local_token < kDispatchGroupingTokensPerTile;
                 ++local_token) {
                const std::uint32_t token = token_begin + local_token;
                if (token >= num_tokens_u32)
                    break;
                for (std::uint32_t rank = 0; rank < world_size; ++rank) {
                    const int destination_rank = static_cast<int>(rank);
                    const std::int32_t owner = owners_global.GetValue(
                        token * world_size + rank);
                    if (owner < 0)
                        continue;
                    const std::int32_t encoded_slot = slots_global.GetValue(
                        token * num_topk_u32 +
                            static_cast<std::uint32_t>(owner));
                    if (encoded_slot < 0)
                        continue;
                    const std::int32_t local_slot =
                        decode_dispatch_local_index(
                            encoded_slot, shard_capacity);
                    if (local_slot < 0 ||
                        (pipeline_source_chunk == 0 &&
                         (static_cast<std::uint64_t>(local_slot) <
                              pipeline_chunk_begin ||
                          static_cast<std::uint64_t>(local_slot) >=
                              pipeline_chunk_end)))
                        continue;
                    __gm__ std::uint8_t* shard = nullptr;
                    if (transport_world_size == 1) {
                        shard = communication_buffer;
                    } else if (destination_rank == transport_world_rank) {
                        shard = reinterpret_cast<__gm__ std::uint8_t*>(
                            transport_local_window_base +
                            dispatch_receive_offset +
                            static_cast<std::uint64_t>(
                                transport_world_rank) *
                                dispatch_receive_shard_bytes);
                    } else {
                        shard = reinterpret_cast<__gm__ std::uint8_t*>(
                            transport_local_window_base +
                            dispatch_staging_offset + rank *
                                dispatch_staging_shard_bytes);
                    }
                    __gm__ std::uint8_t* record = shard +
                        static_cast<std::uint64_t>(local_slot) *
                            token_stride_bytes;
                    for (std::uint64_t byte = 0; byte < vector_bytes;
                         byte += kDispatchProducerVectorTileBytes) {
                        dispatch_copy_gm_to_ub(
                            payload_ub, x + token * token_hidden_bytes + byte,
                            kDispatchProducerVectorTileBytes);
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        dispatch_copy_ub_to_gm(
                            record + token_hidden_offset + byte, payload_ub,
                            kDispatchProducerVectorTileBytes);
                        AscendC::SetFlag<
                            AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                        AscendC::WaitFlag<
                            AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    }
                }
            }
        }
    }
}

__aicore__ inline void direct_dispatch_producer_token_fanout_impl(
    __gm__ const std::uint8_t* x,
    __gm__ std::uint8_t* communication_buffer,
    __gm__ const std::uint8_t* workspace,
    __gm__ const std::int32_t* destination_slots,
    std::uintptr_t transport_local_window_base,
    int transport_world_rank, int transport_world_size,
    std::uint64_t num_tokens, std::uint64_t num_topk,
    std::uint64_t shard_capacity,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t dispatch_staging_offset,
    std::uint64_t dispatch_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t dispatch_group_owner_offset,
    std::uint64_t dispatch_group_tile_count,
    std::uint32_t launch_num_threads,
    std::uint64_t pipeline_chunk_begin,
    std::uint64_t pipeline_chunk_end,
    std::uint32_t pipeline_source_chunk,
    std::uint64_t token_stride_bytes,
    std::uint64_t token_hidden_offset,
    std::uint64_t token_hidden_bytes) {
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const std::uint64_t vector_bytes = token_hidden_bytes -
        token_hidden_bytes % kDispatchTokenFanoutAlignmentBytes;
    if (vector_bytes == 0 ||
        vector_bytes > kDispatchTokenFanoutBufferBytes)
        return;
    const std::uint32_t num_tokens_u32 =
        static_cast<std::uint32_t>(num_tokens);
    const std::uint32_t num_topk_u32 =
        static_cast<std::uint32_t>(num_topk);
    const std::uint32_t tile_count_u32 =
        static_cast<std::uint32_t>(dispatch_group_tile_count);
    const std::uint32_t world_size =
        static_cast<std::uint32_t>(transport_world_size);

    auto* payload_ub = dispatch_ub_payload();
    AscendC::GlobalTensor<std::int32_t> owners_global;
    AscendC::GlobalTensor<std::int32_t> slots_global;
    owners_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            dispatch_group_owner_offset),
        num_tokens_u32 * world_size);
    slots_global.SetGlobalBuffer(
        const_cast<__gm__ std::int32_t*>(destination_slots),
        num_tokens_u32 * num_topk_u32);

    const std::uint32_t block_index =
        static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_count =
        static_cast<std::uint32_t>(AscendC::GetBlockNum());
    const std::uint32_t subgroups_per_block =
        launch_num_threads / kTopkSubgroupWidth;
    const std::uint32_t tile_stride =
        block_count * subgroups_per_block;
    const std::uint32_t source_tile_begin = pipeline_source_chunk != 0 ?
        static_cast<std::uint32_t>(pipeline_chunk_begin) : 0;
    const std::uint32_t source_tile_end = pipeline_source_chunk != 0 ?
        static_cast<std::uint32_t>(pipeline_chunk_end) : tile_count_u32;
    for (std::uint32_t subgroup = 0; subgroup < subgroups_per_block;
         ++subgroup) {
        for (std::uint32_t tile = source_tile_begin +
                 block_index * subgroups_per_block + subgroup;
             tile < source_tile_end; tile += tile_stride) {
            const std::uint32_t token_begin =
                tile * static_cast<std::uint32_t>(
                    kDispatchGroupingTokensPerTile);
            for (std::uint32_t local_token = 0;
                 local_token < kDispatchGroupingTokensPerTile;
                 ++local_token) {
                const std::uint32_t token = token_begin + local_token;
                if (token >= num_tokens_u32)
                    break;
                // Resolve the destination set before starting the source
                // transfer. The resident row is then fanned out without
                // re-reading owner/slot metadata between stores.
                std::int32_t destination_ranks[
                    kDispatchTokenFanoutMaximumWorldSize];
                std::int32_t destination_local_slots[
                    kDispatchTokenFanoutMaximumWorldSize];
                std::uint32_t destination_count = 0;
                for (std::uint32_t rank = 0; rank < world_size; ++rank) {
                    const int destination_rank = static_cast<int>(rank);
                    const std::int32_t owner = owners_global.GetValue(
                        token * world_size + rank);
                    if (owner < 0)
                        continue;
                    const std::int32_t encoded_slot = slots_global.GetValue(
                        token * num_topk_u32 +
                            static_cast<std::uint32_t>(owner));
                    if (encoded_slot < 0)
                        continue;
                    const std::int32_t local_slot =
                        decode_dispatch_local_index(
                            encoded_slot, shard_capacity);
                    if (local_slot < 0 ||
                        (pipeline_source_chunk == 0 &&
                         (static_cast<std::uint64_t>(local_slot) <
                              pipeline_chunk_begin ||
                          static_cast<std::uint64_t>(local_slot) >=
                              pipeline_chunk_end)))
                        continue;
                    destination_ranks[destination_count] =
                        static_cast<std::int32_t>(destination_rank);
                    destination_local_slots[destination_count] = local_slot;
                    ++destination_count;
                }
                dispatch_copy_gm_to_ub(
                    payload_ub,
                    x + static_cast<std::uint64_t>(token) *
                        token_hidden_bytes,
                    static_cast<std::uint32_t>(vector_bytes));
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

                for (std::uint32_t index = 0;
                     index < destination_count; ++index) {
                    const int destination_rank =
                        destination_ranks[index];
                    const std::int32_t local_slot =
                        destination_local_slots[index];
                    __gm__ std::uint8_t* shard = nullptr;
                    if (transport_world_size == 1) {
                        shard = communication_buffer;
                    } else if (destination_rank == transport_world_rank) {
                        shard = reinterpret_cast<__gm__ std::uint8_t*>(
                            transport_local_window_base +
                            dispatch_receive_offset +
                            static_cast<std::uint64_t>(
                                transport_world_rank) *
                                dispatch_receive_shard_bytes);
                    } else {
                        shard = reinterpret_cast<__gm__ std::uint8_t*>(
                            transport_local_window_base +
                            dispatch_staging_offset +
                            static_cast<std::uint64_t>(destination_rank) *
                                dispatch_staging_shard_bytes);
                    }
                    __gm__ std::uint8_t* record = shard +
                        static_cast<std::uint64_t>(local_slot) *
                            token_stride_bytes;
                    dispatch_copy_ub_to_gm(
                        record + token_hidden_offset, payload_ub,
                        static_cast<std::uint32_t>(vector_bytes));
                }
                // All destination records consume the same resident token
                // payload.  Queue their GM stores first, then wait once
                // before the next source load can overwrite UB.
                AscendC::SetFlag<
                    AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<
                    AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }
}

DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_producer_record_body(
    __gm__ const std::uint8_t* x,
    __gm__ const std::uint8_t* scale_factors,
    __gm__ const std::int64_t* topk_indices,
    __gm__ const float* topk_weights,
    __gm__ std::uint8_t* communication_buffer,
    __gm__ std::uint8_t* workspace,
    __gm__ std::int32_t* destination_slots,
    std::uintptr_t transport_local_window_base,
    int transport_world_rank, int transport_world_size,
    CoreModeFlags mode_flags, std::uint32_t use_grouping,
    std::uint64_t num_tokens, std::uint64_t num_experts,
    std::uint64_t num_topk, std::uint64_t shard_capacity,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t dispatch_staging_offset,
    std::uint64_t dispatch_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t dispatch_group_owner_offset,
    std::uint64_t dispatch_group_tile_offset,
    std::uint64_t dispatch_group_tile_count,
    std::uint64_t token_stride_bytes,
    std::uint64_t token_hidden_offset,
    std::uint64_t token_hidden_bytes,
    std::uint64_t hidden_copy_begin,
    std::uint64_t pipeline_chunk_begin,
    std::uint64_t pipeline_chunk_end,
    std::uint32_t pipeline_source_chunk,
    std::uint64_t token_scale_factor_offset,
    std::uint64_t token_scale_factor_bytes,
    std::uint64_t scale_factor_token_stride,
    std::uint64_t scale_factor_pack_stride,
    std::uint64_t token_topk_index_offset,
    std::uint64_t token_topk_weight_offset,
    std::uint64_t token_source_metadata_offset) {
    auto* status = reinterpret_cast<__gm__ std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::simt::load_observed(status) != 0)
        return;
    const std::uint32_t num_tokens_u32 =
        static_cast<std::uint32_t>(num_tokens);
    const std::uint32_t num_experts_u32 =
        static_cast<std::uint32_t>(num_experts);
    const std::uint32_t num_topk_u32 =
        static_cast<std::uint32_t>(num_topk);
    const std::uint32_t tile_count_u32 =
        static_cast<std::uint32_t>(dispatch_group_tile_count);
    const std::uint32_t world_size =
        static_cast<std::uint32_t>(transport_world_size);
    const std::uint32_t num_local_experts = num_experts_u32 / world_size;
    const bool cached = (mode_flags &
        (CoreModeFlags{1} <<
         static_cast<std::uint8_t>(CoreMode::kCached))) != 0;
    const bool grouped = use_grouping != 0 &&
        num_topk <= kTopkSubgroupWidth &&
        transport_world_size <= static_cast<int>(kTopkSubgroupWidth);
    if (grouped) {
        auto* owners = reinterpret_cast<__gm__ const std::int32_t*>(
            workspace + dispatch_group_owner_offset);
        auto* tile_offsets = reinterpret_cast<__gm__ const std::uint64_t*>(
            workspace + dispatch_group_tile_offset);
        const auto grid = direct_subgroup_grid_stride(
            blockIdx.x, threadIdx.x, gridDim.x, blockDim.x,
            kTopkSubgroupWidth);
        const std::uint32_t lane = grid.lane;
        const std::uint32_t source_tile_begin = pipeline_source_chunk != 0 ?
            static_cast<std::uint32_t>(pipeline_chunk_begin) : 0;
        const std::uint32_t source_tile_end = pipeline_source_chunk != 0 ?
            static_cast<std::uint32_t>(pipeline_chunk_end) : tile_count_u32;
        for (std::uint32_t tile = source_tile_begin + grid.first;
             tile < source_tile_end; tile += grid.stride) {
            std::int32_t rank_cursor = 0;
            if (!cached && lane < world_size)
                rank_cursor = static_cast<std::int32_t>(
                    transport::simt::load_observed(
                        &tile_offsets[tile * world_size + lane]));
            const std::uint32_t token_begin =
                tile * static_cast<std::uint32_t>(
                    kDispatchGroupingTokensPerTile);
            for (std::uint32_t local_token = 0;
                 local_token < kDispatchGroupingTokensPerTile;
                 ++local_token) {
                const std::uint32_t token = token_begin + local_token;
                if (token >= num_tokens_u32)
                    break;
                const std::int32_t rank_owner = lane < world_size ?
                    owners[token * world_size + lane] : -1;
                std::int32_t destination_rank = -1;
                std::uint32_t logical_index = 0;
                if (lane < num_topk_u32) {
                    logical_index = token * num_topk_u32 + lane;
                    const std::int64_t expert = topk_indices[logical_index];
                    if (expert >= 0 &&
                        static_cast<std::uint64_t>(expert) < num_experts_u32)
                        destination_rank = static_cast<std::int32_t>(
                            static_cast<std::uint64_t>(expert) /
                            num_local_experts);
                }
                const std::uint32_t source_lane = destination_rank >= 0 ?
                    static_cast<std::uint32_t>(destination_rank) : 0;
                const std::int32_t master_lane = asc_shfl(
                    rank_owner, source_lane, kTopkSubgroupWidth);
                const std::int32_t local_slot = asc_shfl(
                    rank_cursor, source_lane, kTopkSubgroupWidth);
                std::int32_t encoded_slot = -1;
                if (destination_rank >= 0) {
                    encoded_slot = cached ? destination_slots[logical_index] :
                        encode_dispatch_source_index(
                            static_cast<std::uint64_t>(transport_world_rank),
                            shard_capacity,
                            static_cast<std::uint64_t>(local_slot));
                    if (!cached)
                        destination_slots[logical_index] = encoded_slot;
                }
                if (destination_rank >= 0 &&
                    lane == static_cast<std::uint32_t>(master_lane) &&
                    local_slot >= 0 &&
                    (pipeline_source_chunk != 0 ||
                     (static_cast<std::uint64_t>(local_slot) >=
                          pipeline_chunk_begin &&
                      static_cast<std::uint64_t>(local_slot) <
                          pipeline_chunk_end)))
                    direct_dispatch_write_record(
                        x, scale_factors, topk_indices, topk_weights,
                        communication_buffer, transport_local_window_base,
                        transport_world_rank, transport_world_size,
                        token, destination_rank, master_lane, encoded_slot,
                        num_topk, shard_capacity, dispatch_receive_offset,
                        dispatch_receive_shard_bytes, dispatch_staging_offset,
                        dispatch_staging_shard_bytes, token_stride_bytes,
                        token_hidden_offset, token_hidden_bytes,
                        hidden_copy_begin,
                        token_scale_factor_offset, token_scale_factor_bytes,
                        scale_factor_token_stride, scale_factor_pack_stride,
                        token_topk_index_offset, token_topk_weight_offset,
                        token_source_metadata_offset);
                if (!cached && lane < world_size && rank_owner >= 0)
                    ++rank_cursor;
            }
        }
        return;
    }

    const std::uint32_t logical_count = num_tokens_u32 * world_size;
    const auto grid = direct_data_grid_stride(
        blockIdx.x, threadIdx.x, gridDim.x, blockDim.x);
    for (std::uint32_t logical = grid.first; logical < logical_count;
         logical += grid.stride) {
        const std::uint32_t token = logical / world_size;
        const int destination_rank =
            static_cast<int>(logical % world_size);
        std::int32_t master_lane = -1;
        std::int32_t encoded_slot = -1;
        for (std::uint32_t lane = 0; lane < num_topk_u32; ++lane) {
            const std::uint32_t index = token * num_topk_u32 + lane;
            const std::int64_t expert = topk_indices[index];
            if (expert < 0 ||
                static_cast<int>(static_cast<std::uint64_t>(expert) /
                                 num_local_experts) != destination_rank)
                continue;
            if (master_lane < 0) {
                master_lane = static_cast<std::int32_t>(lane);
                encoded_slot = destination_slots[index];
            }
        }
        const auto local_slot = decode_dispatch_local_index(
            encoded_slot, shard_capacity);
        if (master_lane >= 0 && local_slot >= 0 &&
            static_cast<std::uint64_t>(local_slot) >= pipeline_chunk_begin &&
            static_cast<std::uint64_t>(local_slot) < pipeline_chunk_end)
            direct_dispatch_write_record(
                x, scale_factors, topk_indices, topk_weights,
                communication_buffer, transport_local_window_base,
                transport_world_rank, transport_world_size,
                token, destination_rank, master_lane, encoded_slot,
                num_topk, shard_capacity, dispatch_receive_offset,
                dispatch_receive_shard_bytes, dispatch_staging_offset,
                dispatch_staging_shard_bytes, token_stride_bytes,
                token_hidden_offset, token_hidden_bytes, hidden_copy_begin,
                token_scale_factor_offset, token_scale_factor_bytes,
                scale_factor_token_stride, scale_factor_pack_stride,
                token_topk_index_offset, token_topk_weight_offset,
                token_source_metadata_offset);
    }
}







DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_pipeline_stage_simt(
    __gm__ DispatchPipelineState* pipeline,
    DispatchPipelineDiagnosticStage stage, std::uint32_t detail) {
    transport::simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(
            &pipeline->diagnostic_stage),
        static_cast<std::uint32_t>(stage));
    transport::simt::store_published(&pipeline->diagnostic_detail, detail);
}

DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_pipeline_fail_simt(
    __gm__ DispatchPipelineState* pipeline,
    __gm__ DispatchPipelineSlot* slot, __gm__ std::uint8_t* workspace,
    std::uint64_t workspace_status_offset, int world_rank,
    std::uint64_t generation, std::uint32_t detail) {
    auto* status = reinterpret_cast<__gm__ std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::simt::load_observed(status) == 0) {
        const auto failure = make_dispatch_protocol_failure(
            world_rank, DispatchProtocolStage::kProducer, generation,
            DispatchProtocolError::kInvalidControl);
        transport::simt::store_published(status, failure.scratch_status);
    }
    transport::simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(
            &pipeline->terminal_error),
        static_cast<std::uint32_t>(
            transport::DeviceTransportError::kCompletionTimeout));
    transport::simt::store_published(&pipeline->diagnostic_detail, detail);
    if (slot != nullptr)
        transport::simt::store_published(
            reinterpret_cast<__gm__ std::uint32_t*>(&slot->state),
            static_cast<std::uint32_t>(
                DispatchPipelineSlotState::kFailed));
    transport::simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(
            &pipeline->producer_worker_state),
        static_cast<std::uint32_t>(
            DispatchPipelineWorkerState::kFailed));
    transport::simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(
            &pipeline->release_worker_state),
        static_cast<std::uint32_t>(
            DispatchPipelineWorkerState::kFailed));
    transport::simt::system_fence();
}


DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_producer_release_body(
    __gm__ std::uint8_t* communication_buffer,
    __gm__ std::uint8_t* workspace,
    std::uint32_t transport_abi_version,
    std::uint32_t transport_struct_size,
    std::uint64_t transport_capabilities,
    std::uintptr_t transport_local_window_base,
    std::uintptr_t transport_channel_table,
    std::uintptr_t transport_peer_address_table,
    std::uint32_t transport_topology_abi_version,
    std::uint32_t transport_topology_struct_size,
    int transport_world_rank, int transport_world_size,
    int transport_scale_up_rank, int transport_scale_up_size,
    int transport_scale_out_rank, int transport_scale_out_size,
    std::uint32_t transport_scale_up_direct,
    std::uint32_t transport_topology_kind,
    std::uint64_t transport_topology_epoch,
    std::uintptr_t transport_backend_context,
    std::uint64_t generation, std::uint64_t timeout_cycles,
    std::uint64_t dispatch_control_offset,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t dispatch_staging_offset,
    std::uint64_t dispatch_staging_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t workspace_local_count_offset,
    std::uint64_t workspace_rank_counts_offset,
    std::uint64_t dispatch_pipeline_offset,
    std::uint64_t dispatch_pipeline_bytes,
    std::uint64_t dispatch_group_tile_offset,
    std::uint64_t dispatch_group_tile_count,
    std::uint64_t pipeline_chunk_begin,
    std::uint64_t pipeline_chunk_end,
    std::uint32_t pipeline_chunk_index,
    std::uint32_t pipeline_final_chunk,
    std::uint32_t pipeline_source_chunk,
    std::uint64_t token_stride_bytes,
    std::uint32_t release_segment_value) {
#if DEEP_EP_ASCEND_ACQUIRE_DIAGNOSTICS
    __gm__ transport::TransportStageProfile* release_profile = nullptr;
    const std::uint64_t release_profile_start = __asc_simt_vf::clock();
#endif
    if (threadIdx.x != 0)
        return;
    const auto release_segment =
        static_cast<DirectReleaseSegment>(release_segment_value);
    const bool release_all = release_segment == DirectReleaseSegment::kAll;
    const bool release_payload =
        release_all || release_segment == DirectReleaseSegment::kPayload;
    const bool release_control =
        release_all || release_segment == DirectReleaseSegment::kControl;
    const bool release_barrier =
        release_all || release_segment == DirectReleaseSegment::kBarrier;
    const auto context = make_hybrid_dispatch_context(
        transport_abi_version, transport_struct_size,
        transport_capabilities, transport_local_window_base,
        transport_channel_table, transport_peer_address_table,
        transport_topology_abi_version, transport_topology_struct_size,
        transport_world_rank, transport_world_size,
        transport_scale_up_rank, transport_scale_up_size,
        transport_scale_out_rank, transport_scale_out_size,
        transport_scale_up_direct, transport_topology_kind,
        transport_topology_epoch, transport_backend_context);
    transport::DeviceTransportFacade transport(context, 0);
#if DEEP_EP_ASCEND_ACQUIRE_DIAGNOSTICS
    {
        auto* staged = reinterpret_cast<__gm__ transport::StagedTransportContext*>(
            context.backend_context);
        if (context.abi_version == transport::kDeviceTransportAbiVersion &&
            context.struct_size == sizeof(transport::DeviceTransportContext) &&
            staged != nullptr &&
            transport::simt::load_observed(&staged->abi_version) ==
                transport::kTransportCommandAbiVersion &&
            transport::simt::load_observed(&staged->struct_size) ==
                sizeof(transport::StagedTransportContext) &&
            transport::simt::load_observed(&staged->cann_compatibility) ==
                transport::kStagedTransportCannCompatibility &&
            transport::simt::load_observed(&staged->stage_profile) != 0 &&
            transport::simt::load_observed(&staged->stage_profile_bytes) ==
                sizeof(transport::TransportStageProfile)) {
            release_profile = reinterpret_cast<
                __gm__ transport::TransportStageProfile*>(
                staged->stage_profile);
        }
    }
#endif
    const auto status_address = reinterpret_cast<transport::DeviceAddress>(
        workspace + workspace_status_offset);
    if (transport.load_acquire(status_address) != 0)
        return;
    auto* destination_counts = reinterpret_cast<__gm__ std::uint64_t*>(
        workspace + workspace_rank_counts_offset);
    auto* tile_offsets = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + dispatch_group_tile_offset);
    const std::uint64_t source_tile_begin = pipeline_chunk_begin;
    const std::uint64_t source_tile_end = pipeline_chunk_end;
    const std::uint64_t world_size =
        static_cast<std::uint64_t>(transport_world_size);
    if (transport_world_size == 1) {
        if (release_control && pipeline_final_chunk != 0)
            transport.store_release(
                reinterpret_cast<transport::DeviceAddress>(
                    workspace + workspace_local_count_offset),
                destination_counts[0]);
        return;
    }

    const bool pipeline_enabled = dispatch_pipeline_bytes ==
        sizeof(DispatchPipelineState);
    __gm__ DispatchPipelineSlot* pipeline_slot = nullptr;
    if (release_payload && pipeline_enabled) {
        auto* pipeline = reinterpret_cast<__gm__ DispatchPipelineState*>(
            workspace + dispatch_pipeline_offset);
        const auto slot_index =
            pipeline_chunk_index % kDispatchPipelineSlotCount;
        transport::simt::store_published(
            &pipeline->abi_version, kDispatchPipelineAbiVersion);
        transport::simt::store_published(
            &pipeline->struct_size,
            static_cast<std::uint32_t>(sizeof(DispatchPipelineState)));
        transport::simt::store_published(
            &pipeline->generation, generation);
        if (pipeline_chunk_index == 0) {
            transport::simt::store_published(
                &pipeline->chunk_slots,
                pipeline_chunk_end - pipeline_chunk_begin);
            transport::simt::store_published(
                &pipeline->completed_chunks, 0U);
            transport::simt::store_published(
                reinterpret_cast<__gm__ std::uint32_t*>(
                    &pipeline->terminal_error),
                static_cast<std::uint32_t>(
                    transport::DeviceTransportError::kNone));
        }
        if (pipeline_final_chunk != 0)
            transport::simt::store_published(
                &pipeline->chunk_count, pipeline_chunk_index + 1U);
        pipeline_slot = &pipeline->slots[slot_index];
        pipeline_slot->request.abi_version =
            transport::kDeviceRequestAbiVersion;
        pipeline_slot->request.state =
            transport::DeviceRequestState::kEmpty;
        pipeline_slot->request.command_begin = 0;
        pipeline_slot->request.command_end = 0;
        pipeline_slot->request.queue_generation = 0;
        pipeline_slot->request.consumed_target = 0;
        pipeline_slot->request.terminal_error =
            transport::DeviceTransportError::kNone;
        pipeline_slot->chunk_begin = pipeline_chunk_begin;
        pipeline_slot->chunk_end = pipeline_chunk_end;
        pipeline_slot->chunk_index = pipeline_chunk_index;
        pipeline_slot->state = DispatchPipelineSlotState::kInFlight;
    }

    auto* control_slots = reinterpret_cast<__gm__ DispatchControlSlot*>(
        transport_local_window_base + dispatch_control_offset);
    if (release_payload) {
        for (int destination_rank = 0;
             destination_rank < transport_world_size; ++destination_rank) {
            const std::uint64_t total_count =
                destination_counts[destination_rank];
            std::uint64_t chunk_slot_begin = pipeline_chunk_begin;
            std::uint64_t chunk_slot_end = total_count < pipeline_chunk_end ?
                total_count : pipeline_chunk_end;
            if (pipeline_source_chunk != 0) {
                chunk_slot_begin =
                    tile_offsets[source_tile_begin * world_size +
                                 static_cast<std::uint64_t>(destination_rank)];
                chunk_slot_end = source_tile_end ==
                        dispatch_group_tile_count ?
                    total_count :
                    tile_offsets[source_tile_end * world_size +
                                 static_cast<std::uint64_t>(destination_rank)];
            }
            const std::uint64_t count =
                chunk_slot_end > chunk_slot_begin ?
                    chunk_slot_end - chunk_slot_begin : 0;
            if (destination_rank == transport_world_rank) {
                if (release_all && pipeline_final_chunk != 0) {
                    transport.store_release(
                        reinterpret_cast<transport::DeviceAddress>(
                            &control_slots[transport_world_rank].count),
                        total_count);
                    transport.store_release(
                        reinterpret_cast<transport::DeviceAddress>(
                            &control_slots[transport_world_rank].generation),
                        generation);
                }
                continue;
            }
            transport::TeamPeer route{};
            if (!transport::device::detail::
                    checked_device_team_peer_for_world_rank(
                        context.topology, destination_rank, &route)) {
                record_dispatch_protocol_error(
                    context, status_address, destination_rank,
                    DispatchProtocolStage::kProducer, generation,
                    DispatchProtocolError::kInvalidControl);
                return;
            }
            if (count != 0) {
                const auto source = static_cast<transport::DeviceAddress>(
                        transport_local_window_base + dispatch_staging_offset +
                        static_cast<std::uint64_t>(destination_rank) *
                            dispatch_staging_shard_bytes +
                    chunk_slot_begin * token_stride_bytes);
                const auto destination = static_cast<transport::DeviceAddress>(
                        transport_local_window_base + dispatch_receive_offset +
                        static_cast<std::uint64_t>(transport_world_rank) *
                            dispatch_receive_shard_bytes +
                    chunk_slot_begin * token_stride_bytes);
                release_protocol::put_staged_records_striped(
                    transport, route, destination, source, count,
                    token_stride_bytes);
            }
        }
        if (pipeline_slot == nullptr) {
            release_protocol::flush_payload(transport);
        } else {
            transport.flush_async(
                transport::TransportTeam::kWorld, transport_world_rank,
                transport::CooperationScope::kDevice,
                &pipeline_slot->request);
        }
    }
    if (release_payload && pipeline_final_chunk == 0)
        return;
    if (release_control && !release_all) {
        transport.store_release(
            reinterpret_cast<transport::DeviceAddress>(
                &control_slots[transport_world_rank].count),
            destination_counts[transport_world_rank]);
        transport.store_release(
            reinterpret_cast<transport::DeviceAddress>(
                &control_slots[transport_world_rank].generation),
            generation);
    }
    if (release_control) {
        for (int destination_rank = 0;
             destination_rank < transport_world_size; ++destination_rank) {
            if (destination_rank == transport_world_rank)
                continue;
            const std::uint64_t count = destination_counts[destination_rank];
            transport::TeamPeer route{};
            if (!transport::device::detail::
                    checked_device_team_peer_for_world_rank(
                        context.topology, destination_rank, &route)) {
                record_dispatch_protocol_error(
                    context, status_address, destination_rank,
                    DispatchProtocolStage::kProducer, generation,
                    DispatchProtocolError::kInvalidControl);
                return;
            }
            const auto remote_slot = static_cast<transport::DeviceAddress>(
                transport_local_window_base + dispatch_control_offset +
                static_cast<std::uint64_t>(transport_world_rank) *
                    sizeof(DispatchControlSlot));
            release_protocol::publish_control_and_release(
                transport, route, remote_slot + sizeof(std::uint64_t), count,
                remote_slot, generation,
                transport::sync_layout::kDispatchReleaseSignalIndex);
#if DEEP_EP_ASCEND_ACQUIRE_DIAGNOSTICS
            if (release_profile != nullptr &&
                destination_rank < 16) {
                transport::simt::store_published(
                    &release_profile->release_peer_publish_cycles[
                        destination_rank],
                    __asc_simt_vf::clock() - release_profile_start);
                transport::simt::store_published(
                    &release_profile->release_peer_publish_count,
                    static_cast<std::uint32_t>(destination_rank + 1));
            }
#endif
        }
    }
    if (release_barrier) {
        if (DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY) {
            // Complete the generation's ordered count/generation/signal WQEs
            // before the service publishes consumed_generation.  The former
            // signal-only path returned with outstanding requests and merely
            // moved that latency into the consumer acquire loop.
            release_protocol::flush_payload(transport);
        } else {
            transport.device_barrier(
                transport::kWorldTeamMask,
                reinterpret_cast<transport::DeviceAddress>(
                    &control_slots[transport_world_rank].generation),
                timeout_cycles);
        }
    }
}

#define DEEP_EP_ASCEND_DISPATCH_RELEASE_ARGUMENTS \
    communication_buffer, workspace, transport_abi_version, \
    transport_struct_size, transport_capabilities, \
    transport_local_window_base, transport_channel_table, \
    transport_peer_address_table, transport_topology_abi_version, \
    transport_topology_struct_size, transport_world_rank, \
    transport_world_size, transport_scale_up_rank, \
    transport_scale_up_size, transport_scale_out_rank, \
    transport_scale_out_size, transport_scale_up_direct, \
    transport_topology_kind, transport_topology_epoch, \
    transport_backend_context, generation, timeout_cycles, \
    dispatch_control_offset, dispatch_receive_offset, \
    dispatch_receive_shard_bytes, dispatch_staging_offset, \
    dispatch_staging_shard_bytes, workspace_status_offset, \
    workspace_local_count_offset, workspace_rank_counts_offset, \
    dispatch_pipeline_offset, dispatch_pipeline_bytes, \
    dispatch_group_tile_offset, dispatch_group_tile_count




template <std::uint32_t TileBytes>
__aicore__ inline void direct_dispatch_epilogue_vector_payload_impl(
    __gm__ std::uint8_t* communication_buffer,
    __gm__ const std::uint8_t* workspace,
    __gm__ std::uint8_t* recv_x,
    __gm__ const std::int32_t* source_metadata,
    std::uintptr_t transport_local_window_base,
    int transport_world_size, bool expanded,
    std::uint64_t num_topk, std::uint64_t shard_capacity,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t workspace_rank_counts_offset,
    std::uint64_t workspace_rank_values_offset,
    std::uint64_t token_stride_bytes,
    std::uint64_t token_hidden_offset,
    std::uint64_t token_hidden_bytes,
    std::uint64_t token_topk_index_offset) {
    const auto* status = reinterpret_cast<__gm__ const std::uint64_t*>(
        workspace + workspace_status_offset);
    if (transport::aicore::load_device(status) != 0)
        return;
    const auto consumer_copy_plan = dispatch_consumer_copy_plan(
        token_hidden_bytes, TileBytes, 32);
    if (!consumer_copy_plan.valid || consumer_copy_plan.vector_bytes == 0)
        return;

    auto* payload_ub = dispatch_ub_payload();
    AscendC::GlobalTensor<std::uint64_t> source_counts_global;
    AscendC::GlobalTensor<std::uint64_t> source_bases_global;
    AscendC::GlobalTensor<std::int32_t> source_metadata_global;
    AscendC::GlobalTensor<std::int64_t> record_topk_global;
    const std::uint32_t num_topk_u32 =
        static_cast<std::uint32_t>(num_topk);
    const std::uint32_t shard_capacity_u32 =
        static_cast<std::uint32_t>(shard_capacity);
    const std::uint32_t world_size =
        static_cast<std::uint32_t>(transport_world_size);
    source_counts_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint64_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            workspace_rank_counts_offset),
        world_size);
    source_bases_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint64_t*>(
            const_cast<__gm__ std::uint8_t*>(workspace) +
            workspace_rank_values_offset),
        world_size);
    if (expanded)
        source_metadata_global.SetGlobalBuffer(
            const_cast<__gm__ std::int32_t*>(source_metadata),
            static_cast<std::uint64_t>(world_size) * shard_capacity_u32 *
                (2 + num_topk_u32));

    const std::uint32_t block_index =
        static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_count =
        static_cast<std::uint32_t>(AscendC::GetBlockNum());
    if (world_size == 0)
        return;
    const std::uint32_t last_source_rank = world_size - 1;
    const std::uint32_t total_records = static_cast<std::uint32_t>(
        source_bases_global.GetValue(last_source_rank) +
        source_counts_global.GetValue(last_source_rank));
    const std::uint32_t copies_per_record =
        expanded ? num_topk_u32 : 1;
    const std::uint32_t logical_count =
        total_records * copies_per_record;
    // Keep two tiles in flight across record boundaries, including when a
    // whole record fits in one tile. Each free-buffer event is consumed before
    // MTE2 overwrites that buffer and produced after MTE3 finishes reading it.
    std::uint32_t buffer_index = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (std::uint32_t logical = block_index; logical < logical_count;
         logical += block_count) {
        const std::uint32_t compact_record =
            logical / copies_per_record;
        const std::uint32_t lane =
            expanded ? logical % copies_per_record : 0;
        std::uint32_t source_rank = 0;
        std::uint32_t source_slot = 0;
        bool found_source = false;
        for (; source_rank < world_size; ++source_rank) {
            const std::uint64_t source_base =
                source_bases_global.GetValue(source_rank);
            if (compact_record >= source_base &&
                static_cast<std::uint64_t>(compact_record) - source_base <
                    source_counts_global.GetValue(source_rank)) {
                source_slot = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(compact_record) - source_base);
                found_source = true;
                break;
            }
        }
        if (!found_source)
            continue;
        const std::uint64_t compact_slot = compact_record;
        __gm__ std::uint8_t* source_shard =
            transport_world_size == 1 ? communication_buffer :
            reinterpret_cast<__gm__ std::uint8_t*>(
                transport_local_window_base + dispatch_receive_offset +
                static_cast<std::uint64_t>(source_rank) *
                    dispatch_receive_shard_bytes);
        __gm__ std::uint8_t* record =
            source_shard + source_slot * token_stride_bytes;
        std::uint64_t destination = compact_slot;
        if (expanded) {
            record_topk_global.SetGlobalBuffer(
                reinterpret_cast<__gm__ std::int64_t*>(
                    record + token_topk_index_offset),
                num_topk_u32);
            const std::int32_t mapped = source_metadata_global.GetValue(
                compact_slot * (2 + num_topk_u32) + 2 + lane);
            if (record_topk_global.GetValue(lane) < 0 || mapped < 0)
                continue;
            destination = static_cast<std::uint64_t>(mapped);
        }
        for (std::uint64_t byte = 0;
             byte < consumer_copy_plan.vector_bytes; byte += TileBytes) {
            const std::uint32_t copy_bytes = static_cast<std::uint32_t>(
                consumer_copy_plan.vector_bytes - byte < TileBytes ?
                    consumer_copy_plan.vector_bytes - byte : TileBytes);
            const auto event_id = buffer_index == 0 ? EVENT_ID0 : EVENT_ID1;
            auto* tile_ub = payload_ub + buffer_index * TileBytes;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
            dispatch_copy_gm_to_ub(
                tile_ub, record + token_hidden_offset + byte, copy_bytes);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
            dispatch_copy_ub_to_gm(
                recv_x + destination * token_hidden_bytes + byte,
                tile_ub, copy_bytes);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
            buffer_index ^= 1;
        }
    }
    // Also consume unused initial events for empty or one-tile workloads.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void direct_dispatch_epilogue_vector_payload_select(
    __gm__ std::uint8_t* communication_buffer,
    __gm__ const std::uint8_t* workspace,
    __gm__ std::uint8_t* recv_x,
    __gm__ const std::int32_t* source_metadata,
    std::uintptr_t transport_local_window_base,
    int transport_world_size, bool expanded,
    std::uint64_t num_topk, std::uint64_t shard_capacity,
    std::uint64_t dispatch_receive_offset,
    std::uint64_t dispatch_receive_shard_bytes,
    std::uint64_t workspace_status_offset,
    std::uint64_t workspace_rank_counts_offset,
    std::uint64_t workspace_rank_values_offset,
    std::uint64_t token_stride_bytes,
    std::uint64_t token_hidden_offset,
    std::uint64_t token_hidden_bytes,
    std::uint64_t token_topk_index_offset,
    std::uint32_t consumer_tile_bytes) {
    switch (consumer_tile_bytes) {
        case 1024:
            direct_dispatch_epilogue_vector_payload_impl<1024>(
                communication_buffer, workspace, recv_x, source_metadata,
                transport_local_window_base, transport_world_size, expanded,
                num_topk, shard_capacity, dispatch_receive_offset,
                dispatch_receive_shard_bytes, workspace_status_offset,
                workspace_rank_counts_offset, workspace_rank_values_offset,
                token_stride_bytes, token_hidden_offset, token_hidden_bytes,
                token_topk_index_offset);
            return;
        case 2048:
            direct_dispatch_epilogue_vector_payload_impl<2048>(
                communication_buffer, workspace, recv_x, source_metadata,
                transport_local_window_base, transport_world_size, expanded,
                num_topk, shard_capacity, dispatch_receive_offset,
                dispatch_receive_shard_bytes, workspace_status_offset,
                workspace_rank_counts_offset, workspace_rank_values_offset,
                token_stride_bytes, token_hidden_offset, token_hidden_bytes,
                token_topk_index_offset);
            return;
        case 4096:
            direct_dispatch_epilogue_vector_payload_impl<4096>(
                communication_buffer, workspace, recv_x, source_metadata,
                transport_local_window_base, transport_world_size, expanded,
                num_topk, shard_capacity, dispatch_receive_offset,
                dispatch_receive_shard_bytes, workspace_status_offset,
                workspace_rank_counts_offset, workspace_rank_values_offset,
                token_stride_bytes, token_hidden_offset, token_hidden_bytes,
                token_topk_index_offset);
            return;
        case 8192:
            direct_dispatch_epilogue_vector_payload_impl<8192>(
                communication_buffer, workspace, recv_x, source_metadata,
                transport_local_window_base, transport_world_size, expanded,
                num_topk, shard_capacity, dispatch_receive_offset,
                dispatch_receive_shard_bytes, workspace_status_offset,
                workspace_rank_counts_offset, workspace_rank_values_offset,
                token_stride_bytes, token_hidden_offset, token_hidden_bytes,
                token_topk_index_offset);
            return;
        default:
            direct_dispatch_epilogue_vector_payload_impl<512>(
                communication_buffer, workspace, recv_x, source_metadata,
                transport_local_window_base, transport_world_size, expanded,
                num_topk, shard_capacity, dispatch_receive_offset,
                dispatch_receive_shard_bytes, workspace_status_offset,
                workspace_rank_counts_offset, workspace_rank_values_offset,
                token_stride_bytes, token_hidden_offset, token_hidden_bytes,
                token_topk_index_offset);
            return;
    }
}

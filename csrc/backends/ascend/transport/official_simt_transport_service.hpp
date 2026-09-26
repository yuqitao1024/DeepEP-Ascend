#pragma once

// Use the public API from the selected CANN installation, including its paired
// internal headers. Do not vendor or modify the official WQE implementation.
#include "c_api/sync/sync.h"
#include "hcomm/hcomm_simt.h"
#include "simt_intrinsics.hpp"
#include "transport_commands.hpp"

namespace deep_ep::ascend::transport::official_simt {

// The executor owns all channels for the duration of a service invocation.
// Exactly one lane posts and drains; no channel is shared with the AICore
// backend in an official-SIMT build. Producer queues and request generations
// keep their existing meaning.
class Executor {
public:
    __simt_callee__ inline Executor(__gm__ StagedTransportContext* staged,
                                    __gm__ TransportCommandQueue* queue,
                                    __gm__ DeviceChannelTable* table,
                                    std::uint64_t local_base,
                                    std::uint32_t scale_up_size,
                                    std::uint32_t scale_out_size)
        : staged_(staged),
          queue_(queue),
          table_(table),
          local_base_(local_base),
          scale_up_size_(scale_up_size),
          scale_out_size_(scale_out_size) {}

    __simt_callee__ inline bool fail(DeviceTransportError error, int peer, std::uint32_t channel, std::int32_t backend_status = 0) {
        auto* output = reinterpret_cast<__gm__ DeviceTransportDiagnostic*>(queue_->diagnostic);
        if (simt::load_observed(reinterpret_cast<__gm__ std::uint32_t*>(&output->error)) == 0) {
            output->command_index = index_;
            output->opcode = opcode_;
            output->peer = peer;
            output->world_peer = peer;
            output->team = TransportTeam::kWorld;
            output->channel = channel;
            output->backend_status = static_cast<std::uint32_t>(backend_status);
            output->generation = queue_->generation;
            asc_threadfence();
            output->error = error;
            asc_threadfence();
        }
        return false;
    }

    __simt_callee__ inline std::uint64_t channel_handle(int peer, std::uint32_t channel) {
        if (peer < 0 || static_cast<std::uint32_t>(peer) >= table_->member_count ||
            static_cast<std::uint32_t>(peer) == table_->self_member || channel >= table_->channel_count)
            return 0;
        auto* handles = reinterpret_cast<__gm__ std::uint64_t*>(table_->channels);
        return simt::load_observed(handles + static_cast<std::uint64_t>(peer) * table_->channel_count + channel);
    }

    __simt_callee__ inline bool checked_result(std::int32_t result, int peer, std::uint32_t channel) {
        return result == 0 || fail(DeviceTransportError::kCompletionFailure, peer, channel, result);
    }

    __simt_callee__ inline bool drain() {
        for (std::uint32_t peer = 0; peer < table_->member_count; ++peer) {
            if (peer == table_->self_member)
                continue;
            for (std::uint32_t channel = 0; channel < table_->channel_count; ++channel) {
                const auto handle = channel_handle(peer, channel);
                if (handle == 0)
                    return fail(DeviceTransportError::kInvalidChannel, peer, channel);
                if (!checked_result(hcomm_.Drain(handle), peer, channel))
                    return false;
            }
        }
        asc_threadfence();
        return true;
    }

    __simt_callee__ inline std::uint64_t remote_target(int peer, std::uint64_t logical, std::uint64_t bytes) {
        if (logical < local_base_ || bytes > table_->window_bytes || logical - local_base_ > table_->window_bytes - bytes)
            return 0;
        auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(table_->remote_bases);
        const auto base = simt::load_observed(bases + peer);
        return base == 0 ? 0 : base + logical - local_base_;
    }

    __simt_callee__ inline bool add(int peer, std::uint64_t target, std::uint64_t value) {
        const auto handle = channel_handle(peer, 0);
        const auto offset = static_cast<std::uint64_t>(peer) * sizeof(std::uint64_t);
        // Standalone barrier contexts use their registered sync window for
        // atomic fetch results, just as the native service does.
        const auto fetch = staged_->fetch_results != 0 && offset + sizeof(std::uint64_t) <= staged_->fetch_result_bytes
            ? staged_->fetch_results + offset
            : table_->local_sync_base + offset;
        if (handle == 0 || fetch == 0 || (staged_->fetch_results == 0 && offset + sizeof(std::uint64_t) > table_->sync_bytes))
            return fail(DeviceTransportError::kInvalidAddress, peer, 0);
        return checked_result(
            hcomm_.AtomicFAA<std::uint64_t>(handle, reinterpret_cast<__gm__ void*>(target), reinterpret_cast<__gm__ void*>(fetch), value),
            peer,
            0);
    }

    __simt_callee__ inline bool in_phase(std::uint32_t peer, std::uint32_t phase) {
        if (peer == table_->self_member)
            return false;
        return phase == 0 ? peer % scale_up_size_ == table_->self_member % scale_up_size_
                          : peer / scale_up_size_ == table_->self_member / scale_up_size_;
    }

    __simt_callee__ inline bool barrier(__gm__ const TransportCommand* current, __gm__ TransportServiceState* state) {
        if (table_->member_count > 64 || scale_up_size_ == 0 || table_->sync_bytes < sync_layout::sync_window_bytes(table_->member_count))
            return fail(DeviceTransportError::kInvalidProtocol, 0, 0);
        const auto generation = simt::load_observed(&state->barrier_generation) + 1;
        auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(table_->remote_sync_bases);
        // Preserve the existing scale-out then scale-up barrier protocol.
        for (std::uint32_t phase = 0; phase < 2; ++phase) {
            const auto mask = phase == 0 ? kScaleOutTeamMask : kScaleUpTeamMask;
            const auto size = phase == 0 ? scale_out_size_ : scale_up_size_;
            if ((current->options & mask) == 0 || size <= 1)
                continue;
            const auto row = (static_cast<std::uint64_t>(sync_layout::kLogicalSignalCount) + phase) * table_->member_count;
            std::uint64_t pending = 0;
            for (std::uint32_t peer = 0; peer < table_->member_count; ++peer) {
                if (!in_phase(peer, phase))
                    continue;
                pending |= std::uint64_t{1} << peer;
                const auto address = simt::load_observed(bases + peer) + (row + table_->self_member) * sizeof(std::uint64_t);
                if (!add(peer, address, 1))
                    return false;
            }
            if (!drain())
                return false;
            const auto begin = static_cast<std::uint64_t>(clock());
            const auto retries = state->default_retry_limit == 0 ? std::uint64_t{1000000} : state->default_retry_limit;
            std::uint64_t retry = 0;
            while (pending != 0) {
                simt::poll_nop();
                for (std::uint32_t peer = 0; peer < table_->member_count; ++peer) {
                    if ((pending & (std::uint64_t{1} << peer)) == 0)
                        continue;
                    const auto address = table_->local_sync_base + (row + peer) * sizeof(std::uint64_t);
                    if (simt::load_observed(reinterpret_cast<__gm__ std::uint64_t*>(address)) >= generation)
                        pending &= ~(std::uint64_t{1} << peer);
                }
                if (pending == 0)
                    break;
                ++retry;
                const bool timed_out = current->timeout_cycles == 0
                    ? retry >= retries
                    : static_cast<std::uint64_t>(clock()) - begin >= current->timeout_cycles;
                if (timed_out) {
                    for (std::uint32_t peer = 0; peer < table_->member_count; ++peer)
                        if ((pending & (std::uint64_t{1} << peer)) != 0)
                            return fail(DeviceTransportError::kCompletionTimeout, peer, 0);
                }
            }
            asc_threadfence();
        }
        simt::store_published(&state->barrier_generation, generation);
        return true;
    }

    __simt_callee__ inline bool post(__gm__ const TransportCommand* current) {
        const int peer = current->world_peer;
        const auto channel = current->channel;
        const auto handle = channel_handle(peer, channel);
        if (handle == 0)
            return fail(DeviceTransportError::kInvalidChannel, peer, channel);
        const auto bytes = current->opcode == TransportCommandOpcode::kPut ? current->bytes : sizeof(std::uint64_t);
        std::uint64_t target = 0;
        if (current->opcode == TransportCommandOpcode::kSignal && current->action_kind != RemoteActionKind::kSignalAdd) {
            if (current->signal_index >= sync_layout::kLogicalSignalCount)
                return fail(DeviceTransportError::kInvalidAddress, peer, channel);
            auto* bases = reinterpret_cast<__gm__ std::uint64_t*>(table_->remote_sync_bases);
            const auto offset =
                (static_cast<std::uint64_t>(current->signal_index) * table_->member_count + table_->self_member) * sizeof(std::uint64_t);
            if (offset + sizeof(std::uint64_t) > table_->sync_bytes)
                return fail(DeviceTransportError::kInvalidAddress, peer, channel);
            target = simt::load_observed(bases + peer) + offset;
        } else {
            const auto logical =
                current->opcode == TransportCommandOpcode::kSignal ? local_base_ + current->symmetric_offset : current->destination;
            target = remote_target(peer, logical, bytes);
        }
        if (target == 0 || bytes > 0xffffffffULL)
            return fail(DeviceTransportError::kInvalidAddress, peer, channel);
        // Immediate commits avoid stranded deferred WQEs at flush/barrier or
        // a full SQ. Batching is a separate experiment, not part of this A/B.
        if (current->opcode == TransportCommandOpcode::kPut)
            return checked_result(
                hcomm_.WriteNbi(handle, reinterpret_cast<__gm__ void*>(target), reinterpret_cast<__gm__ void*>(current->source), bytes),
                peer,
                channel);
        if (current->opcode == TransportCommandOpcode::kPutValue64 ||
            (current->opcode == TransportCommandOpcode::kSignal && current->action_kind == RemoteActionKind::kSignalSet))
            return checked_result(
                hcomm_.WriteValueNbi<std::uint64_t>(handle, reinterpret_cast<__gm__ void*>(target), current->value), peer, channel);
        if (current->opcode == TransportCommandOpcode::kRemoteAdd64 || current->opcode == TransportCommandOpcode::kSignal)
            return add(peer, target, current->value);
        return fail(DeviceTransportError::kUnsupportedOperation, peer, channel);
    }

    __simt_callee__ inline void run() {
        auto* state = reinterpret_cast<__gm__ TransportServiceState*>(queue_->service_state);
        auto* diagnostic = reinterpret_cast<__gm__ DeviceTransportDiagnostic*>(queue_->diagnostic);
        if (simt::load_observed(reinterpret_cast<__gm__ std::uint32_t*>(&diagnostic->error)) != 0)
            return;
        const auto count = simt::load_observed(&queue_->count);
        index_ = simt::load_observed(&state->consumed_count);
        if (index_ > count || simt::load_observed(&state->active) != 0) {
            fail(DeviceTransportError::kInvalidQueue, 0, 0);
            return;
        }
        simt::store_published(&state->active, std::uint32_t{1});
        simt::store_published(&state->consumed_generation, std::uint64_t{0});
        bool success = checked_result(hcomm_.Init(nullptr, 0), 0, 0);
        bool drained = index_ == count;
        auto* commands = reinterpret_cast<__gm__ TransportCommand*>(queue_->commands);
        for (; success && index_ < count; ++index_) {
            auto* current = commands + index_;
            opcode_ = current->opcode;
            if (opcode_ == TransportCommandOpcode::kFlush)
                success = drain();
            else if (opcode_ == TransportCommandOpcode::kBarrier)
                success = barrier(current, state);
            else
                success = post(current);
            drained = opcode_ == TransportCommandOpcode::kFlush || opcode_ == TransportCommandOpcode::kBarrier;
            if (success)
                simt::store_published(&state->consumed_count, index_ + 1);
        }
        if (success && !drained) {
            opcode_ = TransportCommandOpcode::kFlush;
            success = drain();
        }
        asc_threadfence();
        simt::store_published(&state->active, std::uint32_t{0});
        simt::store_published(&state->consumed_generation, success ? queue_->generation : std::uint64_t{0});
        asc_threadfence();
    }

private:
    AscendC::simt::Hcomm<> hcomm_;
    __gm__ StagedTransportContext* staged_;
    __gm__ TransportCommandQueue* queue_;
    __gm__ DeviceChannelTable* table_;
    std::uint64_t local_base_;
    std::uint32_t scale_up_size_, scale_out_size_;
    std::uint32_t index_ = 0;
    TransportCommandOpcode opcode_ = TransportCommandOpcode::kNone;
};

__simt_vf__ inline void execute_vf(__gm__ StagedTransportContext* staged,
                                   __gm__ TransportCommandQueue* queue,
                                   __gm__ DeviceChannelTable* table,
                                   std::uint64_t local_base,
                                   std::uint32_t scale_up_size,
                                   std::uint32_t scale_out_size) {
    if (threadIdx.x == 0) {
        Executor executor(staged, queue, table, local_base, scale_up_size, scale_out_size);
        executor.run();
    }
}

}  // namespace deep_ep::ascend::transport::official_simt

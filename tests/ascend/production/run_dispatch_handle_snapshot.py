"""Real-device regression for verified descriptor snapshots (two ranks)."""
import copy
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import deep_ep
from deep_ep.buffers import elastic


def main():
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(minutes=3))
    group = dist.group.WORLD
    assert dist.get_world_size(group) == 2
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    size = deep_ep.ElasticBuffer.get_buffer_size_hint(
        group, 32, 128, 2, use_fp8_dispatch=False, allow_hybrid_mode=False)
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=size, allow_hybrid_mode=False, explicitly_destroy=True)
    x = torch.ones((32, 128), dtype=torch.bfloat16, device=device)
    topk = torch.tensor([[0, 2]] * 32, dtype=torch.int64, device=device)
    args = dict(topk_idx=topk, num_experts=4, num_max_tokens_per_rank=32,
                num_sms=1, expert_alignment=1)
    original_fingerprint = elastic._ascend_descriptor_fingerprint
    native_runtime = buffer.runtime

    def forbid_second_readback(_):
        raise AssertionError('completion must reuse the verified snapshot')

    def launch(**extra):
        elastic._ascend_descriptor_fingerprint = forbid_second_readback
        try:
            return buffer.dispatch(x, **args, **extra)
        finally:
            elastic._ascend_descriptor_fingerprint = original_fingerprint

    def snapshot(descriptor):
        generation, observed = native_runtime.get_dispatch_handle_snapshot(descriptor)
        return generation, tuple(observed)

    try:
        result = launch()
        handle = result[3]
        descriptor = handle.token_metadata_at_forward
        verified = snapshot(descriptor)
        assert verified == (handle._ascend_generation, handle._ascend_descriptor_fingerprint)
        assert verified[1] == tuple(descriptor.cpu().tolist())
        assert buffer.runtime.get_dispatch_handle_generation(descriptor) == verified[0]
        assert snapshot(descriptor.clone()) == (0, ())
        assert snapshot(descriptor[:-1]) == (0, ())
        assert snapshot(descriptor.view(torch.int32)) == (0, ())

        saved = descriptor.clone()
        descriptor[0].bitwise_xor_(1)
        torch.npu.synchronize()
        assert snapshot(descriptor) == (0, ())
        assert not buffer._reconcile_ascend_handle(handle)
        assert handle._ascend_descriptor_fingerprint == verified[1]
        descriptor.copy_(saved)
        torch.npu.synchronize()
        assert snapshot(descriptor) == verified

        result2 = launch()
        handle2 = result2[3]
        assert handle2._ascend_generation > verified[0]
        assert snapshot(descriptor) == (0, ())
        assert not buffer._reconcile_ascend_handle(handle)

        result3 = launch(async_with_compute_stream=True, allocate_on_comm_stream=True)
        pending_handle = result3[3]
        pending_descriptor = pending_handle.token_metadata_at_forward
        assert snapshot(pending_descriptor) == (0, ())
        copied_event = copy.copy(result3[-1].event)
        copied_event.current_stream_wait()
        committed = snapshot(pending_descriptor)
        assert committed[0] > handle2._ascend_generation
        elastic._ascend_descriptor_fingerprint = forbid_second_readback
        try:
            result3[-1].current_stream_wait()
        finally:
            elastic._ascend_descriptor_fingerprint = original_fingerprint
        assert committed == (pending_handle._ascend_generation,
                             pending_handle._ascend_descriptor_fingerprint)
        assert snapshot(handle2.token_metadata_at_forward) == (0, ())
        buffer.destroy()
        assert snapshot(pending_descriptor) == (0, ())
        print(f'SNAPSHOT_BOUNDARIES_PASSED rank={dist.get_rank()}', flush=True)
    finally:
        elastic._ascend_descriptor_fingerprint = original_fingerprint
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

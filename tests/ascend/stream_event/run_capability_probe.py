import os
import pathlib
import tempfile

import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load


SOURCE = pathlib.Path(__file__).with_name("capability_probe.cpp")


def main():
    cann_root = pathlib.Path(os.environ["ASCEND_HOME_PATH"])
    torch_npu_root = pathlib.Path(torch_npu.__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="deepep-stream-event-") as build_dir:
        module = load(
            name="deepep_stream_event_capability",
            sources=[str(SOURCE)],
            extra_include_paths=[
                str(torch_npu_root / "include"),
                str(cann_root / "aarch64-linux" / "include"),
            ],
            extra_ldflags=[
                f"-L{torch_npu_root / 'lib'}",
                "-ltorch_npu",
                f'-L{cann_root / "aarch64-linux" / "lib64"}',
                "-lascendcl",
            ],
            build_directory=build_dir,
            verbose=True,
        )
        stream_id, device_index, device_type = \
            module.probe_stream_event_capability()
        stream = torch.npu.Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)
        with torch.npu.stream(stream):
            wrapped_tensor = torch.ones((1,), device=f"npu:{device_index}")
        if wrapped_tensor.device.index != device_index:
            raise RuntimeError("Python stream context used the wrong NPU device")

    print("PASS stream_event_capability no_global_sync=1 allocator_record=1 python_stream=1")


if __name__ == "__main__":
    main()

import os
import pathlib
import tempfile

import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBE = pathlib.Path(__file__).with_name("production_compile_probe.cpp")
RUNTIME = ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"


def main():
    cann_root = pathlib.Path(os.environ["ASCEND_HOME_PATH"])
    torch_npu_root = pathlib.Path(torch_npu.__file__).resolve().parent
    with tempfile.TemporaryDirectory(
            prefix="deepep-stream-event-production-") as build_dir:
        module = load(
            name="deepep_stream_event_production_compile",
            sources=[str(PROBE), str(RUNTIME)],
            extra_cflags=["-std=c++17"],
            extra_include_paths=[
                str(ROOT),
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
        if not module.production_stream_event_callbacks_ready():
            raise RuntimeError("production stream event callbacks are unavailable")

    print("PASS stream_event_production_compile callbacks=1")


if __name__ == "__main__":
    main()

import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class DispatchChunkScheduleTest(unittest.TestCase):
    def test_source_chunks_order_release_reuse_epilogue_and_cleanup(self):
        source = (ROOT / "csrc/backends/ascend/elastic/dispatch.asc").read_text()
        start = source.index('extern "C" int deep_ep_ascend_launch_dispatch(')
        end = source.index('\nextern "C" int deep_ep_ascend_launch_dispatch_epilogue(', start)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "dispatch_pipeline_under_test.hpp").write_text(source[start:end])
            binary = path / "probe"
            compiled = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", f"-I{path}",
                 str(ROOT / "tests/ascend/dispatch_chunk_schedule_probe.cpp"),
                 "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

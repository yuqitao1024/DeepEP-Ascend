import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("deep_ep_setup", ROOT / "setup.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class BuildEntrypointTest(unittest.TestCase):
    def test_invalid_platform_fails_before_build_dependencies(self):
        environ = os.environ.copy()
        environ["DEEP_EP_PLATFORM"] = "rocm"

        result = subprocess.run(
            [sys.executable, str(ROOT / "setup.py"), "--name"],
            cwd=ROOT,
            env=environ,
            capture_output=True,
            text=True,
            check=False)

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValueError: DEEP_EP_PLATFORM must be cuda or ascend", output)
        for discovery_failure in ("No module named 'torch'", "CUDA_HOME", "CUDAToolkit",
                                  "find_nccl_root", "find_nvshmem_root"):
            self.assertNotIn(discovery_failure, output)


class BuildPlatformTest(unittest.TestCase):
    def test_default_is_cuda(self):
        self.assertEqual(SETUP.get_build_platform({}), "cuda")

    def test_explicit_platforms(self):
        self.assertEqual(SETUP.get_build_platform({"DEEP_EP_PLATFORM": "cuda"}), "cuda")
        self.assertEqual(SETUP.get_build_platform({"DEEP_EP_PLATFORM": "ascend"}), "ascend")

    def test_invalid_platform_fails_early(self):
        with self.assertRaisesRegex(ValueError, "DEEP_EP_PLATFORM must be cuda or ascend"):
            SETUP.get_build_platform({"DEEP_EP_PLATFORM": "rocm"})

    def test_ascend_extension_uses_the_root_cmake_project(self):
        self.assertTrue(hasattr(SETUP, "CMakeExtension"))
        extension = SETUP.make_extension("ascend")
        self.assertIsInstance(extension, SETUP.CMakeExtension)
        self.assertEqual(extension.name, "deep_ep._C")
        self.assertEqual(extension.cmake_source_dir, str(ROOT))
        self.assertEqual(extension.sources, [])
        self.assertIn(
            "DEEP_EP_ASCEND_TESTING", (ROOT / "setup.py").read_text())

    def test_ascend_cmake_target_has_the_production_source_and_link_graph(self):
        source = (ROOT / "CMakeLists.txt").read_text()
        marker = 'if(DEEP_EP_PLATFORM STREQUAL "ascend")'
        self.assertIn(marker, source)
        self.assertIn("LANGUAGES ASC CXX", source)
        ascend = source[source.index(marker):]

        for production_source in (
                "csrc/python_api.cpp",
                "csrc/backends/ascend/elastic/barrier.asc",
                "csrc/backends/ascend/elastic/dispatch.asc",
                "csrc/backends/ascend/elastic/combine.asc",
                "csrc/backends/ascend/elastic/runtime.cpp",
                "csrc/backends/ascend/runtime/cann_runtime.cpp",
                "csrc/backends/ascend/transport/cann_transport.cpp"):
            self.assertIn(production_source, ascend)
        for marker in (
                "--npu-arch=dav-3510",
                "DEEP_EP_PLATFORM_ASCEND=1", "DEEP_EP_ASCEND_STAGED_URMA=1",
                "DEEP_EP_ASCEND_AICORE_URMA_SERVICE=1",
                "DEEP_EP_ASCEND_TESTING",
                "hcomm", "ascendcl", "c_sec", "torch_npu"):
            self.assertIn(marker, ascend)
        for forbidden in (".cu", "nvshmem", "nccl"):
            self.assertNotIn(forbidden, ascend.lower())


if __name__ == "__main__":
    unittest.main()

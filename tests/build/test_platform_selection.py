import importlib.util
import os
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
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

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required to construct extensions")
    def test_ascend_extension_has_no_cuda_sources_or_libraries(self):
        extension = SETUP.make_extension("ascend")
        self.assertEqual(extension.sources, ["csrc/python_api.cpp"])
        self.assertIn(("DEEP_EP_PLATFORM_ASCEND", "1"), extension.define_macros)
        forbidden = " ".join(extension.sources + list(extension.libraries or []) +
                             list(extension.library_dirs or []))
        for name in ("cuda", "nccl", "nvshmem", "cann", "hccl"):
            self.assertNotIn(name, forbidden.lower())


if __name__ == "__main__":
    unittest.main()

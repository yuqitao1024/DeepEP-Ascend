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
    def test_ascend_extension_is_pure_and_has_exact_host_build_fields(self):
        from torch.utils.cpp_extension import CppExtension

        extension = SETUP.make_extension("ascend")
        dependency_fields = (
            "sources", "include_dirs", "libraries", "library_dirs",
            "extra_compile_args", "extra_link_args", "extra_objects",
            "define_macros", "runtime_library_dirs", "depends",
            "export_symbols", "swig_opts", "undef_macros",
        )

        # CppExtension adds PyTorch's CPU extension defaults. Remove only the
        # values from an equivalent dependency-free construction so the
        # assertions below describe DeepEP's contribution, independently of
        # the installed PyTorch layout.
        baseline = CppExtension(name="deep_ep_baseline", sources=[])

        def without_baseline(field):
            remaining = list(getattr(extension, field, None) or [])
            for value in getattr(baseline, field, None) or []:
                remaining.remove(value)
            return remaining

        normalized = {field: without_baseline(field) for field in dependency_fields}
        self.assertEqual(normalized["sources"], ["csrc/python_api.cpp"])
        self.assertEqual(normalized["include_dirs"], [str(ROOT / "deep_ep" / "include")])
        self.assertEqual(normalized["libraries"], [])
        self.assertEqual(normalized["library_dirs"], [])
        self.assertEqual(
            normalized["extra_compile_args"],
            ["-O3", "-std=c++17", "-Wno-deprecated-declarations"])
        for field in ("extra_link_args", "extra_objects", "runtime_library_dirs",
                      "depends", "export_symbols", "swig_opts", "undef_macros"):
            self.assertEqual(normalized[field], [], field)
        self.assertEqual(normalized["define_macros"], [("DEEP_EP_PLATFORM_ASCEND", "1")])

        forbidden = " ".join(
            str(value)
            for field in dependency_fields
            for value in normalized[field]
        )
        for name in ("cuda", "nccl", "nvshmem", "cann", "hccl"):
            self.assertNotIn(name, forbidden.lower())


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


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

    def test_ascend_testing_mode_normalizes_to_a_boolean_build_flag(self):
        expected = {
            None: "0",
            "": "0",
            "0": "0",
            "ON": "0",
            "true": "0",
            "2": "0",
            "1": "1",
        }
        for value, expected_flag in expected.items():
            with self.subTest(value=value):
                environ = {}
                if value is not None:
                    environ["DEEP_EP_ASCEND_TESTING"] = value
                self.assertEqual(
                    SETUP.get_ascend_testing_mode(environ), expected_flag)

    def test_ascend_configure_passes_the_normalized_testing_flag(self):
        torch = type("Torch", (), {
            "utils": type("Utils", (), {"cmake_prefix_path": "/torch"})})()
        torch_npu = type("TorchNpu", (), {"__file__": "/torch_npu/__init__.py"})()
        extension = SETUP.make_extension("ascend")

        for value, expected_flag in (("unexpected", "0"), ("1", "1")):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory:
                    pybind11_dir = pathlib.Path(directory) / "pybind11" / "cmake"
                    pybind11 = type("Pybind11", (), {
                        "get_cmake_dir": staticmethod(lambda: str(pybind11_dir))})()
                    build = object.__new__(SETUP.CMakeBuild)
                    build.build_temp = directory
                    build.get_ext_fullpath = lambda _: str(
                        pathlib.Path(directory) / "_C.so")
                    with mock.patch.dict(
                            sys.modules, {
                                "torch": torch,
                                "torch_npu": torch_npu,
                                "pybind11": pybind11,
                            }):
                        with mock.patch.object(
                                SETUP.subprocess, "check_call") as check_call:
                            with mock.patch.dict(
                                    os.environ, {"DEEP_EP_ASCEND_TESTING": value},
                                    clear=False):
                                build.build_extension(extension)

                configure = check_call.call_args_list[0].args[0]
                self.assertIn(
                    f"-DDEEP_EP_ASCEND_TESTING={expected_flag}", configure)
                self.assertIn(
                    f"-Dpybind11_DIR={pybind11_dir.resolve()}", configure)

    def test_ascend_configure_pins_pybind_to_build_python(self):
        torch = type("Torch", (), {
            "utils": type("Utils", (), {"cmake_prefix_path": "/torch"})})()
        torch_npu = type(
            "TorchNpu", (), {"__file__": "/torch_npu/__init__.py"})()
        pybind11 = type("Pybind11", (), {
            "get_cmake_dir": staticmethod(lambda: "/pybind11/cmake")})()
        extension = SETUP.make_extension("ascend")

        with tempfile.TemporaryDirectory() as directory:
            build = object.__new__(SETUP.CMakeBuild)
            build.build_temp = directory
            build.get_ext_fullpath = lambda _: str(
                pathlib.Path(directory) / "_C.so")
            with mock.patch.dict(
                    sys.modules, {
                        "torch": torch,
                        "torch_npu": torch_npu,
                        "pybind11": pybind11,
                    }):
                with mock.patch.object(
                        SETUP.subprocess, "check_call") as check_call:
                    with mock.patch.object(
                            SETUP.sys, "executable", "/venv/bin/python"):
                        build.build_extension(extension)

        configure = check_call.call_args_list[0].args[0]
        self.assertIn("-DPYTHON_EXECUTABLE=/venv/bin/python", configure)
        self.assertNotIn("-DPython_EXECUTABLE=/venv/bin/python", configure)

    def test_ascend_build_reports_missing_pybind11(self):
        torch = type("Torch", (), {
            "utils": type("Utils", (), {"cmake_prefix_path": "/torch"})})()
        torch_npu = type("TorchNpu", (), {"__file__": "/torch_npu/__init__.py"})()
        extension = SETUP.make_extension("ascend")

        with tempfile.TemporaryDirectory() as directory:
            build = object.__new__(SETUP.CMakeBuild)
            build.build_temp = directory
            build.get_ext_fullpath = lambda _: str(pathlib.Path(directory) / "_C.so")
            with mock.patch.dict(
                    sys.modules, {
                        "torch": torch,
                        "torch_npu": torch_npu,
                        "pybind11": None,
                    }):
                with self.assertRaisesRegex(
                        RuntimeError, "Ascend builds require pybind11"):
                    build.build_extension(extension)

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
                "DEEP_EP_ASCEND_AICORE_WQE_CALLEE=__aicore__",
                "DEEP_EP_ASCEND_TESTING=$<BOOL:${DEEP_EP_ASCEND_TESTING}>",
                "hcomm", "ascendcl", "c_sec", "torch_npu"):
            self.assertIn(marker, ascend)
        for forbidden in (".cu", "nvshmem", "nccl"):
            self.assertNotIn(forbidden, ascend.lower())

    def test_ascend_pybind_module_disables_pybind_lto_extras_only_for_ascend(self):
        source = (ROOT / "CMakeLists.txt").read_text()
        cuda_marker = 'if(DEEP_EP_PLATFORM STREQUAL "cuda")'
        ascend_marker = 'if(DEEP_EP_PLATFORM STREQUAL "ascend")'
        cuda = source[source.index(cuda_marker):source.index(ascend_marker)]
        ascend = source[source.rindex(ascend_marker):]

        self.assertIn("pybind11_add_module(_C csrc/python_api.cpp)", cuda)
        self.assertRegex(
            ascend,
            r"pybind11_add_module\(\s*_C\s+SHARED\s+NO_EXTRAS\s+"
            r"csrc/python_api\.cpp")


if __name__ == "__main__":
    unittest.main()

import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/platform/config_probe.cpp"


class PlatformConfigTest(unittest.TestCase):
    def compile(self, *definitions: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "config_probe"
            command = ["c++", "-std=c++17", f"-I{ROOT}", *definitions,
                       str(PROBE), "-o", str(binary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                result.stdout = subprocess.check_output([str(binary)], text=True)
            return result

    def test_cuda_only(self):
        result = self.compile("-DDEEP_EP_PLATFORM_CUDA=1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "cuda")

    def test_ascend_only(self):
        result = self.compile("-DDEEP_EP_PLATFORM_ASCEND=1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ascend")

    def test_both_platforms_are_rejected(self):
        result = self.compile("-DDEEP_EP_PLATFORM_CUDA=1",
                              "-DDEEP_EP_PLATFORM_ASCEND=1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one DeepEP platform", result.stderr)

    def test_missing_platform_is_rejected(self):
        result = self.compile()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one DeepEP platform", result.stderr)


if __name__ == "__main__":
    unittest.main()

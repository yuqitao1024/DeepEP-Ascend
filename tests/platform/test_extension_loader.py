import os
import pathlib
import sys
import tempfile
import types
import unittest

from extension_loader import load_extension


class ExtensionLoaderIsolationTest(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "deep_ep", "deep_ep._C")
        }
        self.saved_path = os.environ.get("DEEP_EP_EXTENSION_PATH")
        sys.modules["torch"] = types.ModuleType("torch")

    def tearDown(self):
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if self.saved_path is None:
            os.environ.pop("DEEP_EP_EXTENSION_PATH", None)
        else:
            os.environ["DEEP_EP_EXTENSION_PATH"] = self.saved_path

    def load_temporary_extension(self, source="marker = 'loaded'\n"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = pathlib.Path(directory.name) / "extension.py"
        path.write_text(source)
        os.environ["DEEP_EP_EXTENSION_PATH"] = str(path)
        return load_extension()

    def test_restores_existing_package_modules(self):
        package = types.ModuleType("deep_ep")
        extension = types.ModuleType("deep_ep._C")
        sys.modules["deep_ep"] = package
        sys.modules["deep_ep._C"] = extension

        loaded = self.load_temporary_extension()

        self.assertEqual(loaded.marker, "loaded")
        self.assertIs(sys.modules["deep_ep"], package)
        self.assertIs(sys.modules["deep_ep._C"], extension)

    def test_removes_temporary_package_modules(self):
        sys.modules.pop("deep_ep", None)
        sys.modules.pop("deep_ep._C", None)

        loaded = self.load_temporary_extension()

        self.assertEqual(loaded.marker, "loaded")
        self.assertNotIn("deep_ep", sys.modules)
        self.assertNotIn("deep_ep._C", sys.modules)

    def test_restores_existing_package_modules_after_load_failure(self):
        package = types.ModuleType("deep_ep")
        extension = types.ModuleType("deep_ep._C")
        sys.modules["deep_ep"] = package
        sys.modules["deep_ep._C"] = extension

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.load_temporary_extension("raise RuntimeError('load failed')\n")

        self.assertIs(sys.modules["deep_ep"], package)
        self.assertIs(sys.modules["deep_ep._C"], extension)

    def test_removes_temporary_package_modules_after_load_failure(self):
        sys.modules.pop("deep_ep", None)
        sys.modules.pop("deep_ep._C", None)

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.load_temporary_extension("raise RuntimeError('load failed')\n")

        self.assertNotIn("deep_ep", sys.modules)
        self.assertNotIn("deep_ep._C", sys.modules)


if __name__ == "__main__":
    unittest.main()

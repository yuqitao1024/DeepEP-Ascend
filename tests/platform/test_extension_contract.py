import unittest

from api_contract import (COMMON_BUFFER_METHODS, COMMON_MODULE_NAMES,
                          CUDA_ONLY_BUFFER_METHODS, CUDA_ONLY_MODULE_NAMES)
from extension_loader import load_extension


_C = load_extension()


class ExtensionContractTest(unittest.TestCase):
    def test_platform_name(self):
        self.assertIn(_C.get_platform(), ("cuda", "ascend"))

    def test_common_names(self):
        for name in COMMON_MODULE_NAMES:
            self.assertTrue(hasattr(_C, name), name)
        for name in COMMON_BUFFER_METHODS:
            self.assertTrue(hasattr(_C.ElasticBuffer, name), name)

    def test_platform_specific_names(self):
        if _C.get_platform() == "cuda":
            for name in CUDA_ONLY_MODULE_NAMES:
                self.assertTrue(hasattr(_C, name), name)
            for name in CUDA_ONLY_BUFFER_METHODS:
                self.assertTrue(hasattr(_C.ElasticBuffer, name), name)
        else:
            for name in CUDA_ONLY_MODULE_NAMES:
                self.assertFalse(hasattr(_C, name), name)
            for name in CUDA_ONLY_BUFFER_METHODS:
                self.assertFalse(hasattr(_C.ElasticBuffer, name), name)


if __name__ == "__main__":
    unittest.main()

import unittest

from api_contract import (ASCEND_ELASTIC_BUFFER_METHODS, ASCEND_MODULE_NAMES,
                          CUDA_BUFFER_METHODS, CUDA_CONFIG_METHODS,
                          CUDA_ELASTIC_BUFFER_METHODS,
                          CUDA_EVENT_HANDLE_METHODS, CUDA_MODULE_NAMES)
from extension_loader import load_extension


_C = load_extension()


def public_names(owner):
    return {name for name in vars(owner) if not name.startswith("_")}


class ExtensionContractTest(unittest.TestCase):
    def test_platform_name(self):
        self.assertIn(_C.get_platform(), ("cuda", "ascend"))

    def test_exact_platform_surface(self):
        if _C.get_platform() == "cuda":
            module_names = CUDA_MODULE_NAMES
            buffer_methods = CUDA_ELASTIC_BUFFER_METHODS
        else:
            module_names = ASCEND_MODULE_NAMES
            buffer_methods = ASCEND_ELASTIC_BUFFER_METHODS

        self.assertSetEqual(public_names(_C), module_names)
        self.assertSetEqual(public_names(_C.ElasticBuffer), buffer_methods)

        if _C.get_platform() == "cuda":
            self.assertSetEqual(public_names(_C.EventHandle), CUDA_EVENT_HANDLE_METHODS)
            self.assertSetEqual(public_names(_C.Buffer), CUDA_BUFFER_METHODS)
            self.assertSetEqual(public_names(_C.Config), CUDA_CONFIG_METHODS)


if __name__ == "__main__":
    unittest.main()

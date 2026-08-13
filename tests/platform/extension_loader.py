import importlib.util
import os
import pathlib
import sys
import types


def load_extension():
    try:
        path = pathlib.Path(os.environ["DEEP_EP_EXTENSION_PATH"]).resolve()
    except KeyError as error:
        raise RuntimeError(
            "DEEP_EP_EXTENSION_PATH must name the built DeepEP extension") from error
    if not path.is_file():
        raise RuntimeError(f"Extension does not exist: {path}")

    # Importing torch first makes its shared libraries available to the extension.
    import torch  # noqa: F401

    package = types.ModuleType("deep_ep")
    package.__path__ = []
    sys.modules["deep_ep"] = package
    spec = importlib.util.spec_from_file_location("deep_ep._C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create an extension loader for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deep_ep._C"] = module
    spec.loader.exec_module(module)
    return module

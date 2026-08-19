import runpy
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MATRIX = ROOT / "tests/ascend/production/run_fp8_dispatch_combine.py"


def _round_to_bfloat16(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


class _Tensor:
    def __init__(self, values, dtype):
        self.values = list(values) if isinstance(values, list) else values
        self.dtype = dtype

    def to(self, dtype):
        values = self.values
        if dtype == _Torch.bfloat16:
            if isinstance(values, list):
                values = [_round_to_bfloat16(value) for value in values]
            else:
                values = _round_to_bfloat16(values)
        return _Tensor(values, dtype)

    def sum(self):
        return _Tensor(sum(self.values), self.dtype)

    def item(self):
        return self.values


class _Torch:
    bfloat16 = "bfloat16"
    float32 = "float32"

    @staticmethod
    def tensor(values, dtype):
        return _Tensor(values, dtype).to(dtype)


def test_bf16_combine_reference_quantizes_each_contribution_before_sum():
    runtime = runpy.run_path(str(RUNTIME_MATRIX))

    result = runtime["_bf16_reduction_reference"](_Torch, [301, 302])

    assert result == 600.0

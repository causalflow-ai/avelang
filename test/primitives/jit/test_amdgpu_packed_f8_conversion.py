#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_rocm


SUPPORTED_ARCHES = {"gfx940", "gfx941", "gfx942", "gfx950"}


def supports_packed_f8_conversion() -> bool:
    if not has_rocm():
        return False
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0]
    return arch in SUPPORTED_ARCHES


@avelang.jit
def kernel_packed_f8_conversion(
    src_a: S.Tensor((1,), S.f32),
    src_b: S.Tensor((1,), S.f32),
    old: S.Tensor((1,), S.u32),
    fp8_low: S.Tensor((1,), S.u32),
    bf8_high: S.Tensor((1,), S.u32),
):
    fp8_low[0] = S.amdgpu.cvt_pk_fp8_f32(src_a[0], src_b[0], old[0], 0)
    bf8_high[0] = S.amdgpu.cvt_pk_bf8_f32(src_a[0], src_b[0], old[0], 1)


def packed_pair(src_a: torch.Tensor, src_b: torch.Tensor, dtype: torch.dtype) -> int:
    values = torch.cat((src_a, src_b)).to(dtype).view(torch.uint8)
    return values[0].item() | (values[1].item() << 8)


@unittest.skipUnless(
    supports_packed_f8_conversion(),
    "Requires a gfx940/gfx941/gfx942 ROCm GPU.",
)
class TestAMDGPUPackedF8Conversion(unittest.TestCase):
    def test_packed_fp8_and_bf8_conversion(self):
        src_a = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        src_b = torch.tensor([2.0], dtype=torch.float32, device="cuda")
        old = torch.tensor([0x12345678], dtype=torch.int32, device="cuda")
        fp8_low = torch.empty_like(old)
        bf8_high = torch.empty_like(old)

        kernel_packed_f8_conversion[lambda: ((1, 1, 1), (1, 1, 1))](
            src_a, src_b, old, fp8_low, bf8_high
        )

        old_bits = old.item() & 0xFFFFFFFF
        expected_fp8_low = (old_bits & 0xFFFF0000) | packed_pair(
            src_a, src_b, torch.float8_e4m3fnuz
        )
        expected_bf8_high = (old_bits & 0x0000FFFF) | (
            packed_pair(src_a, src_b, torch.float8_e5m2fnuz) << 16
        )

        self.assertEqual(fp8_low.item() & 0xFFFFFFFF, expected_fp8_low)
        self.assertEqual(bf8_high.item() & 0xFFFFFFFF, expected_bf8_high)


if __name__ == "__main__":
    unittest.main()

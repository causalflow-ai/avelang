"""FP4 scale formats and device dequantization against the Petit reference."""

import unittest
from dataclasses import replace

import avelang
import avelang.language as al
import torch
from avelang.testing import has_rocm
from avelang_kernels.amdgpu.fp4_gemm.config import FP4GemmConfig
from avelang_kernels.amdgpu.fp4_gemm.dequant import make_dequant
from avelang_kernels.amdgpu.fp4_gemm.solution import (
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    SolutionId,
)
from avelang_kernels.amdgpu.fp4_gemm.utils import (
    _petit_format,
    decode_fp4_reference,
    process_fp4_scales,
)

FP4_VARIANTS = (
    (torch.float16, MatmulMfmaType.FP16, MatmulElementB.NVFP4),
    (torch.bfloat16, MatmulMfmaType.BF16, MatmulElementB.NVFP4),
    (torch.bfloat16, MatmulMfmaType.BF16, MatmulElementB.MXFP4),
)


def _assert_float_bits_equal(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    not_nan = ~torch.isnan(expected)
    assert torch.equal(actual[not_nan].view(torch.uint8), expected[not_nan].view(torch.uint8))


def make_dequant_probe(config):
    dequant_scales, dequant_with_scale = make_dequant(config)

    @avelang.jit
    def dequant_probe(
        scales: al.Tensor((256,), al.u16),
        qword: al.Tensor((1,), al.u32),
        decoded_scales: al.Tensor((256,), al.u32),
        decoded_weights: al.Tensor((256, 2, 4), al.u32),
    ):
        idx = al.block_id(0) * 64 + al.thread_id(0)
        weights = al.make_local((4,), al.u32)
        scale_pair = dequant_scales(scales[idx])
        decoded_scales[idx] = scale_pair
        for i in al.range(2):
            scale = al.convert(scale_pair >> (i * 16), al.u16)
            dequant_with_scale(qword[0], scale, weights)
            decoded_weights[idx, i] = weights

    return dequant_probe


class TestAMDGPUFP4Format(unittest.TestCase):
    def test_mxfp4_scales(self):
        qweight = torch.full((64, 128), 0x22, dtype=torch.uint8)  # FP4 = 1
        scales = torch.full((64, 8), 127, dtype=torch.uint8)
        scales[:, 0] = 0
        scales[:, 1] = 254
        scales[:, 2] = 255
        packed = process_fp4_scales(scales, element_b=MatmulElementB.MXFP4)
        self.assertEqual(packed.shape, (4, 64, 2))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertTrue(packed.is_contiguous())
        weight = decode_fp4_reference(qweight, scales, element_b=MatmulElementB.MXFP4)
        self.assertTrue(torch.all(weight[:, :32] == 2.0**-127))
        self.assertTrue(torch.all(weight[:, 32:64] == 2.0**127))
        self.assertTrue(torch.isnan(weight[:, 64:96]).all())
        self.assertTrue(torch.all(weight[:, 96:] == 1.0))


@unittest.skipUnless(has_rocm(), "Requires ROCm")
class TestAMDGPUFP4Dequant(unittest.TestCase):
    def test_matches_petit_packed_dequant(self):
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        # Cover every byte in both positions, including zero/255 and the
        # high-precision exponent overflow at 248/249. No Petit dependency.
        low = torch.arange(256, dtype=torch.int32)
        high = (low * 73 + 19) % 256
        packed = low | (high << 8)
        qword = _petit_format(torch.tensor([0x11111111], dtype=torch.int32))
        for dtype, mfma, element_b in FP4_VARIANTS:
            # Also exercise the FP16 intermediate on the current GPU.
            for intermediate_arch in dict.fromkeys((arch, "gfx90a")):
                for high_precision in (False, True):
                    with self.subTest(
                        dtype=dtype,
                        format=element_b,
                        arch=intermediate_arch,
                        high_precision=high_precision,
                    ):
                        features = MatmulFeatures.GRID
                        if high_precision:
                            features |= MatmulFeatures.HIGH_PRECISION
                        solution = replace(
                            SolutionId.from_int(0x122111040402),
                            element_b=element_b,
                            mfma_type=mfma,
                            features=features,
                        )
                        is_mx = element_b == MatmulElementB.MXFP4
                        bits = (low << 7) | (high << 23)
                        if is_mx and high_precision:
                            bits = bits + 0x03800380
                        scale_dtype = torch.bfloat16 if is_mx else torch.float16
                        expected_scales = bits.view(scale_dtype).reshape(256, 2).float()

                        scales = torch.empty((256,), dtype=torch.uint32, device="cuda")
                        weights = torch.empty((256, 2, 4), dtype=torch.uint32, device="cuda")
                        config = FP4GemmConfig.from_solution(solution, intermediate_arch)
                        make_dequant_probe(config)[lambda: ((4, 1, 1), (64, 1, 1))](
                            packed.to(torch.uint16).cuda(),
                            qword.cuda(),
                            scales,
                            weights,
                        )
                        self.assertTrue(torch.equal(scales.cpu().view(torch.int32), bits))

                        # All eight packed FP4 weights are +0.5. Follow
                        # Petit's bias multiply and final FP16/BF16 packing.
                        bias = 32768.0 if dtype == torch.bfloat16 and intermediate_arch == "gfx942" else 16384.0
                        value = torch.full_like(expected_scales, 0.5 / bias)
                        if high_precision:
                            value = value * (bias / 128.0)
                        expected_weights = value * expected_scales
                        if dtype == torch.bfloat16:
                            expected_weights = (expected_weights.view(torch.int32) >> 16).to(torch.int16).view(dtype)
                        else:
                            expected_weights = expected_weights.to(dtype)
                        expected_weights = expected_weights[..., None].expand(256, 2, 8)
                        actual_weights = weights.cpu().view(dtype).reshape(256, 2, 8)
                        # Very small FP32 products depend on device denormal
                        # handling; boundary overflow/zero/Inf remain covered.
                        check = (
                            (expected_scales.abs() >= 2.0**-80)
                            | (expected_scales == 0)
                            | ~torch.isfinite(expected_scales)
                        )
                        _assert_float_bits_equal(actual_weights[check], expected_weights[check])


if __name__ == "__main__":
    unittest.main()

import unittest

import avelang
import avelang.language as al
import torch
from avelang import testing


@avelang.jit
def convert_f16_bf16(
    half: al.Tensor((16,), al.f16),
    brain: al.Tensor((16,), al.bf16),
    half_out: al.Tensor((16,), al.f16),
    brain_out: al.Tensor((16,), al.bf16),
):
    index = al.thread_id(0)
    half_out[index] = al.convert(brain[index], al.f16)
    brain_out[index] = al.convert(half[index], al.bf16)


@unittest.skipUnless(testing.has_cuda(), "GPU not available.")
class TestConvert(unittest.TestCase):
    def test_f16_bf16_conversion(self):
        # Rounding ties, signed zero, subnormals, overflow, and special values.
        values = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                1 + 2**-8,
                1 + 3 * 2**-8,
                2**-25,
                2**-24,
                2**-14,
                -(2**-24),
                65504,
                1e-30,
                1e30,
                float("inf"),
                -float("inf"),
                float("nan"),
            ],
            dtype=torch.float32,
        )
        half = values.to(torch.float16)
        brain = values.to(torch.bfloat16)
        half_out = torch.empty(16, dtype=torch.float16, device="cuda")
        brain_out = torch.empty(16, dtype=torch.bfloat16, device="cuda")
        convert_f16_bf16[lambda: ((1, 1, 1), (16, 1, 1))](
            half.to("cuda"),
            brain.to("cuda"),
            half_out,
            brain_out,
        )
        for actual, expected in (
            (half_out, brain.to(torch.float16)),
            (brain_out, half.to(torch.bfloat16)),
        ):
            actual = actual.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            # Check signed zero; NaN payloads may differ across targets.
            non_nan = ~torch.isnan(expected)
            torch.testing.assert_close(
                actual.view(torch.int16)[non_nan],
                expected.view(torch.int16)[non_nan],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()

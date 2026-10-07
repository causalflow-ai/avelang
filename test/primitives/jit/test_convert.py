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


@avelang.jit
def convert_vector_floats(
    src: al.Tensor((12, 2), al.f32),
    half_out: al.Tensor((12, 2), al.f16),
    brain_out: al.Tensor((12, 2), al.bf16),
    restored_half: al.Tensor((12, 2), al.f32),
    restored_brain: al.Tensor((12, 2), al.f32),
    half_from_brain: al.Tensor((12, 2), al.f16),
    brain_from_half: al.Tensor((12, 2), al.bf16),
):
    lane = al.thread_id(0)
    values = src[lane]
    half = al.convert(values, al.f16)
    brain = al.convert(values, al.bf16)
    half_out[lane] = half
    brain_out[lane] = brain
    restored_half[lane] = al.convert(half, al.f32)
    restored_brain[lane] = al.convert(brain, al.f32)
    half_from_brain[lane] = al.convert(brain, al.f16)
    brain_from_half[lane] = al.convert(half, al.bf16)


@avelang.jit
def convert_vector_integers(
    signed: al.Tensor((2, 4), al.i16),
    unsigned: al.Tensor((2, 4), al.u16),
    out: al.Tensor((8, 2, 4), al.f32),
):
    lane = al.thread_id(0)
    signed_values = signed[lane]
    unsigned_values = unsigned[lane]
    out[0, lane] = al.convert(signed_values, al.f32)
    out[1, lane] = al.convert(unsigned_values, al.f32)
    out[2, lane] = al.convert(al.convert(signed_values, al.u32), al.f32)
    out[3, lane] = al.convert(al.convert(unsigned_values, al.i16), al.f32)
    out[4, lane] = al.convert(al.convert(unsigned_values, al.u8), al.f32)
    out[5, lane] = al.convert(al.convert(signed_values, al.i8), al.f32)
    out[6, lane] = al.convert(al.convert(signed_values, al.i64), al.f32)
    out[7, lane] = al.convert(al.convert(unsigned_values, al.u64), al.f32)


@avelang.jit
def convert_vector_float_to_integer(
    signed_src: al.Tensor((2, 4), al.f32),
    unsigned_src: al.Tensor((2, 4), al.f64),
    signed: al.Tensor((2, 4), al.i32),
    unsigned: al.Tensor((2, 4), al.u64),
):
    lane = al.thread_id(0)
    signed[lane] = al.convert(signed_src[lane], al.i32)
    unsigned[lane] = al.convert(unsigned_src[lane], al.u64)


@avelang.jit
def convert_matrix_value(
    src: al.Tensor((1, 6), al.f32),
    out: al.Tensor((2, 3), al.f16),
):
    matrix = al.view(src[0], al.Tensor((2, 3), al.f32))
    half = al.convert(matrix, al.f16)
    out[0] = half[0]
    out[1] = half[1]


@unittest.skipUnless(testing.has_cuda(), "GPU not available.")
class TestConvert(unittest.TestCase):
    def assert_float_conversion(self, actual, expected):
        actual = actual.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        # Check signed zero; NaN signs may differ across targets.
        non_nan = ~torch.isnan(expected)
        torch.testing.assert_close(
            torch.signbit(actual[non_nan]),
            torch.signbit(expected[non_nan]),
        )

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
            self.assert_float_conversion(actual, expected)

    def test_vector_float_conversion(self):
        src = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                1 + 2**-11 - 2**-23,
                1 + 2**-11,
                1 + 2**-11 + 2**-23,
                1 + 3 * 2**-11,
                1 + 2**-8 - 2**-23,
                1 + 2**-8,
                1 + 2**-8 + 2**-23,
                1 + 3 * 2**-8,
                2**-25,
                2**-24,
                2**-14,
                -(2**-24),
                65504,
                65520,
                1e-30,
                -1e-30,
                float("inf"),
                -float("inf"),
                float("nan"),
                -123.75,
            ],
            dtype=torch.float32,
        ).reshape(12, 2)
        half_out = torch.empty((12, 2), dtype=torch.float16, device="cuda")
        brain_out = torch.empty((12, 2), dtype=torch.bfloat16, device="cuda")
        restored_half = torch.empty((12, 2), dtype=torch.float32, device="cuda")
        restored_brain = torch.empty_like(restored_half)
        half_from_brain = torch.empty((12, 2), dtype=torch.float16, device="cuda")
        brain_from_half = torch.empty((12, 2), dtype=torch.bfloat16, device="cuda")
        convert_vector_floats[lambda: ((1, 1, 1), (12, 1, 1))](
            src.to("cuda"),
            half_out,
            brain_out,
            restored_half,
            restored_brain,
            half_from_brain,
            brain_from_half,
        )
        half = src.to(torch.float16)
        brain = src.to(torch.bfloat16)
        self.assert_float_conversion(half_out, half)
        self.assert_float_conversion(brain_out, brain)
        self.assert_float_conversion(restored_half, half.float())
        self.assert_float_conversion(restored_brain, brain.float())
        self.assert_float_conversion(half_from_brain, brain.to(torch.float16))
        self.assert_float_conversion(brain_from_half, half.to(torch.bfloat16))

    def test_vector_integer_width_and_signedness(self):
        signed = torch.tensor([-32768, -255, -1, 0, 1, 255, 32766, 32767], dtype=torch.int16).reshape(2, 4)
        unsigned = torch.tensor([0, 1, 255, 32767, 32768, 65533, 65534, 65535], dtype=torch.uint16).reshape(2, 4)
        out = torch.empty((8, 2, 4), dtype=torch.float32, device="cuda")
        convert_vector_integers[lambda: ((1, 1, 1), (2, 1, 1))](signed.to("cuda"), unsigned.to("cuda"), out)
        expected = torch.stack(
            [
                signed.float(),
                unsigned.float(),
                signed.to(torch.uint32).float(),
                unsigned.to(torch.int16).float(),
                unsigned.to(torch.uint8).float(),
                signed.to(torch.int8).float(),
                signed.to(torch.int64).float(),
                unsigned.to(torch.uint64).float(),
            ]
        )
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    def test_vector_float_to_signed_and_unsigned_integer(self):
        signed_src = torch.tensor(
            [-32768.75, -127.5, -1.9, -0.9, 0.9, 1.9, 255.5, 65535.5],
            dtype=torch.float32,
        ).reshape(2, 4)
        unsigned_src = torch.tensor(
            [0, 0.9, 1.9, 127.5, 255.5, 65535.75, 2**32 + 0.75, 2**40 + 0.75],
            dtype=torch.float64,
        ).reshape(2, 4)
        signed = torch.empty((2, 4), dtype=torch.int32, device="cuda")
        unsigned = torch.empty((2, 4), dtype=torch.uint64, device="cuda")
        convert_vector_float_to_integer[lambda: ((1, 1, 1), (2, 1, 1))](
            signed_src.to("cuda"),
            unsigned_src.to("cuda"),
            signed,
            unsigned,
        )
        expected_signed = signed_src.to(torch.int32)
        expected_unsigned = unsigned_src.to(torch.uint64)
        torch.testing.assert_close(signed.cpu(), expected_signed, rtol=0, atol=0)
        torch.testing.assert_close(
            unsigned.cpu().view(torch.int64), expected_unsigned.view(torch.int64), rtol=0, atol=0
        )

    def test_multidimensional_vector_shape(self):
        src = torch.tensor([-3.5, -1.0, 0.0, 1.0, 2.5, 65504.0], dtype=torch.float32).reshape(1, 6)
        out = torch.empty((2, 3), dtype=torch.float16, device="cuda")
        convert_matrix_value[lambda: ((1, 1, 1), (1, 1, 1))](src.to("cuda"), out)
        self.assert_float_conversion(out, src.reshape(2, 3).to(torch.float16))


if __name__ == "__main__":
    unittest.main()

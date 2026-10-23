#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_rocm


@avelang.jit
def identity_unsigned(value: S.u32) -> S.u32:
    return value


@avelang.jit
def mixed_pair(unsigned: S.u32, signed: S.i32) -> (S.u32, S.i32):
    return unsigned, signed


@avelang.jit
def kernel_signedness_metadata(
    inp: S.Tensor((1,), S.u32),
    out: S.Tensor((17,), S.i32),
):
    converted = S.convert(S.convert(1, S.u8), S.i32)
    out[0] = (converted << 31) >> 31

    out[1] = S.amdgpu.readfirstlane(inp[0]) >> 31

    signed = S.convert(-5, S.i16)
    unsigned = S.convert(3, S.u8)
    out[2] = signed + unsigned
    out[3] = unsigned + signed

    out[4] = S.convert(signed | unsigned, S.i32)
    out[5] = S.convert(unsigned | signed, S.i32)
    out[6] = S.convert(signed ^ unsigned, S.i32)
    out[7] = S.convert(unsigned ^ signed, S.i32)
    out[8] = S.convert(signed & unsigned, S.i32)
    out[9] = S.convert(unsigned & signed, S.i32)

    unsigned_value = S.convert(-1, S.u32)
    signed_value = S.convert(-1, S.i32)
    out[10] = identity_unsigned(unsigned_value) >> 31
    pair_unsigned, pair_signed = mixed_pair(unsigned_value, signed_value)
    out[11] = pair_unsigned >> 31
    out[12] = pair_signed >> 31

    out[13] = S.convert(-1, S.i8) >> 1
    out[14] = S.convert(-1, S.i16) >> 1
    out[15] = S.convert(0x80, S.u8) >> 7
    out[16] = S.convert(0x8000, S.u16) >> 15


@avelang.jit
def kernel_shuffle_signedness(out: S.Tensor((256,), S.i32)):
    tid = S.thread_id(0)
    value = S.convert(-1, S.u32)
    base = tid * 4
    out[base] = S.shuffle(value, 0, 32) >> 31
    out[base + 1] = S.shuffle_up(value, 0, 32) >> 31
    out[base + 2] = S.shuffle_down(value, 0, 32) >> 31
    out[base + 3] = S.shuffle_xor(value, 0, 32) >> 31


@avelang.jit
def kernel_integer_promotions(out: S.Tensor((9,), S.i64)):
    out[0] = S.convert(S.convert(255, S.u8) + S.convert(1, S.u8), S.i64)
    out[1] = S.convert(
        S.convert(65535, S.u16) + S.convert(1, S.u16), S.i64
    )
    out[2] = S.convert(
        S.convert(-32768, S.i16) + S.convert(0, S.u16), S.i64
    )
    out[3] = S.convert(S.convert(0, S.u16) + S.convert(-1, S.i32), S.i64)

    wrapped = S.convert(-3, S.u32) + 2
    out[4] = S.convert(wrapped, S.i64)

    out[5] = (S.convert(-1, S.i32) + S.convert(0, S.u32)) >> 31
    out[6] = (S.convert(-1, S.i64) + S.convert(0, S.u32)) >> 63
    out[7] = (S.convert(-1, S.u64) + S.convert(0, S.i32)) >> 63

    # u16 promotes to signed i32 before the left shift, so the following
    # right shift is arithmetic.
    out[8] = (S.convert(0x8000, S.u16) << 16) >> 31


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestSignednessMetadata(unittest.TestCase):
    def test_signedness_survives_conversion_intrinsic_and_promotion(self):
        inp = torch.tensor([-2147483648], dtype=torch.int32, device="cuda")
        out = torch.empty(17, dtype=torch.int32, device="cuda")

        kernel_signedness_metadata[lambda: ((1, 1, 1), (1, 1, 1))](inp, out)

        expected = torch.tensor(
            [
                -1,
                1,
                -2,
                -2,
                -5,
                -5,
                -8,
                -8,
                3,
                3,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
            ],
            dtype=torch.int32,
        )
        actual = out.cpu()
        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_shuffle_results_preserve_unsignedness(self):
        out = torch.empty(256, dtype=torch.int32, device="cuda")

        kernel_shuffle_signedness[lambda: ((1, 1, 1), (64, 1, 1))](out)

        expected = torch.ones(256, dtype=torch.int32)
        actual = out.cpu()
        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_cpp_integer_promotions_and_unsigned_wraparound(self):
        out = torch.empty(9, dtype=torch.int64, device="cuda")

        kernel_integer_promotions[lambda: ((1, 1, 1), (1, 1, 1))](out)

        expected = torch.tensor(
            [
                256,
                65536,
                -32768,
                -1,
                0xFFFFFFFF,
                1,
                -1,
                1,
                -1,
            ],
            dtype=torch.int64,
        )
        actual = out.cpu()
        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()

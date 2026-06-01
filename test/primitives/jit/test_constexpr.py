#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


GLOBAL_SIZE = 16
GLOBAL_OFFSET = S.constexpr(3)
GLOBAL_F32_EXACT = -0.5


@avelang.jit
def add_global(x: S.i32) -> S.i32:
    return x + GLOBAL_OFFSET


@avelang.jit
def kernel_global_constexpr(
    input_data: S.Tensor((GLOBAL_SIZE,), S.i32),
    output_data: S.Tensor((GLOBAL_SIZE,), S.i32),
):
    shared_buf = S.make_shared((GLOBAL_SIZE,), S.i32)
    tid = S.thread_id(0)
    shared_buf[tid] = add_global(input_data[tid])
    S.syncthreads()
    output_data[tid] = shared_buf[GLOBAL_SIZE - 1 - tid]


@avelang.jit
def helper_constexpr_return(
    kHighPrecision: S.constexpr,
    kUseLargeScale: S.constexpr,
) -> S.f32:
    if kHighPrecision:
        return S.convert(1.0, S.f32)

    if kUseLargeScale:
        return S.convert(256.0, S.f32)

    return S.convert(128.0, S.f32)


@avelang.jit
def helper_constexpr_value(kSize: S.constexpr) -> S.u32:
    return S.convert(kSize, S.u32)


@avelang.jit
def kernel_constexpr_return_helper(
    out: S.Tensor((4,), S.f32),
    kHighPrecision: S.constexpr,
    kUseLargeScale: S.constexpr,
):
    out[0] = helper_constexpr_return(kHighPrecision, kUseLargeScale)


@avelang.jit
def kernel_constexpr_return_direct(
    out: S.Tensor((4,), S.f32),
    kHighPrecision: S.constexpr,
    kUseLargeScale: S.constexpr,
):
    scale = S.convert(1.0, S.f32)
    if kHighPrecision:
        scale = S.convert(1.0, S.f32)
    elif kUseLargeScale:
        scale = S.convert(256.0, S.f32)
    else:
        scale = S.convert(128.0, S.f32)

    out[0] = scale


@avelang.jit
def kernel_constexpr_call_expr(
    out: S.Tensor((4,), S.u32),
    kSize: S.constexpr,
):
    out[0] = helper_constexpr_value(kSize + 1)


class TestConstexprResolve(unittest.TestCase):
    def test_global_constexpr_injection(self):
        input_data = torch.arange(GLOBAL_SIZE, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((GLOBAL_SIZE,), dtype=torch.int32, device="cuda")

        expected = (input_data + GLOBAL_OFFSET.value).flip(0)

        kernel_global_constexpr[lambda: ((1, 1, 1), (GLOBAL_SIZE, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )
        
    def test_constexpr_return(self):
        out = torch.zeros((4,), dtype=torch.float32, device="cuda")
        kernel_constexpr_return_helper[lambda: ((1, 1, 1), (1, 1, 1))](
            out,
            False,
            True,
        )
        torch.cuda.synchronize()

        self.assertEqual(out.cpu()[0], 256.0)

    def test_constexpr_direct(self):
        out = torch.zeros((4,), dtype=torch.float32, device="cuda")
        kernel_constexpr_return_direct[lambda: ((1, 1, 1), (1, 1, 1))](
            out,
            False,
            True,
        )
        torch.cuda.synchronize()

        self.assertEqual(out.cpu()[0], 256.0)

    def test_constexpr_call_expr_materializes_in_callee(self):
        out = torch.zeros((4,), dtype=torch.int32, device="cuda")
        kernel_constexpr_call_expr[lambda: ((1, 1, 1), (1, 1, 1))](
            out,
            3,
        )
        torch.cuda.synchronize()

        self.assertEqual(out.cpu()[0], 4)


if __name__ == "__main__":
    unittest.main()

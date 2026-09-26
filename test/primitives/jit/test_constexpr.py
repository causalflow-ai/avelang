#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


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
def helper_constexpr_bool_value(flag: S.constexpr) -> S.u32:
    if flag:
        return S.convert(11, S.u32)
    return S.convert(23, S.u32)


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


@avelang.jit
def kernel_constexpr_multiple_specializations(out: S.Tensor((4,), S.u32)):
    out[0] = helper_constexpr_bool_value(True)
    out[1] = helper_constexpr_bool_value(False)


@avelang.jit
def kernel_constexpr_in_loop(out: S.Tensor((4,), S.i32), flag: S.constexpr):
    for i in S.range(2):
        if flag:
            out[i * 2] = 11
            out[i * 2 + 1] = 12
        else:
            out[i * 2] = 21
            out[i * 2 + 1] = 22


class TestConstexprResolve(unittest.TestCase):
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

    def test_constexpr_branch_in_loop(self):
        for flag, expected in ((True, [11, 12, 11, 12]), (False, [21, 22, 21, 22])):
            with self.subTest(flag=flag):
                out = torch.zeros((4,), dtype=torch.int32, device="cuda")
                kernel_constexpr_in_loop[lambda: ((1, 1, 1), (1, 1, 1))](out, flag)
                self.assertEqual(out.cpu().tolist(), expected)

    def test_constexpr_jit_helper_specializes_by_value(self):
        out = torch.zeros((4,), dtype=torch.int32, device="cuda")
        kernel_constexpr_multiple_specializations[
            lambda: ((1, 1, 1), (1, 1, 1))
        ](out)
        torch.cuda.synchronize()

        self.assertEqual(out.cpu().tolist()[:2], [11, 23])


if __name__ == "__main__":
    unittest.main()

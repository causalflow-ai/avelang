#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def constexpr_passthrough(
    x: S.Tensor((M, N), S.i32),
    out: S.Tensor((M, N), S.i32),
    M: S.constexpr,
    N: S.constexpr,
):
    row = S.thread_id(1)
    col = S.thread_id(0)
    out[row, col] = x[row, col] + S.convert(M + N, S.i32)


@avelang.jit
def kernel_constexpr_helper_specialization(
    x: S.Tensor((M, N), S.i32),
    out: S.Tensor((M, N), S.i32),
    M: S.constexpr,
    N: S.constexpr,
):
    constexpr_passthrough(x, out, M, N)


class TestConstexprHelperSpecialization(unittest.TestCase):
    def test_constexpr_parameter_flows_to_helper_and_shape(self):
        M = 2
        N = 4
        x = torch.arange(M * N, dtype=torch.int32, device="cuda").reshape(M, N)
        out = torch.zeros((M, N), dtype=torch.int32, device="cuda")

        kernel_constexpr_helper_specialization[
            lambda: ((1, 1, 1), (N, M, 1))
        ](x, out, M, N)

        expected = x + M + N
        self.assertTrue(
            torch.equal(out.cpu(), expected.cpu()),
            f"Expected: {expected.cpu().tolist()}, Actual: {out.cpu().tolist()}",
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def make_pair(a: S.u32, b: S.u32) -> S.Tensor((2,), S.u32):
    pair = S.make_local((2,), S.u32)
    pair[0] = a
    pair[1] = b
    return pair


@avelang.jit
def make_pairs(
    a: S.u32, b: S.u32
) -> (S.Tensor((2,), S.u32), S.Tensor((2,), S.u32)):
    first = S.make_local((2,), S.u32)
    second = S.make_local((2,), S.u32)
    first[0] = a
    first[1] = b
    second[0] = b
    second[1] = a
    return first, second


@avelang.jit
def kernel_tensor_return(
    a: S.Tensor((1,), S.u32),
    b: S.Tensor((1,), S.u32),
    out: S.Tensor((2,), S.u32),
):
    pair = make_pair(a[0], b[0])
    out[0] = pair[0]
    out[1] = pair[1]


@avelang.jit
def kernel_multiple_tensor_returns(
    a: S.Tensor((1,), S.u32),
    b: S.Tensor((1,), S.u32),
    out: S.Tensor((4,), S.u32),
):
    first, second = make_pairs(a[0], b[0])
    out[0] = first[0]
    out[1] = first[1]
    out[2] = second[0]
    out[3] = second[1]


class TestTensorReturns(unittest.TestCase):
    def test_tensor_return(self):
        a = torch.tensor([5], dtype=torch.int32, device="cuda")
        b = torch.tensor([7], dtype=torch.int32, device="cuda")
        out = torch.zeros((2,), dtype=torch.int32, device="cuda")

        kernel_tensor_return[lambda: ((1, 1, 1), (1, 1, 1))](a, b, out)

        expected = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))

    def test_multiple_tensor_returns(self):
        a = torch.tensor([5], dtype=torch.int32, device="cuda")
        b = torch.tensor([7], dtype=torch.int32, device="cuda")
        out = torch.zeros((4,), dtype=torch.int32, device="cuda")

        kernel_multiple_tensor_returns[lambda: ((1, 1, 1), (1, 1, 1))](
            a, b, out
        )

        expected = torch.tensor([5, 7, 7, 5], dtype=torch.int32, device="cuda")
        self.assertTrue(torch.equal(out.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()

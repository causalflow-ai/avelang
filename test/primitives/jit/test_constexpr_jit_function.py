#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def add_one(x: S.i32) -> S.i32:
    return x + 1


@avelang.jit
def add_two(x: S.i32) -> S.i32:
    return x + 2


@avelang.jit
def apply_op(
    input_data: S.Tensor((64,), S.i32),
    output_data: S.Tensor((64,), S.i32),
    OP: S.constexpr,
):
    tid = S.thread_id(0)
    output_data[tid] = OP(input_data[tid])


class TestConstexprJitFunction(unittest.TestCase):
    def test_constexpr_jit_function_argument(self):
        input_data = torch.arange(64, dtype=torch.int32, device="cuda")
        add_one_output = torch.zeros((64,), dtype=torch.int32, device="cuda")
        add_two_output = torch.zeros((64,), dtype=torch.int32, device="cuda")

        apply_op[lambda: ((1, 1, 1), (64, 1, 1))](
            input_data, add_one_output, add_one
        )
        apply_op[lambda: ((1, 1, 1), (64, 1, 1))](
            input_data, add_two_output, add_two
        )

        self.assertTrue(
            torch.equal(add_one_output.cpu(), (input_data + 1).cpu())
        )
        self.assertTrue(
            torch.equal(add_two_output.cpu(), (input_data + 2).cpu())
        )


if __name__ == "__main__":
    unittest.main()

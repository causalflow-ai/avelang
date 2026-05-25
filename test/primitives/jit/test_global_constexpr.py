#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S

GLOBAL_SIZE = 16
GLOBAL_OFFSET = S.constexpr(3)
GLOBAL_F32_EXACT = -0.5
GLOBAL_COMPARISON_THRESHOLD = 8


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
def kernel_global_exact_float_constant(
    output_data: S.Tensor((4,), S.f32),
):
    shared_buf = S.make_shared((4,), S.f32)
    tid = S.thread_id(0)
    shared_buf[tid] = GLOBAL_F32_EXACT
    S.syncthreads()
    output_data[tid] = shared_buf[tid]


@avelang.jit
def kernel_constexpr_compare_python_global(
    output_data: S.Tensor((1,), S.i32),
    value: S.constexpr,
):
    if value > GLOBAL_COMPARISON_THRESHOLD:
        output_data[0] = 1
    else:
        output_data[0] = 0


@avelang.jit
def compare_global_constexpr_in_device_function(value: S.constexpr) -> S.i32:
    result = 0
    if value > GLOBAL_COMPARISON_THRESHOLD:
        result = 1
    return result


@avelang.jit
def kernel_device_function_compares_python_globals(
    output_data: S.Tensor((1,), S.i32),
    value: S.constexpr,
):
    output_data[0] = compare_global_constexpr_in_device_function(value)


class TestGlobalConstexprInjection(unittest.TestCase):
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

    def test_exact_float_constant_can_implicitly_demote(self):
        output_data = torch.zeros((4,), dtype=torch.float32, device="cuda")
        expected = torch.full((4,), GLOBAL_F32_EXACT, dtype=torch.float32, device="cuda")

        kernel_global_exact_float_constant[lambda: ((1, 1, 1), (4, 1, 1))](output_data)

        self.assertTrue(
            torch.equal(output_data.cpu(), expected.cpu()),
            f"Expected: {expected.tolist()}, Actual: {output_data.tolist()}",
        )

    def test_constexpr_parameter_can_compare_with_python_global(self):
        output_data = torch.zeros((1,), dtype=torch.int32, device="cuda")

        kernel_constexpr_compare_python_global[lambda: ((1, 1, 1), (1, 1, 1))](
            output_data, GLOBAL_COMPARISON_THRESHOLD + 1
        )

        self.assertEqual(output_data.cpu().item(), 1)

    def test_device_function_can_compare_constexpr_with_python_global(self):
        output_data = torch.zeros((1,), dtype=torch.int32, device="cuda")

        kernel_device_function_compares_python_globals[lambda: ((1, 1, 1), (1, 1, 1))](
            output_data, GLOBAL_COMPARISON_THRESHOLD + 1
        )

        self.assertEqual(output_data.cpu().item(), 1)


if __name__ == "__main__":
    unittest.main()

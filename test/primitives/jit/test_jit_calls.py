#!/usr/bin/env python3
import unittest

import torch
import avelang
import avelang.language as S


CAPTURED_BLOCK = 128


def helper(x):
    return x + 1


@avelang.jit
def add_values_jit(
    a: S.f32,
    b: S.f32,
) -> S.f32:
    return a + b


@avelang.jit
def kernel_add(
    a: S.Tensor((64,), S.f32),
    b: S.Tensor((64,), S.f32),
    c: S.Tensor((64,), S.f32),
):
    tid = S.thread_id(0)
    c[tid] = add_values_jit(a[tid], b[tid])


@avelang.jit
def kernel_add_nested(
    a: S.Tensor((64,), S.f32),
    b: S.Tensor((64,), S.f32),
    c: S.Tensor((64,), S.f32),
):
    def add_values_inner(x: S.f32, y: S.f32) -> S.f32:
        return x + y

    tid = S.thread_id(0)
    c[tid] = add_values_inner(a[tid], b[tid])


class TestJitCalls(unittest.TestCase):
    """Test that avelang.jit functions can be called from avelang.jit kernels."""

    def test_jit_function(self):
        WARP_SIZE = 64
        A = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        B = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        C = torch.zeros((WARP_SIZE,), dtype=torch.float32, device="cuda")

        expected = (A + B).to(dtype=torch.float32, device="cpu")
        kernel_add[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

        actual = C.cpu()

        self.assertTrue(
            torch.allclose(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_nested_function(self):
        WARP_SIZE = 64
        A = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        B = torch.randn((WARP_SIZE,), dtype=torch.float32, device="cuda")
        C = torch.zeros((WARP_SIZE,), dtype=torch.float32, device="cuda")

        expected = (A + B).to(dtype=torch.float32, device="cpu")
        kernel_add_nested[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

        actual = C.cpu()

        self.assertTrue(
            torch.allclose(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_tensor_slice_arguments(self):
        @avelang.jit
        def set_pair(pair: S.Tensor((2,), S.u32), value: S.u32) -> S.Tensor((2,), S.u32):
            pair[0] = value
            pair[1] = value + 1
            return pair

        @avelang.jit
        def kernel(out: S.Tensor((4, 2), S.u32), row: S.u32):
            local = S.make_local((2, 2), S.u32)
            shared = S.make_shared((2, 2), S.u32)
            set_pair(local[row], 21)
            set_pair(shared[row], 31)
            out[0] = local[row]
            out[2] = shared[row]
            out[3] = set_pair(set_pair(out[row], 11), 41)

        out = torch.empty((4, 2), dtype=torch.int32, device="cuda")
        kernel[lambda: ((1, 1, 1), (1, 1, 1))](out, 1)
        self.assertEqual(out.cpu().tolist(), [[21, 22], [11, 12], [31, 32], [41, 42]])

    def test_helper_uses_own_capture_scope(self):
        def make_helper(OFFSET):
            @avelang.jit
            def captured_helper(extra: S.constexpr) -> S.i32:
                return OFFSET + extra + CAPTURED_BLOCK

            return captured_helper

        captured_helper = make_helper(7)
        _ = captured_helper.cache_key
        OFFSET = 100

        @avelang.jit
        def kernel(out: S.Tensor((2,), S.i32)):
            CAPTURED_BLOCK = 64
            out[0] = captured_helper(OFFSET)
            out[1] = OFFSET + CAPTURED_BLOCK

        out = torch.empty(2, dtype=torch.int32, device="cuda")
        kernel[lambda: ((1, 1, 1), (1, 1, 1))](out)
        self.assertEqual(out.cpu().tolist(), [235, 164])

    def test_factory_tensor_annotations(self):
        def make_identity(size):
            @avelang.jit
            def identity(values: S.Tensor((size,), S.u32)) -> S.Tensor((size,), S.u32):
                return values

            return identity

        identities = [make_identity(size) for size in (5, 9)]
        self.assertNotEqual(identities[0].cache_key, identities[1].cache_key)
        for size, identity in zip((5, 9), identities):
            self.assertNotIn("size", identity.get_capture_scope())

            @avelang.jit
            def kernel(
                values: S.Tensor((size,), S.u32),
                out: S.Tensor((1,), S.u32),
                index: S.u32,
            ):
                result = identity(values)
                out[0] = result[index]

            values = torch.arange(size, dtype=torch.int32, device="cuda") + 100
            out = torch.empty(1, dtype=torch.int32, device="cuda")
            kernel[lambda: ((1, 1, 1), (64, 1, 1))](values, out, size - 1)
            self.assertEqual(out.item(), 99 + size)

    def test_nested_function_shadows_global(self):
        @avelang.jit
        def kernel_shadow(
            a: S.Tensor((64,), S.f32),
            b: S.Tensor((64,), S.f32),
            c: S.Tensor((64,), S.f32),
        ):
            def helper(x: S.f32, y: S.f32) -> S.f32:
                return x + y

            tid = S.thread_id(0)
            c[tid] = helper(a[tid], b[tid])

        self.assertIsInstance(kernel_shadow.cache_key, str)


if __name__ == "__main__":
    unittest.main()

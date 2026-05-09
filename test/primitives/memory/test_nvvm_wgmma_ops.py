#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_cuda_nvidia


def get_wgmma_device():
    if not has_cuda_nvidia():
        return None
    for device_idx in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(device_idx)
        if major >= 9:
            return device_idx
    return None


@avelang.jit
def kernel_nvvm_wgmma_sync_primitives(out: S.Tensor((128,), S.i32)):
    tid = S.thread_id(0)

    S.nvvm.wgmma_fence_aligned()
    S.nvvm.wgmma_group_sync_aligned()
    S.nvvm.wgmma_wait_group_sync(0)

    out[tid] = S.convert(tid, S.i32)


@avelang.jit
def kernel_nvvm_wgmma(
    a: S.Tensor((64, 16), S.f16),
    b: S.Tensor((16, 16), S.f16),
    out: S.Tensor((64, 16), S.f32),
):
    tid = S.thread_id(0)

    a_stage = S.make_shared((64, 16), S.f16)
    b_stage = S.make_shared((16, 16), S.f16)
    a_shared = S.make_shared((64, 16), S.f16)
    b_shared = S.make_shared((16, 16), S.f16)
    c_shared = S.make_shared((64, 16), S.f32)

    a_offset_bytes = tid * 16
    S.nvvm.cp_async_ca_shared_global(a_stage, a, a_offset_bytes, a_offset_bytes, 16)

    if tid < 32:
        b_offset_bytes = tid * 16
        S.nvvm.cp_async_ca_shared_global(
            b_stage, b, b_offset_bytes, b_offset_bytes, 16
        )

    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    for i in S.range(8):
        linear_idx_a = tid + i * 128
        row_a = linear_idx_a // 16
        col_a = linear_idx_a % 16
        swizzle_a = (row_a // 4) % 2
        swizzled_col_a = col_a
        if swizzle_a == 1:
            if col_a < 8:
                swizzled_col_a = col_a + 8
            else:
                swizzled_col_a = col_a - 8
        a_shared[row_a, swizzled_col_a] = a_stage[row_a, col_a]

    for i in S.range(2):
        linear_idx_b = tid + i * 128
        row_b = linear_idx_b // 16
        col_b = linear_idx_b % 16
        swizzle_b = (row_b // 4) % 2
        swizzled_col_b = col_b
        if swizzle_b == 1:
            if col_b < 8:
                swizzled_col_b = col_b + 8
            else:
                swizzled_col_b = col_b - 8
        b_shared[row_b, swizzled_col_b] = b_stage[row_b, col_b]

    S.syncthreads()

    desc_a = S.nvvm.make_wgmma_descriptor(a_shared, 1, 0, 0, 0)
    desc_b = S.nvvm.make_wgmma_descriptor(b_shared, 1, 0, 0, 0)
    acc = S.nvvm.wgmma_init_accumulator(64, 16)
    result = S.nvvm.wgmma_async(desc_a, desc_b, acc)
    S.nvvm.wgmma_store(result, c_shared)

    S.syncthreads()

    for i in S.range(8):
        linear_idx_c = tid + i * 128
        row_c = linear_idx_c // 16
        col_c = linear_idx_c % 16
        out[row_c, col_c] = c_shared[row_c, col_c]


@avelang.jit
def kernel_nvvm_wgmma_swizzle_128b(
    a: S.Tensor((64, 64), S.f16),
    b: S.Tensor((64, 64), S.f16),
    out: S.Tensor((64, 64), S.f32),
):
    tid = S.thread_id(0)

    a_stage = S.make_shared((64, 64), S.f16)
    b_stage = S.make_shared((64, 64), S.f16)
    a_shared = S.make_shared((64, 64), S.f16)
    b_shared = S.make_shared((64, 64), S.f16)
    c_shared = S.make_shared((64, 64), S.f32)

    # 128 threads x 16B x 4 = 8192B covers 64x64xf16.
    for i in S.range(4):
        off_bytes = (tid + i * 128) * 16
        S.nvvm.cp_async_ca_shared_global(a_stage, a, off_bytes, off_bytes, 16)
        S.nvvm.cp_async_ca_shared_global(b_stage, b, off_bytes, off_bytes, 16)

    S.nvvm.cp_async_commit_group()
    S.nvvm.cp_async_wait_group(0)
    S.syncthreads()

    # Swizzle-128B layout transform for f16 shared inputs:
    # col' = col XOR ((row & 0x7) * 8)
    for i in S.range(32):
        idx = tid + i * 128
        row = idx // 64
        col = idx % 64
        swizzle = (row % 8) * 8
        swizzled_col = col ^ swizzle
        a_shared[row, swizzled_col] = a_stage[row, col]
        b_shared[row, swizzled_col] = b_stage[row, col]

    S.syncthreads()

    desc_a = S.nvvm.make_wgmma_descriptor(a_shared, 3, 0, 0, 0)
    desc_b = S.nvvm.make_wgmma_descriptor(b_shared, 3, 0, 0, 0)
    acc = S.nvvm.wgmma_init_accumulator(64, 64)
    result = S.nvvm.wgmma_async(desc_a, desc_b, acc)
    S.nvvm.wgmma_store(result, c_shared)

    S.syncthreads()

    for i in S.range(32):
        idx = tid + i * 128
        row = idx // 64
        col = idx % 64
        out[row, col] = c_shared[row, col]


@unittest.skipUnless(
    get_wgmma_device() is not None,
    "Requires CUDA on an NVIDIA Hopper-or-newer GPU with WGMMA support.",
)
class TestNVVMWGMMAOps(unittest.TestCase):
    def test_wgmma_sync_primitives(self):
        device_idx = get_wgmma_device()
        self.assertIsNotNone(device_idx)
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        out = torch.zeros((128,), dtype=torch.int32, device=device)
        expected = torch.arange(128, dtype=torch.int32, device=device)

        kernel_nvvm_wgmma_sync_primitives[lambda: ((1, 1, 1), (128, 1, 1))](out)

        actual = out.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_gemm_f32_f16_f16_64x16x16(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        a = torch.randn((64, 16), dtype=torch.float16, device=device)
        b = torch.randn((16, 16), dtype=torch.float16, device=device)
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)

        kernel_nvvm_wgmma[lambda: ((1, 1, 1), (128, 1, 1))](a, b, out)

        actual = out.cpu()
        expected = (a @ b).float().cpu()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-2, atol=1e-2),
            msg=f"GEMM results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )

    def test_gemm_f32_f16_f16_64x64x64_swizzle_128b(self):
        device_idx = get_wgmma_device()
        assert device_idx is not None
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        a = torch.randn((64, 64), dtype=torch.float16, device=device)
        b = torch.randn((64, 64), dtype=torch.float16, device=device)
        out = torch.zeros((64, 64), dtype=torch.float32, device=device)

        kernel_nvvm_wgmma_swizzle_128b[lambda: ((1, 1, 1), (128, 1, 1))](
            a, b, out
        )

        actual = out.cpu()
        expected = (a @ b).float().cpu()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-2, atol=1e-2),
            msg=(
                "WGMMA swizzle_128b GEMM results do not match.\n"
                f"Max absolute difference: {torch.max(torch.abs(actual - expected))}\n"
                f"Expected:\n{expected}\nActual:\n{actual}"
            ),
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_rocm


BF16_BYTES = 2
U32_BYTES = 4


@avelang.jit
def kernel_store_x4_full_rsrc(
    src: S.Tensor((1, 4), S.u32),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
):
    out_memref = S.make_tensor(out_ptr, S.bf16, S.make_layout((m * n,), (1,)))
    out_rsrc = S.amdgpu.make_rsrc(out_memref, m * n * BF16_BYTES)
    zero = S.convert(0, S.u32)
    S.amdgpu.raw_buffer_store_x4(src[0], out_rsrc, zero, zero, 0)


@avelang.jit
def kernel_store_x4_row_subview_rsrc(
    src: S.Tensor((1, 4), S.u32),
    out_ptr: S.Pointer(S.bf16),
    m: S.u32,
    n: S.u32,
    row: S.u32,
):
    out_memref = S.make_tensor(out_ptr, S.bf16, S.make_layout((m * n,), (1,)))
    row_view = S.subview(out_memref, (row * n,), (n,), (1,))
    row_rsrc = S.amdgpu.make_rsrc(row_view, n * BF16_BYTES)
    zero = S.convert(0, S.u32)
    S.amdgpu.raw_buffer_store_x4(src[0], row_rsrc, zero, zero, 0)


@avelang.jit
def kernel_store_x4_u8_subview_rsrc(
    src: S.Tensor((1, 4), S.u32),
    out_ptr: S.Pointer(S.u8),
    n: S.u32,
    offset: S.u32,
):
    out_memref = S.make_tensor(out_ptr, S.u8, S.make_layout((n,), (1,)))
    out_view = S.subview(out_memref, (offset,), (4 * U32_BYTES,), (1,))
    out_rsrc = S.amdgpu.make_rsrc(out_view, 4 * U32_BYTES)
    zero = S.convert(0, S.u32)
    S.amdgpu.raw_buffer_store_x4(src[0], out_rsrc, zero, zero, 0)


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestAMDGPUBufferStoreSubview(unittest.TestCase):
    def test_row_subview_rsrc_honors_subview_offset(self):
        values = torch.tensor(
            [[0x3F803F80, 0x40004000, 0x40404040, 0x40804080]],
            dtype=torch.int32,
            device="cuda",
        ).view(torch.uint32)

        out_full = torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda")
        out_subview = torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda")

        kernel_store_x4_full_rsrc[lambda: ((1, 1, 1), (1, 1, 1))](
            values, out_full, out_full.shape[0], out_full.shape[1]
        )
        kernel_store_x4_row_subview_rsrc[lambda: ((1, 1, 1), (1, 1, 1))](
            values, out_subview, out_subview.shape[0], out_subview.shape[1], 1
        )

        expected = torch.tensor(
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            dtype=torch.bfloat16,
        )

        self.assertTrue(torch.equal(out_full[0].cpu(), expected))
        self.assertTrue(torch.equal(out_subview[1].cpu(), expected))
        self.assertTrue(
            torch.equal(out_subview[0].cpu(), torch.zeros_like(expected)),
            f"Subview-backed descriptor wrote the wrong row:\n{out_subview.cpu()}",
        )

    def test_u8_subview_rsrc_honors_subview_offset(self):
        values = torch.arange(1, 17, dtype=torch.uint8, device="cuda")
        words = values.view(torch.int32).view(1, 4).view(torch.uint32)
        output = torch.zeros(32, dtype=torch.uint8, device="cuda")

        kernel_store_x4_u8_subview_rsrc[
            lambda: ((1, 1, 1), (1, 1, 1))
        ](words, output, output.numel(), 8)

        expected = torch.zeros_like(output)
        expected[8:24] = values
        self.assertTrue(torch.equal(output.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()

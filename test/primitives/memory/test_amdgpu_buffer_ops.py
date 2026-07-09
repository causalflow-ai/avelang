#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
import _avelang_bindings as _C
from avelang.compiler.code_generator import (
    _build_import_module,
    _collect_jit_dependencies,
    _get_function_def,
)
from avelang.testing import has_rocm


@avelang.jit
def kernel_amdgpu_raw_buffer_roundtrip(
    src: S.Tensor((7,), S.i32),
    dst: S.Tensor((7,), S.i32),
    range_bytes: S.i32,
):
    src_rsrc = S.amdgpu.make_rsrc(src, range_bytes)
    dst_rsrc = S.amdgpu.make_rsrc(dst, range_bytes)

    zero = S.convert(0, S.i32)
    offset_x2 = S.convert(4, S.i32)
    offset_x4 = S.convert(12, S.i32)

    value_x1 = S.amdgpu.raw_buffer_load_x1(src_rsrc, zero, zero, 0)
    value_x2 = S.amdgpu.raw_buffer_load_x2(src_rsrc, zero, offset_x2, 0)
    value_x4 = S.amdgpu.raw_buffer_load_x4(src_rsrc, zero, offset_x4, 0)

    S.amdgpu.raw_buffer_store_x1(value_x1, dst_rsrc, zero, zero, 0)
    S.amdgpu.raw_buffer_store_x2(value_x2, dst_rsrc, zero, offset_x2, 0)
    S.amdgpu.raw_buffer_store_x4(value_x4, dst_rsrc, zero, offset_x4, 0)


@avelang.jit
def kernel_amdgpu_raw_buffer_bf16_view(
    src: S.Tensor((8,), S.bf16),
    dst: S.Tensor((8,), S.bf16),
    range_bytes: S.i32,
):
    src_rsrc = S.amdgpu.make_rsrc(src, range_bytes)

    zero = S.convert(0, S.i32)
    packed = S.amdgpu.raw_buffer_load_x4(src_rsrc, zero, zero, 0)
    frag = S.view(packed, S.Tensor((2, 4, 1), S.bf16))

    for i in S.range(2):
        for j in S.range(4):
            dst[i * 4 + j] = frag[i, j, 0]


@avelang.jit
def kernel_amdgpu_raw_buffer_load_lds_offset(
    src: S.Tensor((2,), S.i32),
    dst: S.Tensor((1,), S.i32),
    range_bytes: S.i32,
    lds_offset: S.i32,
):
    src_rsrc = S.amdgpu.make_rsrc(src, range_bytes)
    shared = S.make_shared((2,), S.i32)

    zero = S.convert(0, S.i32)
    src_offset = S.convert(4, S.i32)
    shared[0] = S.convert(-1, S.i32)
    shared[1] = S.convert(-1, S.i32)
    S.amdgpu.raw_buffer_load_x1_lds(src_rsrc, shared, 4, zero, src_offset, lds_offset, 0)
    S.amdgpu.s_waitcnt(0, 7, 15)
    dst[0] = shared[1]


@avelang.jit
def kernel_amdgpu_readfirstlane(out: S.Tensor((128,), S.i32)):
    tid = S.thread_id(0)
    tid_i32 = S.convert(tid, S.i32)
    out[tid] = S.amdgpu.readfirstlane(tid_i32)


@avelang.jit
def kernel_amdgpu_eager_materialize(out: S.Tensor((1,), S.i32)):
    value = S.convert(17, S.i32)
    S.amdgpu.eager_materialize_i32(value)
    out[0] = value


def generate_mlir(jit_fn) -> str:
    jit_deps = _collect_jit_dependencies(jit_fn)
    import_module = _build_import_module([jit_fn, *jit_deps])

    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(import_module)

    for dep in jit_deps:
        dep_func = _get_function_def(dep.parse())
        generator.visit_function_def(dep_func, "[]", "jit")

    kernel_func = _get_function_def(jit_fn.parse())
    generator.visit_function_def(kernel_func, "[]", "kernel")
    return generator.get_mlir()


class TestAMDGPUBufferOpsMLIR(unittest.TestCase):
    def test_eager_materialize_i32_generates_call(self):
        mlir = generate_mlir(kernel_amdgpu_eager_materialize)
        self.assertIn("_avelang_amdgpu_eager_materialize_i32", mlir)


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestAMDGPUBufferOps(unittest.TestCase):
    def test_raw_buffer_roundtrip(self):
        src = torch.arange(7, dtype=torch.int32, device="cuda") * 17 - 9
        dst = torch.full((7,), -1, dtype=torch.int32, device="cuda")
        range_bytes = src.numel() * src.element_size()

        kernel_amdgpu_raw_buffer_roundtrip[lambda: ((1, 1, 1), (1, 1, 1))](src, dst, range_bytes)

        self.assertTrue(
            torch.equal(dst.cpu(), src.cpu()),
            f"Expected: {src.tolist()}, Actual: {dst.tolist()}",
        )

    def test_make_rsrc_generates_pointer_and_range(self):
        mlir = generate_mlir(kernel_amdgpu_raw_buffer_roundtrip)

        self.assertIn(
            "ave.memref.extract_aligned_pointer_as_index",
            mlir,
        )
        self.assertGreaterEqual(mlir.count("vector.insert %arg2"), 2, mlir)
        self.assertEqual(mlir.count("ave.gpu.amdgpu_raw_buffer_load"), 3, mlir)
        self.assertEqual(mlir.count("ave.gpu.amdgpu_raw_buffer_store"), 3, mlir)

    def test_raw_buffer_bf16_view_extract(self):
        src = torch.randn((8,), dtype=torch.bfloat16, device="cuda")
        dst = torch.zeros((8,), dtype=torch.bfloat16, device="cuda")
        range_bytes = src.numel() * src.element_size()

        kernel_amdgpu_raw_buffer_bf16_view[lambda: ((1, 1, 1), (1, 1, 1))](src, dst, range_bytes)

        self.assertTrue(
            torch.equal(dst.cpu(), src.cpu()),
            f"Expected: {src.tolist()}, Actual: {dst.tolist()}",
        )

    def test_raw_buffer_load_lds_offset(self):
        src = torch.tensor([17, 29], dtype=torch.int32, device="cuda")
        dst = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        range_bytes = src.numel() * src.element_size()

        kernel_amdgpu_raw_buffer_load_lds_offset[lambda: ((1, 1, 1), (1, 1, 1))](
            src,
            dst,
            range_bytes,
            4,
        )

        self.assertEqual(dst.item(), 29)

    def test_readfirstlane(self):
        out = torch.full((128,), -1, dtype=torch.int32, device="cuda")

        kernel_amdgpu_readfirstlane[lambda: ((1, 1, 1), (128, 1, 1))](out)

        expected = torch.cat(
            [
                torch.zeros((64,), dtype=torch.int32),
                torch.full((64,), 64, dtype=torch.int32),
            ]
        )
        self.assertTrue(
            torch.equal(out.cpu(), expected),
            f"Expected: {expected.tolist()}, Actual: {out.cpu().tolist()}",
        )


if __name__ == "__main__":
    unittest.main()

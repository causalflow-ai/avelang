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


def has_arch(*arches):
    return has_rocm() and (
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] in arches
    )


@avelang.jit
def store_scattered_bytes(dst: S.Tensor((512,), S.u8), aux: S.constexpr):
    lane = S.convert(S.thread_id(0), S.u32)
    resource = S.amdgpu.make_rsrc(dst, 256)
    # Byte 1 of each word; lanes >=64 fall outside the descriptor.
    S.amdgpu.raw_buffer_store_u8(S.convert(lane + 128, S.u8), resource, lane * 4, 1, aux)


@avelang.jit
def kernel_amdgpu_raw_buffer_roundtrip(
    src: S.Tensor((7,), S.i32),
    dst: S.Tensor((7,), S.i32),
    range_bytes: S.i32,
):
    src_rsrc = S.amdgpu.make_rsrc(src, range_bytes)
    dst_rsrc = S.amdgpu.make_rsrc(dst, range_bytes)

    # Buffer offsets use the low 32 bits of these wider values.
    zero = S.convert(0x100000000, S.u64)
    offset_x2 = S.convert(0x100000004, S.u64)
    offset_x4 = S.convert(0x10000000C, S.u64)

    value_x1 = S.amdgpu.raw_buffer_load_x1(src_rsrc, zero, zero, 0)
    value_x2 = S.amdgpu.raw_buffer_load_x2(src_rsrc, zero, offset_x2, 0)
    value_x4 = S.amdgpu.raw_buffer_load_x4(src_rsrc, zero, offset_x4, 0)

    S.amdgpu.raw_buffer_store_x1(S.convert(value_x1, S.u64), dst_rsrc, zero, zero, 0)
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
def kernel_amdgpu_readfirstlane(out: S.Tensor((128,), S.i32)):
    tid = S.thread_id(0)
    tid_i32 = S.convert(tid, S.i32)
    out[tid] = S.amdgpu.readfirstlane(tid_i32)


@avelang.jit
def copy_to_lds_4(src: S.Tensor((256,), S.u32), out: S.Tensor((256,), S.u32), aux: S.constexpr):
    tid = S.thread_id(0)
    wave = S.amdgpu.readfirstlane(S.convert(tid // 64, S.u32))
    storage = S.make_shared((256,), S.u32)
    resource = S.amdgpu.make_rsrc(src, 1024)
    S.amdgpu.raw_buffer_load_x1_lds(resource, storage, 4, S.convert(tid * 4, S.u32), 0, wave * 256, aux)
    S.amdgpu.s_waitcnt(0, 0, 0)
    S.syncthreads()
    out[tid] = storage[(tid + 64) % 256]


@avelang.jit
def copy_to_lds_16(src: S.Tensor((256, 4), S.u32), out: S.Tensor((256, 4), S.u32), aux: S.constexpr):
    tid = S.thread_id(0)
    wave = S.amdgpu.readfirstlane(S.convert(tid // 64, S.u32))
    storage = S.make_shared((1024,), S.u32)
    resource = S.amdgpu.make_rsrc(src, 4096)
    S.amdgpu.raw_buffer_load_x4_lds(resource, storage, 16, S.convert(tid * 16, S.u32), 0, wave * 1024, aux)
    S.amdgpu.s_waitcnt(0, 0, 0)
    S.syncthreads()
    values = S.view(storage, S.u32, S.make_layout((256, 4), (4, 1)))
    out[tid] = values[(tid + 64) % 256]


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


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestAMDGPUBufferOps(unittest.TestCase):
    def test_byte_store_preserves_neighbors_and_obeys_resource_bounds(self):
        for aux in (0, 16, 17):
            with self.subTest(aux=aux):
                dst = torch.full((512,), 37, dtype=torch.uint8, device="cuda")
                store_scattered_bytes[lambda: ((1, 1, 1), (128, 1, 1))](dst, aux)
                expected = torch.full_like(dst, 37)
                expected[1:256:4] = torch.arange(128, 192, dtype=torch.uint8, device="cuda")
                torch.testing.assert_close(dst, expected, rtol=0, atol=0)

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

    @unittest.skipUnless(has_arch("gfx942", "gfx950"), "4-byte LDS transfer test requires gfx942 or gfx950")
    def test_4_byte_lds_copy_and_cross_wave_visibility(self):
        src = torch.arange(256, device="cuda", dtype=torch.int32)
        for aux in (0, 16, 17):
            with self.subTest(aux=aux):
                out = torch.empty_like(src)
                copy_to_lds_4[lambda: ((1, 1, 1), (256, 1, 1))](src, out, aux)
                torch.testing.assert_close(out, src.roll(-64, 0), rtol=0, atol=0)

    @unittest.skipUnless(has_arch("gfx950"), "16-byte LDS transfers require gfx950")
    def test_16_byte_lds_copy_and_cross_wave_visibility(self):
        src = torch.arange(1024, device="cuda", dtype=torch.int32).reshape(256, 4)
        for aux in (0, 16, 17):
            with self.subTest(aux=aux):
                out = torch.empty_like(src)
                copy_to_lds_16[lambda: ((1, 1, 1), (256, 1, 1))](src, out, aux)
                torch.testing.assert_close(out, src.roll(-64, 0), rtol=0, atol=0)

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

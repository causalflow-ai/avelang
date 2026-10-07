"""FP4 GEMM correctness, pipeline regressions, and device components."""

import unittest
from dataclasses import replace

import avelang
import avelang.language as al
import torch
from avelang.testing import has_rocm
from avelang_kernels.amdgpu import fp4_gemm_transposed_b
from avelang_kernels.amdgpu.fp4_gemm.config import (
    PACK_FACTOR,
    FP4GemmConfig,
)
from avelang_kernels.amdgpu.fp4_gemm.memory_ops import make_memory_ops
from avelang_kernels.amdgpu.fp4_gemm.reduce import make_block_reduce
from avelang_kernels.amdgpu.fp4_gemm.solution import (
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    SolutionId,
)
from avelang_kernels.amdgpu.fp4_gemm.utils import (
    decode_fp4_reference,
    process_fp4_scales,
    repack_fp4,
)
from avelang_kernels.amdgpu.fp4_gemm.writeback import make_writeback

FP4_VARIANTS = (
    (torch.float16, MatmulMfmaType.FP16, MatmulElementB.NVFP4),
    (torch.bfloat16, MatmulMfmaType.BF16, MatmulElementB.NVFP4),
    (torch.bfloat16, MatmulMfmaType.BF16, MatmulElementB.MXFP4),
)


def _assert_float_bits_equal(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    not_nan = ~torch.isnan(expected)
    assert torch.equal(actual[not_nan].view(torch.uint8), expected[not_nan].view(torch.uint8))


@unittest.skipUnless(has_rocm(), "Requires ROCm")
class TestAMDGPUFP4GEMM(unittest.TestCase):
    def _check_random_gemm(
        self,
        m,
        n,
        k,
        seed,
        solution_id=None,
        *,
        dtype=torch.bfloat16,
        element_b=MatmulElementB.NVFP4,
        high_precision=None,
    ):
        torch.manual_seed(seed)
        a = torch.randn((m, k), dtype=dtype, device="cuda")
        qweight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        if element_b == MatmulElementB.MXFP4:
            scales = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
        else:
            scales = (torch.rand((n, k // 16), device="cuda") * 3.5 + 0.25).to(torch.float8_e4m3fn)
        packed_weight = repack_fp4(qweight)
        packed_scales = process_fp4_scales(scales, element_b=element_b)
        global_scale = 0.25

        # Guard the M tail and use NaNs to catch unwritten output elements.
        guard_size = 32
        storage = torch.full(
            (m * n + 2 * guard_size,),
            123.0,
            dtype=dtype,
            device="cuda",
        )
        out = storage[guard_size:-guard_size].view(m, n)
        out.fill_(float("nan"))
        actual = fp4_gemm_transposed_b(
            a,
            packed_weight,
            packed_scales,
            global_scale,
            out=out,
            solution_id=solution_id,
            element_b=element_b,
            high_precision=high_precision,
        )
        expected = (a.float() @ (decode_fp4_reference(qweight, scales, element_b=element_b) * global_scale).T).to(dtype)
        tolerance = 2e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        for guard in (storage[:guard_size], storage[-guard_size:]):
            torch.testing.assert_close(guard, torch.full_like(guard, 123.0), rtol=0, atol=0)

    def test_matches_torch_reference(self):
        # M, N, K, seed, solution_id. None exercises the default selection.
        cases = (
            (16, 64, 256, 17, None),
            (64, 128, 256, 1234, None),
            (96, 64, 512, 2026, None),
            (256, 256, 256, 31, None),
            (200, 128, 256, 37, 0x122111020408),
            (80, 192, 256, 39, 0x212111020606),
            (160, 256, 256, 41, 0x12211101100A),
            (224, 192, 256, 43, 0x122111010C0E),
            (224, 256, 256, 47, 0x12211101100E),
        )
        for m, n, k, seed, solution_id in cases:
            with self.subTest(m=m, n=n, k=k, solution_id=solution_id):
                self._check_random_gemm(m, n, k, seed, solution_id)

    def test_pipeline_k_tail(self):
        # One/two LDS buffers, with one through five K iterations.
        for solution_id in (0x122111040402, 0x122111040404):
            config = FP4GemmConfig.from_solution(SolutionId.from_int(solution_id))
            for tiles in range(1, 6):
                with self.subTest(solution_id=hex(solution_id), k_tiles=tiles):
                    self._check_random_gemm(
                        config.group_m + 1,
                        2 * config.group_n,
                        tiles * config.group_k,
                        100 + tiles,
                        solution_id,
                    )

    def test_formats_and_precision(self):
        # Default selection and explicit K-warp 1/2/4 solutions.
        for dtype, mfma, element_b in FP4_VARIANTS:
            for high_precision in (False, True):
                with self.subTest(dtype=dtype, format=element_b, high_precision=high_precision):
                    self._check_random_gemm(
                        17,
                        128,
                        512,
                        73,
                        dtype=dtype,
                        element_b=element_b,
                        high_precision=high_precision,
                    )
                for base_id in (0x122111040402, 0x212111040204, 0x411111040202):
                    solution = replace(
                        SolutionId.from_int(base_id),
                        element_b=element_b,
                        mfma_type=mfma,
                        features=MatmulFeatures.GRID | (MatmulFeatures.HIGH_PRECISION if high_precision else 0),
                    )
                    with self.subTest(solution_id=hex(int(solution))):
                        self._check_random_gemm(
                            solution.group_m + 1,
                            2 * solution.group_n,
                            5 * solution.group_k,
                            79,
                            solution,
                            dtype=dtype,
                            element_b=element_b,
                        )

    def test_small_m_and_partial_blocks(self):
        # Sub-wave M sizes and partial blocks with K-warp 1/2/4.
        for solution_id in (0x122111020804, 0x212111040204, 0x411111040202):
            config = FP4GemmConfig.from_solution(SolutionId.from_int(solution_id))
            for m in (1, 15, 17, config.group_m - 1, config.group_m + 1):
                with self.subTest(m=m, solution_id=hex(solution_id)):
                    self._check_random_gemm(m, 2 * config.group_n, 512, 200 + m, solution_id)

    def test_high_precision_preserves_small_weights(self):
        qweight = torch.full((64, 128), 0x11, dtype=torch.uint8, device="cuda")
        weight = repack_fp4(qweight)
        for dtype, element_b, source_scale, a_value, global_scale in (
            (torch.float16, MatmulElementB.NVFP4, 2.0**-9, 1.0, 1.0),
            (torch.bfloat16, MatmulElementB.MXFP4, 7, 2.0**60, 2.0**60),
        ):
            with self.subTest(dtype=dtype, format=element_b):
                is_mx = element_b == MatmulElementB.MXFP4
                a = torch.full((1, 256), a_value, dtype=dtype, device="cuda")
                scales = torch.full(
                    (64, 8 if is_mx else 16),
                    source_scale,
                    dtype=torch.uint8 if is_mx else torch.float8_e4m3fn,
                    device="cuda",
                )
                packed = process_fp4_scales(scales, element_b=element_b)
                actual = fp4_gemm_transposed_b(
                    a, weight, packed, global_scale, element_b=element_b, high_precision=True
                )
                expected = torch.full_like(actual, 128.0 if is_mx else 0.25)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pipeline_lds_reuse(self):
        # In this single-buffer config, writeback overlaps A's last K atom
        # in LDS. Exercise many blocks and K iterations before reusing it.
        for solution_id in (0x122011020408, 0x122011040404):
            for seed in (83, 89, 97):
                with self.subTest(solution_id=hex(solution_id), seed=seed):
                    self._check_random_gemm(
                        1024,
                        1024,
                        16384,
                        seed,
                        solution_id,
                        dtype=torch.float16,
                    )

    def test_fragment_mapping(self):
        n = k = 256
        row = torch.arange(n, device="cuda")[:, None]
        col = torch.arange(k, device="cuda")[None, :]
        # Distinguish lanes within a fragment as well as neighboring K/N tiles.
        codes = ((3 * row + 5 * col + row // 16 + 7 * (col // 16)) % 16).to(torch.uint8)
        qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scale_values = torch.tensor((0.5, 1, 2, 4, 8), device="cuda")
        packed_weight = repack_fp4(qweight)

        # I @ W.T exposes every dequantized value, independently of packed layout.
        values = torch.tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
            dtype=torch.float32,
            device="cuda",
        )
        for dtype, mfma, element_b in FP4_VARIANTS:
            a = torch.eye(k, dtype=dtype, device="cuda")
            scale_group_size = 32 if element_b == MatmulElementB.MXFP4 else 16
            scale_col = torch.arange(k // scale_group_size, device="cuda")[None, :]
            numeric_scales = scale_values[(row // 16 + 3 * scale_col) % 5]
            scales = (
                (numeric_scales.log2() + 127).to(torch.uint8)
                if element_b == MatmulElementB.MXFP4
                else numeric_scales.to(torch.float8_e4m3fn)
            )
            packed_scales = process_fp4_scales(scales, element_b=element_b)
            weight = values[codes.long()] * numeric_scales.repeat_interleave(scale_group_size, dim=1)
            expected = (weight.T * 0.25).to(dtype)
            for high_precision in (False, True):
                # Multiple N atoms, both buffering modes, and K-warp reduction.
                for base_id in (0x12211101100A, 0x12211101100E, 0x411111040202):
                    solution = replace(
                        SolutionId.from_int(base_id),
                        element_b=element_b,
                        mfma_type=mfma,
                        features=MatmulFeatures.GRID | (MatmulFeatures.HIGH_PRECISION if high_precision else 0),
                    )
                    with self.subTest(solution_id=hex(int(solution))):
                        actual = fp4_gemm_transposed_b(
                            a,
                            packed_weight,
                            packed_scales,
                            0.25,
                            solution_id=solution,
                            element_b=element_b,
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_global_scale_tensor_is_graph_safe(self):
        for dtype, _, element_b in FP4_VARIANTS:
            for high_precision in (False, True):
                with self.subTest(dtype=dtype, format=element_b, high_precision=high_precision):
                    self._check_graph_scale(dtype, element_b, high_precision)

    def _check_graph_scale(self, dtype, element_b, high_precision):
        m, n, k = 16, 64, 256
        torch.manual_seed(53)
        a = torch.randn((m, k), dtype=dtype, device="cuda")
        qweight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        if element_b == MatmulElementB.MXFP4:
            scales = torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda")
        else:
            scales = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn, device="cuda")
        packed_weight = repack_fp4(qweight)
        packed_scales = process_fp4_scales(scales, element_b=element_b)
        weight = decode_fp4_reference(qweight, scales, element_b=element_b)
        global_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
        output = torch.empty((m, n), dtype=dtype, device="cuda")

        fp4_gemm_transposed_b(
            a,
            packed_weight,
            packed_scales,
            global_scale,
            out=output,
            element_b=element_b,
            high_precision=high_precision,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp4_gemm_transposed_b(
                a,
                packed_weight,
                packed_scales,
                global_scale,
                out=output,
                element_b=element_b,
                high_precision=high_precision,
            )

        for scale in (0.25, 0.5):
            global_scale.fill_(scale)
            graph.replay()
            torch.cuda.synchronize()
            expected = (a.float() @ (weight * scale).T).to(dtype)
            torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def make_descriptor_probe(config):
    _, advance_global_ptr, *_ = make_memory_ops(config)

    @avelang.jit
    def descriptor_probe(
        source: al.Tensor((3, 4), al.u32),
        out: al.Tensor((3, 4), al.u32),
        n: al.u32,
    ):
        resources = al.make_local((3, 4), al.u32)
        for i in al.range(3):
            resources[i] = source[i]
        for _ in al.range(2):
            advance_global_ptr(resources[0], resources[1], resources[2], n)
        for i in al.range(3):
            out[i] = resources[i]

    return descriptor_probe


def make_reduction_probe(config):
    shm_words, block_reduce = make_block_reduce(config)
    SHM_U32 = shm_words
    WARP_TILES_M = config.warp_tiles_m
    WARP_TILES_N = config.warp_tiles_n
    WARP_PARTITION_N = config.warp_partition_n
    WARP_PARTITION_M = config.warp_partition_m

    @avelang.jit
    def reduction_probe(
        values: al.Tensor((4,), al.f32),
        out: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        wtid = al.thread_id(0) % 64
        wid = al.thread_id(0) // 64
        warp_k = wid // WARP_PARTITION_N // WARP_PARTITION_M
        warp_n = wid % WARP_PARTITION_N
        warp_m = (wid // WARP_PARTITION_N) % WARP_PARTITION_M
        shm = al.make_shared((SHM_U32,), al.u32)
        acc = al.make_local((WARP_TILES_M, WARP_TILES_N, 4), al.f32)
        for tile_m in al.range(WARP_TILES_M):
            for tile_n in al.range(WARP_TILES_N):
                for i in al.range(4):
                    acc[tile_m, tile_n, i] = values[warp_k]
        block_reduce(shm, warp_m, warp_n, warp_k, wtid, acc)
        if warp_k == 0 and wtid == 0:
            for tile_m in al.range(WARP_TILES_M):
                for tile_n in al.range(WARP_TILES_N):
                    out[tile_m, tile_n] = acc[tile_m, tile_n]

    return reduction_probe


def make_writeback_probe(config):
    shm_words, write_result = make_writeback(config)
    THREADS = config.threads
    WARP_TILES_M = config.warp_tiles_m
    WARP_TILES_N = config.warp_tiles_n
    WARP_PARTITION_N = config.warp_partition_n
    WARP_PARTITION_M = config.warp_partition_m
    GROUP_M = config.group_m
    GROUP_N = config.group_n
    SHM_U32 = shm_words

    @avelang.jit
    def writeback_probe(
        values: al.Tensor((THREADS, WARP_TILES_M, WARP_TILES_N, 4), al.f32),
        out: al.Pointer(al.u16),
        m: al.u32,
        alpha: al.Tensor((1,), al.f32),
    ):
        tid = al.thread_id(0)
        wtid, wid = tid % 64, tid // 64
        warp_k = wid // WARP_PARTITION_N // WARP_PARTITION_M
        warp_n = wid % WARP_PARTITION_N
        warp_m = (wid // WARP_PARTITION_N) % WARP_PARTITION_M
        shm = al.make_shared((SHM_U32,), al.u32)
        acc = al.make_local((WARP_TILES_M, WARP_TILES_N, 4), al.f32)
        for tile_m in al.range(WARP_TILES_M):
            for tile_n in al.range(WARP_TILES_N):
                acc[tile_m, tile_n] = values[tid, tile_m, tile_n]
        output = al.make_tensor(out, al.u16, al.make_layout((GROUP_M * GROUP_N,), (1,)))
        resource = al.amdgpu.make_rsrc(output, m * GROUP_N * 2)
        write_result(
            resource,
            shm,
            GROUP_N,
            warp_m,
            warp_n,
            warp_k,
            wtid,
            tid,
            alpha[0],
            acc,
        )

    return writeback_probe


@unittest.skipUnless(has_rocm(), "Requires ROCm")
class TestAMDGPUFP4GEMMComponents(unittest.TestCase):
    def test_writeback_rounding_and_partitions(self):
        # FP16/BF16 ties, signed zero, subnormals, overflow, infinities and NaN.
        bits = torch.tensor(
            [
                0x3F807FFF,
                0x3F808000,
                0x3F808001,
                0x3F818000,
                0xBF808000,
                0xBF818000,
                0x3F800FFF,
                0x3F801000,
                0x3F801001,
                0x3F803000,
                0,
                0x80000000,
                0x33800000,
                0x33000000,
                0x33000001,
                0x00018000,
                0x477FE000,
                0xC77FE000,
                0x47800000,
                0x7F800000,
                0xFF800000,
                0x7FC00000,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # Single/even/uneven partitions, partial vector batches and K-warp 4.
        for base_id in (
            0x122111040402,
            0x212111020606,
            0x122111011008,
            0x12211101100A,
            0x411111040202,
        ):
            for dtype, mfma in ((torch.float16, MatmulMfmaType.FP16), (torch.bfloat16, MatmulMfmaType.BF16)):
                solution = replace(SolutionId.from_int(base_id), mfma_type=mfma)
                config = FP4GemmConfig.from_solution(solution)
                row = torch.arange(config.group_m)[:, None]
                col = torch.arange(config.group_n)[None, :]
                values = bits[(row * 17 + col * 13 + row // 16 * 7 + col // 16 * 3) % bits.numel()]
                tid = torch.arange(config.threads)[:, None, None, None]
                wid, lane = tid // 64, tid % 64
                tile_m = torch.arange(config.warp_tiles_m)[None, :, None, None]
                tile_n = torch.arange(config.warp_tiles_n)[None, None, :, None]
                rows = (
                    ((wid // config.warp_partition_n) % config.warp_partition_m) * config.warp_tiles_m * 16
                    + tile_m * 16
                    + lane % 16
                )
                cols = (
                    (wid % config.warp_partition_n) * config.warp_tiles_n * 16
                    + tile_n * 16
                    + lane // 16 * 4
                    + torch.arange(4)
                )
                source = values[rows, cols].contiguous().cuda()
                # Nonzero K warps must not write their accumulators.
                warp_k = torch.arange(config.threads) // 64 // config.warp_partition_n // config.warp_partition_m
                source[warp_k != 0] = float("nan")
                for m, alpha in ((config.group_m, 1.0), (config.group_m - 3, -0.75)):
                    with self.subTest(solution_id=hex(int(solution)), m=m, alpha=alpha):
                        storage = torch.full((m * config.group_n + 64,), 123.0, dtype=dtype, device="cuda")
                        output = storage[32:-32].view(m, config.group_n)
                        output.fill_(float("nan"))
                        scale = torch.tensor([alpha], dtype=torch.float32, device="cuda")
                        make_writeback_probe(config)[lambda threads=config.threads: ((1, 1, 1), (threads, 1, 1))](
                            source, output, m, scale
                        )
                        actual = output.cpu()
                        expected = (values[:m] * alpha).to(dtype)
                        _assert_float_bits_equal(actual, expected)
                        for guard in (storage[:32], storage[-32:]):
                            torch.testing.assert_close(guard, torch.full_like(guard, 123.0), rtol=0, atol=0)

    def test_descriptor_pointer_carry(self):
        config = FP4GemmConfig.from_solution(SolutionId.from_int(0x122111040402))
        n = 128
        source = torch.tensor(
            [[0xFFFFFF00, 0x12345678, 0x432100FF, 0x27000]] * 3,
            dtype=torch.uint32,
            device="cuda",
        )
        out = torch.empty_like(source)
        make_descriptor_probe(config)[lambda: ((1, 1, 1), (1, 1, 1))](source, out, n)
        steps = (
            config.group_k * config.element_a_bytes,
            n * config.group_k // PACK_FACTOR * 4,
            n * config.group_k // config.scale_group_size,
        )
        expected = []
        for step in steps:
            pointer = 0x12345678FFFFFF00 + 2 * step
            expected.append([pointer & 0xFFFFFFFF, pointer >> 32, 0x432100FF, 0x27000])
        self.assertTrue(torch.equal(out.cpu(), torch.tensor(expected, dtype=torch.uint32)))

    def test_reduction_matches_petit_addition_order(self):
        config = FP4GemmConfig.from_solution(SolutionId.from_int(0x411111040202))
        # Petit evaluates acc + (rd + wr), not (acc + rd) + wr.
        values = torch.tensor([0.0, 1e20, 3.0, -1e20], dtype=torch.float32, device="cuda")
        out = torch.empty(config.accumulator_shape, dtype=torch.float32, device="cuda")
        make_reduction_probe(config)[lambda: ((1, 1, 1), (config.threads, 1, 1))](values, out)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

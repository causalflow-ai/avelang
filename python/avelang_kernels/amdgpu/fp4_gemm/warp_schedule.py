"""Warp-level FP4 GEMM scheduling for AMDGPU."""

import avelang
import avelang.language as al

from .config import LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N, LAYOUT_N, QUANT_VEC_SIZE, TILE

from .dequant import make_dequant
from .solution import MatmulMfmaType


def make_mfma(mfma_type):
    MFMA_TYPE = mfma_type
    FP16 = MatmulMfmaType.FP16

    @avelang.jit
    def mfma(
        b: al.Tensor((2,), al.u32),
        a: al.Tensor((2,), al.u32),
        c: al.Tensor((4,), al.f32),
    ) -> al.Tensor((4,), al.f32):
        if MFMA_TYPE == FP16:
            return al.amdgpu.mfma_16x16x16_f16_f32(b, a, c)
        else:
            return al.amdgpu.mfma_16x16x16_bf16_f32(b, a, c)

    return mfma


def make_warp_schedule(config, memory_layout, read_shm_a, read_shm_b, mfma):
    dequant_scales, dequant_with_scale = make_dequant(config)
    WARP_TILES_M = config.warp_tiles_m
    WARP_TILES_N = config.warp_tiles_n
    READ_BATCH_A = config.read_batch_a
    WARP_ATOM_K = config.warp_atom_k
    WARP_ATOM_N = config.warp_atom_n
    SHM_U32 = memory_layout.shm_u32

    @avelang.jit
    def prefetch(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        data_a: al.Tensor((WARP_TILES_M, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
    ):
        tile_idx_k = warp_k * WARP_ATOM_K
        read_shm_a(shm, stage, warp_m, tile_idx_k, wtid, data_a)
        read_shm_b(shm, stage, qword, packed_scale, warp_n, tile_idx_k, 0, wtid)

    @avelang.jit
    def matmul(
        data_a: al.Tensor((WARP_TILES_M, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((4,), al.u32),
        warp_idx_n: al.u32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        scale_pair = dequant_scales(packed_scale[0, 0])

        # Petit WarpAccumLayout / WarpMatmulRegALayout: qword's K part
        # selects A, while its N part selects the accumulator and scale.
        accum = al.view(
            acc, al.f32,
            al.make_layout(
                (WARP_TILES_M, WARP_ATOM_N, (LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N), 4),
                (WARP_TILES_N * 4, LAYOUT_N // TILE * 4, (0, 4), 1),
            ),
        )
        reg_a = al.view(
            data_a, al.u32,
            al.make_layout(
                (WARP_TILES_M, (LAYOUT_ELEMENTS_K, LAYOUT_ELEMENTS_N), 2, 2),
                (READ_BATCH_A * 4, (4, 0), 2, 1),
            ),
        )
        frag_b = al.view(data_b, al.Tensor((2, 2), al.u32))

        for j in al.range(QUANT_VEC_SIZE):
            scale = al.convert(scale_pair >> ((j // LAYOUT_ELEMENTS_K) * 16), al.u16)
            dequant_with_scale(qword[0, j], scale, data_b)
            for tile_m in al.range(WARP_TILES_M):
                for i in al.range(2):
                    accum[tile_m, warp_idx_n, j] = mfma(frag_b[i], reg_a[tile_m, j, i], accum[tile_m, warp_idx_n, j])

    @avelang.jit
    def pipeline_compute(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        data_a: al.Tensor((WARP_TILES_M, READ_BATCH_A, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((4,), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        for warp_idx_k in al.range(WARP_ATOM_K):
            tile_idx_k = warp_k * WARP_ATOM_K + warp_idx_k
            for warp_idx_n in al.range(WARP_ATOM_N):
                matmul(data_a, qword, packed_scale, data_b, warp_idx_n, acc)
                if warp_idx_n + 1 < WARP_ATOM_N:
                    read_shm_b(shm, stage, qword, packed_scale, warp_n, tile_idx_k, warp_idx_n + 1, wtid)
            if warp_idx_k + 1 < WARP_ATOM_K:
                tile_idx_k = tile_idx_k + 1
                read_shm_a(shm, stage, warp_m, tile_idx_k, wtid, data_a)
                read_shm_b(shm, stage, qword, packed_scale, warp_n, tile_idx_k, 0, wtid)

    return prefetch, pipeline_compute

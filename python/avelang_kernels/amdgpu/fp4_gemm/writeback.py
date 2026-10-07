"""FP4 GEMM result writeback for AMDGPU."""

import avelang
import avelang.language as al

from .config import MAX_SHM_BYTES, TILE, U16_BYTES, U32_BYTES, UINT4_BYTES, WARP_SIZE
from .solution import MatmulMfmaType
from .utils import _ceildiv

RESULT_VEC_SIZE = UINT4_BYTES // U16_BYTES


def make_writeback(config):
    WARP_TILES_M = config.warp_tiles_m
    WARP_TILES_N = config.warp_tiles_n
    WARP_PARTITION_M = config.warp_partition_m
    WARP_PARTITION_N = config.warp_partition_n
    WARP_PARTITION_K = config.warp_partition_k
    GROUP_N = config.group_n
    THREADS = config.threads
    MFMA_TYPE = config.mfma_type
    FP16 = MatmulMfmaType.FP16
    max_tiles_m = min(WARP_TILES_M, MAX_SHM_BYTES // (WARP_PARTITION_M * TILE * GROUP_N * U16_BYTES))
    CHUNK_TILES_M = WARP_TILES_M // _ceildiv(WARP_TILES_M, max_tiles_m)
    SHM_U32 = CHUNK_TILES_M * WARP_PARTITION_M * TILE * GROUP_N * U16_BYTES // U32_BYTES
    # Keep a short final chunk contiguous in LDS by interleaving M warps.
    INTERLEAVE_M_WARPS = WARP_TILES_M % CHUNK_TILES_M != 0

    @avelang.jit
    def pack_to_shm(
        shm: al.Tensor((SHM_U32,), al.u32),
        tile_m_base: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        alpha: al.f32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        tile_m_stride = (WARP_PARTITION_M if INTERLEAVE_M_WARPS else 1) * TILE * GROUP_N * U16_BYTES // U32_BYTES
        warp_m_stride = (1 if INTERLEAVE_M_WARPS else CHUNK_TILES_M) * TILE * GROUP_N * U16_BYTES // U32_BYTES
        alpha_pair = al.full((1, 2), alpha, al.f32)
        acc_pairs = al.view(acc, al.Tensor((WARP_TILES_M, WARP_TILES_N, 2, 2), al.f32))
        packed = al.make_local((2,), al.u32)
        shm_tiles = al.view(
            shm,
            al.u32,
            al.make_layout(
                (CHUNK_TILES_M, WARP_TILES_N, WARP_PARTITION_M, WARP_PARTITION_N, (TILE, WARP_SIZE // TILE), 2),
                (
                    tile_m_stride,
                    TILE * U16_BYTES // U32_BYTES,
                    warp_m_stride,
                    GROUP_N // WARP_PARTITION_N * U16_BYTES // U32_BYTES,
                    (GROUP_N * U16_BYTES // U32_BYTES, 2),
                    1,
                ),
            ),
        )

        if WARP_PARTITION_K == 1 or warp_k == 0:
            for tile_m in al.range(CHUNK_TILES_M):
                acc_m = tile_m_base + tile_m
                if acc_m < WARP_TILES_M:
                    for tile_n in al.range(WARP_TILES_N):
                        for i in al.range(2):
                            pair = acc_pairs[acc_m, tile_n, i] * alpha_pair[0]
                            if MFMA_TYPE == FP16:
                                packed[i] = al.amdgpu.cvt_pk_f16_f32(pair[0], pair[1])
                            else:
                                packed[i] = al.amdgpu.cvt_pk_bf16_f32(pair[0], pair[1])
                        shm_tiles[tile_m, tile_n, warp_m, warp_n, wtid] = packed

    @avelang.jit
    def write_result(
        c_rsrc: al.Tensor((4,), al.u32),
        shm: al.Tensor((SHM_U32,), al.u32),
        n: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        tid: al.u32,
        alpha: al.f32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        shm_vectors = al.view(shm, al.Tensor((SHM_U32 // 4, 4), al.u32))
        stores_per_thread = (SHM_U32 // 4 + THREADS - 1) // THREADS
        vectors_per_row = GROUP_N // RESULT_VEC_SIZE
        # Each interleaved warp segment holds one M tile instead of a full chunk.
        warp_vectors = (1 if INTERLEAVE_M_WARPS else CHUNK_TILES_M) * TILE * vectors_per_row
        for tile_m_base in al.range(0, WARP_TILES_M, CHUNK_TILES_M):
            pack_to_shm(shm, tile_m_base, warp_m, warp_n, warp_k, wtid, alpha, acc)
            al.syncthreads()

            # All threads copy contiguous 16-byte vectors from LDS to C.
            pending_vectors = (
                al.min(CHUNK_TILES_M, WARP_TILES_M - tile_m_base) * WARP_PARTITION_M * TILE * vectors_per_row
            )
            for i in al.range(stores_per_thread):
                idx = tid + i * THREADS
                if idx < pending_vectors:
                    warp_slot = idx // warp_vectors
                    warp_vector = idx % warp_vectors
                    output_warp_m = warp_slot % WARP_PARTITION_M if INTERLEAVE_M_WARPS else warp_slot
                    output_tile_m = tile_m_base + (warp_slot // WARP_PARTITION_M if INTERLEAVE_M_WARPS else 0)
                    row = (output_warp_m * WARP_TILES_M + output_tile_m) * TILE + warp_vector // vectors_per_row
                    vector_col = warp_vector % vectors_per_row
                    al.amdgpu.raw_buffer_store_x4(
                        shm_vectors[idx], c_rsrc, (row * n // RESULT_VEC_SIZE + vector_col) * UINT4_BYTES, 0, 0
                    )
            if tile_m_base + CHUNK_TILES_M < WARP_TILES_M:
                al.syncthreads()

    return SHM_U32, write_result

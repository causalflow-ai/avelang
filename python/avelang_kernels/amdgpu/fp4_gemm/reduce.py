"""Cross-warp K reduction for AMDGPU FP4 GEMM."""

import avelang
import avelang.language as al

from .config import ACCUM_VALUES, WARP_SIZE


def make_block_reduce(config):
    shm_words = config.warp_tiles_n * (config.num_warps // 2) * WARP_SIZE * ACCUM_VALUES
    WARP_TILES_M = config.warp_tiles_m
    WARP_TILES_N = config.warp_tiles_n
    WARP_PARTITION_M = config.warp_partition_m
    WARP_PARTITION_N = config.warp_partition_n
    WARP_PARTITION_K = config.warp_partition_k
    SHM_U32 = shm_words

    @avelang.jit
    def block_reduce(
        shm: al.Tensor((SHM_U32,), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        if WARP_PARTITION_K > 1:
            reduction = al.view(
                shm,
                al.Tensor(
                    (WARP_TILES_N, WARP_PARTITION_K // 2, WARP_PARTITION_M, WARP_PARTITION_N, WARP_SIZE, ACCUM_VALUES),
                    al.f32,
                ),
            )
            for tile_m in al.range(WARP_TILES_M):
                red_offset = WARP_PARTITION_K // 2
                while red_offset:
                    if red_offset <= warp_k and warp_k < 2 * red_offset:
                        for tile_n in al.range(WARP_TILES_N):
                            if red_offset < WARP_PARTITION_K // 2:
                                acc[tile_m, tile_n] = acc[tile_m, tile_n] + (
                                    reduction[tile_n, warp_k, warp_m, warp_n, wtid]
                                    + reduction[tile_n, warp_k - red_offset, warp_m, warp_n, wtid]
                                )
                            reduction[tile_n, warp_k - red_offset, warp_m, warp_n, wtid] = acc[tile_m, tile_n]
                    al.syncthreads()
                    red_offset = red_offset // 2

                if warp_k == 0:
                    for tile_n in al.range(WARP_TILES_N):
                        acc[tile_m, tile_n] = acc[tile_m, tile_n] + reduction[tile_n, 0, warp_m, warp_n, wtid]
                al.syncthreads()

    return shm_words if config.warp_partition_k > 1 else 0, block_reduce

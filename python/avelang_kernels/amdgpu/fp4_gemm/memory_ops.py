"""Global and shared-memory operations for AMDGPU FP4 GEMM."""

from dataclasses import dataclass

import avelang
import avelang.language as al

from .config import (
    LAYOUT_ELEMENTS_K,
    LAYOUT_K,
    LAYOUT_N,
    MAX_SHM_BYTES,
    PACKED_VALUES_PER_VECTOR,
    PACK_FACTOR,
    PIPELINE_STAGES,
    QUANT_VEC_SIZE,
    SCALE_VEC_SIZE,
    TILE,
    U16_BYTES,
    U32_BYTES,
    UINT4_BYTES,
    WARP_SIZE,
)
from .utils import _ceildiv

SHM_DISCARD_VEC = (160 * 1024) // UINT4_BYTES


@dataclass(frozen=True, slots=True)
class MemoryLayout:
    global_loads_a: int
    global_loads_b: int
    global_loads_scale: int
    shm_u32: int
    shm_stages: int


def make_memory_ops(config):
    a_vector_count = config.group_m * config.group_k // config.vec_size
    b_vector_count = config.group_k * config.group_n // PACKED_VALUES_PER_VECTOR
    scale_vector_count = config.group_k * config.group_n // (config.scale_group_size * SCALE_VEC_SIZE)
    shm_data_u32 = (a_vector_count + b_vector_count + scale_vector_count) * QUANT_VEC_SIZE
    shm_stages = min(PIPELINE_STAGES, MAX_SHM_BYTES // (shm_data_u32 * U32_BYTES))
    memory_layout = MemoryLayout(
        global_loads_a=_ceildiv(a_vector_count, config.threads),
        global_loads_b=_ceildiv(b_vector_count, config.threads),
        global_loads_scale=_ceildiv(scale_vector_count, config.threads),
        shm_u32=shm_stages * shm_data_u32,
        shm_stages=shm_stages,
    )
    GROUP_K = config.group_k
    GROUP_N = config.group_n
    ELEMENT_A_BYTES = config.element_a_bytes
    VEC_SIZE = config.vec_size
    SCALE_GROUP_SIZE = config.scale_group_size
    THREADS = config.threads
    WARP_TILES_M = config.warp_tiles_m
    READ_BATCH_A = config.read_batch_a
    WARP_ATOM_N = config.warp_atom_n
    GLOBAL_LOADS_A = memory_layout.global_loads_a
    GLOBAL_LOADS_B = memory_layout.global_loads_b
    GLOBAL_LOADS_SCALE = memory_layout.global_loads_scale
    A_VECTOR_COUNT = a_vector_count
    B_VECTOR_COUNT = b_vector_count
    SCALE_VECTOR_COUNT = scale_vector_count
    SHM_DATA_U32 = shm_data_u32
    SHM_U32 = memory_layout.shm_u32

    @avelang.jit
    def a_shm_index(tile_idx_m: al.u32, tile_idx_k: al.u32, batch_id: al.u32, row: al.u32, col: al.u32) -> al.u32:
        base = tile_idx_m * TILE * GROUP_K // VEC_SIZE + tile_idx_k * TILE * LAYOUT_K // VEC_SIZE + batch_id * WARP_SIZE
        xor_stride = col * LAYOUT_ELEMENTS_K + batch_id
        return (base + col * TILE + row) ^ xor_stride

    @avelang.jit
    def a_shm_store_index(tid: al.u32, steps: al.u32) -> al.u32:
        row_block = GROUP_K // VEC_SIZE
        row = tid // row_block + (THREADS // row_block) * steps
        col = tid % row_block
        col_in_tile = col % (LAYOUT_K // VEC_SIZE)
        return a_shm_index(
            row // TILE, col // (LAYOUT_K // VEC_SIZE),
            col_in_tile % LAYOUT_ELEMENTS_K, row % TILE,
            col_in_tile // LAYOUT_ELEMENTS_K,
        )

    @avelang.jit
    def advance_global_ptr(
        a_rsrc: al.Tensor((4,), al.u32), b_rsrc: al.Tensor((4,), al.u32), scale_rsrc: al.Tensor((4,), al.u32), n: al.u32
    ):
        # Descriptor words 0/1 contain the 64-bit base pointer. Preserve
        # words 2/3 (range/config), exactly as Petit AdvanceGlobalPtr does.
        a_ptr = al.view(a_rsrc, al.Tensor((2,), al.u64))
        b_ptr = al.view(b_rsrc, al.Tensor((2,), al.u64))
        scale_ptr = al.view(scale_rsrc, al.Tensor((2,), al.u64))
        a_ptr[0] = a_ptr[0] + al.convert(GROUP_K * ELEMENT_A_BYTES, al.u64)
        b_ptr[0] = b_ptr[0] + al.convert(n * GROUP_K // PACK_FACTOR * U32_BYTES, al.u64)
        scale_ptr[0] = scale_ptr[0] + al.convert(n * GROUP_K // SCALE_GROUP_SIZE, al.u64)

    @avelang.jit
    def load_global(
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        scale_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        k: al.u32,
        tid: al.u32,
        reg_a: al.Tensor((GLOBAL_LOADS_A, 4), al.u32),
        reg_b: al.Tensor((GLOBAL_LOADS_B, 4), al.u32),
        reg_scale: al.Tensor((GLOBAL_LOADS_SCALE, 4), al.u32),
    ):
        for i in al.range(GLOBAL_LOADS_A):
            idx = tid + i * THREADS
            row = idx // (GROUP_K // VEC_SIZE)
            col = (idx % (GROUP_K // VEC_SIZE)) * VEC_SIZE
            offset = (row * k + col) * ELEMENT_A_BYTES
            reg_a[i] = al.amdgpu.raw_buffer_load_x4(a_rsrc, offset, 0, 0)

        b_row_size = LAYOUT_K * GROUP_N // PACKED_VALUES_PER_VECTOR
        for i in al.range(GLOBAL_LOADS_B):
            idx = tid + i * THREADS
            row = idx // b_row_size
            col = idx % b_row_size
            offset = (row * (n * LAYOUT_K // PACKED_VALUES_PER_VECTOR) + col) * UINT4_BYTES
            reg_b[i] = al.amdgpu.raw_buffer_load_x4(b_rsrc, offset, 0, 0)

        scale_row_size = LAYOUT_K * GROUP_N // (SCALE_GROUP_SIZE * SCALE_VEC_SIZE)
        for i in al.range(GLOBAL_LOADS_SCALE):
            idx = tid + i * THREADS
            row = idx // scale_row_size
            col = idx % scale_row_size
            offset = (row * (n * LAYOUT_K // (SCALE_GROUP_SIZE * SCALE_VEC_SIZE)) + col) * UINT4_BYTES
            reg_scale[i] = al.amdgpu.raw_buffer_load_x4(scale_rsrc, offset, 0, 0)

    @avelang.jit
    def store_shm(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        reg_a: al.Tensor((GLOBAL_LOADS_A, 4), al.u32),
        reg_b: al.Tensor((GLOBAL_LOADS_B, 4), al.u32),
        reg_scale: al.Tensor((GLOBAL_LOADS_SCALE, 4), al.u32),
        tid: al.u32,
    ):
        shm_vec = al.view(shm, al.Tensor((SHM_U32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE), al.u32))
        stage_vec_offset = stage * (SHM_DATA_U32 // QUANT_VEC_SIZE)
        discard_vec = al.convert(SHM_DISCARD_VEC, al.u32)
        for i in al.range(GLOBAL_LOADS_A):
            idx = tid + i * THREADS
            a_idx = a_shm_store_index(tid, i)
            store_idx = stage_vec_offset + a_idx
            if A_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(idx < A_VECTOR_COUNT, store_idx, discard_vec)
            shm_vec[store_idx] = reg_a[i]

        for i in al.range(GLOBAL_LOADS_B):
            idx = tid + i * THREADS
            store_idx = stage_vec_offset + A_VECTOR_COUNT + idx
            if B_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(idx < B_VECTOR_COUNT, store_idx, discard_vec)
            shm_vec[store_idx] = reg_b[i]

        for i in al.range(GLOBAL_LOADS_SCALE):
            idx = tid + i * THREADS
            store_idx = stage_vec_offset + A_VECTOR_COUNT + B_VECTOR_COUNT + idx
            if SCALE_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(idx < SCALE_VECTOR_COUNT, store_idx, discard_vec)
            shm_vec[store_idx] = reg_scale[i]

    @avelang.jit
    def read_shm_a(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        warp_m: al.u32,
        tile_idx_k: al.u32,
        wtid: al.u32,
        data_a: al.Tensor((WARP_TILES_M, READ_BATCH_A, 4), al.u32),
    ):
        shm_vec = al.view(shm, al.Tensor((SHM_U32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE), al.u32))
        stage_vec_offset = stage * (SHM_DATA_U32 // QUANT_VEC_SIZE)
        for batch in al.range(READ_BATCH_A):
            for tile_m in al.range(WARP_TILES_M):
                a_idx = a_shm_index(warp_m * WARP_TILES_M + tile_m, tile_idx_k, batch, wtid % TILE, wtid // TILE)
                data_a[tile_m, batch] = shm_vec[stage_vec_offset + a_idx]

    @avelang.jit
    def read_shm_scale(
        shm: al.Tensor((SHM_U32,), al.u32), stage: al.u32, group_k: al.u32, group_n: al.u32, wtid: al.u32
    ) -> al.u16:
        shm_u16 = al.view(shm, al.Tensor((SHM_U32 * (U32_BYTES // U16_BYTES),), al.u16))
        scale_storage = al.subview(
            shm_u16,
            (
                stage * SHM_DATA_U32 * (U32_BYTES // U16_BYTES)
                + (A_VECTOR_COUNT + B_VECTOR_COUNT) * (UINT4_BYTES // U16_BYTES),
            ),
            (SCALE_VECTOR_COUNT * (UINT4_BYTES // U16_BYTES),),
            (1,),
        )
        scale_tiles = al.view(
            scale_storage,
            al.Tensor(
                (GROUP_K // LAYOUT_K, GROUP_N // LAYOUT_N, LAYOUT_K * LAYOUT_N // SCALE_GROUP_SIZE // U16_BYTES), al.u16
            ),
        )
        scale_lane = wtid
        if SCALE_GROUP_SIZE == 32:
            scale_lane = (wtid // 32) * (LAYOUT_N // 2) + wtid % (LAYOUT_N // 2)
        return scale_tiles[group_k, group_n, scale_lane]

    @avelang.jit
    def read_shm_b(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        warp_n: al.u32,
        tile_idx_k: al.u32,
        warp_idx_n: al.u32,
        wtid: al.u32,
    ):
        b_storage = al.subview(
            shm, (stage * SHM_DATA_U32 + A_VECTOR_COUNT * QUANT_VEC_SIZE,), (B_VECTOR_COUNT * QUANT_VEC_SIZE,), (1,)
        )
        b_tiles = al.view(
            b_storage, al.Tensor((GROUP_K // LAYOUT_K, GROUP_N // LAYOUT_N, WARP_SIZE, QUANT_VEC_SIZE), al.u32)
        )
        tile_idx_n = warp_n * WARP_ATOM_N + warp_idx_n
        packed_scale[0, 0] = read_shm_scale(shm, stage, tile_idx_k, tile_idx_n, wtid)
        qword[0] = b_tiles[tile_idx_k, tile_idx_n, wtid]

    return memory_layout, advance_global_ptr, load_global, store_shm, read_shm_a, read_shm_b

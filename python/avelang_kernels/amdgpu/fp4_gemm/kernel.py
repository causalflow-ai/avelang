"""AMDGPU FP4 GEMM device kernel and pipeline."""

from functools import cache

import avelang
import avelang.language as al

from .config import (
    ACCUM_VALUES,
    LAYOUT_K,
    MAX_SHM_BYTES,
    PACK_FACTOR,
    PIPELINE_STAGES,
    TILE,
    U16_BYTES,
    U32_BYTES,
    WARP_SIZE,
)
from .memory_ops import make_memory_ops
from .reduce import make_block_reduce
from .warp_schedule import make_mfma, make_warp_schedule
from .writeback import make_writeback

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


@cache
def make_kernel(config):
    mfma = make_mfma(config.mfma_type)
    memory_layout, advance_global_ptr, load_global, store_shm, read_shm_a, read_shm_b = make_memory_ops(config)
    result_shm_u32, write_result = make_writeback(config)
    prefetch, pipeline_compute = make_warp_schedule(config, memory_layout, read_shm_a, read_shm_b, mfma)
    reduction_shm_u32, block_reduce = make_block_reduce(config)
    shm_words = max(memory_layout.shm_u32, reduction_shm_u32, result_shm_u32)
    if shm_words * U32_BYTES > MAX_SHM_BYTES:
        raise ValueError(f"configuration requires {shm_words * U32_BYTES} bytes of LDS")
    SHM_U32 = shm_words
    SINGLE_BUFFER = memory_layout.shm_stages == 1
    global_loads = memory_layout.global_loads_a + memory_layout.global_loads_b + memory_layout.global_loads_scale
    MFMA_COUNT = config.warp_tiles_m * config.warp_tiles_n * (config.group_k // TILE // config.warp_partition_k)
    mfmas_per_load = MFMA_COUNT // global_loads
    MFMAS_PER_LOAD = 4 if mfmas_per_load > 12 else 2 if mfmas_per_load > 6 else 1
    GLOBAL_LOADS = global_loads
    GROUP_M = config.group_m
    GROUP_N = config.group_n
    GROUP_K = config.group_k
    WARP_PARTITION_M = config.warp_partition_m
    WARP_PARTITION_N = config.warp_partition_n
    WARP_PARTITION_K = config.warp_partition_k
    WARP_TILES_M = config.warp_tiles_m
    WARP_TILES_N = config.warp_tiles_n
    ELEMENT_A_BYTES = config.element_a_bytes
    SCALE_GROUP_SIZE = config.scale_group_size
    GLOBAL_SCALE_FACTOR = config.global_scale_factor
    READ_BATCH_A = config.read_batch_a
    GLOBAL_LOADS_A = memory_layout.global_loads_a
    GLOBAL_LOADS_B = memory_layout.global_loads_b
    GLOBAL_LOADS_SCALE = memory_layout.global_loads_scale
    PIPELINE_SHM_U32 = memory_layout.shm_u32
    RESULT_SHM_U32 = result_shm_u32
    REDUCTION_SHM_U32 = reduction_shm_u32

    @avelang.jit
    def hot_loop_scheduler():
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, GLOBAL_LOADS_A + 2, 0)
        for _ in al.range(GLOBAL_LOADS):
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, MFMAS_PER_LOAD, 0)
        for _ in al.range(GLOBAL_LOADS):
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, MFMAS_PER_LOAD * 2, 0)
        al.amdgpu.sched_barrier(0)

    @avelang.jit
    def single_buffer_pipeline(
        shm: al.Tensor((PIPELINE_SHM_U32,), al.u32),
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        scale_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        k: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        tid: al.u32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        reg_a = al.make_local((PIPELINE_STAGES, GLOBAL_LOADS_A, 4), al.u32)
        reg_b = al.make_local((PIPELINE_STAGES, GLOBAL_LOADS_B, 4), al.u32)
        reg_scale = al.make_local((PIPELINE_STAGES, GLOBAL_LOADS_SCALE, 4), al.u32)
        data_a = al.make_local((WARP_TILES_M, READ_BATCH_A, 4), al.u32)
        qword = al.make_local((1, 4), al.u32)
        packed_scale = al.make_local((1, 1), al.u16)
        data_b = al.make_local((4,), al.u32)

        k_total = k // GROUP_K
        load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[0], reg_b[0], reg_scale[0])
        al.amdgpu.sched_barrier(0x7DF)
        al.amdgpu.s_waitcnt(0, 7, 15)
        store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

        if k_total > 1:
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[1], reg_b[1], reg_scale[1])

        k_idx = al.convert(0, al.u32)
        while k_idx + 3 < k_total:
            al.amdgpu.sched_barrier(0)
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()

            prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[0], reg_b[0], reg_scale[0])
            pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)
            al.syncthreads()
            store_shm(shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid)
            hot_loop_scheduler()

            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[1], reg_b[1], reg_scale[1])
            pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)
            al.syncthreads()
            store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)
            hot_loop_scheduler()
            k_idx = k_idx + 2

        # Epilogue: statically unroll up to three remaining K tiles.
        advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
        al.syncthreads()
        prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
        if k_idx + 2 < k_total:
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[0], reg_b[0], reg_scale[0])
        pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)
        if k_idx + 1 < k_total:
            al.syncthreads()
            store_shm(shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid)
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)
            if k_idx + 2 < k_total:
                al.syncthreads()
                store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

        if k_idx + 2 < k_total:
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)

    @avelang.jit
    def double_buffer_pipeline(
        shm: al.Tensor((PIPELINE_SHM_U32,), al.u32),
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        scale_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        k: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        tid: al.u32,
        acc: al.Tensor((WARP_TILES_M, WARP_TILES_N, 4), al.f32),
    ):
        reg_a = al.make_local((PIPELINE_STAGES, GLOBAL_LOADS_A, 4), al.u32)
        reg_b = al.make_local((PIPELINE_STAGES, GLOBAL_LOADS_B, 4), al.u32)
        reg_scale = al.make_local((PIPELINE_STAGES, GLOBAL_LOADS_SCALE, 4), al.u32)
        data_a = al.make_local((WARP_TILES_M, READ_BATCH_A, 4), al.u32)
        qword = al.make_local((1, 4), al.u32)
        packed_scale = al.make_local((1, 1), al.u16)
        data_b = al.make_local((4,), al.u32)

        k_total = k // GROUP_K
        load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[0], reg_b[0], reg_scale[0])
        al.amdgpu.sched_barrier(0x7DF)
        al.amdgpu.s_waitcnt(0, 7, 15)
        store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

        if k_total > 1:
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[1], reg_b[1], reg_scale[1])

        k_idx = al.convert(0, al.u32)
        while k_idx + 3 < k_total:
            al.amdgpu.sched_barrier(0)
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()

            prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[0], reg_b[0], reg_scale[0])
            store_shm(shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid)
            pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)
            hot_loop_scheduler()

            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            prefetch(shm, 1, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[1], reg_b[1], reg_scale[1])
            store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)
            pipeline_compute(shm, 1, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)
            hot_loop_scheduler()
            k_idx = k_idx + 2

        # Epilogue: statically unroll up to three remaining K tiles.
        advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
        al.syncthreads()
        prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
        if k_idx + 2 < k_total:
            load_global(a_rsrc, b_rsrc, scale_rsrc, n, k, tid, reg_a[0], reg_b[0], reg_scale[0])
        if k_idx + 1 < k_total:
            store_shm(shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid)
        pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)

        if k_idx + 1 < k_total:
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            prefetch(shm, 1, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            if k_idx + 2 < k_total:
                store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)
            pipeline_compute(shm, 1, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)

        if k_idx + 2 < k_total:
            advance_global_ptr(a_rsrc, b_rsrc, scale_rsrc, n)
            al.syncthreads()
            prefetch(shm, 0, data_a, qword, packed_scale, warp_m, warp_n, warp_k, wtid)
            pipeline_compute(shm, 0, data_a, qword, packed_scale, data_b, warp_m, warp_n, warp_k, wtid, acc)

    @avelang.jit
    def fp4_gemm_kernel(
        A: al.Pointer(al.u16),
        B: al.Pointer(al.u32),
        scales: al.Pointer(al.u8),
        global_scale: al.Tensor((1,), al.f32),
        C: al.Pointer(al.u16),
        m: al.u32,
        n: al.u32,
        k: al.u32,
    ):
        tid = al.thread_id(0)
        wid = tid // WARP_SIZE
        wtid = tid % WARP_SIZE
        warp_k = wid // WARP_PARTITION_N // WARP_PARTITION_M
        warp_n = wid % WARP_PARTITION_N
        warp_m = (wid // WARP_PARTITION_N) % WARP_PARTITION_M

        group_m = al.block_id(0)
        group_n = al.block_id(1)
        alpha = global_scale[0] * al.convert(GLOBAL_SCALE_FACTOR, al.f32)

        block_row = group_m * GROUP_M
        valid_m = al.min(m - block_row, al.convert(GROUP_M, al.u32))

        # Like Petit, descriptor bases advance with K while ranges cover one tile.
        a_tensor = al.make_tensor(A, al.u16, al.make_layout((m * k,), (1,)))
        b_tensor = al.make_tensor(B, al.u32, al.make_layout((n * k // PACK_FACTOR,), (1,)))
        s_tensor = al.make_tensor(scales, al.u8, al.make_layout((n * k // SCALE_GROUP_SIZE,), (1,)))
        c_tensor = al.make_tensor(C, al.u16, al.make_layout((m * n,), (1,)))

        # Subview offsets/extents are in elements; resource ranges are in bytes.
        a_block_elements = (valid_m - 1) * k + GROUP_K
        a_block = al.subview(a_tensor, (block_row * k,), (a_block_elements,), (1,))
        b_block_elements = ((GROUP_K // LAYOUT_K - 1) * n + GROUP_N) * LAYOUT_K // PACK_FACTOR
        b_block = al.subview(b_tensor, (group_n * GROUP_N * LAYOUT_K // PACK_FACTOR,), (b_block_elements,), (1,))
        scale_block_elements = (GROUP_K // SCALE_GROUP_SIZE - 1) * n + GROUP_N
        scale_block = al.subview(
            s_tensor, (group_n * GROUP_N * LAYOUT_K // SCALE_GROUP_SIZE,), (scale_block_elements,), (1,)
        )
        c_block_elements = (valid_m - 1) * n + GROUP_N
        c_block = al.subview(c_tensor, (block_row * n + group_n * GROUP_N,), (c_block_elements,), (1,))
        a_rsrc = al.make_local((1, 4), al.u32)
        b_rsrc = al.make_local((1, 4), al.u32)
        scale_rsrc = al.make_local((1, 4), al.u32)
        a_rsrc[0] = al.amdgpu.make_rsrc(a_block, a_block_elements * ELEMENT_A_BYTES)
        b_rsrc[0] = al.amdgpu.make_rsrc(b_block, b_block_elements * U32_BYTES)
        scale_rsrc[0] = al.amdgpu.make_rsrc(scale_block, scale_block_elements)
        c_rsrc = al.amdgpu.make_rsrc(c_block, c_block_elements * U16_BYTES)

        shm = al.make_shared((SHM_U32,), al.u32)
        pipeline_shm = al.subview(shm, (0,), (PIPELINE_SHM_U32,), (1,))
        result_shm = al.subview(shm, (0,), (RESULT_SHM_U32,), (1,))

        acc = al.make_local((WARP_TILES_M, WARP_TILES_N, ACCUM_VALUES), al.f32)
        for tile_m in al.range(WARP_TILES_M):
            for tile_n in al.range(WARP_TILES_N):
                for acc_idx in al.range(ACCUM_VALUES):
                    acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

        if SINGLE_BUFFER:
            single_buffer_pipeline(
                pipeline_shm, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, warp_m, warp_n, warp_k, wtid, tid, acc
            )
        else:
            double_buffer_pipeline(
                pipeline_shm, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, warp_m, warp_n, warp_k, wtid, tid, acc
            )

        al.syncthreads()
        if WARP_PARTITION_K > 1:
            reduction_shm = al.subview(shm, (0,), (REDUCTION_SHM_U32,), (1,))
            block_reduce(reduction_shm, warp_m, warp_n, warp_k, wtid, acc)
        write_result(c_rsrc, result_shm, n, warp_m, warp_n, warp_k, wtid, tid, alpha, acc)

    return fp4_gemm_kernel

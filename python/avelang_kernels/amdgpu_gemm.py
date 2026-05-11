import torch
import torch.nn as nn

import avelang 
import avelang.language as al


WARP_SIZE = 64
NUM_WARPS = 4
GROUP_M = 128
GROUP_N = 128
GROUP_K = 64
WARP_PER_ROW = 2
WARP_PER_COL = 2
WARP_MAT_M = GROUP_M // WARP_PER_ROW
WARP_MAT_N = GROUP_N // WARP_PER_COL
M_TILES_PER_WARP = WARP_MAT_M // 16
N_TILES_PER_WARP = WARP_MAT_N // 16
VEC_SIZE = 8
THREADS = WARP_SIZE * NUM_WARPS
REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
BF16_BYTES = 2
SHM_PAD_ROWS = 4
SHM_PAD_BF16 = 16
SHM_GROUPS_A = GROUP_M // SHM_PAD_ROWS
SHM_GROUPS_B = GROUP_N // SHM_PAD_ROWS
SHM_GROUP_BF16 = SHM_PAD_ROWS * GROUP_K + SHM_PAD_BF16
SHM_GROUP_WORDS = SHM_GROUP_BF16 // 2
SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16
SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16
SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


@avelang.jit
def _wgm_mapping(m: al.u32, n: al.u32) -> (al.u32, al.u32):
    linear_group_id = al.block_id(0)
    m_groups = m // GROUP_M
    n_groups = n // GROUP_N

    total_groups = m_groups * n_groups

    cu_count = al.convert(38 * 8, al.u32)
    wgm_xcc = al.convert(8, al.u32)
    workgroup_mapping = al.convert(32, al.u32)

    linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
    cu_base = (linear_group_id // cu_count) * cu_count
    cu_xcc = (linear_group_id % cu_count) // wgm_xcc
    cu_base = cu_base + cu_xcc

    cu_tail_limit = (total_groups // cu_count) * cu_count
    active_cu = (
        (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
    )
    cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
    linear_group_mapped = cu_base + cu_xcc_stride

    linear_group_id = (
        linear_group_mapped
        if (linear_group_id < linear_group_limit)
        else linear_group_id
    )

    group_m = linear_group_id // n_groups
    group_n = linear_group_id - group_m * n_groups

    mapping_block = group_m // workgroup_mapping
    mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
    mapping_groups = m_groups // workgroup_mapping
    mapping_tail = m_groups % workgroup_mapping
    mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail

    mapping_span = (
        mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
    )

    group_n = mapping_linear // mapping_span
    group_m = mapping_linear % mapping_span
    group_m = group_m + mapping_block * workgroup_mapping

    return group_m, group_n


@avelang.jit
def _load_global_a(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

    for i in al.range(REG_ROWS_A):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_offset_stride,
            0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))

        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _load_global_b(
    src_rsrc: al.Tensor((4,), al.u32),
    k: al.u32,
    group_row: al.u32,
    k_idx: al.u32,
    tid: al.u32,
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
):
    row = tid // SHM_CHUNKS_PER_ROW
    col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
    thread_offset = (row * k + col) * BF16_BYTES
    tile_offset = (group_row * GROUP_N * k + k_idx * GROUP_K) * BF16_BYTES
    thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES

    for i in al.range(REG_ROWS_B):
        packed = al.amdgpu.raw_buffer_load_x4(
            src_rsrc,
            thread_offset,
            tile_offset + i * thread_offset_stride,
            0,
        )
        frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))

        for v in al.range(VEC_SIZE):
            reg[i, v] = frag[v]


@avelang.jit
def _store_shm_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_A // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )

    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )

    for i in al.range(REG_ROWS_A):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _store_shm_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16),
    tid: al.u32,
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_TOTAL_BF16_B // VEC_SIZE, 4),
            (VEC_SIZE // 2, 1),
        ),
    )

    row = tid // SHM_CHUNKS_PER_ROW
    row_group = row // SHM_PAD_ROWS
    row_in_group = row - row_group * SHM_PAD_ROWS
    chunk = tid - row * SHM_CHUNKS_PER_ROW
    shm_chunk = (
        row_group * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
        + row_in_group * SHM_CHUNKS_PER_ROW
        + chunk
    )
    shm_chunk_stride = (
        (THREADS * VEC_SIZE // GROUP_K // SHM_PAD_ROWS)
        * (SHM_GROUP_WORDS // (VEC_SIZE // 2))
    )

    for i in al.range(REG_ROWS_B):
        packed = al.view(reg[i], al.Tensor((4,), al.u32))
        shm_vec[shm_chunk + i * shm_chunk_stride] = packed


@avelang.jit
def _load_shm_to_regs_batch_a(
    shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_GROUPS_A, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )

    row_start = row_base + (wtid % 16) * M_TILES_PER_WARP
    chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)

    for tile in al.range(M_TILES_PER_WARP):
        row = row_start + tile
        row_group = row // SHM_PAD_ROWS
        row_in_group = row - row_group * SHM_PAD_ROWS
        data[tile] = shm_vec[row_group, row_in_group, chunk_base]


@avelang.jit
def _load_shm_to_regs_batch_b(
    shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16),
    row_base: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (SHM_GROUPS_B, SHM_PAD_ROWS, SHM_CHUNKS_PER_ROW, 4),
            (SHM_GROUP_WORDS, GROUP_K // 2, 4, 1),
        ),
    )

    row_start = row_base + (wtid % 16) * N_TILES_PER_WARP
    chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)

    for tile in al.range(N_TILES_PER_WARP):
        row = row_start + tile
        row_group = row // SHM_PAD_ROWS
        row_in_group = row - row_group * SHM_PAD_ROWS
        data[tile] = shm_vec[row_group, row_in_group, chunk_base]


@avelang.jit
def _matmul_from_regs_batch(
    data_a: al.Tensor((M_TILES_PER_WARP, 4), al.u32),
    data_b: al.Tensor((N_TILES_PER_WARP, 4), al.u32),
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            frag_a = al.view(data_a[tile_m], al.Tensor((2, 4, 1), al.bf16))
            frag_b = al.view(data_b[tile_n], al.Tensor((2, 4, 1), al.bf16))
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[0],
                frag_b[0],
                acc[tile_m, tile_n],
            )
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_a[1],
                frag_b[1],
                acc[tile_m, tile_n],
            )


@avelang.jit
def _write_results(
    dst_rsrc: al.Tensor((4,), al.u32),
    n: al.u32,
    group_m: al.u32,
    group_n: al.u32,
    wtid: al.u32,
    warp_row: al.u32,
    warp_col: al.u32,
    acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
):
    lane_row_group = wtid // 16
    lane_col = wtid % 16
    warp_offset = (
        (group_m * GROUP_M + warp_row * WARP_MAT_M) * n
        + group_n * GROUP_N
        + warp_col * WARP_MAT_N
    ) * BF16_BYTES

    for tile_m in al.range(M_TILES_PER_WARP):
        for acc_idx in al.range(4):
            row_offset = (
                lane_row_group * (4 * M_TILES_PER_WARP)
                + acc_idx * M_TILES_PER_WARP
                + tile_m
            ) * n
            col_offset = lane_col * N_TILES_PER_WARP
            thread_offset = (row_offset + col_offset) * BF16_BYTES

            lo0 = al.bitcast(acc[tile_m, 0, acc_idx], al.u32)
            hi0 = al.bitcast(acc[tile_m, 1, acc_idx], al.u32)
            lo1 = al.bitcast(acc[tile_m, 2, acc_idx], al.u32)
            hi1 = al.bitcast(acc[tile_m, 3, acc_idx], al.u32)
            packed = al.full((2,), 0, al.u32)
            packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
            packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
            al.amdgpu.raw_buffer_store_x2(packed, dst_rsrc, thread_offset, warp_offset, 0)


@avelang.jit
def _hot_loop_scheduler():
    for _ in al.range(8):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)

    for _ in al.range(8):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)

    for _ in al.range(8):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)


@avelang.jit
def _gemm_pipeline_transposed_b_kernel(
    A: al.Pointer(al.bf16),
    B: al.Pointer(al.bf16),
    C: al.Pointer(al.bf16),
    m: al.u32,
    n: al.u32,
    k: al.u32,
):
    tid = al.thread_id(0)
    wid = tid // WARP_SIZE
    wtid = tid % WARP_SIZE
    warp_row = wid // WARP_PER_COL
    warp_col = wid % WARP_PER_COL

    m_groups = (m + GROUP_M - 1) // GROUP_M
    n_groups = (n + GROUP_N - 1) // GROUP_N
    group_m, group_n = _wgm_mapping(m, n)

    a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
    b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
    c_tensor = al.make_tensor(C, al.bf16, al.make_layout((m * n,), (1,)))
    a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
    b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)
    c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

    shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
    shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
    reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
    reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)
    data_a0 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_a1 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
    data_b0 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    data_b1 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
    acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)

    for tile_m in al.range(M_TILES_PER_WARP):
        for tile_n in al.range(N_TILES_PER_WARP):
            for acc_idx in al.range(4):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    k_total = k // GROUP_K

    _load_global_a(a_rsrc, k, group_m, 0, tid, reg_a)
    _load_global_b(b_rsrc, k, group_n, 0, tid, reg_b)
    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _load_global_a(a_rsrc, k, group_m, 1, tid, reg_a)
    _load_global_b(b_rsrc, k, group_n, 1, tid, reg_b)

    for k_idx in al.range(0, k_total - 3, 2):
        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, k_idx + 2, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, k_idx + 2, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
        _matmul_from_regs_batch(data_a0, data_b0, acc)
        al.syncthreads()

        _store_shm_a(shm_a, reg_a, tid)
        _store_shm_b(shm_b, reg_b, tid)
        _load_global_a(a_rsrc, k, group_m, k_idx + 3, tid, reg_a)
        _load_global_b(b_rsrc, k, group_n, k_idx + 3, tid, reg_b)
        al.syncthreads()

        _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
        _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
        _matmul_from_regs_batch(data_a1, data_b1, acc)

        _hot_loop_scheduler()
        _hot_loop_scheduler()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)
    al.syncthreads()

    _store_shm_a(shm_a, reg_a, tid)
    _store_shm_b(shm_b, reg_b, tid)
    al.syncthreads()

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 0, wtid, data_a0)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 0, wtid, data_b0)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _load_shm_to_regs_batch_a(shm_a, warp_row * WARP_MAT_M, 1, wtid, data_a1)
    _load_shm_to_regs_batch_b(shm_b, warp_col * WARP_MAT_N, 1, wtid, data_b1)
    _matmul_from_regs_batch(data_a0, data_b0, acc)
    _matmul_from_regs_batch(data_a1, data_b1, acc)

    _write_results(c_rsrc, n, group_m, group_n, wtid, warp_row, warp_col, acc)


def gemm_pipeline_transposed_b(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if A.dtype != torch.bfloat16:
        raise ValueError(f"A must have dtype torch.bfloat16, got {A.dtype}.")
    if B.dtype != torch.bfloat16:
        raise ValueError(f"B must have dtype torch.bfloat16, got {B.dtype}.")

    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[0]
    if m % GROUP_M != 0:
        raise ValueError(f"M must be a multiple of {GROUP_M}, got {m}.")
    if n % GROUP_N != 0:
        raise ValueError(f"N must be a multiple of {GROUP_N}, got {n}.")
    if k % GROUP_K != 0:
        raise ValueError(f"K must be a multiple of {GROUP_K}, got {k}.")

    out_shape = (m, n)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=A.device)
    elif out.shape != out_shape:
        raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}.")
    elif out.dtype != torch.bfloat16:
        raise ValueError(f"out must have dtype torch.bfloat16, got {out.dtype}.")
    elif out.device != A.device:
        raise ValueError(f"out must be on device {A.device}, got {out.device}.")

    m_groups = m // GROUP_M
    n_groups = n // GROUP_N
    grid_size = m_groups * n_groups
    block_size = WARP_SIZE * NUM_WARPS

    _gemm_pipeline_transposed_b_kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](A, B, out, m, n, k)
    return out

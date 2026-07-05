"""AMDGPU BF16 flash attention debug helpers through masked softmax."""

import math
import avelang
import avelang.language as al
import torch

WARP_SIZE = 64
NUM_WARPS = 4
THREADS = WARP_SIZE * NUM_WARPS

BLOCK_ROWS = 128
BLOCK_COLS = 128
HEAD_DIM = 64
VEC_SIZE = 16 // 2
BF16_BYTES = 2
U128_BYTES = VEC_SIZE * BF16_BYTES

SHM_Q_WORDS = BLOCK_ROWS * HEAD_DIM // 2
SHM_K_WORDS = BLOCK_COLS * HEAD_DIM // 2
SHM_Q_VECS = BLOCK_ROWS * HEAD_DIM // VEC_SIZE
SHM_K_VECS = BLOCK_COLS * HEAD_DIM // VEC_SIZE

WARP_ROWS = 32
HALF_WARP_ROWS = 16
SCORE_TILE_COLS = 16
Q_BATCHES = HEAD_DIM // SCORE_TILE_COLS
K_BATCHES = BLOCK_COLS // WARP_ROWS
O_BATCHES = HEAD_DIM // WARP_ROWS
NEG_INF = -1.0e30
SCALE_LOG2 = math.log2(math.e) / math.sqrt(HEAD_DIM)


def _validate_qk_shape(q: torch.Tensor, k: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            "q and k must be rank-4 tensors shaped [batch, heads, seq, dim] "
            f"(got q.ndim={q.ndim}, k.ndim={k.ndim})"
        )
    if q.shape != k.shape:
        raise ValueError(f"q and k must have identical shapes (got {q.shape}, {k.shape})")
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"BF16 flash attention debug currently requires head_dim={HEAD_DIM} "
            f"(got {q.shape[-1]})"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError(f"q and k must be torch.bfloat16 (got q={q.dtype}, k={k.dtype})")
    if q.device.type != "cuda" or k.device.type != "cuda":
        raise ValueError(f"q and k must be CUDA tensors (got q={q.device}, k={k.device})")
    if q.device != k.device:
        raise ValueError(f"q and k must be on the same device (got {q.device}, {k.device})")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")


def flash_attn_validate_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    _validate_qk_shape(q, k)
    if v.ndim != 4:
        raise ValueError(
            "v must be a rank-4 tensor shaped [batch, heads, seq, dim] "
            f"(got v.ndim={v.ndim})"
        )
    if v.shape != q.shape:
        raise ValueError(f"v must have shape {q.shape} (got {v.shape})")
    if v.dtype != torch.bfloat16:
        raise TypeError(f"v must be torch.bfloat16 (got {v.dtype})")
    if v.device != q.device:
        raise ValueError(f"v must be on {q.device} (got {v.device})")
    if not v.is_contiguous():
        raise ValueError("v must be contiguous")


def _validate_packed_flash_attn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr: torch.Tensor,
) -> list[int]:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "q, k, and v must be rank-3 packed tensors shaped [tokens, heads, dim] "
            f"(got q.ndim={q.ndim}, k.ndim={k.ndim}, v.ndim={v.ndim})"
        )
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"BF16 flash attention currently requires head_dim={HEAD_DIM} "
            f"(got q={q.shape[-1]}, k={k.shape[-1]}, v={v.shape[-1]})"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            f"q, k, and v must have the same token dimension (got {q.shape[0]}, {k.shape[0]}, {v.shape[0]})"
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"k and v must have the same number of KV heads (got {k.shape[1]}, {v.shape[1]})")
    if q.shape[1] == 0 or k.shape[1] == 0:
        raise ValueError("q_heads and kv_heads must be nonzero")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads (got {q.shape[1]} and {k.shape[1]})")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError(
            f"q, k, and v must be torch.bfloat16 (got q={q.dtype}, k={k.dtype}, v={v.dtype})"
        )
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise ValueError(
            f"q, k, and v must be CUDA tensors (got q={q.device}, k={k.device}, v={v.device})"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"q, k, and v must be on the same device (got {q.device}, {k.device}, {v.device})"
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must be contiguous")

    if not isinstance(seq_ptr, torch.Tensor):
        raise TypeError("seq_ptr must be torch.Tensor")
    if seq_ptr.ndim != 1:
        raise ValueError(f"seq_ptr must be rank-1 (got seq_ptr.ndim={seq_ptr.ndim})")
    if seq_ptr.numel() < 2:
        raise ValueError("seq_ptr must have at least two elements")
    if seq_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"seq_ptr must be torch.int32 or torch.int64 (got {seq_ptr.dtype})")
    if not seq_ptr.is_contiguous():
        raise ValueError("seq_ptr must be contiguous")

    seq_ptr_cpu = seq_ptr.detach().to(device="cpu", dtype=torch.int64).tolist()
    if seq_ptr_cpu[0] != 0:
        raise ValueError(f"seq_ptr must start at 0 (got {seq_ptr_cpu[0]})")
    if seq_ptr_cpu[-1] != q.shape[0]:
        raise ValueError(f"seq_ptr must end at total_tokens={q.shape[0]} (got {seq_ptr_cpu[-1]})")
    for idx in range(len(seq_ptr_cpu) - 1):
        if seq_ptr_cpu[idx] > seq_ptr_cpu[idx + 1]:
            raise ValueError("seq_ptr must be nondecreasing")
    return seq_ptr_cpu


@avelang.jit
def _tiled_layout_clear_qk_tiles(
    q_tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    k_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    query_row: al.u32,
    tid_u32: al.u32,
):
    zero_bf16 = al.convert(0.0, al.bf16)
    for dim in al.range(HEAD_DIM):
        if query_row < BLOCK_ROWS:
            q_tile[query_row, dim] = zero_bf16
        if tid_u32 < BLOCK_COLS:
            k_tile[tid_u32, dim] = zero_bf16


@avelang.jit
def _tiled_layout_clear_k_tile(
    k_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    tid_u32: al.u32,
):
    zero_bf16 = al.convert(0.0, al.bf16)
    if tid_u32 < BLOCK_COLS:
        for dim in al.range(HEAD_DIM):
            k_tile[tid_u32, dim] = zero_bf16


@avelang.jit
def _tiled_layout_fetch_global_q(
    ret: al.Tensor((HEAD_DIM,), al.bf16),
    q_rsrc: al.Tensor((1, 4), al.i32),
    query_row: al.u32,
):
    zero_i32 = al.convert(0, al.i32)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    for dim_vec in al.range(HEAD_DIM // VEC_SIZE):
        q_base = query_row * head_dim_u32
        q_offset = al.convert(
            (q_base + al.convert(dim_vec * VEC_SIZE, al.u32)) * bf16_bytes_u32,
            al.i32,
        )
        q_data = al.amdgpu.raw_buffer_load_x4(q_rsrc[0], zero_i32, q_offset, 0)
        q_frag = al.view(q_data, al.Tensor((2, 4, 1), al.bf16))
        for subfrag in al.range(2):
            for elem in al.range(4):
                ret[dim_vec * VEC_SIZE + subfrag * 4 + elem] = q_frag[subfrag, elem, 0]


@avelang.jit
def _tiled_layout_fetch_global_k(
    ret: al.Tensor((HEAD_DIM,), al.bf16),
    k_rsrc: al.Tensor((1, 4), al.i32),
    key_row: al.u32,
):
    zero_i32 = al.convert(0, al.i32)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    for dim_vec in al.range(HEAD_DIM // VEC_SIZE):
        k_base = key_row * head_dim_u32
        k_offset = al.convert(
            (k_base + al.convert(dim_vec * VEC_SIZE, al.u32)) * bf16_bytes_u32,
            al.i32,
        )
        k_data = al.amdgpu.raw_buffer_load_x4(k_rsrc[0], zero_i32, k_offset, 0)
        k_frag = al.view(k_data, al.Tensor((2, 4, 1), al.bf16))
        for subfrag in al.range(2):
            for elem in al.range(4):
                ret[dim_vec * VEC_SIZE + subfrag * 4 + elem] = k_frag[subfrag, elem, 0]


@avelang.jit
def _tiled_layout_store_shm_q(
    q_tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    loaded_q: al.Tensor((HEAD_DIM,), al.bf16),
    query_row: al.u32,
):
    for dim in al.range(HEAD_DIM):
        q_tile[query_row, dim] = loaded_q[dim]


@avelang.jit
def _tiled_layout_store_shm_k(
    k_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    loaded_k: al.Tensor((HEAD_DIM,), al.bf16),
    key_row: al.u32,
):
    for dim in al.range(HEAD_DIM):
        k_tile[key_row, dim] = loaded_k[dim]


@avelang.jit
def _tiled_layout_store_debug_rows(
    debug_tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    tid_u32: al.u32,
):
    idx = tid_u32
    vecs_per_row = HEAD_DIM // VEC_SIZE
    for _ in al.range(SHM_Q_VECS // THREADS):
        row = idx // vecs_per_row
        col = idx % vecs_per_row
        for elem in al.range(VEC_SIZE):
            debug_tile[row, col * VEC_SIZE + elem] = tile[row, col * VEC_SIZE + elem]
        idx += THREADS


@avelang.jit
def _tiled_layout_store_debug_cols(
    debug_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    tid_u32: al.u32,
):
    idx = tid_u32
    vecs_per_row = HEAD_DIM // VEC_SIZE
    for _ in al.range(SHM_K_VECS // THREADS):
        row = idx // vecs_per_row
        col = idx % vecs_per_row
        for elem in al.range(VEC_SIZE):
            debug_tile[row, col * VEC_SIZE + elem] = tile[row, col * VEC_SIZE + elem]
        idx += THREADS


@avelang.jit
def _tiled_layout_fetch_reg_q(
    ret: al.Tensor((4,), al.bf16),
    q_tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    query_row: al.u32,
    pack_col: al.u32,
):
    for elem in al.range(4):
        ret[elem] = q_tile[query_row, pack_col + elem]


@avelang.jit
def _tiled_layout_fetch_reg_k(
    ret: al.Tensor((4,), al.bf16),
    k_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    k_row: al.u32,
    pack_col: al.u32,
):
    for elem in al.range(4):
        ret[elem] = k_tile[k_row, pack_col + elem]


@avelang.jit
def _gemm_qk(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    q_tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    k_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    query_row: al.u32,
    key_row_local: al.u32,
    lane_half: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    q_frag0 = al.make_local((4,), al.bf16)
    q_frag1 = al.make_local((4,), al.bf16)
    k_frag0 = al.make_local((4,), al.bf16)
    k_frag1 = al.make_local((4,), al.bf16)

    for k_batch in al.range(K_BATCHES):
        for t in al.range(16):
            score_acc[k_batch, t] = zero_f32

    for k_batch in al.range(K_BATCHES):
        for q_batch in al.range(Q_BATCHES):
            q_batch_base = al.convert(q_batch * SCORE_TILE_COLS, al.u32)
            k_batch_base = al.convert(k_batch * WARP_ROWS, al.u32)
            for mfma_k in al.range(2):
                pack_col = q_batch_base + al.convert(mfma_k * VEC_SIZE, al.u32) + lane_half * 4
                k_row = k_batch_base + key_row_local
                if mfma_k == 0:
                    _tiled_layout_fetch_reg_q(q_frag0, q_tile, query_row, pack_col)
                    _tiled_layout_fetch_reg_k(k_frag0, k_tile, k_row, pack_col)
                else:
                    _tiled_layout_fetch_reg_q(q_frag1, q_tile, query_row, pack_col)
                    _tiled_layout_fetch_reg_k(k_frag1, k_tile, k_row, pack_col)
            score_acc[k_batch] = al.amdgpu.mfma_f32_32x32x8_bf16(k_frag0, q_frag0, score_acc[k_batch])
            score_acc[k_batch] = al.amdgpu.mfma_f32_32x32x8_bf16(k_frag1, q_frag1, score_acc[k_batch])


@avelang.jit
def _flash_attn_debug_qk_tile_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    q_debug_ptr: al.Pointer(al.bf16),
    k_debug_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
):
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_cols_u32 = al.convert(BLOCK_COLS, al.u32)

    q_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    q_debug_layout = al.make_layout((BLOCK_ROWS, HEAD_DIM), (HEAD_DIM, 1))
    k_debug_layout = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    out_layout = al.make_layout((BLOCK_ROWS, BLOCK_COLS), (BLOCK_COLS, 1))

    q = al.make_tensor(q_ptr, al.bf16, q_layout)
    k = al.make_tensor(k_ptr, al.bf16, q_layout)
    q_debug = al.make_tensor(q_debug_ptr, al.bf16, q_debug_layout)
    k_debug = al.make_tensor(k_debug_ptr, al.bf16, k_debug_layout)
    out = al.make_tensor(out_ptr, al.f32, out_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    perm_high = (lane_col >> al.convert(2, al.u32)) & al.convert(0x7, al.u32)
    perm_rot = ((perm_high & al.convert(0x1, al.u32)) << al.convert(2, al.u32)) | (
        perm_high >> al.convert(1, al.u32)
    )
    key_row_local = (lane_col & al.convert(0x3, al.u32)) | (perm_rot << al.convert(2, al.u32))

    shm_words = al.make_shared((SHM_Q_WORDS + SHM_K_WORDS,), al.u32)
    q_words = al.subview(shm_words, (0,), (SHM_Q_WORDS,), (1,))
    k_words = al.subview(shm_words, (SHM_Q_WORDS,), (SHM_K_WORDS,), (1,))
    layout_q = al.make_layout((BLOCK_ROWS, HEAD_DIM), (HEAD_DIM, 1))
    layout_k = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    q_tile = al.view(q_words, al.bf16, layout_q)
    k_tile = al.view(k_words, al.bf16, layout_k)
    loaded_q = al.make_local((HEAD_DIM,), al.bf16)
    loaded_k = al.make_local((HEAD_DIM,), al.bf16)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    total_bytes = seq_len * head_dim_u32 * bf16_bytes_u32
    q_rsrc = al.amdgpu.make_rsrc(q, total_bytes)
    k_rsrc = al.amdgpu.make_rsrc(k, total_bytes)
    q_rsrc_buf = al.make_local((1, 4), al.i32)
    k_rsrc_buf = al.make_local((1, 4), al.i32)
    q_rsrc_buf[0] = q_rsrc
    k_rsrc_buf[0] = k_rsrc

    _tiled_layout_clear_qk_tiles(q_tile, k_tile, query_row, tid_u32)

    if lane_col < warp_rows_u32:
        _tiled_layout_fetch_global_q(loaded_q, q_rsrc_buf, query_row)
        _tiled_layout_store_shm_q(q_tile, loaded_q, query_row)

    if tid_u32 < block_cols_u32:
        _tiled_layout_fetch_global_k(loaded_k, k_rsrc_buf, tid_u32)
        _tiled_layout_store_shm_k(k_tile, loaded_k, tid_u32)

    al.syncthreads()

    _tiled_layout_store_debug_rows(q_debug, q_tile, tid_u32)
    _tiled_layout_store_debug_cols(k_debug, k_tile, tid_u32)

    al.syncthreads()

    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    _gemm_qk(score_acc, q_tile, k_tile, query_row, key_row_local, lane_half)

    if query_row < BLOCK_ROWS:
        for k_batch in al.range(K_BATCHES):
            col_base = al.convert(k_batch * WARP_ROWS, al.u32) + lane_half * HALF_WARP_ROWS
            for t in al.range(16):
                out[query_row, col_base + t] = score_acc[k_batch, t]


@avelang.jit
def _flash_attn_debug_apply_causal_mask_kernel(
    scores_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    neg_inf = al.convert(NEG_INF, al.f32)

    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    half_warp_rows_u32 = al.convert(HALF_WARP_ROWS, al.u32)
    block_rows_u32 = al.convert(BLOCK_ROWS, al.u32)

    layout = al.make_layout((BLOCK_ROWS, BLOCK_COLS), (BLOCK_COLS, 1))
    scores = al.make_tensor(scores_ptr, al.f32, layout)
    out = al.make_tensor(out_ptr, al.f32, layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    for k_batch in al.range(K_BATCHES):
        col_block_base = al.convert(k_batch * WARP_ROWS, al.u32) + lane_half * half_warp_rows_u32
        for t in al.range(16):
            masked_score = zero_f32
            global_col = col_block_base + t
            if query_row < seq_len:
                masked_score = neg_inf
                if global_col < seq_len:
                    if query_row >= global_col:
                        masked_score = scores[query_row, global_col]
            out[query_row, global_col] = masked_score


@avelang.jit
def _compute_row_max(score_acc: al.Tensor((K_BATCHES, 16), al.f32)) -> al.f32:
    neg_inf = al.convert(NEG_INF, al.f32)
    local_max = neg_inf
    for k_batch in al.range(K_BATCHES):
        for t in al.range(16):
            if score_acc[k_batch, t] > local_max:
                local_max = score_acc[k_batch, t]
    partner_max = al.shuffle_xor(local_max, 32, 64)
    block_max = local_max
    if partner_max > block_max:
        block_max = partner_max
    return block_max


@avelang.jit
def _compute_ptilde(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    row_max: al.f32,
    scale_log2: al.f32,
 ) -> al.f32:
    zero_f32 = al.convert(0.0, al.f32)
    local_sum = zero_f32
    row_max_scaled = row_max * scale_log2
    for k_batch in al.range(K_BATCHES):
        for t in al.range(16):
            score_acc[k_batch, t] = al.exp2(score_acc[k_batch, t] * scale_log2 - row_max_scaled)
            local_sum = local_sum + score_acc[k_batch, t]
    return local_sum


@avelang.jit
def _compute_row_sum(local_sum: al.f32) -> al.f32:
    return local_sum + al.shuffle_xor(local_sum, 32, 64)


@avelang.jit
def _multiply_alpha_o(alpha: al.f32, out_acc: al.Tensor((HEAD_DIM,), al.f32)):
    for d in al.range(HEAD_DIM):
        out_acc[d] = out_acc[d] * alpha


@avelang.jit
def _apply_causal_mask(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    global_query_row: al.u32,
    key_base: al.u32,
    lane_half: al.u32,
    seq_len: al.u32,
):
    neg_inf = al.convert(NEG_INF, al.f32)
    half_warp_rows_u32 = al.convert(HALF_WARP_ROWS, al.u32)
    for k_batch in al.range(K_BATCHES):
        col_block_base = key_base + al.convert(k_batch * WARP_ROWS, al.u32) + lane_half * half_warp_rows_u32
        for t in al.range(16):
            global_col = col_block_base + t
            if global_query_row >= seq_len or global_col >= seq_len or global_query_row < global_col:
                score_acc[k_batch, t] = neg_inf


@avelang.jit
def _compute_local_row_max(score_acc: al.Tensor((K_BATCHES, 16), al.f32)) -> al.f32:
    neg_inf = al.convert(NEG_INF, al.f32)
    local_max = neg_inf
    for k_batch in al.range(K_BATCHES):
        for t in al.range(16):
            if score_acc[k_batch, t] > local_max:
                local_max = score_acc[k_batch, t]
    return local_max


@avelang.jit
def _gemm_o(
    out_acc: al.Tensor((HEAD_DIM,), al.f32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    v_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    lane_half: al.u32,
    key_base: al.u32,
    seq_len: al.u32,
):
    half_warp_rows_u32 = al.convert(HALF_WARP_ROWS, al.u32)
    for k_batch in al.range(K_BATCHES):
        local_col_base = al.convert(k_batch * WARP_ROWS, al.u32)
        for t in al.range(16):
            p0 = score_acc[k_batch, t]
            # Both halves of the wave must execute the shuffle so lane_half==0
            # can legally read the partner half.
            p1 = al.shuffle_xor(score_acc[k_batch, t], 32, 64)
            row0 = local_col_base + t
            row1 = local_col_base + half_warp_rows_u32 + t
            if lane_half == 0:
                if key_base + row0 < seq_len:
                    for d in al.range(HEAD_DIM):
                        out_acc[d] = out_acc[d] + p0 * al.convert(v_tile[row0, d], al.f32)
                if key_base + row1 < seq_len:
                    for d in al.range(HEAD_DIM):
                        out_acc[d] = out_acc[d] + p1 * al.convert(v_tile[row1, d], al.f32)


@avelang.jit
def _gemm_o_mfma(
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    v_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    out_row_local: al.u32,
    lane_half: al.u32,
):
    score_frag = al.make_local((4,), al.bf16)
    v_frag = al.make_local((4,), al.bf16)

    for o_batch in al.range(O_BATCHES):
        out_acc[o_batch] = al.full((16,), 0.0, al.f32)
        out_col = al.convert(o_batch * WARP_ROWS, al.u32) + out_row_local
        for k_batch in al.range(K_BATCHES):
            for k_slice in al.range(4):
                row_base = (
                    al.convert(k_batch * WARP_ROWS, al.u32)
                    + al.convert(k_slice * VEC_SIZE, al.u32)
                    + lane_half * 4
                )
                for elem in al.range(4):
                    score_frag[elem] = al.convert(score_acc[k_batch, k_slice * 4 + elem], al.bf16)
                    v_frag[elem] = v_tile[row_base + elem, out_col]
                out_acc[o_batch] = al.amdgpu.mfma_f32_32x32x8_bf16(v_frag, score_frag, out_acc[o_batch])


@avelang.jit
def _pack_gemm_o_score_mfma(
    packed_score: al.Tensor((K_BATCHES, 16), al.f32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    lane_half: al.u32,
):
    for k_batch in al.range(K_BATCHES):
        for elem in al.range(4):
            lo0 = score_acc[k_batch, elem]
            lo1 = score_acc[k_batch, 4 + elem]
            hi0 = score_acc[k_batch, 8 + elem]
            hi1 = score_acc[k_batch, 12 + elem]
            partner_lo0 = al.shuffle_xor(lo0, 32, 64)
            partner_lo1 = al.shuffle_xor(lo1, 32, 64)
            partner_hi0 = al.shuffle_xor(hi0, 32, 64)
            partner_hi1 = al.shuffle_xor(hi1, 32, 64)
            if lane_half == 0:
                packed_score[k_batch, elem] = lo0
                packed_score[k_batch, 4 + elem] = hi0
                packed_score[k_batch, 8 + elem] = partner_lo0
                packed_score[k_batch, 12 + elem] = partner_hi0
            else:
                packed_score[k_batch, elem] = partner_lo1
                packed_score[k_batch, 4 + elem] = partner_hi1
                packed_score[k_batch, 8 + elem] = lo1
                packed_score[k_batch, 12 + elem] = hi1


@avelang.jit
def _store_gemm_o_mfma(
    out_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    query_row: al.u32,
    lane_half: al.u32,
):
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    half_warp_rows_u32 = al.convert(HALF_WARP_ROWS, al.u32)
    out_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    out = al.make_tensor(out_ptr, al.f32, out_layout)
    dim_block = lane_half * half_warp_rows_u32
    for o_batch in al.range(O_BATCHES):
        dim_base = al.convert(o_batch * WARP_ROWS, al.u32) + dim_block
        for t in al.range(16):
            out[query_row, dim_base + t] = out_acc[o_batch, t]


@avelang.jit
def _normalize_softmax_lse(
    out_acc: al.Tensor((HEAD_DIM,), al.f32),
    out_ptr: al.Pointer(al.bf16),
    seq_len: al.u32,
    global_query_row: al.u32,
    lane_half: al.u32,
    l: al.f32,
):
    one_f32 = al.convert(1.0, al.f32)
    if lane_half == 0:
        head_dim_u32 = al.convert(HEAD_DIM, al.u32)
        out_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
        out = al.make_tensor(out_ptr, al.bf16, out_layout)
        inv_l = one_f32 / l
        for d in al.range(HEAD_DIM):
            out[global_query_row, d] = al.convert(out_acc[d] * inv_l, al.bf16)


@avelang.jit
def _flash_attn_debug_softmax_kernel(
    scores_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)
    scale_log2 = al.convert(SCALE_LOG2, al.f32)

    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    half_warp_rows_u32 = al.convert(HALF_WARP_ROWS, al.u32)
    block_rows_u32 = al.convert(BLOCK_ROWS, al.u32)

    layout = al.make_layout((BLOCK_ROWS, BLOCK_COLS), (BLOCK_COLS, 1))
    scores = al.make_tensor(scores_ptr, al.f32, layout)
    out = al.make_tensor(out_ptr, al.f32, layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    score_acc = al.make_local((K_BATCHES, 16), al.f32)

    valid_q = query_row < seq_len
    for k_batch in al.range(K_BATCHES):
        col_block_base = al.convert(k_batch * WARP_ROWS, al.u32) + lane_half * half_warp_rows_u32
        for t in al.range(16):
            score_acc[k_batch, t] = zero_f32
            if valid_q:
                score_acc[k_batch, t] = scores[query_row, col_block_base + t]
    block_max = zero_f32
    row_sum = one_f32
    if valid_q:
        block_max = _compute_row_max(score_acc)
        local_sum = _compute_ptilde(score_acc, block_max, scale_log2)
        row_sum = _compute_row_sum(local_sum)
    inv_row_sum = zero_f32
    if valid_q:
        inv_row_sum = one_f32 / row_sum

    if query_row < block_rows_u32:
        for k_batch in al.range(K_BATCHES):
            col_base = al.convert(k_batch * WARP_ROWS, al.u32) + lane_half * half_warp_rows_u32
            for t in al.range(16):
                if valid_q:
                    out[query_row, col_base + t] = score_acc[k_batch, t] * inv_row_sum
                else:
                    out[query_row, col_base + t] = zero_f32


@avelang.jit
def _flash_attn_debug_gemm_o_kernel(
    probs_ptr: al.Pointer(al.f32),
    v_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    key_base = al.convert(0, al.u32)

    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_cols_u32 = al.convert(BLOCK_COLS, al.u32)
    two_u32 = al.convert(2, al.u32)
    one_u32 = al.convert(1, al.u32)
    three_u32 = al.convert(0x3, al.u32)
    seven_u32 = al.convert(0x7, al.u32)
    one_mask_u32 = al.convert(0x1, al.u32)

    probs_layout = al.make_layout((BLOCK_ROWS, BLOCK_COLS), (BLOCK_COLS, 1))
    out_layout = al.make_layout((BLOCK_ROWS, head_dim_u32), (head_dim_u32, 1))
    v_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    probs = al.make_tensor(probs_ptr, al.f32, probs_layout)
    out = al.make_tensor(out_ptr, al.f32, out_layout)
    v = al.make_tensor(v_ptr, al.bf16, v_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col
    perm_high = (lane_col >> two_u32) & seven_u32
    perm_rot = ((perm_high & one_mask_u32) << two_u32) | (perm_high >> one_u32)
    out_row_local = (lane_col & three_u32) | (perm_rot << two_u32)

    shm_words = al.make_shared((SHM_K_WORDS,), al.u32)
    layout_kv = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    kv_tile = al.view(shm_words, al.bf16, layout_kv)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    total_bytes = seq_len * head_dim_u32 * bf16_bytes_u32
    v_rsrc = al.amdgpu.make_rsrc(v, total_bytes)
    v_rsrc_buf = al.make_local((1, 4), al.i32)
    v_rsrc_buf[0] = v_rsrc

    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, v_rsrc_buf, tid_u32, key_base)
    al.syncthreads()

    score_acc_rowmajor = al.make_local((K_BATCHES, 16), al.f32)
    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    out_acc = al.make_local((O_BATCHES, 16), al.f32)
    for o_batch in al.range(O_BATCHES):
        out_acc[o_batch] = al.full((16,), 0.0, al.f32)

    for k_batch in al.range(K_BATCHES):
        col_block_base = al.convert(k_batch * WARP_ROWS, al.u32) + lane_half * al.convert(
            HALF_WARP_ROWS, al.u32
        )
        for t in al.range(16):
            score_acc_rowmajor[k_batch, t] = zero_f32
            if query_row < seq_len:
                score_acc_rowmajor[k_batch, t] = probs[query_row, col_block_base + t]

    if query_row < seq_len:
        _pack_gemm_o_score_mfma(score_acc, score_acc_rowmajor, lane_half)
        _gemm_o_mfma(out_acc, score_acc, kv_tile, out_row_local, lane_half)
        _store_gemm_o_mfma(out_ptr, seq_len, out_acc, query_row, lane_half)


@avelang.jit
def _flash_attn_debug_store_o_kernel(
    pre_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)

    pre_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    pre = al.make_tensor(pre_ptr, al.f32, pre_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    out_acc = al.make_local((HEAD_DIM,), al.f32)
    for d in al.range(HEAD_DIM):
        out_acc[d] = zero_f32
        if query_row < seq_len and lane_half == 0:
            out_acc[d] = pre[query_row, d]

    if query_row < seq_len:
        _normalize_softmax_lse(out_acc, out_ptr, seq_len, query_row, lane_half, one_f32)


@avelang.jit
def _flash_attn_debug_normalize_o_kernel(
    pre_ptr: al.Pointer(al.f32),
    l_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    one_f32 = al.convert(1.0, al.f32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)

    pre_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    l_layout = al.make_layout((seq_len,), (1,))
    pre = al.make_tensor(pre_ptr, al.f32, pre_layout)
    l_in = al.make_tensor(l_ptr, al.f32, l_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    out_acc = al.make_local((HEAD_DIM,), al.f32)
    l = one_f32
    for d in al.range(HEAD_DIM):
        out_acc[d] = zero_f32
        if query_row < seq_len and lane_half == 0:
            out_acc[d] = pre[query_row, d]
    if query_row < seq_len and lane_half == 0:
        l = l_in[query_row]

    if query_row < seq_len:
        _normalize_softmax_lse(out_acc, out_ptr, seq_len, query_row, lane_half, l)


@avelang.jit
def _flash_attn_debug_single_tile_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    pre_out_ptr: al.Pointer(al.f32),
    l_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    scale_log2 = al.convert(SCALE_LOG2, al.f32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_cols_u32 = al.convert(BLOCK_COLS, al.u32)
    two_u32 = al.convert(2, al.u32)
    one_u32 = al.convert(1, al.u32)
    three_u32 = al.convert(0x3, al.u32)
    seven_u32 = al.convert(0x7, al.u32)
    one_mask_u32 = al.convert(0x1, al.u32)
    key_base = al.convert(0, al.u32)

    l_layout = al.make_layout((seq_len,), (1,))
    l_out = al.make_tensor(l_ptr, al.f32, l_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    perm_high = (lane_col >> two_u32) & seven_u32
    perm_rot = ((perm_high & one_mask_u32) << two_u32) | (perm_high >> one_u32)
    key_row_local = (lane_col & three_u32) | (perm_rot << two_u32)

    shm_words = al.make_shared((SHM_Q_WORDS + SHM_K_WORDS,), al.u32)
    q_words = al.subview(shm_words, (0,), (SHM_Q_WORDS,), (1,))
    kv_words = al.subview(shm_words, (SHM_Q_WORDS,), (SHM_K_WORDS,), (1,))
    layout_q = al.make_layout((BLOCK_ROWS, HEAD_DIM), (HEAD_DIM, 1))
    layout_kv = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    global_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    q_tile = al.view(q_words, al.bf16, layout_q)
    kv_tile = al.view(kv_words, al.bf16, layout_kv)
    q = al.make_tensor(q_ptr, al.bf16, global_layout)
    k = al.make_tensor(k_ptr, al.bf16, global_layout)
    v = al.make_tensor(v_ptr, al.bf16, global_layout)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    total_bytes = seq_len * head_dim_u32 * bf16_bytes_u32
    q_rsrc = al.amdgpu.make_rsrc(q, total_bytes)
    k_rsrc = al.amdgpu.make_rsrc(k, total_bytes)
    v_rsrc = al.amdgpu.make_rsrc(v, total_bytes)
    q_rsrc_buf = al.make_local((1, 4), al.i32)
    k_rsrc_buf = al.make_local((1, 4), al.i32)
    v_rsrc_buf = al.make_local((1, 4), al.i32)
    q_rsrc_buf[0] = q_rsrc
    k_rsrc_buf[0] = k_rsrc
    v_rsrc_buf[0] = v_rsrc

    _tiled_layout_clear_qk_tiles(q_tile, kv_tile, query_row, tid_u32)
    _load_q_tile(q_tile, q_rsrc_buf, query_row, query_row, lane_col)
    al.syncthreads()

    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, k_rsrc_buf, tid_u32, key_base)
    al.syncthreads()

    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    score_acc_mfma = al.make_local((K_BATCHES, 16), al.f32)
    out_acc = al.make_local((O_BATCHES, 16), al.f32)
    for o_batch in al.range(O_BATCHES):
        out_acc[o_batch] = al.full((16,), 0.0, al.f32)

    _gemm_qk(score_acc, q_tile, kv_tile, query_row, key_row_local, lane_half)
    _apply_causal_mask(score_acc, query_row, key_base, lane_half, seq_len)

    l = zero_f32
    block_max = _compute_row_max(score_acc)
    mi_new = block_max
    if block_max <= al.convert(NEG_INF, al.f32):
        mi_new = zero_f32
    local_sum = _compute_ptilde(score_acc, mi_new, scale_log2)
    l = _compute_row_sum(local_sum)

    al.syncthreads()
    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, v_rsrc_buf, tid_u32, key_base)
    al.syncthreads()

    _pack_gemm_o_score_mfma(score_acc_mfma, score_acc, lane_half)
    _gemm_o_mfma(out_acc, score_acc_mfma, kv_tile, key_row_local, lane_half)

    if lane_half == 0 and query_row < seq_len:
        l_out[query_row] = l
    if query_row < seq_len:
        _store_gemm_o_mfma(pre_out_ptr, seq_len, out_acc, query_row, lane_half)


@avelang.jit
def _flash_attn_debug_softmax_stats_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    localmax_ptr: al.Pointer(al.f32),
    blockmax_ptr: al.Pointer(al.f32),
    l_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    scale_log2 = al.convert(SCALE_LOG2, al.f32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    two_u32 = al.convert(2, al.u32)
    one_u32 = al.convert(1, al.u32)
    three_u32 = al.convert(0x3, al.u32)
    seven_u32 = al.convert(0x7, al.u32)
    one_mask_u32 = al.convert(0x1, al.u32)
    key_base = al.convert(0, al.u32)

    localmax_layout = al.make_layout((BLOCK_ROWS, 2), (2, 1))
    scalar_layout = al.make_layout((BLOCK_ROWS,), (1,))
    localmax_out = al.make_tensor(localmax_ptr, al.f32, localmax_layout)
    blockmax_out = al.make_tensor(blockmax_ptr, al.f32, scalar_layout)
    l_out = al.make_tensor(l_ptr, al.f32, scalar_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col

    perm_high = (lane_col >> two_u32) & seven_u32
    perm_rot = ((perm_high & one_mask_u32) << two_u32) | (perm_high >> one_u32)
    key_row_local = (lane_col & three_u32) | (perm_rot << two_u32)

    shm_words = al.make_shared((SHM_Q_WORDS + SHM_K_WORDS,), al.u32)
    q_words = al.subview(shm_words, (0,), (SHM_Q_WORDS,), (1,))
    kv_words = al.subview(shm_words, (SHM_Q_WORDS,), (SHM_K_WORDS,), (1,))
    layout_q = al.make_layout((BLOCK_ROWS, HEAD_DIM), (HEAD_DIM, 1))
    layout_kv = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    global_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    q_tile = al.view(q_words, al.bf16, layout_q)
    kv_tile = al.view(kv_words, al.bf16, layout_kv)
    q = al.make_tensor(q_ptr, al.bf16, global_layout)
    k = al.make_tensor(k_ptr, al.bf16, global_layout)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    total_bytes = seq_len * head_dim_u32 * bf16_bytes_u32
    q_rsrc = al.amdgpu.make_rsrc(q, total_bytes)
    k_rsrc = al.amdgpu.make_rsrc(k, total_bytes)
    q_rsrc_buf = al.make_local((1, 4), al.i32)
    k_rsrc_buf = al.make_local((1, 4), al.i32)
    q_rsrc_buf[0] = q_rsrc
    k_rsrc_buf[0] = k_rsrc

    _tiled_layout_clear_qk_tiles(q_tile, kv_tile, query_row, tid_u32)
    _load_q_tile(q_tile, q_rsrc_buf, query_row, query_row, lane_col)
    al.syncthreads()

    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, k_rsrc_buf, tid_u32, key_base)
    al.syncthreads()

    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    _gemm_qk(score_acc, q_tile, kv_tile, query_row, key_row_local, lane_half)
    _apply_causal_mask(score_acc, query_row, key_base, lane_half, seq_len)

    local_max = _compute_local_row_max(score_acc)
    partner_max = al.shuffle_xor(local_max, 32, 64)
    block_max = local_max
    if partner_max > block_max:
        block_max = partner_max
    mi_new = block_max
    if block_max <= al.convert(NEG_INF, al.f32):
        mi_new = zero_f32
    local_sum = _compute_ptilde(score_acc, mi_new, scale_log2)
    l = _compute_row_sum(local_sum)

    if query_row < al.convert(BLOCK_ROWS, al.u32):
        localmax_out[query_row, lane_half] = local_max
        if lane_half == 0:
            blockmax_out[query_row] = block_max
            l_out[query_row] = l


@avelang.jit
def _flash_attn_tile_offdiag_step_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    pre_ptr: al.Pointer(al.f32),
    mi_ptr: al.Pointer(al.f32),
    l_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
    q_tile_idx: al.u32,
    key_tile_idx: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    scale_log2 = al.convert(SCALE_LOG2, al.f32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_cols_u32 = al.convert(BLOCK_COLS, al.u32)
    block_rows_u32 = al.convert(BLOCK_ROWS, al.u32)
    two_u32 = al.convert(2, al.u32)
    one_u32 = al.convert(1, al.u32)
    three_u32 = al.convert(0x3, al.u32)
    seven_u32 = al.convert(0x7, al.u32)
    one_mask_u32 = al.convert(0x1, al.u32)

    q_base = q_tile_idx * block_rows_u32
    key_base = key_tile_idx * block_cols_u32
    pre_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    scalar_layout = al.make_layout((seq_len,), (1,))
    pre_state = al.make_tensor(pre_ptr, al.f32, pre_layout)
    mi_state = al.make_tensor(mi_ptr, al.f32, scalar_layout)
    l_state = al.make_tensor(l_ptr, al.f32, scalar_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col
    global_query_row = q_base + query_row

    if q_base >= seq_len:
        return

    perm_high = (lane_col >> two_u32) & seven_u32
    perm_rot = ((perm_high & one_mask_u32) << two_u32) | (perm_high >> one_u32)
    key_row_local = (lane_col & three_u32) | (perm_rot << two_u32)

    shm_words = al.make_shared((SHM_Q_WORDS + SHM_K_WORDS,), al.u32)
    q_words = al.subview(shm_words, (0,), (SHM_Q_WORDS,), (1,))
    kv_words = al.subview(shm_words, (SHM_Q_WORDS,), (SHM_K_WORDS,), (1,))
    layout_q = al.make_layout((BLOCK_ROWS, HEAD_DIM), (HEAD_DIM, 1))
    layout_kv = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    global_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    q_tile = al.view(q_words, al.bf16, layout_q)
    kv_tile = al.view(kv_words, al.bf16, layout_kv)
    q = al.make_tensor(q_ptr, al.bf16, global_layout)
    k = al.make_tensor(k_ptr, al.bf16, global_layout)
    v = al.make_tensor(v_ptr, al.bf16, global_layout)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    total_bytes = seq_len * head_dim_u32 * bf16_bytes_u32
    q_rsrc = al.amdgpu.make_rsrc(q, total_bytes)
    k_rsrc = al.amdgpu.make_rsrc(k, total_bytes)
    v_rsrc = al.amdgpu.make_rsrc(v, total_bytes)
    q_rsrc_buf = al.make_local((1, 4), al.i32)
    k_rsrc_buf = al.make_local((1, 4), al.i32)
    v_rsrc_buf = al.make_local((1, 4), al.i32)
    q_rsrc_buf[0] = q_rsrc
    k_rsrc_buf[0] = k_rsrc
    v_rsrc_buf[0] = v_rsrc

    _tiled_layout_clear_qk_tiles(q_tile, kv_tile, query_row, tid_u32)
    _load_q_tile(q_tile, q_rsrc_buf, query_row, global_query_row, lane_col)
    al.syncthreads()

    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, k_rsrc_buf, tid_u32, key_base)
    al.syncthreads()

    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    out_acc = al.make_local((HEAD_DIM,), al.f32)
    mi = zero_f32
    l = zero_f32
    for d in al.range(HEAD_DIM):
        out_acc[d] = zero_f32

    if global_query_row < seq_len:
        mi = mi_state[global_query_row]
        l = l_state[global_query_row]
        if lane_half == 0:
            for d in al.range(HEAD_DIM):
                out_acc[d] = pre_state[global_query_row, d]

    _gemm_qk(score_acc, q_tile, kv_tile, query_row, key_row_local, lane_half)

    if global_query_row < seq_len:
        block_max = _compute_row_max(score_acc)
        mi_new = block_max
        if mi > block_max:
            mi_new = mi
        local_sum = _compute_ptilde(score_acc, mi_new, scale_log2)
        row_sum = _compute_row_sum(local_sum)
        scaling = al.exp2((mi - mi_new) * scale_log2)
        l = scaling * l + row_sum
        if lane_half == 0:
            _multiply_alpha_o(scaling, out_acc)
        mi = mi_new

    if lane_half == 0 and global_query_row < seq_len:
        mi_state[global_query_row] = mi
        l_state[global_query_row] = l

    al.syncthreads()
    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, v_rsrc_buf, tid_u32, key_base)
    al.syncthreads()

    if global_query_row < seq_len:
        _gemm_o(out_acc, score_acc, kv_tile, lane_half, key_base, seq_len)

    if lane_half == 0 and global_query_row < seq_len:
        for d in al.range(HEAD_DIM):
            pre_state[global_query_row, d] = out_acc[d]


@avelang.jit
def _flash_attn_tile_diag_step_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    pre_ptr: al.Pointer(al.f32),
    mi_ptr: al.Pointer(al.f32),
    l_ptr: al.Pointer(al.f32),
    seq_len: al.u32,
    q_tile_idx: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    scale_log2 = al.convert(SCALE_LOG2, al.f32)
    head_dim_u32 = al.convert(HEAD_DIM, al.u32)
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_cols_u32 = al.convert(BLOCK_COLS, al.u32)
    block_rows_u32 = al.convert(BLOCK_ROWS, al.u32)
    two_u32 = al.convert(2, al.u32)
    one_u32 = al.convert(1, al.u32)
    three_u32 = al.convert(0x3, al.u32)
    seven_u32 = al.convert(0x7, al.u32)
    one_mask_u32 = al.convert(0x1, al.u32)

    q_base = q_tile_idx * block_rows_u32
    pre_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    scalar_layout = al.make_layout((seq_len,), (1,))
    pre_state = al.make_tensor(pre_ptr, al.f32, pre_layout)
    mi_state = al.make_tensor(mi_ptr, al.f32, scalar_layout)
    l_state = al.make_tensor(l_ptr, al.f32, scalar_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col
    global_query_row = q_base + query_row

    if q_base >= seq_len:
        return

    perm_high = (lane_col >> two_u32) & seven_u32
    perm_rot = ((perm_high & one_mask_u32) << two_u32) | (perm_high >> one_u32)
    key_row_local = (lane_col & three_u32) | (perm_rot << two_u32)

    shm_words = al.make_shared((SHM_Q_WORDS + SHM_K_WORDS,), al.u32)
    q_words = al.subview(shm_words, (0,), (SHM_Q_WORDS,), (1,))
    kv_words = al.subview(shm_words, (SHM_Q_WORDS,), (SHM_K_WORDS,), (1,))
    layout_q = al.make_layout((BLOCK_ROWS, HEAD_DIM), (HEAD_DIM, 1))
    layout_kv = al.make_layout((BLOCK_COLS, HEAD_DIM), (HEAD_DIM, 1))
    global_layout = al.make_layout((seq_len, head_dim_u32), (head_dim_u32, 1))
    q_tile = al.view(q_words, al.bf16, layout_q)
    kv_tile = al.view(kv_words, al.bf16, layout_kv)
    q = al.make_tensor(q_ptr, al.bf16, global_layout)
    k = al.make_tensor(k_ptr, al.bf16, global_layout)
    v = al.make_tensor(v_ptr, al.bf16, global_layout)
    bf16_bytes_u32 = al.convert(BF16_BYTES, al.u32)
    total_bytes = seq_len * head_dim_u32 * bf16_bytes_u32
    q_rsrc = al.amdgpu.make_rsrc(q, total_bytes)
    k_rsrc = al.amdgpu.make_rsrc(k, total_bytes)
    v_rsrc = al.amdgpu.make_rsrc(v, total_bytes)
    q_rsrc_buf = al.make_local((1, 4), al.i32)
    k_rsrc_buf = al.make_local((1, 4), al.i32)
    v_rsrc_buf = al.make_local((1, 4), al.i32)
    q_rsrc_buf[0] = q_rsrc
    k_rsrc_buf[0] = k_rsrc
    v_rsrc_buf[0] = v_rsrc

    _tiled_layout_clear_qk_tiles(q_tile, kv_tile, query_row, tid_u32)
    _load_q_tile(q_tile, q_rsrc_buf, query_row, global_query_row, lane_col)
    al.syncthreads()

    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, k_rsrc_buf, tid_u32, q_base)
    al.syncthreads()

    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    out_acc = al.make_local((HEAD_DIM,), al.f32)
    for d in al.range(HEAD_DIM):
        out_acc[d] = zero_f32

    _gemm_qk(score_acc, q_tile, kv_tile, query_row, key_row_local, lane_half)
    _apply_causal_mask(score_acc, global_query_row, q_base, lane_half, seq_len)

    block_max = _compute_row_max(score_acc)
    mi = block_max
    if block_max <= al.convert(NEG_INF, al.f32):
        mi = zero_f32
    local_sum = _compute_ptilde(score_acc, mi, scale_log2)
    l = _compute_row_sum(local_sum)

    if lane_half == 0 and global_query_row < seq_len:
        mi_state[global_query_row] = mi
        l_state[global_query_row] = l

    al.syncthreads()
    _tiled_layout_clear_k_tile(kv_tile, tid_u32)
    _load_kv_tile(kv_tile, v_rsrc_buf, tid_u32, q_base)
    al.syncthreads()

    _gemm_o(out_acc, score_acc, kv_tile, lane_half, q_base, seq_len)

    if lane_half == 0 and global_query_row < seq_len:
        for d in al.range(HEAD_DIM):
            pre_state[global_query_row, d] = out_acc[d]


@avelang.jit
def _flash_attn_debug_index_kernel(
    row_out_ptr: al.Pointer(al.u32),
    global_row_out_ptr: al.Pointer(al.u32),
    valid_out_ptr: al.Pointer(al.u32),
    seq_len: al.u32,
    idx_q: al.u32,
):
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_rows_u32 = al.convert(BLOCK_ROWS, al.u32)
    row_layout = al.make_layout((BLOCK_ROWS,), (1,))
    row_out = al.make_tensor(row_out_ptr, al.u32, row_layout)
    global_row_out = al.make_tensor(global_row_out_ptr, al.u32, row_layout)
    valid_out = al.make_tensor(valid_out_ptr, al.u32, row_layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col
    global_query_row = idx_q * block_rows_u32 + query_row
    valid_q = global_query_row < seq_len

    if lane_half == 0 and query_row < block_rows_u32:
        row_out[query_row] = query_row
        global_row_out[query_row] = global_query_row
        if valid_q:
            valid_out[query_row] = al.convert(1, al.u32)
        else:
            valid_out[query_row] = al.convert(0, al.u32)


@avelang.jit
def _flash_attn_debug_valid_loop_kernel(
    out_ptr: al.Pointer(al.u32),
    seq_len: al.u32,
):
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    layout = al.make_layout((BLOCK_ROWS,), (1,))
    out = al.make_tensor(out_ptr, al.u32, layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col
    valid_q = query_row < seq_len
    one_u32 = al.convert(1, al.u32)
    zero_u32 = al.convert(0, al.u32)

    if lane_half == 0 and query_row < al.convert(BLOCK_ROWS, al.u32):
        out[query_row] = zero_u32

    for _ in al.range(one_u32):
        if lane_half == 0 and query_row < al.convert(BLOCK_ROWS, al.u32):
            if valid_q:
                out[query_row] = one_u32


@avelang.jit
def _flash_attn_debug_dynamic_range_kernel(
    out_ptr: al.Pointer(al.u32),
    count: al.u32,
):
    warp_size_u32 = al.convert(WARP_SIZE, al.u32)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    block_rows_u32 = al.convert(BLOCK_ROWS, al.u32)
    layout = al.make_layout((BLOCK_ROWS,), (1,))
    out = al.make_tensor(out_ptr, al.u32, layout)

    tid = al.thread_id(0)
    tid_u32 = al.convert(tid, al.u32)
    wid = tid_u32 // warp_size_u32
    wtid = tid_u32 % warp_size_u32
    lane_col = wtid % warp_rows_u32
    lane_half = wtid // warp_rows_u32
    query_row = wid * warp_rows_u32 + lane_col
    zero_u32 = al.convert(0, al.u32)
    one_u32 = al.convert(1, al.u32)

    if lane_half == 0 and query_row < block_rows_u32:
        out[query_row] = zero_u32

    for _ in al.range(count):
        if lane_half == 0 and query_row < block_rows_u32:
            out[query_row] = one_u32


@avelang.jit
def _load_q_tile(
    q_tile: al.Tensor((BLOCK_ROWS, HEAD_DIM), al.bf16),
    q_rsrc: al.Tensor((1, 4), al.i32),
    query_row_local: al.u32,
    global_query_row: al.u32,
    lane_col: al.u32,
):
    loaded_q = al.make_local((HEAD_DIM,), al.bf16)
    warp_rows_u32 = al.convert(WARP_ROWS, al.u32)
    if lane_col < warp_rows_u32:
        _tiled_layout_fetch_global_q(loaded_q, q_rsrc, global_query_row)
        _tiled_layout_store_shm_q(q_tile, loaded_q, query_row_local)


@avelang.jit
def _load_kv_tile(
    kv_tile: al.Tensor((BLOCK_COLS, HEAD_DIM), al.bf16),
    kv_rsrc: al.Tensor((1, 4), al.i32),
    tid_u32: al.u32,
    key_base: al.u32,
):
    loaded_kv = al.make_local((HEAD_DIM,), al.bf16)
    block_cols_u32 = al.convert(BLOCK_COLS, al.u32)
    if tid_u32 < block_cols_u32:
        global_row = key_base + tid_u32
        _tiled_layout_fetch_global_k(loaded_kv, kv_rsrc, global_row)
        _tiled_layout_store_shm_k(kv_tile, loaded_kv, tid_u32)


def _run_flash_attn_tiles(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    v_head: torch.Tensor,
    pre: torch.Tensor,
    mi: torch.Tensor,
    l: torch.Tensor,
    seq_len: int,
) -> None:
    row_tiles = math.ceil(seq_len / BLOCK_ROWS)
    for q_tile_idx in range(row_tiles):
        for key_tile_idx in range(q_tile_idx, -1, -1):
            if key_tile_idx == q_tile_idx:
                _flash_attn_tile_diag_step_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
                    q_head,
                    k_head,
                    v_head,
                    pre,
                    mi,
                    l,
                    seq_len,
                    q_tile_idx,
                )
            else:
                _flash_attn_tile_offdiag_step_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
                    q_head,
                    k_head,
                    v_head,
                    pre,
                    mi,
                    l,
                    seq_len,
                    q_tile_idx,
                    key_tile_idx,
                )


def flash_attn_debug_qk_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
    print_loads: bool = False,
) -> torch.Tensor:
    _validate_qk_shape(q, k)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    q_debug, k_debug, out = _flash_attn_debug_qk_tile_outputs(q, k, batch_idx, head_idx)
    _flash_attn_debug_qk_tile_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        q[batch_idx, head_idx].contiguous(),
        k[batch_idx, head_idx].contiguous(),
        q_debug,
        k_debug,
        out,
        seq_len,
    )
    if print_loads:
        print("loaded_q_row0", q_debug[0, :8].float().cpu().tolist())
        print("loaded_q_row1", q_debug[1, :8].float().cpu().tolist())
        print("loaded_k_row0", k_debug[0, :8].float().cpu().tolist())
        print("loaded_k_row1", k_debug[1, :8].float().cpu().tolist())
    return out


def _flash_attn_debug_qk_tile_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int,
    head_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_head = q[batch_idx, head_idx].contiguous()
    k_head = k[batch_idx, head_idx].contiguous()
    q_debug = torch.zeros((BLOCK_ROWS, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    k_debug = torch.zeros((BLOCK_COLS, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    out = torch.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=torch.float32, device=q.device)
    return q_debug, k_debug, out


def flash_attn_debug_qk_loads(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_qk_shape(q, k)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    q_debug, k_debug, out = _flash_attn_debug_qk_tile_outputs(q, k, batch_idx, head_idx)
    _flash_attn_debug_qk_tile_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        q[batch_idx, head_idx].contiguous(),
        k[batch_idx, head_idx].contiguous(),
        q_debug,
        k_debug,
        out,
        seq_len,
    )
    return q_debug, k_debug


def flash_attn_debug_qk_tile_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    _validate_qk_shape(q, k)

    q_tile = torch.zeros((BLOCK_ROWS, HEAD_DIM), dtype=torch.float32, device=q.device)
    k_tile = torch.zeros((BLOCK_COLS, HEAD_DIM), dtype=torch.float32, device=q.device)

    valid_rows = min(q.shape[2], BLOCK_ROWS)
    valid_cols = min(k.shape[2], BLOCK_COLS)
    q_tile[:valid_rows] = q[batch_idx, head_idx, :valid_rows].to(torch.float32)
    k_tile[:valid_cols] = k[batch_idx, head_idx, :valid_cols].to(torch.float32)
    return q_tile @ k_tile.transpose(0, 1)


def flash_attn_debug_softmax_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    masked_scores = flash_attn_debug_apply_causal_mask_tile(q, k, batch_idx, head_idx)

    out = torch.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=torch.float32, device=q.device)
    _flash_attn_debug_softmax_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        masked_scores,
        out,
        q.shape[2],
    )
    return out


def flash_attn_debug_apply_causal_mask_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    _validate_qk_shape(q, k)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    q_debug, k_debug, scores = _flash_attn_debug_qk_tile_outputs(q, k, batch_idx, head_idx)
    _flash_attn_debug_qk_tile_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        q[batch_idx, head_idx].contiguous(),
        k[batch_idx, head_idx].contiguous(),
        q_debug,
        k_debug,
        scores,
        seq_len,
    )

    out = torch.empty_like(scores)
    _flash_attn_debug_apply_causal_mask_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        scores,
        out,
        seq_len,
    )
    return out


def flash_attn_debug_softmax_tile_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    scores = flash_attn_debug_qk_tile_reference(q, k, batch_idx, head_idx)
    valid_len = q.shape[2]
    scores = scores.clone()
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for row in range(BLOCK_ROWS):
        if row >= valid_len:
            scores[row].zero_()
            continue
        scores[row, valid_len:].fill_(NEG_INF)
        scores[row, row + 1 : valid_len].fill_(NEG_INF)
    probs = torch.zeros_like(scores)
    probs[:valid_len, :valid_len] = torch.softmax(scores[:valid_len, :valid_len] * scale, dim=-1)
    return probs


def flash_attn_debug_gemm_o_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    flash_attn_validate_shape(q, k, v)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    probs = flash_attn_debug_softmax_tile(q, k, batch_idx, head_idx)
    out = torch.zeros((BLOCK_ROWS, HEAD_DIM), dtype=torch.float32, device=q.device)
    _flash_attn_debug_gemm_o_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        probs,
        v[batch_idx, head_idx].contiguous(),
        out,
        seq_len,
    )
    return out


def flash_attn_debug_gemm_o_tile_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    flash_attn_validate_shape(q, k, v)
    probs = flash_attn_debug_softmax_tile_reference(q, k, batch_idx, head_idx)
    v_tile = torch.zeros((BLOCK_COLS, HEAD_DIM), dtype=torch.float32, device=q.device)
    valid_cols = min(v.shape[2], BLOCK_COLS)
    v_tile[:valid_cols] = v[batch_idx, head_idx, :valid_cols].to(torch.float32)
    return probs @ v_tile


def flash_attn_debug_store_o(
    pre: torch.Tensor,
) -> torch.Tensor:
    if pre.ndim != 2 or pre.shape[1] != HEAD_DIM:
        raise ValueError(f"pre must have shape [seq, {HEAD_DIM}] (got {tuple(pre.shape)})")
    if pre.dtype != torch.float32:
        raise TypeError(f"pre must be torch.float32 (got {pre.dtype})")
    if pre.device.type != "cuda":
        raise ValueError(f"pre must be CUDA (got {pre.device})")
    if not pre.is_contiguous():
        raise ValueError("pre must be contiguous")

    out = torch.zeros((pre.shape[0], HEAD_DIM), dtype=torch.bfloat16, device=pre.device)
    _flash_attn_debug_store_o_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        pre,
        out,
        pre.shape[0],
    )
    return out


def flash_attn_debug_normalize_o(
    pre: torch.Tensor,
    l: torch.Tensor,
) -> torch.Tensor:
    if pre.ndim != 2 or pre.shape[1] != HEAD_DIM:
        raise ValueError(f"pre must have shape [seq, {HEAD_DIM}] (got {tuple(pre.shape)})")
    if l.ndim != 1 or l.shape[0] != pre.shape[0]:
        raise ValueError(f"l must have shape [{pre.shape[0]}] (got {tuple(l.shape)})")
    if pre.dtype != torch.float32 or l.dtype != torch.float32:
        raise TypeError(f"pre and l must be torch.float32 (got pre={pre.dtype}, l={l.dtype})")
    if pre.device != l.device or pre.device.type != "cuda":
        raise ValueError(f"pre and l must be CUDA on same device (got pre={pre.device}, l={l.device})")
    if not pre.is_contiguous() or not l.is_contiguous():
        raise ValueError("pre and l must be contiguous")

    out = torch.zeros((pre.shape[0], HEAD_DIM), dtype=torch.bfloat16, device=pre.device)
    _flash_attn_debug_normalize_o_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        pre,
        l,
        out,
        pre.shape[0],
    )
    return out


def flash_attn_debug_single_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_attn_validate_shape(q, k, v)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    pre_out = torch.zeros((seq_len, HEAD_DIM), dtype=torch.float32, device=q.device)
    l = torch.zeros((seq_len,), dtype=torch.float32, device=q.device)
    _flash_attn_debug_single_tile_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        q[batch_idx, head_idx].contiguous(),
        k[batch_idx, head_idx].contiguous(),
        v[batch_idx, head_idx].contiguous(),
        pre_out,
        l,
        seq_len,
    )
    return pre_out, l


def flash_attn_debug_softmax_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_qk_shape(q, k)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    localmax = torch.zeros((BLOCK_ROWS, 2), dtype=torch.float32, device=q.device)
    blockmax = torch.zeros((BLOCK_ROWS,), dtype=torch.float32, device=q.device)
    l = torch.zeros((BLOCK_ROWS,), dtype=torch.float32, device=q.device)
    _flash_attn_debug_softmax_stats_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        q[batch_idx, head_idx].contiguous(),
        k[batch_idx, head_idx].contiguous(),
        localmax,
        blockmax,
        l,
        seq_len,
    )
    return localmax, blockmax, l


def flash_attn_debug_full_kernel_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_attn_validate_shape(q, k, v)

    batch_size, num_heads, seq_len, _ = q.shape
    if not (0 <= batch_idx < batch_size):
        raise ValueError(f"batch_idx must be in [0, {batch_size}) (got {batch_idx})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx must be in [0, {num_heads}) (got {head_idx})")

    pre_out = torch.zeros((seq_len, HEAD_DIM), dtype=torch.float32, device=q.device)
    mi = torch.zeros((seq_len,), dtype=torch.float32, device=q.device)
    l = torch.zeros((seq_len,), dtype=torch.float32, device=q.device)
    _run_flash_attn_tiles(
        q[batch_idx, head_idx].contiguous(),
        k[batch_idx, head_idx].contiguous(),
        v[batch_idx, head_idx].contiguous(),
        pre_out,
        mi,
        l,
        seq_len,
    )
    return pre_out, l


def flash_attn_debug_index_state(
    seq_len: int,
    idx_q: int = 0,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row = torch.zeros((BLOCK_ROWS,), dtype=torch.uint32, device=device)
    global_row = torch.zeros((BLOCK_ROWS,), dtype=torch.uint32, device=device)
    valid = torch.zeros((BLOCK_ROWS,), dtype=torch.uint32, device=device)
    _flash_attn_debug_index_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        row,
        global_row,
        valid,
        seq_len,
        idx_q,
    )
    return row, global_row, valid


def flash_attn_debug_valid_loop_state(
    seq_len: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    out = torch.zeros((BLOCK_ROWS,), dtype=torch.uint32, device=device)
    _flash_attn_debug_valid_loop_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        out,
        seq_len,
    )
    return out


def flash_attn_debug_dynamic_range_state(
    count: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    out = torch.zeros((BLOCK_ROWS,), dtype=torch.uint32, device=device)
    _flash_attn_debug_dynamic_range_kernel[lambda: ((1, 1, 1), (THREADS, 1, 1))](
        out,
        count,
    )
    return out


def flash_attn_debug_apply_causal_mask_tile_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    scores = flash_attn_debug_qk_tile_reference(q, k, batch_idx, head_idx)
    valid_len = q.shape[2]
    masked = scores.clone()
    for row in range(BLOCK_ROWS):
        if row >= valid_len:
            masked[row].zero_()
            continue
        masked[row, valid_len:].fill_(NEG_INF)
        masked[row, row + 1 : valid_len].fill_(NEG_INF)
    return masked


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_ptr_cpu = _validate_packed_flash_attn_inputs(q, k, v, seq_ptr)
    if out is None:
        out = torch.empty_like(q)
    else:
        if out.shape != q.shape:
            raise ValueError(f"out must have shape {q.shape} (got {out.shape})")
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out must be torch.bfloat16 (got {out.dtype})")
        if out.device != q.device:
            raise ValueError(f"out must be on {q.device} (got {out.device})")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    gqa_ratio = q_heads // kv_heads
    for seq_idx in range(len(seq_ptr_cpu) - 1):
        seq_begin = seq_ptr_cpu[seq_idx]
        seq_end = seq_ptr_cpu[seq_idx + 1]
        seq_len = seq_end - seq_begin
        if seq_len == 0:
            continue
        for q_head_idx in range(q_heads):
            kv_head_idx = q_head_idx // gqa_ratio
            q_head = q[seq_begin:seq_end, q_head_idx].contiguous()
            k_head = k[seq_begin:seq_end, kv_head_idx].contiguous()
            v_head = v[seq_begin:seq_end, kv_head_idx].contiguous()
            pre = torch.zeros((seq_len, HEAD_DIM), dtype=torch.float32, device=q.device)
            mi = torch.zeros((seq_len,), dtype=torch.float32, device=q.device)
            l = torch.zeros((seq_len,), dtype=torch.float32, device=q.device)
            _run_flash_attn_tiles(q_head, k_head, v_head, pre, mi, l, seq_len)
            out[seq_begin:seq_end, q_head_idx].copy_(
                flash_attn_debug_normalize_o(pre.contiguous(), l.contiguous())
            )
    return out


__all__ = [
    "flash_attn",
    "flash_attn_debug_apply_causal_mask_tile",
    "flash_attn_debug_apply_causal_mask_tile_reference",
    "flash_attn_debug_gemm_o_tile",
    "flash_attn_debug_gemm_o_tile_reference",
    "flash_attn_debug_full_kernel_state",
    "flash_attn_debug_index_state",
    "flash_attn_debug_normalize_o",
    "flash_attn_debug_single_tile",
    "flash_attn_debug_store_o",
    "flash_attn_debug_dynamic_range_state",
    "flash_attn_debug_valid_loop_state",
    "flash_attn_debug_qk_loads",
    "flash_attn_debug_qk_tile",
    "flash_attn_debug_qk_tile_reference",
    "flash_attn_debug_softmax_stats",
    "flash_attn_debug_softmax_tile",
    "flash_attn_debug_softmax_tile_reference",
]

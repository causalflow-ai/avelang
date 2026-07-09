"""AMDGPU BF16 flash attention debug helpers through masked softmax."""

import math
import avelang
import avelang.language as al
import torch

WARP_SIZE = 64
NUM_WARPS = 8
THREADS = WARP_SIZE * NUM_WARPS

BLOCK_ROWS = 256
BLOCK_COLS = 64
HEAD_DIM = 128
Q_HEADS = 8
KV_HEADS = 1
VEC_SIZE = 16 // 2
BF16_BYTES = 2
U128_BYTES = VEC_SIZE * BF16_BYTES
U32_BYTES = 4

K_SLICES = 2
SHM_Q_WORDS = BLOCK_ROWS * HEAD_DIM // 2
SHM_K_WORDS = BLOCK_COLS * HEAD_DIM // 2
Q_STAGE_ROWS = BLOCK_ROWS // K_SLICES
Q_STAGE_DATA_WORDS = Q_STAGE_ROWS * HEAD_DIM // 2
Q_STAGE_PADDING_U32 = (Q_STAGE_ROWS * HEAD_DIM) // (WARP_SIZE * 2) * (U128_BYTES // U32_BYTES)
SHM_Q_STAGE_WORDS = Q_STAGE_DATA_WORDS + Q_STAGE_PADDING_U32
K_SLICE_ROWS = BLOCK_COLS // K_SLICES
K_SLICE_DATA_WORDS = K_SLICE_ROWS * HEAD_DIM // 2
K_SLICE_PADDING_U32 = (K_SLICE_ROWS * HEAD_DIM) // (WARP_SIZE * 2) * (U128_BYTES // U32_BYTES)
K_PAGE_WORDS = K_SLICE_DATA_WORDS + K_SLICE_PADDING_U32
V_COLS_U32 = HEAD_DIM // 2
V_PADDING_U32 = 1
V_LOGICAL_ROW_STRIDE_U32 = V_COLS_U32
V_ROW_GROUPS = BLOCK_COLS // 4
V_SLICE_ROW_GROUPS = K_SLICE_ROWS // 4
V_TILE_HALF_PADDING_U2 = V_ROW_GROUPS * V_PADDING_U32
V_HALF_PADDING_U2 = V_SLICE_ROW_GROUPS * V_PADDING_U32
V_TILE_HALF_U2 = V_ROW_GROUPS * V_LOGICAL_ROW_STRIDE_U32 + V_TILE_HALF_PADDING_U2
V_HALF_U2 = V_SLICE_ROW_GROUPS * V_LOGICAL_ROW_STRIDE_U32 + V_HALF_PADDING_U2
V_SHM_U2 = 2 * V_HALF_U2
SHM_V_WORDS = V_SHM_U2 * 2
V_TILE_SHM_U2 = 2 * V_TILE_HALF_U2
SHM_V_TILE_WORDS = V_TILE_SHM_U2 * 2
SHM_V_OFFSET_WORDS = K_SLICES * K_PAGE_WORDS
SHM_QV_WORDS = max(SHM_Q_STAGE_WORDS, SHM_V_TILE_WORDS)
SHM_KV_WORDS = SHM_V_OFFSET_WORDS + SHM_QV_WORDS
SHM_O_WORDS = BLOCK_ROWS * HEAD_DIM // 2
SHM_O_VECS = BLOCK_ROWS * HEAD_DIM // VEC_SIZE
SHM_WORK_WORDS = max(SHM_KV_WORDS, SHM_O_WORDS)
SHM_Q_VECS = BLOCK_ROWS * HEAD_DIM // VEC_SIZE
SHM_K_VECS = BLOCK_COLS * HEAD_DIM // VEC_SIZE
V_LOAD_GLOBAL = BLOCK_COLS * HEAD_DIM // (VEC_SIZE * THREADS)
V_ROWPAIR_GROUP_STRIDE = THREADS // V_COLS_U32
V_ROWPAIR_GROUPS_PER_WARP = WARP_SIZE // V_COLS_U32

WARP_ROWS = 32
HALF_WARP_ROWS = 16
SCORE_TILE_COLS = 16
QK_TILE_DWORDS = SCORE_TILE_COLS // 2
QK_ROW_DATA_WORDS = HEAD_DIM // 2
QK_ROW_PADDING_WORDS = U128_BYTES // U32_BYTES
QK_ROW_STRIDE_WORDS = QK_ROW_DATA_WORDS + QK_ROW_PADDING_WORDS
QK_TILE_ROW_STRIDE_WORDS = WARP_ROWS * QK_ROW_STRIDE_WORDS
Q_BATCHES = HEAD_DIM // SCORE_TILE_COLS
K_BATCHES = BLOCK_COLS // WARP_ROWS
O_BATCHES = HEAD_DIM // WARP_ROWS
QK_MFMAS_PER_SLICE = Q_BATCHES * 2
QK_MAJOR_MFMAS_PER_SLICE = (QK_MFMAS_PER_SLICE * 3) // 4
QK_MINOR_MFMAS_PER_SLICE = QK_MFMAS_PER_SLICE - QK_MAJOR_MFMAS_PER_SLICE
QK_MAJOR_K_BATCHES = (K_BATCHES * 3) // 4
QK_TOTAL_MFMAS = K_BATCHES * Q_BATCHES * 2
QK_MAJOR_MFMAS = (QK_TOTAL_MFMAS * 3) // 4
QK_MINOR_MFMAS = QK_TOTAL_MFMAS - QK_MAJOR_MFMAS
GEMM_O_MFMAS = O_BATCHES * K_BATCHES * 4
NEG_INF = -1.0e30
SCALE_LOG2 = math.log2(math.e) / math.sqrt(HEAD_DIM)

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200
SCHED_MASK_TRANS = 0x400
SCHED_MASK_VALU = 0x1
SCHED_INST_ALU_LIGHT = 2
SCHED_INST_ALU_MEDIUM = 4
SCHED_INST_TRANS_HEAVY = 1


def _mfma_per_issue(inst_mfma: int, inst_issue: int) -> int:
    ratio = inst_mfma // inst_issue
    if ratio > 12:
        return 4
    if ratio > 6:
        return 2
    return 1


QK_MAJOR_MFMA_PER_ISSUE = _mfma_per_issue(QK_MAJOR_MFMAS_PER_SLICE, SCHED_INST_ALU_LIGHT)
QK_MINOR_MFMA_PER_ISSUE = _mfma_per_issue(QK_MINOR_MFMAS_PER_SLICE, SCHED_INST_ALU_LIGHT)
GEMM_O_MFMA_PER_ISSUE = _mfma_per_issue(
    GEMM_O_MFMAS,
    SCHED_INST_ALU_MEDIUM + SCHED_INST_TRANS_HEAVY,
)


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
    if q.shape[1] != Q_HEADS or k.shape[1] != KV_HEADS:
        raise ValueError(
            f"BF16 flash attention currently requires q_heads={Q_HEADS}, kv_heads={KV_HEADS} "
            f"(got q_heads={q.shape[1]}, kv_heads={k.shape[1]})"
        )
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
def _hot_loop_scheduler_qk_major():
    for _ in al.range(SCHED_INST_ALU_LIGHT):
        al.amdgpu.sched_group_barrier(SCHED_MASK_VALU, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, QK_MAJOR_MFMA_PER_ISSUE, 0)
    al.amdgpu.sched_barrier(0)


@avelang.jit
def _hot_loop_scheduler_qk_minor():
    for _ in al.range(SCHED_INST_ALU_LIGHT):
        al.amdgpu.sched_group_barrier(SCHED_MASK_VALU, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, QK_MINOR_MFMA_PER_ISSUE, 0)
    al.amdgpu.sched_barrier(0)


@avelang.jit
def _hot_loop_scheduler_gemm_o():
    for _ in al.range(SCHED_INST_ALU_MEDIUM):
        al.amdgpu.sched_group_barrier(SCHED_MASK_VALU, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, GEMM_O_MFMA_PER_ISSUE, 0)
    for _ in al.range(SCHED_INST_TRANS_HEAVY):
        al.amdgpu.sched_group_barrier(SCHED_MASK_TRANS, 1, 0)
        al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, GEMM_O_MFMA_PER_ISSUE, 0)
    al.amdgpu.sched_barrier(0)


@avelang.jit
def _load_q_reg_words_lds_strided(
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    q_words: al.Tensor((SHM_Q_STAGE_WORDS,), al.u32),
    q_rsrc: al.Tensor((4,), al.u32),
    query_row_local: al.u32,
    global_query_row: al.u32,
    row_stride_words: al.u32,
    wid: al.u32,
    lane_col: al.u32,
    lane_half: al.u32,
):
    words_per_row = HEAD_DIM // 2
    q_stage_rows = Q_STAGE_ROWS
    warps_per_stage = NUM_WARPS // K_SLICES
    tid = al.thread_id(0)
    q_base = global_query_row - query_row_local
    q_stage_id = wid // warps_per_stage
    stage_wid = wid % warps_per_stage
    stage_query_row = stage_wid * WARP_ROWS + lane_col
    q_word_tile = al.view(
        q_words,
        al.u32,
        al.make_layout(
            (Q_STAGE_ROWS // WARP_ROWS, Q_BATCHES, WARP_ROWS, 2, 4),
            (QK_TILE_ROW_STRIDE_WORDS, QK_TILE_DWORDS, QK_ROW_STRIDE_WORDS, 4, 1),
        ),
    )

    for q_stage in al.range(K_SLICES):
        stage_base = q_stage * q_stage_rows
        for word_batch in al.range(Q_STAGE_DATA_WORDS // THREADS):
            row = tid // words_per_row
            dest_col_word = tid % words_per_row
            tile_col = dest_col_word // QK_TILE_DWORDS
            dword_in_tile = dest_col_word - tile_col * QK_TILE_DWORDS
            col_word = (
                tile_col * QK_TILE_DWORDS
                + (dword_in_tile & 0x1)
                + ((dword_in_tile >> 1) & 0x1)
                * (QK_TILE_DWORDS // 2)
                + (dword_in_tile >> 2) * 2
            )
            global_vword = row * row_stride_words + col_word
            global_sword = (
                q_base + stage_base + word_batch * (THREADS // (HEAD_DIM // 2))
            ) * row_stride_words
            global_voffset = global_vword * U32_BYTES
            global_soffset = global_sword * U32_BYTES
            lds_row = wid + (word_batch * NUM_WARPS)
            lds_word = lds_row * QK_ROW_STRIDE_WORDS
            lds_offset = lds_word * U32_BYTES
            al.amdgpu.raw_buffer_load_x1_lds(
                q_rsrc,
                q_words,
                U32_BYTES,
                global_voffset,
                global_soffset,
                lds_offset,
                0,
            )
        al.amdgpu.s_waitcnt(0, 7, 15)
        al.syncthreads()
        if q_stage == q_stage_id:
            for q_batch in al.range(Q_BATCHES):
                packed = q_word_tile[stage_wid, q_batch, lane_col, lane_half]
                q_regs[q_batch, 0, 0] = packed[0]
                q_regs[q_batch, 0, 1] = packed[1]
                q_regs[q_batch, 1, 0] = packed[2]
                q_regs[q_batch, 1, 1] = packed[3]
        al.syncthreads()


@avelang.jit
def _fetch_k_reg_words_page(
    k_regs: al.Tensor((K_BATCHES, Q_BATCHES, 2, 2), al.u32),
    k_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    key_row_local: al.u32,
    lane_half: al.u32,
    k_batch: al.u32,
):
    k_word_layout = al.make_layout(
        (K_BATCHES, Q_BATCHES, WARP_ROWS, 2, 4),
        (K_PAGE_WORDS, QK_TILE_DWORDS, QK_ROW_STRIDE_WORDS, 4, 1),
    )
    k_word_tile = al.view(k_words, al.u32, k_word_layout)
    for q_batch in al.range(Q_BATCHES):
        packed = k_word_tile[k_batch, q_batch, key_row_local, lane_half]
        k_regs[k_batch, q_batch, 0, 0] = packed[0]
        k_regs[k_batch, q_batch, 0, 1] = packed[1]
        k_regs[k_batch, q_batch, 1, 0] = packed[2]
        k_regs[k_batch, q_batch, 1, 1] = packed[3]


@avelang.jit
def _fetch_k_reg_words_page_small(
    k_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    k_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    key_row_local: al.u32,
    lane_half: al.u32,
    k_batch: al.u32,
):
    k_word_layout = al.make_layout(
        (K_BATCHES, Q_BATCHES, WARP_ROWS, 2, 4),
        (K_PAGE_WORDS, QK_TILE_DWORDS, QK_ROW_STRIDE_WORDS, 4, 1),
    )
    k_word_tile = al.view(k_words, al.u32, k_word_layout)
    for q_batch in al.range(Q_BATCHES):
        packed = k_word_tile[k_batch, q_batch, key_row_local, lane_half]
        k_regs[q_batch, 0, 0] = packed[0]
        k_regs[q_batch, 0, 1] = packed[1]
        k_regs[q_batch, 1, 0] = packed[2]
        k_regs[q_batch, 1, 1] = packed[3]


@avelang.jit
def _gemm_qk_word_regs_batch0_small(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    k_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
):
    zero_f32 = al.convert(0.0, al.f32)
    for t in al.range(16):
        score_acc[0, t] = zero_f32

    for mfma_idx in al.range(QK_MAJOR_MFMAS_PER_SLICE):
        q_batch = mfma_idx // 2
        mfma_k = mfma_idx - q_batch * 2
        q_frag = al.view(q_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        k_frag = al.view(k_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        score_acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag, q_frag, score_acc[0])
    _hot_loop_scheduler_qk_major()

    for mfma_idx in al.range(QK_MAJOR_MFMAS_PER_SLICE, QK_MFMAS_PER_SLICE):
        q_batch = mfma_idx // 2
        mfma_k = mfma_idx - q_batch * 2
        q_frag = al.view(q_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        k_frag = al.view(k_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        score_acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag, q_frag, score_acc[0])
    _hot_loop_scheduler_qk_minor()


@avelang.jit
def _gemm_qk_word_regs_batch1_small(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    k_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
):
    zero_f32 = al.convert(0.0, al.f32)
    for t in al.range(16):
        score_acc[1, t] = zero_f32

    for mfma_idx in al.range(QK_MAJOR_MFMAS_PER_SLICE):
        q_batch = mfma_idx // 2
        mfma_k = mfma_idx - q_batch * 2
        q_frag = al.view(q_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        k_frag = al.view(k_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        score_acc[1] = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag, q_frag, score_acc[1])
    _hot_loop_scheduler_qk_major()

    for mfma_idx in al.range(QK_MAJOR_MFMAS_PER_SLICE, QK_MFMAS_PER_SLICE):
        q_batch = mfma_idx // 2
        mfma_k = mfma_idx - q_batch * 2
        q_frag = al.view(q_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        k_frag = al.view(k_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        score_acc[1] = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag, q_frag, score_acc[1])
    _hot_loop_scheduler_qk_minor()


@avelang.jit
def _gemm_qk_word_regs_batch_major(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    k_regs: al.Tensor((K_BATCHES, Q_BATCHES, 2, 2), al.u32),
    k_batch: al.u32,
):
    for mfma_idx in al.range(QK_MAJOR_MFMAS_PER_SLICE):
        q_batch = mfma_idx // 2
        mfma_k = mfma_idx - q_batch * 2
        q_frag = al.view(q_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        k_frag = al.view(k_regs[k_batch, q_batch, mfma_k], al.Tensor((4,), al.bf16))
        score_acc[k_batch] = al.amdgpu.mfma_32x32x8_bf16_f32(
            k_frag,
            q_frag,
            score_acc[k_batch],
        )


@avelang.jit
def _gemm_qk_word_regs_batch_minor(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    k_regs: al.Tensor((K_BATCHES, Q_BATCHES, 2, 2), al.u32),
    k_batch: al.u32,
):
    for mfma_idx in al.range(QK_MAJOR_MFMAS_PER_SLICE, QK_MFMAS_PER_SLICE):
        q_batch = mfma_idx // 2
        mfma_k = mfma_idx - q_batch * 2
        q_frag = al.view(q_regs[q_batch, mfma_k], al.Tensor((4,), al.bf16))
        k_frag = al.view(k_regs[k_batch, q_batch, mfma_k], al.Tensor((4,), al.bf16))
        score_acc[k_batch] = al.amdgpu.mfma_32x32x8_bf16_f32(
            k_frag,
            q_frag,
            score_acc[k_batch],
        )


@avelang.jit
def _gemm_qk_word_regs_batch(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    k_regs: al.Tensor((K_BATCHES, Q_BATCHES, 2, 2), al.u32),
    k_batch: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    for t in al.range(16):
        score_acc[k_batch, t] = zero_f32

    _gemm_qk_word_regs_batch_major(score_acc, q_regs, k_regs, k_batch)
    _hot_loop_scheduler_qk_major()
    _gemm_qk_word_regs_batch_minor(score_acc, q_regs, k_regs, k_batch)
    _hot_loop_scheduler_qk_minor()


@avelang.jit
def _softmax_xor_permute_f32(value: al.f32, wtid: al.u32) -> al.f32:
    index = (wtid ^ 32) * 4
    bits = al.amdgpu.ds_permute_b32(index, al.bitcast(value, al.u32))
    return al.bitcast(bits, al.f32)


@avelang.jit
def _compute_row_max_batch(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    k_batch: al.u32,
) -> al.f32:
    neg_inf = al.convert(NEG_INF, al.f32)
    local_max = neg_inf
    for t in al.range(16):
        score = score_acc[k_batch, t]
        local_max = al.fmax(local_max, score)
    partner_max = al.shuffle_xor(local_max, 32, 64)
    return al.fmax(local_max, partner_max)


@avelang.jit
def _compute_ptilde_and_row_sum_batch(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    k_batch: al.u32,
    row_max: al.f32,
    scale_log2: al.f32,
) -> al.f32:
    row_max_scaled = row_max * scale_log2
    local_sum = al.convert(0.0, al.f32)
    for t in al.range(16):
        p = al.exp2(al.fma(score_acc[k_batch, t], scale_log2, -row_max_scaled))
        score_acc[k_batch, t] = p
        local_sum = local_sum + p
    return local_sum


@avelang.jit
def _compute_row_max_batch0(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    wtid: al.u32,
) -> al.f32:
    neg_inf = al.convert(NEG_INF, al.f32)
    local_max = neg_inf
    for t in al.range(16):
        score = score_acc[0, t]
        local_max = al.fmax(local_max, score)
    partner_max = _softmax_xor_permute_f32(local_max, wtid)
    return al.fmax(local_max, partner_max)


@avelang.jit
def _compute_row_max_batch1(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    wtid: al.u32,
) -> al.f32:
    neg_inf = al.convert(NEG_INF, al.f32)
    local_max = neg_inf
    for t in al.range(16):
        score = score_acc[1, t]
        local_max = al.fmax(local_max, score)
    partner_max = _softmax_xor_permute_f32(local_max, wtid)
    return al.fmax(local_max, partner_max)


@avelang.jit
def _compute_ptilde_and_row_sum_batch0(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    row_max: al.f32,
    scale_log2: al.f32,
) -> al.f32:
    row_max_scaled = row_max * scale_log2
    local_sum = al.convert(0.0, al.f32)
    for t in al.range(16):
        p = al.exp2(al.fma(score_acc[0, t], scale_log2, -row_max_scaled))
        score_acc[0, t] = p
        local_sum = local_sum + p
    return local_sum


@avelang.jit
def _compute_ptilde_and_row_sum_batch1(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    row_max: al.f32,
    scale_log2: al.f32,
) -> al.f32:
    row_max_scaled = row_max * scale_log2
    local_sum = al.convert(0.0, al.f32)
    for t in al.range(16):
        p = al.exp2(al.fma(score_acc[1, t], scale_log2, -row_max_scaled))
        score_acc[1, t] = p
        local_sum = local_sum + p
    return local_sum


@avelang.jit
def _multiply_alpha_o_mfma(alpha: al.f32, out_acc: al.Tensor((O_BATCHES, 16), al.f32)):
    zero_f32 = al.convert(0.0, al.f32)
    for o_batch in al.range(O_BATCHES):
        for t in al.range(16):
            out_acc[o_batch, t] = al.fma(out_acc[o_batch, t], alpha, zero_f32)


@avelang.jit
def _apply_lower_causal_mask_batch0_at_key_base(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    global_query_row: al.u32,
    actual_key_base: al.u32,
    lane_half: al.u32,
):
    neg_inf = al.convert(NEG_INF, al.f32)
    col_block_base = actual_key_base + lane_half * 4
    rel = global_query_row - col_block_base
    score_acc[0, 0] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 0], 0)
    score_acc[0, 1] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 1], 1)
    score_acc[0, 2] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 2], 2)
    score_acc[0, 3] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 3], 3)
    score_acc[0, 4] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 4], 8)
    score_acc[0, 5] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 5], 9)
    score_acc[0, 6] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 6], 10)
    score_acc[0, 7] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 7], 11)
    score_acc[0, 8] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 8], 16)
    score_acc[0, 9] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 9], 17)
    score_acc[0, 10] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 10], 18)
    score_acc[0, 11] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 11], 19)
    score_acc[0, 12] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 12], 24)
    score_acc[0, 13] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 13], 25)
    score_acc[0, 14] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 14], 26)
    score_acc[0, 15] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[0, 15], 27)


@avelang.jit
def _apply_lower_causal_mask_batch1_at_key_base(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    global_query_row: al.u32,
    actual_key_base: al.u32,
    lane_half: al.u32,
):
    neg_inf = al.convert(NEG_INF, al.f32)
    col_block_base = actual_key_base + lane_half * 4
    rel = global_query_row - col_block_base
    score_acc[1, 0] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 0], 0)
    score_acc[1, 1] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 1], 1)
    score_acc[1, 2] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 2], 2)
    score_acc[1, 3] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 3], 3)
    score_acc[1, 4] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 4], 8)
    score_acc[1, 5] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 5], 9)
    score_acc[1, 6] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 6], 10)
    score_acc[1, 7] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 7], 11)
    score_acc[1, 8] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 8], 16)
    score_acc[1, 9] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 9], 17)
    score_acc[1, 10] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 10], 18)
    score_acc[1, 11] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 11], 19)
    score_acc[1, 12] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 12], 24)
    score_acc[1, 13] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 13], 25)
    score_acc[1, 14] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 14], 26)
    score_acc[1, 15] = al.amdgpu.mask_lower_bound_f32(rel, neg_inf, score_acc[1, 15], 27)


@avelang.jit
def _fetch_v_global_slice_packed_strided(
    g_v: al.Tensor((4,), al.u32),
    v_rsrc: al.Tensor((4,), al.u32),
    wid: al.u32,
    wtid: al.u32,
    key_base: al.u32,
    row_stride_words: al.u32,
    v_slice: al.u32,
):
    col_u32 = wtid % V_COLS_U32
    rowpair_group = (
        wid * V_ROWPAIR_GROUPS_PER_WARP
        + wtid // V_COLS_U32
    )
    row_group = v_slice * K_SLICE_ROWS + rowpair_group * 4
    row0 = row_group
    row1 = row_group + 1
    row2 = row_group + 2
    row3 = row_group + 3
    offset0 = ((key_base + row0) * row_stride_words + col_u32) * U32_BYTES
    offset1 = ((key_base + row1) * row_stride_words + col_u32) * U32_BYTES
    offset2 = ((key_base + row2) * row_stride_words + col_u32) * U32_BYTES
    offset3 = ((key_base + row3) * row_stride_words + col_u32) * U32_BYTES
    g_v[0] = al.bitcast(al.amdgpu.raw_buffer_load_x1(v_rsrc, offset0, 0, 0), al.u32)
    g_v[1] = al.bitcast(al.amdgpu.raw_buffer_load_x1(v_rsrc, offset1, 0, 0), al.u32)
    g_v[2] = al.bitcast(al.amdgpu.raw_buffer_load_x1(v_rsrc, offset2, 0, 0), al.u32)
    g_v[3] = al.bitcast(al.amdgpu.raw_buffer_load_x1(v_rsrc, offset3, 0, 0), al.u32)


@avelang.jit
def _store_v_global_slice_packed(
    work_words: al.Tensor((SHM_WORK_WORDS,), al.u32),
    g_v: al.Tensor((4,), al.u32),
    wid: al.u32,
    wtid: al.u32,
    v_slice: al.u32,
):
    work_u2 = al.view(
        work_words,
        al.u32,
        al.make_layout(
            (SHM_WORK_WORDS // 2, 2),
            (2, 1),
        ),
    )
    col_u32 = wtid % V_COLS_U32
    rowpair_group = wid * V_ROWPAIR_GROUPS_PER_WARP + wtid // V_COLS_U32
    even0 = al.amdgpu.perm(g_v[1], g_v[0], 0x05040100)
    even1 = al.amdgpu.perm(g_v[3], g_v[2], 0x05040100)
    odd0 = al.amdgpu.perm(g_v[1], g_v[0], 0x07060302)
    odd1 = al.amdgpu.perm(g_v[3], g_v[2], 0x07060302)
    shm_idx = rowpair_group * V_LOGICAL_ROW_STRIDE_U32 + col_u32
    addr_adjust = al.convert(wid == 0, al.u32)
    even_base = SHM_V_OFFSET_WORDS // 2 + addr_adjust
    odd_base = SHM_V_OFFSET_WORDS // 2 + V_HALF_U2 + addr_adjust
    local_idx = shm_idx - addr_adjust
    even_idx = even_base + local_idx
    odd_idx = odd_base + local_idx
    work_u2[even_idx, 0] = even0
    work_u2[even_idx, 1] = even1
    work_u2[odd_idx, 0] = odd0
    work_u2[odd_idx, 1] = odd1


@avelang.jit
def _fetch_v_reg_words_batch_compact(
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    v_words: al.Tensor((SHM_V_WORDS,), al.u32),
    out_row_local: al.u32,
    wtid: al.u32,
    k_batch: al.u32,
):
    v_pair = al.view(
        v_words,
        al.u32,
        al.make_layout(
            (2, O_BATCHES, WARP_ROWS // 2, 2, 2, 2, 2),
            (
                (SCORE_TILE_COLS // 4) * V_LOGICAL_ROW_STRIDE_U32 * 2,
                (WARP_ROWS // 2) * 2,
                2,
                V_LOGICAL_ROW_STRIDE_U32 * 2,
                V_HALF_U2 * 2,
                (SCORE_TILE_COLS // 4 // 2) * V_LOGICAL_ROW_STRIDE_U32 * 2,
                1,
            ),
        ),
    )
    lane_half = wtid // WARP_ROWS
    lane_pair = (wtid % WARP_ROWS) // 2
    lane_parity = wtid & 1

    for o_batch in al.range(O_BATCHES):
        for v_chunk in al.range(2):
            packed = v_pair[v_chunk, o_batch, lane_pair, lane_half, lane_parity]
            v_regs[o_batch, v_chunk * 2, 0] = packed[0, 0]
            v_regs[o_batch, v_chunk * 2, 1] = packed[0, 1]
            v_regs[o_batch, v_chunk * 2 + 1, 0] = packed[1, 0]
            v_regs[o_batch, v_chunk * 2 + 1, 1] = packed[1, 1]


@avelang.jit
def _pack_gemm_o_score_mfma_batch(
    packed_score: al.Tensor((16,), al.f32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    k_batch: al.u32,
    lane_half: al.u32,
):
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
            packed_score[elem] = lo0
            packed_score[4 + elem] = hi0
            packed_score[8 + elem] = partner_lo0
            packed_score[12 + elem] = partner_hi0
        else:
            packed_score[elem] = partner_lo1
            packed_score[4 + elem] = partner_hi1
            packed_score[8 + elem] = lo1
            packed_score[12 + elem] = hi1


@avelang.jit
def _gemm_o_mfma_v_word_regs_batch_compact(
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    packed_score: al.Tensor((16,), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
):
    score_frag = al.make_local((4,), al.bf16)

    for o_batch in al.range(O_BATCHES):
        for k_slice in al.range(4):
            v_frag = al.view(v_regs[o_batch, k_slice], al.Tensor((4,), al.bf16))
            for elem in al.range(4):
                score_frag[elem] = al.convert(packed_score[k_slice * 4 + elem], al.bf16)
            out_acc[o_batch] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[o_batch])
    _hot_loop_scheduler_gemm_o()


@avelang.jit
def _gemm_o_mfma_v_word_regs_batch0_direct(
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
):
    score_frag = al.make_local((4,), al.bf16)

    for k_slice in al.range(4):
        v_frag = al.view(v_regs[0, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[0, k_slice * 4 + elem], al.bf16)
        out_acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[0])
    for k_slice in al.range(4):
        v_frag = al.view(v_regs[1, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[0, k_slice * 4 + elem], al.bf16)
        out_acc[1] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[1])
    for k_slice in al.range(4):
        v_frag = al.view(v_regs[2, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[0, k_slice * 4 + elem], al.bf16)
        out_acc[2] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[2])
    for k_slice in al.range(4):
        v_frag = al.view(v_regs[3, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[0, k_slice * 4 + elem], al.bf16)
        out_acc[3] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[3])
    _hot_loop_scheduler_gemm_o()


@avelang.jit
def _gemm_o_mfma_v_word_regs_batch1_direct(
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
):
    score_frag = al.make_local((4,), al.bf16)

    for k_slice in al.range(4):
        v_frag = al.view(v_regs[0, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[1, k_slice * 4 + elem], al.bf16)
        out_acc[0] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[0])
    for k_slice in al.range(4):
        v_frag = al.view(v_regs[1, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[1, k_slice * 4 + elem], al.bf16)
        out_acc[1] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[1])
    for k_slice in al.range(4):
        v_frag = al.view(v_regs[2, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[1, k_slice * 4 + elem], al.bf16)
        out_acc[2] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[2])
    for k_slice in al.range(4):
        v_frag = al.view(v_regs[3, k_slice], al.Tensor((4,), al.bf16))
        for elem in al.range(4):
            score_frag[elem] = al.convert(score_acc[1, k_slice * 4 + elem], al.bf16)
        out_acc[3] = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag, score_frag, out_acc[3])
    _hot_loop_scheduler_gemm_o()


@avelang.jit
def _flash_attn_update_o_batch(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    softmax_state: al.Tensor((2,), al.f32),
    k_batch: al.u32,
    partition_idx: al.u32,
    lane_half: al.u32,
    scale_log2: al.f32,
) -> (al.f32, al.f32):
    zero_f32 = al.convert(0.0, al.f32)
    mi = softmax_state[0]
    l = softmax_state[1]
    block_max = _compute_row_max_batch(score_acc, k_batch)
    mi_new = block_max
    if partition_idx != 0:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            mi_new = mi
        elif mi > block_max:
            mi_new = mi
    if block_max <= al.convert(NEG_INF, al.f32):
        if partition_idx == 0:
            mi_new = zero_f32
        else:
            mi_new = mi

    local_sum = _compute_ptilde_and_row_sum_batch(score_acc, k_batch, mi_new, scale_log2)
    row_sum = local_sum
    if partition_idx == 0:
        l = row_sum
    else:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            l = l + row_sum
        else:
            scaling = al.exp2((mi - mi_new) * scale_log2)
            l = al.fma(scaling, l, row_sum)
            _multiply_alpha_o_mfma(scaling, out_acc)
    mi = mi_new

    _pack_gemm_o_score_mfma_batch(score_acc_mfma, score_acc, k_batch, lane_half)
    _gemm_o_mfma_v_word_regs_batch_compact(out_acc, score_acc_mfma, v_regs)
    softmax_state[0] = mi
    softmax_state[1] = l
    return mi, l


@avelang.jit
def _flash_attn_update_o_batch0(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    softmax_state: al.Tensor((2,), al.f32),
    partition_idx: al.u32,
    lane_half: al.u32,
    wtid: al.u32,
    scale_log2: al.f32,
) -> (al.f32, al.f32):
    zero_f32 = al.convert(0.0, al.f32)
    mi = softmax_state[0]
    l = softmax_state[1]
    block_max = _compute_row_max_batch0(score_acc, wtid)
    mi_new = block_max
    if partition_idx != 0:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            mi_new = mi
        elif mi > block_max:
            mi_new = mi
    if block_max <= al.convert(NEG_INF, al.f32):
        if partition_idx == 0:
            mi_new = zero_f32
        else:
            mi_new = mi

    local_sum = _compute_ptilde_and_row_sum_batch0(score_acc, mi_new, scale_log2)
    row_sum = local_sum
    if partition_idx == 0:
        l = row_sum
    else:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            l = l + row_sum
        else:
            scaling = al.exp2((mi - mi_new) * scale_log2)
            l = al.fma(scaling, l, row_sum)
            _multiply_alpha_o_mfma(scaling, out_acc)
    mi = mi_new

    _gemm_o_mfma_v_word_regs_batch0_direct(out_acc, score_acc, v_regs)
    softmax_state[0] = mi
    softmax_state[1] = l
    return mi, l


@avelang.jit
def _flash_attn_update_o_batch1(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    softmax_state: al.Tensor((2,), al.f32),
    partition_idx: al.u32,
    lane_half: al.u32,
    wtid: al.u32,
    scale_log2: al.f32,
) -> (al.f32, al.f32):
    zero_f32 = al.convert(0.0, al.f32)
    mi = softmax_state[0]
    l = softmax_state[1]
    block_max = _compute_row_max_batch1(score_acc, wtid)
    mi_new = block_max
    if partition_idx != 0:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            mi_new = mi
        elif mi > block_max:
            mi_new = mi
    if block_max <= al.convert(NEG_INF, al.f32):
        if partition_idx == 0:
            mi_new = zero_f32
        else:
            mi_new = mi

    local_sum = _compute_ptilde_and_row_sum_batch1(score_acc, mi_new, scale_log2)
    row_sum = local_sum
    if partition_idx == 0:
        l = row_sum
    else:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            l = l + row_sum
        else:
            scaling = al.exp2((mi - mi_new) * scale_log2)
            l = al.fma(scaling, l, row_sum)
            _multiply_alpha_o_mfma(scaling, out_acc)
    mi = mi_new

    _gemm_o_mfma_v_word_regs_batch1_direct(out_acc, score_acc, v_regs)
    softmax_state[0] = mi
    softmax_state[1] = l
    return mi, l


@avelang.jit
def _flash_attn_prepare_o_batch0(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    softmax_state: al.Tensor((2,), al.f32),
    partition_idx: al.u32,
    lane_half: al.u32,
    wtid: al.u32,
    scale_log2: al.f32,
) -> (al.f32, al.f32):
    zero_f32 = al.convert(0.0, al.f32)
    mi = softmax_state[0]
    l = softmax_state[1]
    block_max = _compute_row_max_batch0(score_acc, wtid)
    mi_new = block_max
    if partition_idx != 0:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            mi_new = mi
        elif mi > block_max:
            mi_new = mi
    if block_max <= al.convert(NEG_INF, al.f32):
        if partition_idx == 0:
            mi_new = zero_f32
        else:
            mi_new = mi

    local_sum = _compute_ptilde_and_row_sum_batch0(score_acc, mi_new, scale_log2)
    row_sum = local_sum
    if partition_idx == 0:
        l = row_sum
    else:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            l = l + row_sum
        else:
            scaling = al.exp2((mi - mi_new) * scale_log2)
            l = al.fma(scaling, l, row_sum)
            _multiply_alpha_o_mfma(scaling, out_acc)
    mi = mi_new

    softmax_state[0] = mi
    softmax_state[1] = l
    return mi, l


@avelang.jit
def _flash_attn_prepare_o_batch1(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    softmax_state: al.Tensor((2,), al.f32),
    partition_idx: al.u32,
    lane_half: al.u32,
    wtid: al.u32,
    scale_log2: al.f32,
) -> (al.f32, al.f32):
    zero_f32 = al.convert(0.0, al.f32)
    mi = softmax_state[0]
    l = softmax_state[1]
    block_max = _compute_row_max_batch1(score_acc, wtid)
    mi_new = block_max
    if partition_idx != 0:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            mi_new = mi
        elif mi > block_max:
            mi_new = mi
    if block_max <= al.convert(NEG_INF, al.f32):
        if partition_idx == 0:
            mi_new = zero_f32
        else:
            mi_new = mi

    local_sum = _compute_ptilde_and_row_sum_batch1(score_acc, mi_new, scale_log2)
    row_sum = local_sum
    if partition_idx == 0:
        l = row_sum
    else:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            l = l + row_sum
        else:
            scaling = al.exp2((mi - mi_new) * scale_log2)
            l = al.fma(scaling, l, row_sum)
            _multiply_alpha_o_mfma(scaling, out_acc)
    mi = mi_new

    softmax_state[0] = mi
    softmax_state[1] = l
    return mi, l


@avelang.jit
def _flash_attn_prepare_o_batch(
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    score_acc_mfma: al.Tensor((16,), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    softmax_state: al.Tensor((2,), al.f32),
    k_batch: al.u32,
    partition_idx: al.u32,
    lane_half: al.u32,
    scale_log2: al.f32,
) -> (al.f32, al.f32):
    zero_f32 = al.convert(0.0, al.f32)
    mi = softmax_state[0]
    l = softmax_state[1]
    block_max = _compute_row_max_batch(score_acc, k_batch)
    mi_new = block_max
    if partition_idx != 0:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            mi_new = mi
        elif mi > block_max:
            mi_new = mi
    if block_max <= al.convert(NEG_INF, al.f32):
        if partition_idx == 0:
            mi_new = zero_f32
        else:
            mi_new = mi

    local_sum = _compute_ptilde_and_row_sum_batch(score_acc, k_batch, mi_new, scale_log2)
    row_sum = local_sum
    if partition_idx == 0:
        l = row_sum
    else:
        if (block_max - mi) * scale_log2 <= al.convert(8.0, al.f32):
            l = l + row_sum
        else:
            scaling = al.exp2((mi - mi_new) * scale_log2)
            l = al.fma(scaling, l, row_sum)
            _multiply_alpha_o_mfma(scaling, out_acc)
    mi = mi_new

    _pack_gemm_o_score_mfma_batch(score_acc_mfma, score_acc, k_batch, lane_half)
    softmax_state[0] = mi
    softmax_state[1] = l
    return mi, l


@avelang.jit
def _flash_attn_compute_qk_page(
    k_stage_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    k_rsrc: al.Tensor((4,), al.u32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    tid: al.u32,
    key_row_local: al.u32,
    lane_half: al.u32,
    global_query_row: al.u32,
    seq_len: al.u32,
    actual_key_base: al.u32,
    first_causal_key_base: al.u32,
    kv_row_stride_words: al.u32,
    page: al.u32,
):
    _load_k_page_strided(
        k_stage_words,
        k_rsrc,
        tid,
        actual_key_base,
        kv_row_stride_words,
        page,
    )
    al.amdgpu.s_waitcnt(0, 7, 15)
    al.syncthreads()
    k_regs = al.make_local((Q_BATCHES, 2, 2), al.u32)
    _fetch_k_reg_words_page_small(k_regs, k_stage_words, key_row_local, lane_half, page)
    if page == 0:
        _gemm_qk_word_regs_batch0_small(score_acc, q_regs, k_regs)
    else:
        _gemm_qk_word_regs_batch1_small(score_acc, q_regs, k_regs)
    if actual_key_base >= first_causal_key_base:
        if page == 0:
            _apply_lower_causal_mask_batch0_at_key_base(
                score_acc,
                global_query_row,
                actual_key_base,
                lane_half,
            )
        else:
            _apply_lower_causal_mask_batch1_at_key_base(
                score_acc,
                global_query_row,
                actual_key_base,
                lane_half,
            )


@avelang.jit
def _flash_attn_compute_qk_page_loaded(
    k_stage_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    key_row_local: al.u32,
    lane_half: al.u32,
    global_query_row: al.u32,
    seq_len: al.u32,
    actual_key_base: al.u32,
    first_causal_key_base: al.u32,
    page: al.u32,
):
    k_regs = al.make_local((Q_BATCHES, 2, 2), al.u32)
    _fetch_k_reg_words_page_small(k_regs, k_stage_words, key_row_local, lane_half, page)
    if page == 0:
        _gemm_qk_word_regs_batch0_small(score_acc, q_regs, k_regs)
    else:
        _gemm_qk_word_regs_batch1_small(score_acc, q_regs, k_regs)
    if actual_key_base >= first_causal_key_base:
        if page == 0:
            _apply_lower_causal_mask_batch0_at_key_base(
                score_acc,
                global_query_row,
                actual_key_base,
                lane_half,
            )
        else:
            _apply_lower_causal_mask_batch1_at_key_base(
                score_acc,
                global_query_row,
                actual_key_base,
                lane_half,
            )


@avelang.jit
def _flash_attn_packed_pair_step_wg0(
    k_stage_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    work_words: al.Tensor((SHM_WORK_WORDS,), al.u32),
    v_words: al.Tensor((SHM_V_WORDS,), al.u32),
    k_rsrc: al.Tensor((4,), al.u32),
    v_rsrc: al.Tensor((4,), al.u32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    g_v0: al.Tensor((4,), al.u32),
    g_v1: al.Tensor((4,), al.u32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    softmax_state: al.Tensor((2,), al.f32),
    tid: al.u32,
    wid: al.u32,
    wtid: al.u32,
    key_row_local: al.u32,
    lane_half: al.u32,
    global_query_row: al.u32,
    seq_len: al.u32,
    pair_idx: al.u32,
    max_k_slice: al.u32,
    actual_odd_key_base: al.u32,
    actual_next_even_key_base: al.u32,
    first_causal_key_base: al.u32,
    kv_row_stride_words: al.u32,
    scale_log2: al.f32,
):
    odd_slice = pair_idx * 2 + 1
    next_even_slice = odd_slice + 1

    _flash_attn_compute_qk_page_loaded(
        k_stage_words,
        q_regs,
        score_acc,
        key_row_local,
        lane_half,
        global_query_row,
        seq_len,
        actual_odd_key_base,
        first_causal_key_base,
        1,
    )
    al.syncthreads()
    _fetch_v_global_slice_packed_strided(
        g_v1,
        v_rsrc,
        wid,
        wtid,
        actual_odd_key_base,
        kv_row_stride_words,
        0,
    )
    al.syncthreads()

    al.amdgpu.s_waitcnt(0, 7, 15)
    _store_v_global_slice_packed(work_words, g_v0, wid, wtid, 0)
    al.syncthreads()
    _fetch_v_reg_words_batch_compact(v_regs, v_words, key_row_local, wtid, 0)
    _flash_attn_update_o_batch0(
        score_acc,
        out_acc,
        v_regs,
        softmax_state,
        pair_idx * 2,
        lane_half,
        wtid,
        scale_log2,
    )
    al.syncthreads()

    al.amdgpu.eager_materialize_i32(pair_idx)
    al.syncthreads()
    if next_even_slice <= max_k_slice:
        al.amdgpu.s_waitcnt(0, 7, 15)
        al.syncthreads()
        _flash_attn_compute_qk_page_loaded(
            k_stage_words,
            q_regs,
            score_acc,
            key_row_local,
            lane_half,
            global_query_row,
            seq_len,
            actual_next_even_key_base,
            first_causal_key_base,
            0,
        )
        al.syncthreads()
        _fetch_v_global_slice_packed_strided(
            g_v0,
            v_rsrc,
            wid,
            wtid,
            actual_next_even_key_base,
            kv_row_stride_words,
            0,
        )
        al.syncthreads()
        al.amdgpu.eager_materialize_i32(next_even_slice)
        al.syncthreads()
        al.amdgpu.s_waitcnt(0, 7, 15)
        _store_v_global_slice_packed(work_words, g_v1, wid, wtid, 1)
        al.syncthreads()
        _fetch_v_reg_words_batch_compact(v_regs, v_words, key_row_local, wtid, 1)
        _flash_attn_update_o_batch1(
            score_acc,
            out_acc,
            v_regs,
            softmax_state,
            odd_slice,
            lane_half,
            wtid,
            scale_log2,
        )
        al.syncthreads()
        al.amdgpu.eager_materialize_i32(odd_slice)
        al.syncthreads()


@avelang.jit
def _flash_attn_packed_pair_step_wg1(
    k_stage_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    work_words: al.Tensor((SHM_WORK_WORDS,), al.u32),
    v_words: al.Tensor((SHM_V_WORDS,), al.u32),
    k_rsrc: al.Tensor((4,), al.u32),
    v_rsrc: al.Tensor((4,), al.u32),
    q_regs: al.Tensor((Q_BATCHES, 2, 2), al.u32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    g_v0: al.Tensor((4,), al.u32),
    g_v1: al.Tensor((4,), al.u32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    softmax_state: al.Tensor((2,), al.f32),
    tid: al.u32,
    wid: al.u32,
    wtid: al.u32,
    key_row_local: al.u32,
    lane_half: al.u32,
    global_query_row: al.u32,
    seq_len: al.u32,
    pair_idx: al.u32,
    max_k_slice: al.u32,
    actual_odd_key_base: al.u32,
    actual_next_even_key_base: al.u32,
    first_causal_key_base: al.u32,
    kv_row_stride_words: al.u32,
    scale_log2: al.f32,
):
    odd_slice = pair_idx * 2 + 1
    next_even_slice = odd_slice + 1

    _fetch_v_global_slice_packed_strided(
        g_v1,
        v_rsrc,
        wid,
        wtid,
        actual_odd_key_base,
        kv_row_stride_words,
        0,
    )
    al.syncthreads()
    _flash_attn_compute_qk_page_loaded(
        k_stage_words,
        q_regs,
        score_acc,
        key_row_local,
        lane_half,
        global_query_row,
        seq_len,
        actual_odd_key_base,
        first_causal_key_base,
        1,
    )
    al.syncthreads()

    al.amdgpu.s_waitcnt(0, 7, 15)
    _store_v_global_slice_packed(work_words, g_v0, wid, wtid, 0)
    al.syncthreads()
    _fetch_v_reg_words_batch_compact(v_regs, v_words, key_row_local, wtid, 0)
    _flash_attn_update_o_batch0(
        score_acc,
        out_acc,
        v_regs,
        softmax_state,
        pair_idx * 2,
        lane_half,
        wtid,
        scale_log2,
    )
    al.syncthreads()

    al.amdgpu.eager_materialize_i32(pair_idx)
    al.syncthreads()
    if next_even_slice <= max_k_slice:
        _fetch_v_global_slice_packed_strided(
            g_v0,
            v_rsrc,
            wid,
            wtid,
            actual_next_even_key_base,
            kv_row_stride_words,
            0,
        )
        al.syncthreads()
        al.amdgpu.s_waitcnt(0, 7, 15)
        al.syncthreads()
        _flash_attn_compute_qk_page_loaded(
            k_stage_words,
            q_regs,
            score_acc,
            key_row_local,
            lane_half,
            global_query_row,
            seq_len,
            actual_next_even_key_base,
            first_causal_key_base,
            0,
        )
        al.syncthreads()
        al.amdgpu.eager_materialize_i32(next_even_slice)
        al.syncthreads()
        al.amdgpu.s_waitcnt(0, 7, 15)
        _store_v_global_slice_packed(work_words, g_v1, wid, wtid, 1)
        al.syncthreads()
        _fetch_v_reg_words_batch_compact(v_regs, v_words, key_row_local, wtid, 1)
        _flash_attn_update_o_batch1(
            score_acc,
            out_acc,
            v_regs,
            softmax_state,
            odd_slice,
            lane_half,
            wtid,
            scale_log2,
        )
        al.syncthreads()
        al.amdgpu.eager_materialize_i32(odd_slice)
        al.syncthreads()


@avelang.jit
def _flash_attn_packed_drain_epilogue(
    work_words: al.Tensor((SHM_WORK_WORDS,), al.u32),
    v_words: al.Tensor((SHM_V_WORDS,), al.u32),
    score_acc: al.Tensor((K_BATCHES, 16), al.f32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    v_regs: al.Tensor((O_BATCHES, 4, 2), al.u32),
    g_v0: al.Tensor((4,), al.u32),
    g_v1: al.Tensor((4,), al.u32),
    softmax_state: al.Tensor((2,), al.f32),
    wid: al.u32,
    wtid: al.u32,
    key_row_local: al.u32,
    lane_half: al.u32,
    max_k_slice: al.u32,
    scale_log2: al.f32,
):
    epilogue_uses_s1 = (max_k_slice & 1) != 0
    if epilogue_uses_s1:
        _flash_attn_prepare_o_batch1(
            score_acc,
            out_acc,
            softmax_state,
            max_k_slice,
            lane_half,
            wtid,
            scale_log2,
        )
    else:
        _flash_attn_prepare_o_batch0(
            score_acc,
            out_acc,
            softmax_state,
            max_k_slice,
            lane_half,
            wtid,
            scale_log2,
        )
    if epilogue_uses_s1:
        al.amdgpu.s_waitcnt(0, 7, 15)
        _store_v_global_slice_packed(work_words, g_v1, wid, wtid, 1)
        al.syncthreads()
        _fetch_v_reg_words_batch_compact(v_regs, v_words, key_row_local, wtid, 1)
        _gemm_o_mfma_v_word_regs_batch1_direct(out_acc, score_acc, v_regs)
    if not epilogue_uses_s1:
        al.amdgpu.s_waitcnt(0, 7, 15)
        _store_v_global_slice_packed(work_words, g_v0, wid, wtid, 0)
        al.syncthreads()
        _fetch_v_reg_words_batch_compact(v_regs, v_words, key_row_local, wtid, 0)
        _gemm_o_mfma_v_word_regs_batch0_direct(out_acc, score_acc, v_regs)


@avelang.jit
def _store_gemm_o_mfma_normalized_packed_lds(
    out_rsrc: al.Tensor((4,), al.u32),
    o_words: al.Tensor((SHM_O_WORDS,), al.u32),
    out_acc: al.Tensor((O_BATCHES, 16), al.f32),
    tid: al.u32,
    wid: al.u32,
    wtid: al.u32,
    lane_col: al.u32,
    lane_half: al.u32,
    l: al.f32,
):
    vecs_per_row = HEAD_DIM // VEC_SIZE
    o_u2 = al.view(
        o_words,
        al.u32,
        al.make_layout(
            (NUM_WARPS, O_BATCHES * 4, WARP_ROWS, 2, 2),
            (WARP_ROWS * (HEAD_DIM // 4) * 2, 4, (HEAD_DIM // 4) * 2, 2, 1),
        ),
    )
    o_u4 = al.view(
        o_words,
        al.u32,
        al.make_layout(
            (SHM_O_VECS // THREADS, THREADS, 4),
            (THREADS * 4, 4, 1),
        ),
    )
    l_total = l + _softmax_xor_permute_f32(l, wtid)
    inv_l = al.amdgpu.rcp(l_total)

    for o_batch in al.range(O_BATCHES):
        for t_group in al.range(4):
            f0 = out_acc[o_batch, t_group * 4] * inv_l
            f1 = out_acc[o_batch, t_group * 4 + 1] * inv_l
            f2 = out_acc[o_batch, t_group * 4 + 2] * inv_l
            f3 = out_acc[o_batch, t_group * 4 + 3] * inv_l
            packed0 = al.amdgpu.perm(al.bitcast(f1, al.u32), al.bitcast(f0, al.u32), 0x07060302)
            packed1 = al.amdgpu.perm(al.bitcast(f3, al.u32), al.bitcast(f2, al.u32), 0x07060302)
            element_idx = o_batch * 4 + t_group
            packed = al.full((2,), 0, al.u32)
            packed[0] = packed0
            packed[1] = packed1
            o_u2[wid, element_idx, lane_col, lane_half] = packed
    al.syncthreads()

    for store_batch in al.range(SHM_O_VECS // THREADS):
        vec_idx = tid + store_batch * THREADS
        row = vec_idx // vecs_per_row
        col_vec = vec_idx - row * vecs_per_row
        vec = o_u4[store_batch, tid]
        elem_offset = row * Q_HEADS * HEAD_DIM + col_vec * VEC_SIZE
        byte_offset = elem_offset * BF16_BYTES
        al.amdgpu.raw_buffer_store_x4(vec, out_rsrc, byte_offset, 0, 0)
    al.syncthreads()


@avelang.jit
def _actual_k_slice_from_ordinal(
    max_k_slice: al.u32,
    slice_ordinal: al.u32,
    reverse_pass: al.u32,
) -> al.u32:
    return slice_ordinal + reverse_pass * (max_k_slice - slice_ordinal - slice_ordinal)


@avelang.jit
def _flash_attn_packed_process_tile(
    out_ptr: al.Pointer(al.bf16),
    total_tokens: al.u32,
    seq_begin: al.u32,
    q_head_idx: al.u32,
    q_tile_idx: al.u32,
    seq_len: al.u32,
    kv_words: al.Tensor((SHM_WORK_WORDS,), al.u32),
    q_rsrc: al.Tensor((4,), al.u32),
    k_rsrc: al.Tensor((4,), al.u32),
    v_rsrc: al.Tensor((4,), al.u32),
    q_row_stride_words: al.u32,
    kv_row_stride_words: al.u32,
    tid: al.u32,
    wid: al.u32,
    wtid: al.u32,
    lane_col: al.u32,
    lane_half: al.u32,
    query_row: al.u32,
    key_row_local: al.u32,
    scale_log2: al.f32,
    reverse_pass: al.u32,
):
    zero_f32 = al.convert(0.0, al.f32)
    q_base = q_tile_idx * BLOCK_ROWS
    if q_base >= seq_len:
        return

    global_query_row = q_base + query_row

    k_stage_words = al.subview(kv_words, (0,), (SHM_V_OFFSET_WORDS,), (1,))
    q_stage_words = al.subview(kv_words, (SHM_V_OFFSET_WORDS,), (SHM_Q_STAGE_WORDS,), (1,))
    v_words = al.subview(kv_words, (SHM_V_OFFSET_WORDS,), (SHM_V_WORDS,), (1,))

    q_regs = al.make_local((Q_BATCHES, 2, 2), al.u32)
    v_regs = al.make_local((O_BATCHES, 4, 2), al.u32)
    g_v0 = al.make_local((4,), al.u32)
    g_v1 = al.make_local((4,), al.u32)
    score_acc = al.make_local((K_BATCHES, 16), al.f32)
    out_acc = al.make_local((O_BATCHES, 16), al.f32)
    softmax_state = al.make_local((2,), al.f32)
    mi = zero_f32
    l = zero_f32
    softmax_state[0] = mi
    softmax_state[1] = l
    for o_batch in al.range(O_BATCHES):
        out_acc[o_batch] = al.full((16,), 0.0, al.f32)

    _load_q_reg_words_lds_strided(
        q_regs,
        q_stage_words,
        q_rsrc,
        query_row,
        global_query_row,
        q_row_stride_words,
        wid,
        lane_col,
        lane_half,
    )
    seq_last_row = seq_len - 1
    tile_last_row = q_base + BLOCK_ROWS - 1
    last_query_row = tile_last_row
    if seq_last_row < tile_last_row:
        last_query_row = seq_last_row
    max_k_slice = last_query_row // K_SLICE_ROWS
    pair_count = (max_k_slice + 1) // 2
    first_slice = _actual_k_slice_from_ordinal(max_k_slice, 0, reverse_pass)
    second_slice = _actual_k_slice_from_ordinal(max_k_slice, 1, reverse_pass)
    first_key_base = first_slice * K_SLICE_ROWS
    second_key_base = second_slice * K_SLICE_ROWS
    first_causal_key_base = q_base

    _flash_attn_compute_qk_page(
        k_stage_words,
        k_rsrc,
        q_regs,
        score_acc,
        tid,
        key_row_local,
        lane_half,
        global_query_row,
        seq_len,
        first_key_base,
        first_causal_key_base,
        kv_row_stride_words,
        0,
    )
    _fetch_v_global_slice_packed_strided(
        g_v0,
        v_rsrc,
        wid,
        wtid,
        first_key_base,
        kv_row_stride_words,
        0,
    )
    _load_k_page_strided(
        k_stage_words,
        k_rsrc,
        tid,
        second_key_base,
        kv_row_stride_words,
        1,
    )
    _load_k_page_strided(
        k_stage_words,
        k_rsrc,
        tid,
        first_key_base,
        kv_row_stride_words,
        0,
    )
    al.amdgpu.s_waitcnt(0, 7, 15)
    al.syncthreads()
    if wid < (NUM_WARPS // 2):
        _fetch_v_global_slice_packed_strided(
            g_v1,
            v_rsrc,
            wid,
            wtid,
            second_key_base,
            kv_row_stride_words,
            0,
        )

    if wid < (NUM_WARPS // 2):
        for pair_idx in al.range(pair_count):
            odd_slice_pre = pair_idx * 2 + 1
            actual_odd_slice = _actual_k_slice_from_ordinal(max_k_slice, odd_slice_pre, reverse_pass)
            actual_odd_key_base = actual_odd_slice * K_SLICE_ROWS
            if pair_idx != 0:
                if odd_slice_pre <= max_k_slice:
                    _load_k_page_strided(
                        k_stage_words,
                        k_rsrc,
                        tid,
                        actual_odd_key_base,
                        kv_row_stride_words,
                        1,
                    )
                    al.amdgpu.s_waitcnt(0, 7, 15)
                    al.syncthreads()
            next_even_slice_pre = pair_idx * 2 + 2
            actual_next_even_slice = _actual_k_slice_from_ordinal(max_k_slice, next_even_slice_pre, reverse_pass)
            actual_next_even_key_base = actual_next_even_slice * K_SLICE_ROWS
            if next_even_slice_pre <= max_k_slice:
                _load_k_page_strided(
                    k_stage_words,
                    k_rsrc,
                    tid,
                    actual_next_even_key_base,
                    kv_row_stride_words,
                    0,
                )
            _flash_attn_packed_pair_step_wg0(
                k_stage_words,
                kv_words,
                v_words,
                k_rsrc,
                v_rsrc,
                q_regs,
                v_regs,
                g_v0,
                g_v1,
                score_acc,
                out_acc,
                softmax_state,
                tid,
                wid,
                wtid,
                key_row_local,
                lane_half,
                global_query_row,
                seq_len,
                pair_idx,
                max_k_slice,
                actual_odd_key_base,
                actual_next_even_key_base,
                first_causal_key_base,
                kv_row_stride_words,
                scale_log2,
            )
    else:
        for pair_idx in al.range(pair_count):
            odd_slice_pre = pair_idx * 2 + 1
            actual_odd_slice = _actual_k_slice_from_ordinal(max_k_slice, odd_slice_pre, reverse_pass)
            actual_odd_key_base = actual_odd_slice * K_SLICE_ROWS
            if pair_idx != 0:
                if odd_slice_pre <= max_k_slice:
                    _load_k_page_strided(
                        k_stage_words,
                        k_rsrc,
                        tid,
                        actual_odd_key_base,
                        kv_row_stride_words,
                        1,
                    )
                    al.amdgpu.s_waitcnt(0, 7, 15)
                    al.syncthreads()
            next_even_slice_pre = pair_idx * 2 + 2
            actual_next_even_slice = _actual_k_slice_from_ordinal(max_k_slice, next_even_slice_pre, reverse_pass)
            actual_next_even_key_base = actual_next_even_slice * K_SLICE_ROWS
            if next_even_slice_pre <= max_k_slice:
                _load_k_page_strided(
                    k_stage_words,
                    k_rsrc,
                    tid,
                    actual_next_even_key_base,
                    kv_row_stride_words,
                    0,
                )
            _flash_attn_packed_pair_step_wg1(
                k_stage_words,
                kv_words,
                v_words,
                k_rsrc,
                v_rsrc,
                q_regs,
                v_regs,
                g_v0,
                g_v1,
                score_acc,
                out_acc,
                softmax_state,
                tid,
                wid,
                wtid,
                key_row_local,
                lane_half,
                global_query_row,
                seq_len,
                pair_idx,
                max_k_slice,
                actual_odd_key_base,
                actual_next_even_key_base,
                first_causal_key_base,
                kv_row_stride_words,
                scale_log2,
            )

    _flash_attn_packed_drain_epilogue(
        kv_words,
        v_words,
        score_acc,
        out_acc,
        v_regs,
        g_v0,
        g_v1,
        softmax_state,
        wid,
        wtid,
        key_row_local,
        lane_half,
        max_k_slice,
        scale_log2,
    )

    out_elems = total_tokens * Q_HEADS * HEAD_DIM
    out_flat = al.make_tensor(out_ptr, al.bf16, al.make_layout((out_elems,), (1,)))
    remaining_o_rows = seq_len - q_base
    store_o_rows = al.min(remaining_o_rows, al.convert(BLOCK_ROWS, al.u32))
    out_span_elems = (store_o_rows - 1) * Q_HEADS * HEAD_DIM + HEAD_DIM
    out_base_elems = ((seq_begin + q_base) * Q_HEADS + q_head_idx) * HEAD_DIM
    out_view = al.subview(out_flat, (out_base_elems,), (out_span_elems,), (1,))
    out_rsrc = al.amdgpu.make_rsrc(out_view, out_span_elems * BF16_BYTES)

    o_words = al.subview(kv_words, (0,), (SHM_O_WORDS,), (1,))
    _store_gemm_o_mfma_normalized_packed_lds(
        out_rsrc,
        o_words,
        out_acc,
        tid,
        wid,
        wtid,
        lane_col,
        lane_half,
        softmax_state[1],
    )


@avelang.jit
def _flash_attn_packed_kernel(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    seq_ptr: al.Pointer(al.i32),
    out_ptr: al.Pointer(al.bf16),
    total_tokens: al.u32,
    num_seqs: al.u32,
):
    scale_log2 = al.convert(SCALE_LOG2, al.f32)

    physical_q_tile_idx = al.block_id(0)
    physical_q_head_idx = al.block_id(1)
    seq_id = al.block_id(2)
    # al.assume(q_head_idx < Q_HEADS)
    # al.assume(seq_id < num_seqs)

    seq_layout = al.make_layout((num_seqs + 1,), (1,))
    seq_start = al.make_tensor(seq_ptr, al.i32, seq_layout)
    seq_begin = al.convert(seq_start[seq_id], al.u32)
    seq_end = al.convert(seq_start[seq_id + 1], al.u32)
    seq_begin = al.amdgpu.readfirstlane(seq_begin)
    seq_len = al.amdgpu.readfirstlane(seq_end - seq_begin)
    num_q_tiles = al.amdgpu.readfirstlane(
        (seq_len + BLOCK_ROWS - 1) // BLOCK_ROWS
    )
    physical_tiles = al.amdgpu.readfirstlane((num_q_tiles + 1) // 2)
    if physical_q_tile_idx >= physical_tiles:
        return
    # al.assume(physical_q_tile_idx < physical_tiles)

    tid = al.thread_id(0)
    # al.assume(tid < THREADS)
    wid = al.amdgpu.readfirstlane(tid // WARP_SIZE)
    # al.assume(wid < NUM_WARPS)
    wtid = tid % WARP_SIZE
    lane_col = wtid % WARP_ROWS
    lane_half = wtid // WARP_ROWS
    query_row = wid * WARP_ROWS + lane_col

    if wid < (NUM_WARPS // 2):
        al.amdgpu.s_setprio(0)
    else:
        al.amdgpu.s_setprio(1)

    key_row_local = lane_col

    kv_words = al.make_shared((SHM_WORK_WORDS,), al.u32, 16)

    q_elems = total_tokens * Q_HEADS * HEAD_DIM
    kv_elems = total_tokens * KV_HEADS * HEAD_DIM
    q_flat = al.make_tensor(q_ptr, al.bf16, al.make_layout((q_elems,), (1,)))
    k_flat = al.make_tensor(k_ptr, al.bf16, al.make_layout((kv_elems,), (1,)))
    v_flat = al.make_tensor(v_ptr, al.bf16, al.make_layout((kv_elems,), (1,)))

    q_head_idx = physical_q_head_idx
    head_idx_q = physical_q_tile_idx
    merged_head_tile = (physical_q_head_idx & 7) * physical_tiles + physical_q_tile_idx
    q_head_idx = (physical_q_head_idx // 8) * 8 + (merged_head_tile & 7)
    head_idx_q = merged_head_tile >> 3

    kv_head_idx = q_head_idx // (Q_HEADS // KV_HEADS)
    q_row_stride_elems = Q_HEADS * HEAD_DIM
    kv_row_stride_elems = KV_HEADS * HEAD_DIM
    q_span_elems = (seq_len - 1) * q_row_stride_elems + HEAD_DIM
    kv_span_elems = (seq_len - 1) * kv_row_stride_elems + HEAD_DIM
    q_base_elems = (seq_begin * Q_HEADS + q_head_idx) * HEAD_DIM
    kv_base_elems = (seq_begin * KV_HEADS + kv_head_idx) * HEAD_DIM
    q_view = al.subview(q_flat, (q_base_elems,), (q_span_elems,), (1,))
    k_view = al.subview(k_flat, (kv_base_elems,), (kv_span_elems,), (1,))
    v_view = al.subview(v_flat, (kv_base_elems,), (kv_span_elems,), (1,))
    q_rsrc = al.amdgpu.make_rsrc(q_view, q_span_elems * BF16_BYTES)
    k_rsrc = al.amdgpu.make_rsrc(k_view, kv_span_elems * BF16_BYTES)
    v_rsrc = al.amdgpu.make_rsrc(v_view, kv_span_elems * BF16_BYTES)

    q_row_stride_words = q_row_stride_elems // 2
    kv_row_stride_words = kv_row_stride_elems // 2

    mirrored_idx_q = num_q_tiles - 1 - head_idx_q
    num_passes = 1
    if mirrored_idx_q != head_idx_q:
        num_passes = 2

    tile_idx = head_idx_q
    reverse_pass = 0
    for _ in al.range(num_passes):
        _flash_attn_packed_process_tile(
            out_ptr,
            total_tokens,
            seq_begin,
            q_head_idx,
            tile_idx,
            seq_len,
            kv_words,
            q_rsrc,
            k_rsrc,
            v_rsrc,
            q_row_stride_words,
            kv_row_stride_words,
            tid,
            wid,
            wtid,
            lane_col,
            lane_half,
            query_row,
            key_row_local,
            scale_log2,
            reverse_pass,
        )
        tile_idx = num_q_tiles - 1 - tile_idx
        reverse_pass = 1 - reverse_pass


@avelang.jit
def _load_k_page_strided(
    k_words: al.Tensor((SHM_V_OFFSET_WORDS,), al.u32),
    k_rsrc: al.Tensor((4,), al.u32),
    tid: al.u32,
    actual_key_base: al.u32,
    row_stride_words: al.u32,
    page: al.u32,
):
    words_per_row = HEAD_DIM // 2
    page_word_base = page * K_PAGE_WORDS
    wid = al.amdgpu.readfirstlane(tid // WARP_SIZE)
    for word_batch in al.range(K_SLICE_DATA_WORDS // THREADS):
        row = tid // words_per_row
        dest_col_word = tid % words_per_row
        tile_col = dest_col_word // QK_TILE_DWORDS
        dword_in_tile = dest_col_word - tile_col * QK_TILE_DWORDS
        col_word = (
            tile_col * QK_TILE_DWORDS
            + (dword_in_tile & 0x1)
            + ((dword_in_tile >> 1) & 0x1)
            * (QK_TILE_DWORDS // 2)
            + (dword_in_tile >> 2) * 2
        )
        global_vword = row * row_stride_words + col_word
        global_sword = (
            actual_key_base + word_batch * (THREADS // (HEAD_DIM // 2))
        ) * row_stride_words
        lds_row = wid + (word_batch * NUM_WARPS)
        lds_word = page_word_base + lds_row * QK_ROW_STRIDE_WORDS
        global_voffset = global_vword * U32_BYTES
        global_soffset = global_sword * U32_BYTES
        lds_offset = lds_word * U32_BYTES
        al.amdgpu.raw_buffer_load_x1_lds(
            k_rsrc,
            k_words,
            U32_BYTES,
            global_voffset,
            global_soffset,
            lds_offset,
            0,
        )


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

    total_tokens = q.shape[0]
    seq_lens = [seq_ptr_cpu[idx + 1] - seq_ptr_cpu[idx] for idx in range(len(seq_ptr_cpu) - 1)]
    max_seq_len = max(seq_lens, default=0)
    if total_tokens == 0 or max_seq_len == 0:
        return out

    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    seq_ptr_device = seq_ptr.to(device=q.device, dtype=torch.int32)
    row_tiles = math.ceil(max_seq_len / BLOCK_ROWS)
    physical_tiles = math.ceil(row_tiles / 2)
    _flash_attn_packed_kernel[lambda: ((physical_tiles, q_heads, len(seq_ptr_cpu) - 1), (THREADS, 1, 1))](
        q,
        k,
        v,
        seq_ptr_device,
        out,
        total_tokens,
        len(seq_ptr_cpu) - 1,
        num_warps=NUM_WARPS,
    )
    return out


__all__ = [
    "flash_attn"
]

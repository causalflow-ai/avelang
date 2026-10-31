"""AMDGPU-specific language intrinsics."""


class AtomicScope:
    WORKGROUP = 0
    AGENT = 1
    SYSTEM = 2


def atomic_add(voffset, data, tensor, scope):
    pass


def mfma_16x16x16_f16_f32(a, b, c):
    pass


def mfma_16x16x16_bf16_f32(a, b, c):
    pass


def mfma_f32_16x16x16_bf16(a, b, c):
    pass


def mfma_32x32x8_bf16_f32(a, b, c):
    pass


def mfma_f32_32x32x8_bf16(a, b, c):
    pass


def mfma_scale_f32_16x16x128_f8f6f4(
    a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b,
):
    """gfx950 scaled MFMA with packed u32 inputs and an f32x4 accumulator/result.

    cbsz and blgp independently select A/B formats: 0 = FP8 E4M3, 1 = FP8 E5M2,
    2 = FP6 E2M3, 3 = FP6 E3M2, 4 = FP4 E2M1. Inputs must contain 4, 6, or 8
    u32 values per lane, with at least 8 for FP8, 6 for FP6, and 4 for FP4.
    Extra u32 values are ignored, as in the LLVM intrinsic.

    Each u32 scale packs four E8M0 bytes and may vary per lane. opsel_a and
    opsel_b select the scale bytes. Formats and selectors must be compile-time
    integers, with selectors in [0, 3]. All four immediates and the packed
    data/scales are passed directly to the native instruction. Inputs must
    follow its format-specific register/lane layout (CDNA4 ISA 7.1.5.1):
    for g = lane // 16, FP4/FP6 hold K[32*g:32*g+32], while FP8 holds
    K[16*g:16*g+16] followed by K[64+16*g:80+16*g].
    """
    pass


def mfma_scale_f32_32x32x64_f8f6f4(
    a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b,
):
    """gfx950 scaled MFMA with packed u32 inputs and an f32x16 accumulator/result.

    Formats, input widths, scales, selectors, and argument order follow
    mfma_scale_f32_16x16x128_f8f6f4. For g = lane // 32, FP4/FP6 hold
    K[32*g:32*g+32], while FP8 holds K[16*g:16*g+16] followed by
    K[32+16*g:48+16*g]. Each lane accumulates 16 f32 values.
    """
    pass


def make_rsrc(tensor, range_bytes):
    pass


def perm(lhs, rhs, sel):
    pass


def get_dpp(old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl):
    pass


def rcp(value):
    pass


def s_waitcnt(vmcnt, expcnt, lgkmcnt):
    pass


def v_setvskip(mask, skip_id):
    pass


def readfirstlane(value):
    pass


def raw_buffer_load_x1(rsrc, vindex, soffset, aux):
    pass


def raw_buffer_load_x2(rsrc, vindex, soffset, aux):
    pass


def raw_buffer_load_x4(rsrc, vindex, soffset, aux):
    pass


def raw_buffer_load_x1_lds(rsrc, lds_ptr, size, vindex, soffset, offset, aux):
    pass


def raw_buffer_store_x1(vdata, rsrc, vindex, soffset, aux):
    pass


def raw_buffer_store_x2(vdata, rsrc, vindex, soffset, aux):
    pass


def raw_buffer_store_x4(vdata, rsrc, vindex, soffset, aux):
    pass


def cvt_pk_fp8_f32(src0, src1, old, word_sel):
    pass

def cvt_pk_bf8_f32(src0, src1, old, word_sel):
    pass

def sched_group_barrier(mask, size, group_id):
    pass

def sched_barrier(mask):
    pass

def s_setprio(priority):
    pass

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

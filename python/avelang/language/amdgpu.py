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


def bitreverse(value):
    pass


def get_dpp(old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl):
    pass


def ds_swizzle(value, pattern):
    """Permute i32 lanes using a constant AMD DS swizzle pattern."""
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
    """Load one dword (4 bytes) per lane into LDS; size must be 4."""
    pass


def raw_buffer_load_x4_lds(rsrc, lds_ptr, size, vindex, soffset, offset, aux):
    """Load four dwords (16 bytes) per lane into LDS; size must be 16."""
    pass


def raw_buffer_store_u8(value, resource, byte_offset, scalar_offset, aux):
    """Store an i8/u8 using i32 offsets and constant aux in [0, 31]."""
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

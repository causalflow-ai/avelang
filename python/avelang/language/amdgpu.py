"""AMDGPU-specific language intrinsics."""


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


def ds_permute_b32(index, value):
    pass


def rcp(value):
    pass


def s_waitcnt(vmcnt, expcnt, lgkmcnt):
    pass


def readfirstlane(value):
    pass


def eager_materialize_i32(value):
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


def sched_group_barrier(mask, size, group_id):
    pass


def sched_barrier(mask):
    pass


def s_setprio(priority):
    pass

"""NVIDIA-specific language intrinsics."""


def mma_16x8x16_f16_f16(a, b, c):
    pass


def mma_16x8x8_f16_f32(a, b, c):
    pass


def ldmatrix_m8n8_x1_b16(ptr):
    pass


def ldmatrix_m8n8_x1_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x2_b16(ptr):
    pass


def ldmatrix_m8n8_x2_b16_trans(ptr):
    pass


def ldmatrix_m8n8_x4_b16(ptr):
    pass


def ldmatrix_m8n8_x4_b16_trans(ptr):
    pass


def stmatrix_m8n8_x1_b16(ptr, data):
    pass


def stmatrix_m8n8_x1_b16_trans(ptr, data):
    pass


def stmatrix_m8n8_x2_b16(ptr, data):
    pass


def stmatrix_m8n8_x2_b16_trans(ptr, data):
    pass


def stmatrix_m8n8_x4_b16(ptr, data):
    pass


def stmatrix_m8n8_x4_b16_trans(ptr, data):
    pass

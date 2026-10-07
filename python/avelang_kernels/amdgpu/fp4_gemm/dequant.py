"""Device-side FP4 dequantization helpers for AMDGPU."""

import avelang
import avelang.language as al

from .solution import MatmulElementB, MatmulMfmaType


@avelang.jit
def fp4_to_fp16(out: al.Tensor((4,), al.u32), q: al.u32):
    qr = al.amdgpu.bitreverse(q)
    mask = al.convert(0x8E008E00, al.u32)
    out[0] = q & mask
    out[1] = (q << 8) & mask
    out[2] = qr & mask
    out[3] = (qr << 8) & mask


def make_dequant(config):
    ELEMENT_B = config.element_b
    MFMA_TYPE = config.mfma_type
    MXFP4 = MatmulElementB.MXFP4
    FP16 = MatmulMfmaType.FP16
    USE_BF8 = config.arch in ("gfx942", "gfx950")
    HIGH_PRECISION = config.high_precision
    WEIGHT_BIAS = config.weight_bias

    @avelang.jit
    def dequant_scales(packed_scale: al.u16) -> al.u32:
        packed = al.convert(packed_scale, al.u32)
        if ELEMENT_B == MXFP4:
            # Petit DequantFullScale: construct a BF16 pair, then adjust
            # its packed exponents, including zero/255 and overflow behavior.
            bits = ((packed & 0xFF) << 7) | ((packed >> 8) << 23)
            if HIGH_PRECISION:
                bits = bits + al.convert(0x03800380, al.u32)
            return bits
        else:
            # NVFP4 scales are already preprocessed by a factor of 128.
            return al.amdgpu.perm(0, packed, 0x0C010C00) << 7

    @avelang.jit
    def dequant_with_scale_fp16(q: al.u32, scale: al.f16, out: al.Tensor((4,), al.u32)):
        s2 = al.full((1, 2), scale, al.f16)
        bias2 = al.full((1, 2), WEIGHT_BIAS, al.f16)
        fp4_to_fp16(out, q)
        h2 = al.view(out, al.f16, al.make_layout((4, 2), (2, 1)))
        for i in al.range(4):
            if HIGH_PRECISION:
                h2[i] = h2[i] * bias2[0]
            h2[i] = h2[i] * s2[0]

    @avelang.jit
    def dequant_with_scale_impl_bf8_fnuz(q: al.u32, scale: al.f32, out: al.Tensor((4,), al.u32)):
        s2 = al.full((1, 2), scale, al.f32)
        bias_f32_2 = al.full((1, 2), WEIGHT_BIAS, al.f32)
        mask = al.convert(0x8E8E8E8E, al.u32)
        bf8 = al.make_local((2,), al.u32)
        bf8[0] = q & mask
        bf8[1] = al.amdgpu.bitreverse(q) & mask
        out_f2 = al.make_local((4, 2), al.f32)
        for i in al.range(2):
            out_f2[i * 2] = al.amdgpu.cvt_pk_f32_bf8(bf8[i], 0)
            out_f2[i * 2 + 1] = al.amdgpu.cvt_pk_f32_bf8(bf8[i], 1)
        for i in al.range(4):
            if HIGH_PRECISION:
                out_f2[i] = out_f2[i] * bias_f32_2[0]
            out_f2[i] = out_f2[i] * s2[0]
        out_b32 = al.view(out_f2, al.u32, al.make_layout((8,), (1,)))
        out[0] = al.amdgpu.perm(out_b32[3], out_b32[1], 0x07060302)
        out[1] = al.amdgpu.perm(out_b32[2], out_b32[0], 0x07060302)
        out[2] = al.amdgpu.perm(out_b32[7], out_b32[5], 0x07060302)
        out[3] = al.amdgpu.perm(out_b32[6], out_b32[4], 0x07060302)

    @avelang.jit
    def dequant_with_scale_impl_fp16(q: al.u32, scale: al.f32, out: al.Tensor((4,), al.u32)):
        s2 = al.full((1, 2), scale, al.f32)
        bias_f32_2 = al.full((1, 2), WEIGHT_BIAS, al.f32)
        fp4_to_fp16(out, q)
        h2 = al.view(out, al.f16, al.make_layout((4, 2), (2, 1)))
        out_f2 = al.make_local((4, 2), al.f32)
        for i in al.range(4):
            out_f2[i, 0] = al.convert(h2[i, 0], al.f32)
            out_f2[i, 1] = al.convert(h2[i, 1], al.f32)
        for i in al.range(4):
            if HIGH_PRECISION:
                out_f2[i] = out_f2[i] * bias_f32_2[0]
            out_f2[i] = out_f2[i] * s2[0]
        out_b32 = al.view(out_f2, al.u32, al.make_layout((4, 2), (2, 1)))
        for i in al.range(4):
            out[i] = al.amdgpu.perm(out_b32[i, 1], out_b32[i, 0], 0x07060302)

    @avelang.jit
    def dequant_with_scale(q: al.u32, scale: al.u16, out: al.Tensor((4,), al.u32)):
        if MFMA_TYPE == FP16:
            dequant_with_scale_fp16(q, al.bitcast(scale, al.f16), out)
        else:
            scale_f32 = (
                al.convert(al.bitcast(scale, al.bf16), al.f32)
                if ELEMENT_B == MXFP4
                else al.convert(al.bitcast(scale, al.f16), al.f32)
            )
            if USE_BF8:
                dequant_with_scale_impl_bf8_fnuz(q, scale_f32, out)
            else:
                dequant_with_scale_impl_fp16(q, scale_f32, out)

    return dequant_scales, dequant_with_scale

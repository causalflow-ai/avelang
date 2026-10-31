"""gfx950 scaled F8/F6/F4 MFMA instruction contracts."""

import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_gfx950


pytestmark = [
    pytest.mark.skipif(not has_gfx950(), reason="Requires gfx950 scaled MFMA"),
    pytest.mark.parametrize("m", [16, 32]),
]


@avelang.jit
def kernel_scaled_mfma_16(
    a: al.Tensor((64, 8), al.u32),
    sa: al.Tensor((64,), al.u32),
    b: al.Tensor((64, 8), al.u32),
    sb: al.Tensor((64,), al.u32),
    c: al.Tensor((64, 4), al.f32),
    out: al.Tensor((64, 4), al.f32),
    cbsz: al.constexpr,
    blgp: al.constexpr,
    opsel_a: al.constexpr,
    opsel_b: al.constexpr,
    a_regs: al.constexpr,
    b_regs: al.constexpr,
):
    lane = al.thread_id(0)
    a_input = al.subview(a, (0, 0), (64, a_regs), (1, 1))
    b_input = al.subview(b, (0, 0), (64, b_regs), (1, 1))
    out[lane] = al.amdgpu.mfma_scale_f32_16x16x128_f8f6f4(
        a_input[lane], b_input[lane], c[lane], cbsz, blgp,
        opsel_a, sa[lane], opsel_b, sb[lane],
    )


@avelang.jit
def kernel_scaled_mfma_32(
    a: al.Tensor((64, 8), al.u32),
    sa: al.Tensor((64,), al.u32),
    b: al.Tensor((64, 8), al.u32),
    sb: al.Tensor((64,), al.u32),
    c: al.Tensor((64, 16), al.f32),
    out: al.Tensor((64, 16), al.f32),
    cbsz: al.constexpr,
    blgp: al.constexpr,
    opsel_a: al.constexpr,
    opsel_b: al.constexpr,
    a_regs: al.constexpr,
    b_regs: al.constexpr,
):
    lane = al.thread_id(0)
    a_input = al.subview(a, (0, 0), (64, a_regs), (1, 1))
    b_input = al.subview(b, (0, 0), (64, b_regs), (1, 1))
    out[lane] = al.amdgpu.mfma_scale_f32_32x32x64_f8f6f4(
        a_input[lane], b_input[lane], c[lane], cbsz, blgp,
        opsel_a, sa[lane], opsel_b, sb[lane],
    )


# Format code -> (exponent bits, mantissa bits).
FORMATS = ((4, 3), (5, 2), (2, 3), (3, 2), (2, 1))


def make_input(m, fmt, generator):
    exponent_bits, mantissa_bits = FORMATS[fmt]
    bits = 1 + exponent_bits + mantissa_bits
    bias = (1 << (exponent_bits - 1)) - 1
    k = 2048 // m
    # Keep every scaled sum exactly representable in f32, including mixed
    # formats and per-block scales in [1/4, 2]. FP4 still covers every code.
    exponent = torch.randint(
        max(0, bias - 1), min(1 << exponent_bits, bias + {4: 3, 6: 2, 8: 1}[bits]),
        (m, k), generator=generator,
    )
    mantissa = torch.randint(0, 1 << mantissa_bits, (m, k), generator=generator)
    sign = torch.randint(0, 2, (m, k), generator=generator)
    codes = (sign << (bits - 1)) | (exponent << mantissa_bits) | mantissa
    significand = (exponent != 0).double() + mantissa.double() / (1 << mantissa_bits)
    values = (1 - 2 * sign) * significand * torch.exp2(exponent.clamp(min=1).double() - bias)

    # CDNA4 ISA 7.1.5.1: FP8 uses two K16 chunks separated by K/2;
    # FP4/FP6 use one contiguous K32 chunk for both instruction shapes.
    lane = torch.arange(64)[:, None]
    element = torch.arange(32)[None, :]
    group = lane // m
    if bits == 8:
        k_index = 16 * group + (k // 2) * (element // 16) + element % 16
    else:
        k_index = 32 * group + element
    lanes = codes[lane % m, k_index]

    # Dense bit packing includes FP6 values crossing u32 boundaries.
    # Nonzero padding checks that excess registers are ignored.
    words = torch.full((64, 8), 0xDEADBEEF, dtype=torch.int64)
    words[:, :bits] = 0
    for i in range(32):
        word, shift = divmod(i * bits, 32)
        words[:, word] |= (lanes[:, i] << shift) & 0xFFFFFFFF
        if shift + bits > 32:
            words[:, word + 1] |= lanes[:, i] >> (32 - shift)
    return words.to(torch.uint32).cuda(), values, bits


def make_scales(m, selector, operand, vary_scales):
    lane = torch.arange(64)[:, None]
    byte = torch.arange(4)[None, :]
    if vary_scales:
        row, group = lane % m, lane // m
        # Distinguish rows, K32 blocks, A/B, and all four selector bytes.
        pattern = row + group + byte if operand == "a" else 2 * row + 3 * group - byte
        exponents = 125 + pattern % 4
    else:
        exponents = (125 + (byte if operand == "a" else 3 - byte)).expand(64, 4)
    words = torch.sum(exponents << (8 * byte), dim=1).to(torch.uint32).cuda()
    # Scale storage is indexed by logical (row, K32 block), independently
    # of the data format. FP8 data can consume scales from other lanes.
    selected = torch.exp2(exponents[:, selector].double() - 127)
    scales = selected.reshape(64 // m, m).T.repeat_interleave(32, dim=1)
    return words, scales


def check_scaled_mfma(m, cbsz, blgp, opsel_a, opsel_b,
                      a_regs=None, b_regs=None, vary_scales=True):
    generator = torch.Generator().manual_seed(123)
    a, a_values, a_min_regs = make_input(m, cbsz, generator)
    b, b_values, b_min_regs = make_input(m, blgp, generator)
    sa, a_scales = make_scales(m, opsel_a, "a", vary_scales)
    sb, b_scales = make_scales(m, opsel_b, "b", vary_scales)
    c = torch.arange(m * m, dtype=torch.float64).reshape(m, m) / 4

    # Express the reference with B rows and A columns, as in B @ A.T.
    # For 32x32 the 16 registers span four separate groups of four columns.
    lane = torch.arange(64)[:, None]
    register = torch.arange(m * m // 64)[None, :]
    row = lane % m
    column = 4 * (lane // m) + register % 4 + (register // 4) * (256 // m)
    c_fragments = c[row, column].float().contiguous().cuda()
    out = torch.empty_like(c_fragments)
    kernel = kernel_scaled_mfma_16 if m == 16 else kernel_scaled_mfma_32
    kernel[lambda: ((1, 1, 1), (64, 1, 1))](
        a, sa, b, sb, c_fragments, out, cbsz, blgp, opsel_a, opsel_b,
        a_min_regs if a_regs is None else a_regs,
        b_min_regs if b_regs is None else b_regs,
    )
    actual = torch.empty_like(c)
    actual[row, column] = out.cpu().double()
    expected = (b_values * b_scales) @ (a_values * a_scales).T + c
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cbsz", range(5))
@pytest.mark.parametrize("blgp", range(5))
@pytest.mark.parametrize("vary_scales", [False, True])
def test_scaled_mfma_formats(m, cbsz, blgp, vary_scales):
    check_scaled_mfma(m, cbsz, blgp, 1, 2, vary_scales=vary_scales)


@pytest.mark.parametrize("opsel_a", range(4))
@pytest.mark.parametrize("opsel_b", range(4))
def test_scaled_mfma_fp4_selectors(m, opsel_a, opsel_b):
    check_scaled_mfma(m, 4, 4, opsel_a, opsel_b)


@pytest.mark.parametrize("cbsz, blgp", [(0, 4), (2, 1)])
@pytest.mark.parametrize("opsel_a, opsel_b", [(0, 0), (1, 3), (2, 2), (3, 1)])
def test_scaled_mfma_mixed_selectors(m, cbsz, blgp, opsel_a, opsel_b):
    check_scaled_mfma(m, cbsz, blgp, opsel_a, opsel_b)


@pytest.mark.parametrize("cbsz, blgp, a_regs, b_regs", [
    (4, 4, 6, 8),
    (4, 2, 8, 8),
    (3, 4, 8, 6),
])
def test_scaled_mfma_excess_registers(m, cbsz, blgp, a_regs, b_regs):
    check_scaled_mfma(m, cbsz, blgp, 1, 2, a_regs, b_regs)

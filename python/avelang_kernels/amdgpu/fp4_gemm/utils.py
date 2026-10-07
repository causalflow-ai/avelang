"""AveLang's packed FP4 weight and scale formats."""

import torch

from .solution import MatmulElementB, fp4_scale_group_size

WEIGHT_TILE_N = 32
WEIGHT_TILE_K = 256
SCALE_TILE_N = 64
SCALE_TILE_K = 64

_SIGN_OFFSETS = (15, 31, 7, 23, 16, 0, 24, 8)
_VALUE_OFFSETS = (9, 25, 1, 17, 20, 4, 28, 12)


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def packed_weight_shape(n: int, k: int) -> tuple[int, ...]:
    return (k // WEIGHT_TILE_K, 4, n // WEIGHT_TILE_N, 64, 4)


def packed_scale_shape(n: int, k: int, *, element_b: MatmulElementB = MatmulElementB.NVFP4) -> tuple[int, ...]:
    return (k // SCALE_TILE_K, n, SCALE_TILE_K // fp4_scale_group_size(element_b))


def _validate_source_weight(qweight: torch.Tensor) -> tuple[int, int]:
    if qweight.ndim != 2 or qweight.dtype != torch.uint8 or not qweight.is_contiguous():
        raise ValueError("qweight must be a contiguous 2-D torch.uint8 tensor.")
    n, packed_k = qweight.shape
    k = packed_k * 2
    if n % SCALE_TILE_N or k % WEIGHT_TILE_K:
        raise ValueError(f"N must be divisible by {SCALE_TILE_N}; K by {WEIGHT_TILE_K}.")
    return n, k


def _validate_source_scales(scales: torch.Tensor, element_b: MatmulElementB) -> tuple[int, int]:
    scale_group_size = fp4_scale_group_size(element_b)
    dtype = torch.uint8 if element_b == MatmulElementB.MXFP4 else torch.float8_e4m3fn
    if scales.ndim != 2 or scales.dtype != dtype or not scales.is_contiguous():
        raise ValueError(f"scales must be a contiguous 2-D {dtype} tensor.")
    n, scale_k = scales.shape
    k = scale_k * scale_group_size
    if n % SCALE_TILE_N or k % WEIGHT_TILE_K:
        raise ValueError(f"N must be divisible by {SCALE_TILE_N}; K by {WEIGHT_TILE_K}.")
    return n, k


def _petit_format(words: torch.Tensor) -> torch.Tensor:
    encoded = torch.zeros_like(words)
    for index, (sign_offset, value_offset) in enumerate(zip(_SIGN_OFFSETS, _VALUE_OFFSETS)):
        code = (words >> (index * 4)) & 0xF
        magnitude = code & 0x7
        sign = torch.where(magnitude == 0, 0, code >> 3)
        if index >= 4:
            magnitude = ((magnitude & 1) << 2) | (magnitude & 2) | ((magnitude & 4) >> 2)
        encoded |= (sign << sign_offset) | (magnitude << value_offset)
    return encoded


def repack_fp4(qweight: torch.Tensor) -> torch.Tensor:
    """Pack row-major E2M1 nibbles into AveLang's tiled FP4 weight format."""
    n, k = _validate_source_weight(qweight)
    words = _petit_format(qweight.view(torch.int32))
    blocks = words.reshape(n // WEIGHT_TILE_N, 2, 16, k // WEIGHT_TILE_K, 4, 4, 2)
    return blocks.permute(3, 4, 0, 5, 2, 1, 6).contiguous().reshape(packed_weight_shape(n, k))


def process_fp4_scales(scales: torch.Tensor, *, element_b: MatmulElementB = MatmulElementB.NVFP4) -> torch.Tensor:
    """Pack NVFP4 E4M3 or MXFP4 E8M0 (uint8 exponent) block scales."""
    n, k = _validate_source_scales(scales, element_b)
    if element_b == MatmulElementB.MXFP4:
        blocks = scales.reshape(n // 32, 2, 8, 2, k // SCALE_TILE_K, 2)
        return blocks.permute(4, 0, 5, 2, 3, 1).contiguous().reshape(packed_scale_shape(n, k, element_b=element_b))
    half_bits = (scales.to(torch.float16) * 128).view(torch.int16)
    encoded = ((half_bits.to(torch.int32) >> 7) & 0xFF).to(torch.uint8)
    blocks = encoded.reshape(n // SCALE_TILE_N, 2, 2, 8, 2, k // SCALE_TILE_K, 4)
    return blocks.permute(5, 0, 1, 6, 3, 4, 2).contiguous().reshape(packed_scale_shape(n, k))


def decode_fp4_reference(
    qweight: torch.Tensor, scales: torch.Tensor, *, element_b: MatmulElementB = MatmulElementB.NVFP4
) -> torch.Tensor:
    """Decode source FP4 weights and scales to float32 for reference checks."""
    n, k = _validate_source_weight(qweight)
    scale_n, scale_k = _validate_source_scales(scales, element_b)
    if (scale_n, scale_k) != (n, k):
        raise ValueError("qweight and scales describe different matrix shapes.")
    if qweight.device != scales.device:
        raise ValueError("qweight and scales must be on the same device.")

    values = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6), dtype=torch.float32, device=qweight.device
    )
    unpacked = torch.empty((n, k), dtype=torch.float32, device=qweight.device)
    unpacked[:, 0::2] = values[(qweight & 0xF).long()]
    unpacked[:, 1::2] = values[(qweight >> 4).long()]
    if element_b == MatmulElementB.MXFP4:
        scale_values = torch.exp2(scales.float() - 127)
        scale_values = torch.where(scales == 255, float("nan"), scale_values)
    else:
        scale_values = scales.float()
    return unpacked * scale_values.repeat_interleave(fp4_scale_group_size(element_b), dim=1)

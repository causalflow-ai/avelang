"""Compile-time configuration shared by FP4 GEMM components."""

from dataclasses import dataclass
from functools import cache

from .solution import (
    LAYOUT_K,
    LAYOUT_N,
    TILE,
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    SolutionId,
    fp4_scale_group_size,
)

WARP_SIZE = 64
PIPELINE_STAGES = 2
MAX_SHM_BYTES = 64 * 1024
UINT4_BYTES = 16
U32_BYTES = 4
U16_BYTES = 2
FP4_BITS = 4
PACK_FACTOR = 32 // FP4_BITS
QUANT_VEC_SIZE = UINT4_BYTES // U32_BYTES
SCALE_VEC_SIZE = UINT4_BYTES
LAYOUT_ELEMENTS_K = 2
LAYOUT_ELEMENTS_N = 2
PACKED_VALUES_PER_VECTOR = PACK_FACTOR * QUANT_VEC_SIZE
ACCUM_VALUES = 4


@dataclass(frozen=True, slots=True)
class FP4GemmConfig:
    mfma_type: MatmulMfmaType
    element_b: MatmulElementB
    arch: str
    global_scale_factor: float
    high_precision: bool
    element_a_bytes: int
    vec_size: int
    read_batch_a: int
    group_m: int
    group_n: int
    group_k: int
    scale_group_size: int
    warp_partition_m: int
    warp_partition_n: int
    warp_partition_k: int
    num_warps: int
    threads: int
    warp_tiles_m: int
    warp_tiles_n: int
    warp_atom_k: int
    warp_atom_n: int
    weight_bias: float

    @classmethod
    @cache
    def from_solution(cls, solution: SolutionId, arch: str = "gfx942") -> "FP4GemmConfig":
        dequant_bias = 32768.0 if solution.mfma_type == MatmulMfmaType.BF16 and arch == "gfx942" else 16384.0
        high_precision = bool(solution.features & MatmulFeatures.HIGH_PRECISION)
        if high_precision:
            global_scale_factor = 1.0
        elif solution.element_b == MatmulElementB.MXFP4:
            global_scale_factor = 16384.0 if arch in ("gfx950", "gfx1200", "gfx1201") else 32768.0
        else:
            global_scale_factor = dequant_bias / 128.0
        element_a_bytes = 2 if solution.mfma_type in (MatmulMfmaType.FP16, MatmulMfmaType.BF16) else 1
        group_m, group_n, group_k = solution.group_m, solution.group_n, solution.group_k
        threads = solution.num_warps * WARP_SIZE
        vec_size = UINT4_BYTES // element_a_bytes
        scale_group_size = fp4_scale_group_size(solution.element_b)
        warp_tiles_m = solution.tile_m // solution.warp_partition_m
        warp_tiles_n = solution.tile_n // solution.warp_partition_n

        return cls(
            mfma_type=solution.mfma_type,
            element_b=solution.element_b,
            arch=arch,
            global_scale_factor=global_scale_factor,
            high_precision=high_precision,
            element_a_bytes=element_a_bytes,
            vec_size=vec_size,
            read_batch_a=2 * PACK_FACTOR * element_a_bytes // UINT4_BYTES,
            group_m=group_m,
            group_n=group_n,
            group_k=group_k,
            scale_group_size=scale_group_size,
            warp_partition_m=solution.warp_partition_m,
            warp_partition_n=solution.warp_partition_n,
            warp_partition_k=solution.warp_partition_k,
            num_warps=solution.num_warps,
            threads=threads,
            warp_tiles_m=warp_tiles_m,
            warp_tiles_n=warp_tiles_n,
            warp_atom_k=solution.group_k // LAYOUT_K // solution.warp_partition_k,
            warp_atom_n=solution.group_n // LAYOUT_N // solution.warp_partition_n,
            weight_bias=dequant_bias / 128.0 if high_precision else 1.0,
        )

    @property
    def accumulator_shape(self) -> tuple[int, int, int]:
        return (self.warp_tiles_m, self.warp_tiles_n, ACCUM_VALUES)

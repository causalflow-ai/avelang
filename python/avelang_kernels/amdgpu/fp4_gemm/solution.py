"""Solution descriptions for AveLang's AMDGPU FP4 GEMM."""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

TILE = 16
LAYOUT_K = 64
LAYOUT_N = 32


class MatmulFeatures(IntFlag):
    GLOBAL = 0
    GRID = 1
    HIGH_PRECISION = 2


class MatmulElementB(IntEnum):
    INT4 = 0
    NVFP4 = 1
    MXFP4 = 2


def fp4_scale_group_size(element_b: MatmulElementB) -> int:
    if element_b == MatmulElementB.NVFP4:
        return 16
    if element_b == MatmulElementB.MXFP4:
        return 32
    raise ValueError(f"unsupported FP4 format: {element_b}")


class MatmulMfmaType(IntEnum):
    FP16 = 0
    BF16 = 1
    FP8 = 2


class MatmulWarpPartition(IntEnum):
    NK = 0
    COOPERATIVE = 1


@dataclass(frozen=True, slots=True)
class SolutionId:
    tile_m: int
    tile_n: int
    tile_k: int  # Encoded as group_k / (4 * TILE), following Petit.
    features: MatmulFeatures
    element_b: MatmulElementB
    mfma_type: MatmulMfmaType
    warp_partition_m: int
    warp_partition_n: int
    warp_partition_k: int
    warp_partition: MatmulWarpPartition = MatmulWarpPartition.NK

    def __post_init__(self):
        tiles = (self.tile_m, self.tile_n, self.tile_k * 4)
        partitions = (self.warp_partition_m, self.warp_partition_n, self.warp_partition_k)
        if any(value <= 0 or value > 0xFF for value in tiles):
            raise ValueError(f"tile dimensions must be in [1, 255], got {tiles}")
        if any(value <= 0 or value > 0xF for value in partitions):
            raise ValueError(f"warp partitions must be in [1, 15], got {partitions}")
        if self.group_n % (LAYOUT_N * self.warp_partition_n):
            raise ValueError("N tile is incompatible with the warp partition")
        if self.group_k % (LAYOUT_K * self.warp_partition_k):
            raise ValueError("K tile is incompatible with the warp partition")
        if self.tile_m % self.warp_partition_m:
            raise ValueError("M tile is incompatible with the warp partition")

    @classmethod
    def multi_stage(
        cls,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        warp_partition_m: int,
        warp_partition_n: int,
        warp_partition_k: int,
        *,
        features: MatmulFeatures,
        element_b: MatmulElementB,
        mfma_type: MatmulMfmaType,
        warp_partition: MatmulWarpPartition = MatmulWarpPartition.NK,
    ) -> "SolutionId":
        if tile_k % 4:
            raise ValueError("tile_k must be divisible by 4")
        return cls(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k // 4,
            features=features,
            element_b=element_b,
            mfma_type=mfma_type,
            warp_partition_m=warp_partition_m,
            warp_partition_n=warp_partition_n,
            warp_partition_k=warp_partition_k,
            warp_partition=warp_partition,
        )

    @classmethod
    def from_int(cls, value: int) -> "SolutionId":
        if value < 0 or value >> 52:
            raise ValueError(f"invalid AMDGPU FP4 solution id: {value:#x}")
        return cls(
            tile_m=(value >> 0) & 0xFF,
            tile_n=(value >> 8) & 0xFF,
            tile_k=(value >> 16) & 0xFF,
            features=MatmulFeatures((value >> 24) & 0xF),
            element_b=MatmulElementB((value >> 28) & 0xF),
            mfma_type=MatmulMfmaType((value >> 32) & 0xF),
            warp_partition_m=(value >> 36) & 0xF,
            warp_partition_n=(value >> 40) & 0xF,
            warp_partition_k=(value >> 44) & 0xF,
            warp_partition=MatmulWarpPartition((value >> 48) & 0xF),
        )

    def __int__(self) -> int:
        return (
            self.tile_m
            | (self.tile_n << 8)
            | (self.tile_k << 16)
            | (int(self.features) << 24)
            | (int(self.element_b) << 28)
            | (int(self.mfma_type) << 32)
            | (self.warp_partition_m << 36)
            | (self.warp_partition_n << 40)
            | (self.warp_partition_k << 44)
            | (int(self.warp_partition) << 48)
        )

    @property
    def group_m(self) -> int:
        return self.tile_m * TILE

    @property
    def group_n(self) -> int:
        return self.tile_n * TILE

    @property
    def group_k(self) -> int:
        return self.tile_k * 4 * TILE

    @property
    def num_warps(self) -> int:
        return self.warp_partition_m * self.warp_partition_n * self.warp_partition_k


TILE_SHAPES = (
    (4, 2, 8, 4, 1, 1),
    (1, 2, 16, 1, 1, 4),
    (2, 2, 16, 2, 1, 2),
    (2, 2, 16, 1, 1, 4),
    (4, 2, 16, 2, 1, 2),
    (1, 2, 32, 1, 1, 4),
    (2, 2, 32, 2, 1, 2),
    (1, 4, 8, 1, 2, 2),
    (2, 4, 8, 2, 2, 1),
    (4, 4, 8, 2, 2, 1),
    (6, 4, 8, 2, 2, 1),
    (8, 4, 8, 2, 2, 1),
    (10, 4, 8, 2, 2, 1),
    (1, 4, 16, 1, 2, 2),
    (2, 4, 16, 2, 2, 1),
    (4, 4, 16, 2, 2, 1),
    (1, 4, 32, 1, 2, 2),
    (2, 4, 32, 2, 2, 1),
    (4, 6, 8, 2, 1, 2),
    (6, 6, 8, 2, 1, 2),
    (1, 8, 4, 1, 4, 1),
    (8, 8, 4, 2, 2, 1),
    (12, 8, 4, 2, 2, 1),
    (14, 8, 4, 2, 2, 1),
    (16, 8, 4, 2, 2, 1),
    (2, 8, 8, 2, 2, 1),
    (4, 8, 8, 2, 2, 1),
    (5, 8, 8, 1, 2, 2),
    (10, 8, 4, 2, 2, 1),
    (8, 12, 4, 2, 2, 1),
    (10, 12, 4, 2, 2, 1),
    (12, 12, 4, 2, 2, 1),
    (14, 12, 4, 2, 2, 1),
    (16, 12, 4, 2, 2, 1),
    (8, 16, 4, 2, 2, 1),
    (10, 16, 4, 2, 2, 1),
    (12, 16, 4, 2, 2, 1),
    (14, 16, 4, 2, 2, 1),
    (16, 16, 4, 2, 2, 1),
)


def generate_solution_list() -> tuple[SolutionId, ...]:
    type_pairs = (
        (MatmulElementB.NVFP4, MatmulMfmaType.FP16),
        (MatmulElementB.NVFP4, MatmulMfmaType.BF16),
        (MatmulElementB.MXFP4, MatmulMfmaType.BF16),
    )
    return tuple(
        SolutionId.multi_stage(*shape, features=features, element_b=element_b, mfma_type=mfma_type)
        for features in (MatmulFeatures.GRID, MatmulFeatures.GRID | MatmulFeatures.HIGH_PRECISION)
        for element_b, mfma_type in type_pairs
        for shape in TILE_SHAPES
    )


SOLUTION_LIST = generate_solution_list()


def available_solutions(
    n: int, k: int, *, element_b: MatmulElementB, mfma_type: MatmulMfmaType, high_precision: bool = False
) -> tuple[SolutionId, ...]:
    features = MatmulFeatures.GRID
    if high_precision:
        features |= MatmulFeatures.HIGH_PRECISION
    return tuple(
        solution
        for solution in SOLUTION_LIST
        if solution.features == features
        and solution.element_b == element_b
        and solution.mfma_type == mfma_type
        and n % solution.group_n == 0
        and k % solution.group_k == 0
    )


def choose_default_solution(
    m: int,
    n: int,
    k: int,
    *,
    element_b: MatmulElementB = MatmulElementB.NVFP4,
    mfma_type: MatmulMfmaType = MatmulMfmaType.BF16,
    high_precision: bool = False,
) -> SolutionId:
    candidates = available_solutions(n, k, element_b=element_b, mfma_type=mfma_type, high_precision=high_precision)
    if not candidates:
        raise ValueError(f"no FP4 GEMM solution for M={m}, N={n}, K={k}")

    def is_better(candidate: SolutionId, current: SolutionId) -> bool:
        if m <= 64:
            if candidate.tile_m != current.tile_m:
                candidate_m = candidate.group_m
                current_m = current.group_m
                candidate_delta = abs(m % candidate_m - candidate_m // 2)
                current_delta = abs(m % current_m - current_m // 2)
                return candidate_delta < current_delta
            if candidate.warp_partition_k != current.warp_partition_k:
                return candidate.warp_partition_k > current.warp_partition_k
            if candidate.tile_n != current.tile_n:
                return candidate.tile_n < current.tile_n

        candidate_sum = candidate.tile_m + candidate.tile_n
        current_sum = current.tile_m + current.tile_n
        if candidate_sum != current_sum:
            return candidate_sum > current_sum
        if candidate.tile_m != current.tile_m:
            return candidate.tile_m > current.tile_m
        if candidate.tile_k != current.tile_k:
            return candidate.tile_k > current.tile_k
        return int(candidate) > int(current)

    best = candidates[0]
    for candidate in candidates[1:]:
        if is_better(candidate, best):
            best = candidate
    return best

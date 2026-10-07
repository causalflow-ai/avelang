"""Public PyTorch API and kernel dispatch for AMDGPU FP4 GEMM."""

import torch

from .config import FP4GemmConfig
from .kernel import make_kernel
from .solution import SOLUTION_LIST, MatmulElementB, MatmulFeatures, MatmulMfmaType, SolutionId, choose_default_solution
from .utils import SCALE_TILE_N, WEIGHT_TILE_K, WEIGHT_TILE_N, _ceildiv, packed_scale_shape, packed_weight_shape

SUPPORTED_SOLUTIONS = frozenset(SOLUTION_LIST)


def _should_require_high_precision(props) -> bool:
    # For gfx90a, we need to use high precision as the MFMA instructions
    # flush the inputs and output denormals.
    return props.major * 10 + props.minor <= 90


def resolve_solution(
    m: int,
    n: int,
    k: int,
    solution_id: SolutionId | int | None = None,
    *,
    element_b: MatmulElementB = MatmulElementB.NVFP4,
    mfma_type: MatmulMfmaType = MatmulMfmaType.BF16,
    high_precision: bool | None = None,
) -> SolutionId:
    if solution_id is None or solution_id == -1:
        solution = choose_default_solution(
            m, n, k, element_b=element_b, mfma_type=mfma_type, high_precision=bool(high_precision)
        )
    elif isinstance(solution_id, SolutionId):
        solution = solution_id
    else:
        solution = SolutionId.from_int(solution_id)
    if solution not in SUPPORTED_SOLUTIONS:
        raise ValueError(f"unsupported AveLang FP4 solution {int(solution):#x}")
    if solution.element_b != element_b or solution.mfma_type != mfma_type:
        raise ValueError("solution_id does not match the input dtype and FP4 format.")
    if high_precision is not None and bool(solution.features & MatmulFeatures.HIGH_PRECISION) != high_precision:
        raise ValueError("solution_id does not match high_precision.")
    if n % solution.group_n or k % solution.group_k:
        raise ValueError(f"solution {int(solution):#x} is incompatible with N={n}, K={k}")
    return solution


def fp4_gemm_transposed_b(
    A: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_scales: torch.Tensor,
    global_scale: torch.Tensor | float,
    out: torch.Tensor | None = None,
    solution_id: SolutionId | int | None = None,
    *,
    element_b: MatmulElementB = MatmulElementB.NVFP4,
    high_precision: bool | None = None,
) -> torch.Tensor:
    """Compute ``global_scale * (A @ B.T)`` with FP16/BF16 A and FP4 B.

    B's weights and scales must be packed by ``repack_fp4`` and
    ``process_fp4_scales``, using the same ``element_b``. Output has A's dtype.

    ``high_precision`` restores weights before MFMA; its default follows
    the explicit solution or Petit's device policy.
    """
    if (
        A.ndim != 2
        or A.dtype not in (torch.float16, torch.bfloat16)
        or not A.is_contiguous()
        or A.device.type != "cuda"
    ):
        raise ValueError("A must be a contiguous 2-D CUDA FP16/BF16 tensor.")
    if packed_weight.ndim != 5:
        raise ValueError("packed_weight must have 5 dimensions.")
    m, k = A.shape
    n = packed_weight.shape[2] * WEIGHT_TILE_N
    if min(m, n, k) <= 0 or n % SCALE_TILE_N or k % WEIGHT_TILE_K:
        raise ValueError(f"M, N, K must be positive; N divisible by {SCALE_TILE_N}, K by {WEIGHT_TILE_K}.")

    props = torch.cuda.get_device_properties(A.device)
    if high_precision is None and (solution_id is None or solution_id == -1):
        high_precision = _should_require_high_precision(props)
    mfma_type = MatmulMfmaType.FP16 if A.dtype == torch.float16 else MatmulMfmaType.BF16
    solution = resolve_solution(
        m, n, k, solution_id, element_b=element_b, mfma_type=mfma_type, high_precision=high_precision
    )

    if (
        packed_weight.dtype != torch.int32
        or packed_weight.shape != packed_weight_shape(n, k)
        or not packed_weight.is_contiguous()
    ):
        raise ValueError("packed_weight must be contiguous int32 in the repack_fp4 layout.")
    if (
        packed_scales.dtype != torch.uint8
        or packed_scales.shape != packed_scale_shape(n, k, element_b=element_b)
        or not packed_scales.is_contiguous()
    ):
        raise ValueError("packed_scales must be contiguous uint8 in the process_fp4_scales layout.")
    if not (A.device == packed_weight.device == packed_scales.device):
        raise ValueError("All inputs must be on the same device.")

    if isinstance(global_scale, torch.Tensor):
        if global_scale.dtype != torch.float32 or global_scale.numel() != 1 or global_scale.device != A.device:
            raise ValueError("global_scale must contain one float32 value on A's device.")
    else:
        global_scale = torch.tensor([float(global_scale)], dtype=torch.float32, device=A.device)

    if out is None:
        out = torch.empty((m, n), dtype=A.dtype, device=A.device)
    elif out.shape != (m, n) or out.dtype != A.dtype or out.device != A.device or not out.is_contiguous():
        raise ValueError(f"out must be contiguous {A.dtype} with shape {(m, n)} on {A.device}.")

    arch = props.gcnArchName.split(":")[0]
    config = FP4GemmConfig.from_solution(solution, arch)
    fp4_gemm_kernel = make_kernel(config)
    grid_m = _ceildiv(m, config.group_m)
    grid_n = n // config.group_n
    with torch.cuda.device(A.device):
        fp4_gemm_kernel[lambda: ((grid_m, grid_n, 1), (config.threads, 1, 1))](
            A, packed_weight, packed_scales, global_scale.reshape(1), out, m, n, k, num_warps=config.num_warps
        )
    return out

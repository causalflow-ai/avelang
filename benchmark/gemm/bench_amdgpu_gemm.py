#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

from avelang_kernels import amdgpu_gemm


def _ensure_rocm_available(label):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError(f"HIP is not available; {label} benchmark requires ROCm.")


def _validate_shape(m, n, k):
    if m % amdgpu_gemm.GROUP_M != 0:
        raise ValueError(f"M must be a multiple of {amdgpu_gemm.GROUP_M}, got {m}.")
    if n % amdgpu_gemm.GROUP_N != 0:
        raise ValueError(f"N must be a multiple of {amdgpu_gemm.GROUP_N}, got {n}.")
    if k % amdgpu_gemm.GROUP_K != 0:
        raise ValueError(f"K must be a multiple of {amdgpu_gemm.GROUP_K}, got {k}.")


def run_gemm_pipeline_benchmark(m, n, k, warmup, repeat, iters, validate):
    _ensure_rocm_available("gemm_pipeline_transposed_b")
    _validate_shape(m, n, k)

    A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

    # Warmup to JIT-compile and stabilize clocks.
    for _ in range(warmup):
        amdgpu_gemm.gemm_pipeline_transposed_b(A, B, out=C)
    torch.cuda.synchronize()

    # Capture the kernel launch in a CUDAGraph for low-overhead replays.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        amdgpu_gemm.gemm_pipeline_transposed_b(A, B, out=C)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeat):
        for _ in range(iters):
            graph.replay()
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    elapsed = (elapsed_ms * 1.0e-3) / (repeat * iters)

    flops = 2.0 * m * n * k
    tflops = flops / elapsed / 1.0e12

    bytes_moved = (m * k + k * n + m * n) * 2
    bandwidth = bytes_moved / elapsed / 1.0e9

    print(f"M={m} N={n} K={k} time_ms={elapsed * 1.0e3:.4f} tflops={tflops:.3f} bandwidth_gbs={bandwidth:.3f}")

    if validate:
        expected = (A.float() @ B.float().T).to(dtype=torch.bfloat16, device="cpu")
        actual = C.to("cpu")
        max_abs = torch.max(torch.abs(actual - expected)).item()
        if not torch.allclose(actual, expected, rtol=1e-1, atol=1e-1):
            raise AssertionError(f"Validation failed (max abs diff {max_abs}).")
        print(f"validation=max_abs_diff:{max_abs:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="AMDGPU GEMM pipeline benchmark (M,N multiples of 128; K multiple of 64)"
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100, help="Graph replays per timing repeat")
    parser.add_argument("--validate", action="store_true", default=False)
    args = parser.parse_args()

    run_gemm_pipeline_benchmark(args.m, args.n, args.k, args.warmup, args.repeat, args.iters, args.validate)


if __name__ == "__main__":
    main()

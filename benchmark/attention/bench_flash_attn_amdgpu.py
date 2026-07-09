#!/usr/bin/env python3
import argparse
import math

import torch
import torch.nn.functional as F

from avelang_kernels import flash_attn


def _ensure_rocm_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError("HIP is not available; flash attention benchmark requires ROCm.")


def _parse_seq_lens(seq_lens: str | None, batch_size: int, seq_len: int) -> list[int]:
    if seq_lens is None:
        return [seq_len] * batch_size
    values = [int(item.strip()) for item in seq_lens.split(",") if item.strip()]
    if not values:
        raise ValueError("--seq-lens must contain at least one sequence length")
    if any(value < 0 for value in values):
        raise ValueError("--seq-lens must not contain negative sequence lengths")
    return values


def _validate_config(q_heads: int, kv_heads: int, head_dim: int, seq_lens: list[int]) -> None:
    if head_dim != flash_attn.HEAD_DIM:
        raise ValueError(f"head_dim must be {flash_attn.HEAD_DIM}, got {head_dim}.")
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("q_heads and kv_heads must be positive.")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads, got {q_heads} and {kv_heads}.")
    if max(seq_lens, default=0) <= 0:
        raise ValueError("at least one sequence length must be positive.")


def _make_inputs(
    seq_lens: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    total_tokens = sum(seq_lens)
    q = torch.randn((total_tokens, q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    seq_ptr_cpu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device="cpu")
    seq_ptr_device = seq_ptr_cpu.to(device="cuda")
    return q, k, v, seq_ptr_cpu, seq_ptr_device


def _reference_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr_cpu: torch.Tensor,
) -> torch.Tensor:
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    groups = q_heads // kv_heads
    out = torch.empty_like(q, dtype=torch.float32)
    for seq_idx in range(seq_ptr_cpu.numel() - 1):
        seq_begin = int(seq_ptr_cpu[seq_idx].item())
        seq_end = int(seq_ptr_cpu[seq_idx + 1].item())
        if seq_begin == seq_end:
            continue
        q_seq = q[seq_begin:seq_end].permute(1, 0, 2).unsqueeze(0)
        k_seq = k[seq_begin:seq_end].permute(1, 0, 2).unsqueeze(0).repeat_interleave(groups, dim=1)
        v_seq = v[seq_begin:seq_end].permute(1, 0, 2).unsqueeze(0).repeat_interleave(groups, dim=1)
        ref = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True)
        out[seq_begin:seq_end] = ref.squeeze(0).permute(1, 0, 2).to(torch.float32)
    return out


def _launch_avelang_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr_device: torch.Tensor,
    out: torch.Tensor,
    seq_lens: list[int],
) -> None:
    row_tiles = math.ceil(max(seq_lens) / flash_attn.BLOCK_ROWS)
    physical_tiles = math.ceil(row_tiles / 2)
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    flash_attn._flash_attn_packed_kernel[lambda: ((physical_tiles, q_heads, len(seq_lens)), (flash_attn.THREADS, 1, 1))](
        q,
        k,
        v,
        seq_ptr_device,
        out,
        q.shape[0],
        len(seq_lens),
        num_warps=flash_attn.NUM_WARPS,
    )


def _time_avelang_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_ptr_device: torch.Tensor,
    out: torch.Tensor,
    seq_lens: list[int],
    warmup: int,
    repeat: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _launch_avelang_flash_attn(q, k, v, seq_ptr_device, out, seq_lens)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        _launch_avelang_flash_attn(q, k, v, seq_ptr_device, out, seq_lens)
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
    return (elapsed_ms * 1.0e-3) / (repeat * iters)


def _causal_attention_flops(seq_lens: list[int], q_heads: int, head_dim: int) -> float:
    causal_pairs = sum(seq_len * (seq_len + 1) // 2 for seq_len in seq_lens)
    return 4.0 * causal_pairs * q_heads * head_dim


def _logical_bytes(seq_lens: list[int], q_heads: int, kv_heads: int, head_dim: int) -> int:
    total_tokens = sum(seq_lens)
    elements = total_tokens * (q_heads + 2 * kv_heads + q_heads) * head_dim
    return elements * 2


def run_benchmark(args: argparse.Namespace) -> None:
    _ensure_rocm_available()
    seq_lens = _parse_seq_lens(args.seq_lens, args.batch_size, args.seq_len)
    _validate_config(args.q_heads, args.kv_heads, args.head_dim, seq_lens)

    q, k, v, seq_ptr_cpu, seq_ptr_device = _make_inputs(
        seq_lens,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.seed,
    )
    out = torch.empty_like(q)

    elapsed = _time_avelang_graph(
        q,
        k,
        v,
        seq_ptr_device,
        out,
        seq_lens,
        args.warmup,
        args.repeat,
        args.iters,
    )

    flops = _causal_attention_flops(seq_lens, args.q_heads, args.head_dim)
    tflops = flops / elapsed / 1.0e12
    bandwidth = _logical_bytes(seq_lens, args.q_heads, args.kv_heads, args.head_dim) / elapsed / 1.0e9
    tokens = sum(seq_lens)

    print(
        "kernel=avelang_flash_attn "
        f"dtype=bf16 tokens={tokens} batch={len(seq_lens)} max_seq_len={max(seq_lens)} "
        f"q_heads={args.q_heads} kv_heads={args.kv_heads} head_dim={args.head_dim} "
        f"time_ms={elapsed * 1.0e3:.4f} tflops={tflops:.3f} logical_bandwidth_gbs={bandwidth:.3f}"
    )

    if args.validate:
        expected = _reference_sdpa(q, k, v, seq_ptr_cpu)
        actual = out.to(torch.float32)
        max_abs = torch.max(torch.abs(actual - expected)).item()
        if not torch.allclose(actual.cpu(), expected.cpu(), rtol=args.rtol, atol=args.atol):
            raise AssertionError(f"Validation failed (max abs diff {max_abs}).")
        print(f"validation=max_abs_diff:{max_abs:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMDGPU BF16 flash attention benchmark for avelang packed attention"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=None,
        help="Comma-separated sequence lengths. Overrides --batch-size and --seq-len.",
    )
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100, help="Graph replays per timing repeat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()

---
layout: layouts/base.njk
title: "Tutorial 3: Intrinsics"
description: Use AMDGPU MFMA and raw-buffer intrinsics for dynamic-shape GEMM kernels.
---

# Tutorial 3: Intrinsics

This tutorial demonstrates how to use AMDGPU hardware intrinsics to improve GEMM kernels.

## Using Matrix Fused Multiply-Add

Modern GPUs include specialized matrix hardware, such as tensor cores on NVIDIA GPUs and matrix cores on AMD GPUs. On AMD GPUs, MFMA instructions let a wave compute a matrix multiply cooperatively. Each lane contributes operand fragments and receives accumulator fragments in the hardware layout described by the [AMD matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).

This example still computes a full `128 x 128` matrix multiplication, `C = A @ B^T`. Each wave computes one `32 x 32` output tile, and the grid covers the full matrix with `4 x 4` wave tiles. The kernel uses shared memory to move between logical matrix layouts and the lane layouts expected by `mfma_32x32x8_bf16_f32`.

The calculator reports that `v_mfma_f32_32x32x8_bf16` consumes two registers for `A` and two for `B`: each lane contributes four BF16 values, lanes `0..31` provide one half of the `K = 8` slice, and lanes `32..63` provide the other half. To keep the memory path contiguous, this kernel stages `K = 16` at a time. Each lane loads and stores one 16-byte vector for `A` and one for `B`. The shared-memory read returns that same 16-byte vector, which is split into two 8-byte fragments for two MFMA instructions.

```python
import avelang
import avelang.language as al

@avelang.jit
def gemm_mfma_128_bf16_mfma(
    A: al.Tensor((128, 128), al.bf16),
    B: al.Tensor((128, 128), al.bf16),
    C: al.Tensor((128, 128), al.f32),
):
    TILE_M = 32
    TILE_N = 32
    TILE_K = 16
    K_TILES = 8

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * TILE_M
    block_n = al.block_id(0) * TILE_N

    A_vec = al.view(A, al.Tensor((128, 16, 4), al.i32))
    B_vec = al.view(B, al.Tensor((128, 16, 4), al.i32))

    a_smem = al.make_shared((TILE_M * (TILE_K >> 3), TILE_K >> 2), al.i32)
    b_smem = al.make_shared((TILE_N * (TILE_K >> 3), TILE_K >> 2), al.i32)
    C_vec = al.view(C, al.Tensor((128, 32, 4), al.i32))

    c_smem = al.make_shared((32, 32), al.f32)
    c_smem_vec = al.view(c_smem, al.Tensor((32, 8, 4), al.i32))
    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(K_TILES):
        k_vec = kt * 2 + lane_group

        a_smem[lane] = A_vec[block_m + lane_col, k_vec]
        b_smem[lane] = B_vec[block_n + lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_row = lane >> 1
    store_vec_base = (lane & 1) * 4

    for v in al.range(4):
        C_vec[block_m + store_row, (block_n >> 2) + store_vec_base + v] = (
            c_smem_vec[store_row, store_vec_base + v]
        )
```

Launch one wave per `32 x 32` tile:

```python
import torch

M = 128
N = 128
K = 128

A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
B = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
C = torch.empty((M, N), dtype=torch.float32, device="cuda")

gemm_mfma_128_bf16_mfma[lambda: ((4, 4, 1), (64, 1, 1))](A, B, C)
expected = A.to(torch.float32) @ B.to(torch.float32).T
torch.testing.assert_close(C.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
```

The input tiles are written contiguously as packed `i32` vectors. Each lane reads back one 16-byte vector from shared memory, splits it into two 8-byte fragments, and issues two MFMA instructions. The MFMA operands are swapped so the accumulator fragment lines up with row-major `C`; the final shared-memory step writes four contiguous `f32` values at a time.

## Runtime Shapes

So far, the MFMA kernel used static `128 x 128` tensors. For a reusable GEMM entry point, accept raw pointers plus runtime `m`, `n`, and `k`, then build typed tensor views inside the kernel.

`al.Pointer(dtype)` marks a pointer argument. `al.make_tensor(ptr, dtype, layout)` gives that pointer a tensor view, and `al.make_layout()` can use runtime dimensions.

`BLOCK_M`, `BLOCK_N`, and `BLOCK_K` describe the logical tile computed by one program instance. In this first MFMA kernel they are `32`, `32`, and `16`: one wave computes one `32 x 32` output tile and advances through K in 16-BF16 chunks. Keeping them as `al.constexpr` makes the tile shape explicit and lets larger kernels build bigger program tiles by composing multiple MFMA tiles while preserving static shared-memory shapes and loop bounds. The packed shared-memory buffers are sized from these values: `BLOCK_M` or `BLOCK_N` rows, `BLOCK_K / 8` MFMA lane groups, and `BLOCK_K / 4` packed `i32` words per vector.

For now, assume the runtime shapes are aligned to the tile and vector shape: `m` is a multiple of `BLOCK_M`, `n` is a multiple of `BLOCK_N`, and `k` is a multiple of `BLOCK_K`. That keeps every MFMA tile and 16-byte vectorized load in bounds.

```python
BF16_BYTES = 2
F32_BYTES = 4

@avelang.jit
def gemm_runtime_shape(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.f32),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    A_bf16 = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
    B_bf16 = al.make_tensor(B_ptr, al.bf16, al.make_layout((n, k), (k, 1)))
    C = al.make_tensor(C_ptr, al.f32, al.make_layout((m, n), (n, 1)))

    k_vecs = k >> 3
    packed_row_stride = k >> 1

    A_vec = al.view(
        A_bf16,
        al.i32,
        al.make_layout((m, k_vecs, 4), (packed_row_stride, 4, 1)),
    )
    B_vec = al.view(
        B_bf16,
        al.i32,
        al.make_layout((n, k_vecs, 4), (packed_row_stride, 4, 1)),
    )
    C_vec = al.view(
        C,
        al.i32,
        al.make_layout((m, n >> 2, 4), (n, 4, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    a_smem = al.make_shared((BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    c_smem_vec = al.view(
        c_smem,
        al.i32,
        al.make_layout((BLOCK_M, BLOCK_N >> 2, 4), (BLOCK_N, 4, 1)),
    )
    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(k // BLOCK_K):
        k_vec = kt * 2 + lane_group

        a_smem[lane] = A_vec[block_m + lane_col, k_vec]
        b_smem[lane] = B_vec[block_n + lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_row = lane >> 1
    store_vec_base = (lane & 1) * (BLOCK_N >> 3)

    for v in al.range(BLOCK_N >> 3):
        C_vec[block_m + store_row, (block_n >> 2) + store_vec_base + v] = (
            c_smem_vec[store_row, store_vec_base + v]
        )
```

The launch grid now comes from the runtime shape:

```python
m = 128
n = 128
k = 128

A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
C = torch.empty((m, n), dtype=torch.float32, device="cuda")

grid_x = n // 32
grid_y = m // 32

gemm_runtime_shape[lambda: ((grid_x, grid_y, 1), (64, 1, 1))](
    A, B, C, m, n, k, 32, 32, 16
)

expected = A.to(torch.float32) @ B.to(torch.float32).T
torch.testing.assert_close(C.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
```

This is the same MFMA kernel as before, but the row-major tensor layouts come from runtime dimensions. The aligned-shape assumption removes edge guards; the next section shows how raw-buffer loads handle those edge cases while keeping vectorized memory movement.

## Hardware-Guarded Buffer Loads

Dynamic shapes introduce edge tiles. With a ceil-divided grid, the final tile in `M` or `N` may ask some lanes to load rows outside `A` or `B`. AMDGPU raw-buffer operations attach a byte range to a resource descriptor; out-of-range loads return zero, and stores can use the same descriptor form.

The descriptor base must be uniform across the wave. The kernel below creates block-uniform descriptors for `A` and `B`, then creates a uniform row descriptor for each `C` row during writeback. Per-lane row, column, and K positions become byte offsets into those descriptors. It still assumes `K` and the row length of `C` are 16-byte aligned, so each vectorized access belongs to one row.

```python
@avelang.jit
def gemm_runtime_shape_guarded_loads(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.f32),
    m: al.u32,
    n: al.u32,
    k: al.u32,
    BLOCK_M: al.constexpr,
    BLOCK_N: al.constexpr,
    BLOCK_K: al.constexpr,
):
    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    block_m = al.block_id(1) * BLOCK_M
    block_n = al.block_id(0) * BLOCK_N

    A_flat = al.make_tensor(A_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    B_flat = al.make_tensor(B_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    C_flat = al.make_tensor(C_ptr, al.f32, al.make_layout((m * n,), (1,)))

    a_rows = m - block_m
    b_rows = n - block_n
    c_rows = m - block_m
    c_cols = n - block_n

    if a_rows > BLOCK_M:
        a_rows = BLOCK_M

    if b_rows > BLOCK_N:
        b_rows = BLOCK_N

    if c_rows > BLOCK_M:
        c_rows = BLOCK_M

    if c_cols > BLOCK_N:
        c_cols = BLOCK_N

    A_block = al.subview(A_flat, (block_m * k,), (a_rows * k,), (1,))
    B_block = al.subview(B_flat, (block_n * k,), (b_rows * k,), (1,))

    A_rsrc = al.amdgpu.make_rsrc(A_block, a_rows * k * BF16_BYTES)
    B_rsrc = al.amdgpu.make_rsrc(B_block, b_rows * k * BF16_BYTES)

    a_smem = al.make_shared((BLOCK_M * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    b_smem = al.make_shared((BLOCK_N * (BLOCK_K >> 3), BLOCK_K >> 2), al.i32)
    c_smem = al.make_shared((BLOCK_M, BLOCK_N), al.f32)
    c_smem_vec = al.view(
        c_smem,
        al.i32,
        al.make_layout((BLOCK_M, BLOCK_N >> 2, 4), (BLOCK_N, 4, 1)),
    )
    acc = al.full((16,), 0.0, al.f32)

    zero = al.convert(0, al.i32)

    for kt in al.range(k // BLOCK_K):
        k_base = kt * BLOCK_K + lane_group * (BLOCK_K >> 1)
        load_offset = al.convert((lane_col * k + k_base) * BF16_BYTES, al.i32)

        a_smem[lane] = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero, load_offset, 0)
        b_smem[lane] = al.amdgpu.raw_buffer_load_x4(B_rsrc, zero, load_offset, 0)

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))

        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        row_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c_smem[lane_col, row_offset] = acc[r]

    al.syncthreads()

    store_vec = lane
    store_offset = al.convert(store_vec * 4 * F32_BYTES, al.i32)

    for row in al.range(BLOCK_M):
        c_row_range = al.convert(0, al.u32)

        if row < c_rows:
            c_row_range = c_cols * F32_BYTES

        C_row = al.subview(C_flat, ((block_m + row) * n + block_n,), (c_cols,), (1,))
        C_rsrc = al.amdgpu.make_rsrc(C_row, c_row_range)

        if store_vec < (BLOCK_N >> 2):
            al.amdgpu.raw_buffer_store_x4(
                c_smem_vec[row, store_vec],
                C_rsrc,
                zero,
                store_offset,
                0,
            )
```

The input descriptors start at the first `A` and `B` rows owned by the block. Their ranges cover only the valid rows in the edge tile, so lanes assigned past `m` or `n` read zeros. For `C`, the writeback loop creates a uniform row descriptor; its range removes the edge-row and edge-column guards for `raw_buffer_store_x4`. The remaining lane predicate only selects the eight lanes that own vector stores for a row.

Launch with ceil-divided output tiles:

```python
m = 117
n = 121
k = 128

A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
B = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
C = torch.empty((m, n), dtype=torch.float32, device="cuda")

grid_x = (n + 31) // 32
grid_y = (m + 31) // 32

gemm_runtime_shape_guarded_loads[lambda: ((grid_x, grid_y, 1), (64, 1, 1))](
    A, B, C, m, n, k, 32, 32, 16
)

expected = A.to(torch.float32) @ B.to(torch.float32).T
torch.testing.assert_close(C.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
```


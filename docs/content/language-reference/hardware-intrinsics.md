---
layout: layouts/base.njk
title: Hardware Intrinsics
description: Backend namespaces, matrix instructions, raw-buffer operations, and backend options.
---

# Hardware Intrinsics

Ave exposes backend-specific hardware intrinsics as submodules of `avelang.language`. Programs usually import the language package as `al`, then call intrinsics through `al.amdgpu` or `al.nvvm`.

```python
import avelang.language as al
```

These functions are low-level operations. They assume the caller understands the target instruction layout, lane ownership, memory alignment, and synchronization requirements. Prefer ordinary Ave tensor code until a kernel needs a specific hardware instruction or memory operation.

## AMDGPU

AMDGPU intrinsics live under `al.amdgpu`. The main families are MFMA matrix instructions and raw-buffer memory operations.

## MFMA

MFMA instructions let one AMD wave cooperatively compute a matrix multiply. Each lane passes two packed `u32` dwords for `A` and `B`, plus an accumulator fragment `C`, and receives its accumulator fragment in the hardware layout for that instruction.

```python
acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)
```

The currently registered MFMA wrappers are:

```text
mfma_16x16x16_f16_f32
mfma_16x16x16_bf16_f32
mfma_f32_16x16x16_bf16
mfma_32x32x8_bf16_f32
mfma_f32_32x32x8_bf16
mfma_16x16x32_fp8_fp8
mfma_f32_16x16x32_fp8_fp8
```

The name encodes the matrix shape and logical operand types. For example, `mfma_32x32x8_bf16_f32` consumes two packed `u32` dwords for each BF16 input fragment in a `32 x 32 x 8` operation and accumulates into F32. Each lane's dwords contain four BF16 values for `A` and four for `B`, and hold sixteen F32 accumulator values.

Use `al.view` to reinterpret packed words as the fragment shape expected by the instruction:

```python
a_frag = al.view(a_words, al.Tensor((2, 2, 1), al.u32))
b_frag = al.view(b_words, al.Tensor((2, 2, 1), al.u32))

acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)
```

The operand order is part of the layout decision. In the GEMM tutorial, the operands are swapped so the MFMA accumulator fragment can be shuffled back into row-major `C`.

`mfma_16x16x32_fp8_fp8` and its `mfma_f32_...` alias consume two packed `u32` dwords per lane for each input and produce four `f32` accumulator values. They lower to the AMDGPU FP8 MFMA instruction.

## Raw-Buffer Resources

`al.amdgpu.make_rsrc(tensor, range_bytes)` creates a raw-buffer resource descriptor from a tensor view and a byte range.

```python
A_block = al.subview(A_flat, (block_m * k,), (a_rows * k,), (1,))
A_rsrc = al.amdgpu.make_rsrc(A_block, a_rows * k * BF16_BYTES)
```

The descriptor base is expected to be uniform across the wave. Per-lane row, column, and K positions should be represented as byte offsets inside that descriptor, not by making each lane use a different descriptor base.

The byte range is used by hardware bounds checking. Out-of-range raw-buffer loads return zero. Raw-buffer stores can use the same descriptor range to suppress writes outside the valid region.

## Raw-Buffer Loads And Stores

Raw-buffer loads and stores operate on `i32` words. The suffix selects the number of words moved:

```text
raw_buffer_load_x1   -> i32
raw_buffer_load_x2   -> Tensor((2,), i32)
raw_buffer_load_x4   -> Tensor((4,), i32)
raw_buffer_store_x1  <- i32
raw_buffer_store_x2  <- Tensor((2,), i32)
raw_buffer_store_x4  <- Tensor((4,), i32)
```

Each raw-buffer operation takes a resource descriptor, a vector index, a scalar byte offset, and an auxiliary immediate:

```python
zero = al.convert(0, al.i32)
load_offset = al.convert((lane_col * k + k_base) * BF16_BYTES, al.i32)

words = al.amdgpu.raw_buffer_load_x4(A_rsrc, zero, load_offset, 0)
al.amdgpu.raw_buffer_store_x4(words, C_rsrc, zero, store_offset, 0)
```

The common GEMM pattern is to create one descriptor for the block-owned rows, then let each lane compute a byte offset for its vectorized load. For edge tiles, the descriptor range covers only valid rows or columns, so lanes outside the problem shape are handled by the hardware guard.

## Other AMDGPU Helpers

`al.amdgpu.rcp(x)` lowers to an AMDGPU reciprocal instruction for `f32` values.

```python
r = al.amdgpu.rcp(x)
```

`al.amdgpu.perm(lhs, rhs, sel)` exposes the AMDGPU byte-permute intrinsic for 32-bit integer values.

```python
out = al.amdgpu.perm(lhs, rhs, sel)
```

`al.amdgpu.s_waitcnt(vmcnt, expcnt, lgkmcnt)` emits an explicit wait-count instruction. The arguments must be compile-time integers in the hardware ranges `vmcnt=[0,63]`, `expcnt=[0,7]`, and `lgkmcnt=[0,15]`.

```python
al.amdgpu.s_waitcnt(0, 7, 15)
```

`al.amdgpu.sched_group_barrier(mask, size, group_id)` emits the AMDGPU scheduling group barrier operation.

```python
al.amdgpu.sched_group_barrier(2, 4, 1)
```

`al.amdgpu.raw_buffer_load_x1_lds(...)` is the lower-level direct-to-LDS raw-buffer load wrapper. It is intended for carefully scheduled pipelines and requires compile-time constant size and offset arguments.

## NVVM

NVIDIA intrinsics live under `al.nvvm`. The current wrappers cover selected MMA operations and `m8n8`/`b16` matrix load and store operations.

```python
ra = al.nvvm.ldmatrix_m8n8_x4_b16(a_tile)
rb = al.nvvm.ldmatrix_m8n8_x2_b16(b_tile)
acc = al.nvvm.mma_16x8x16_f16_f16(ra, rb, acc)
```

The currently registered MMA wrappers are:

```text
mma_16x8x16_f16_f16
mma_16x8x8_f16_f32
```

The currently registered matrix load wrappers are:

```text
ldmatrix_m8n8_x1_b16
ldmatrix_m8n8_x2_b16
ldmatrix_m8n8_x4_b16
ldmatrix_m8n8_x1_b16_trans
ldmatrix_m8n8_x2_b16_trans
ldmatrix_m8n8_x4_b16_trans
```

The corresponding store wrappers use `stmatrix_m8n8_x1_b16`, `stmatrix_m8n8_x2_b16`, and `stmatrix_m8n8_x4_b16`, with `_trans` variants for the transposed layout.

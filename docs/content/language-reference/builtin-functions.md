---
layout: layouts/base.njk
title: Builtin Functions
description: Builtin math, memory, layout, synchronization, and shuffle functions.
---

# Builtin Functions

Ave exposes compiler-recognized functions under the `avelang.language` package. Programs usually import it as `al`:

```python
import avelang.language as al
```

These functions are part of the language surface. They are parsed by the JIT and lowered to Ave IR, GPU IR, or target-specific intrinsics instead of being ordinary Python calls.

## Math And Conversion

Use `al.convert(value, dtype)` for value conversion. The compiler emits a numeric conversion to the requested scalar type.

```python
acc = al.convert(0.0, al.f32)
a = al.convert(A[row, k], al.f32)
```

Use `al.bitcast(value, dtype)` when the bit pattern should be reinterpreted without a numeric conversion. This is useful when working with packed values or hardware fragments.

```python
word = al.bitcast(bits, al.u32)
```

Ave also provides scalar math functions: `al.abs`, `al.exp2`, `al.exp`, `al.tanh`, `al.log`, `al.log2`, `al.erf`, and `al.sqrt`. Integer extrema are available through `al.min(lhs, rhs)` and `al.max(lhs, rhs)`.

```python
idx = al.min(idx, limit)
y = al.sqrt(x)
```

## Bit Operations

`al.bitreverse(value)` reverses all 32 bits of an `al.i32` or `al.u32` scalar and preserves its type. Bit 0 becomes bit 31, bit 1 becomes bit 30, and so on. The operation is available on both AMD and NVIDIA GPUs.

```python
reversed_word = al.bitreverse(word)
```

For example, `0x00000001` becomes `0x80000000`. Other integer widths, floating-point values, and vectors are not supported.

## Tensor Construction And Memory

`al.make_shared(shape, dtype)` allocates workgroup shared memory. The shape is usually static or derived from `al.constexpr` parameters so the compiler can allocate a fixed shared-memory region.

```python
a_tile = al.make_shared((BLOCK_M, BLOCK_K), al.bf16)
```

`al.make_local(shape, dtype)` allocates private memory for the current program instance. It is commonly used for small fragments or temporary tensors that belong to one lane or thread.

```python
frag = al.make_local((4,), al.f32)
```

`al.full(shape, value, dtype)` creates a tensor value initialized to a scalar. This is the usual way to initialize accumulator fragments.

```python
acc = al.full((16,), 0.0, al.f32)
```

`al.make_tensor(ptr, dtype, layout)` gives a raw pointer an indexable tensor view. It does not allocate memory. The pointer supplies the base address, `dtype` supplies the element type, and the layout supplies the runtime shape and strides.

```python
A = al.make_tensor(A_ptr, al.bf16, al.make_layout((m, k), (k, 1)))
```

The pointer and dynamic-shape type pattern is described in [Pointer Types](../builtin-types/#pointer-types).

## Layouts And Views

`al.make_layout(shape, strides)` describes how logical coordinates map to a linear element index. For a row-major matrix, the row stride is the number of columns and the column stride is one.

```python
layout = al.make_layout((m, n), (n, 1))
```

For coordinates `(i, j)`, that layout maps to:

```text
linear_index = i * n + j
```

`al.view(value, new_type)` reinterprets an existing tensor or value with another tensor type. It does not copy or rearrange data. For example, a contiguous `128 x 128` BF16 matrix can be viewed as 16-byte vectors:

```python
A_vec = al.view(A, al.Tensor((128, 16, 4), al.i32))
```

When the target layout is dynamic, use the three-argument form with an explicit layout:

```python
A_vec = al.view(A, al.i32, packed_layout)
```

Detailed tensor shape, stride, and packed-view examples live in [Builtin Types](../builtin-types/#tensor-types).

## Subviews

`al.subview(base, offsets, sizes, strides)` creates a view into an existing tensor or memref. Offsets choose the starting coordinate, sizes choose the logical extent, and strides choose how the subview steps through the base.

```python
row = al.subview(matrix, (r, 0), (1, cols), (1, 1))
```

Subviews are useful for describing the part of a larger tensor owned by a block. They are also used to build block-local descriptors for hardware buffer loads and stores.

```python
A_block = al.subview(A_flat, (block_m * k,), (rows * k,), (1,))
```

## Coordinate Functions

`al.block_id`, `al.thread_id`, `al.block_dim`, and `al.grid_dim` expose launch coordinates. The dimension argument must be the constant integer `0`, `1`, or `2`. Coordinate usage is covered in [Functions And Control Flow](../functions-control-flow/#gpu-coordinates).

## Synchronization

`al.syncthreads()` is a workgroup barrier. Use it after writing shared memory and before other threads read those writes.

```python
tile[tid] = x[idx]
al.syncthreads()
y[idx] = tile[(tid + 1) & 31]
```

Use another barrier before reusing the same shared-memory tile for a later loop iteration.

## Shuffle

`al.shuffle`, `al.shuffle_up`, `al.shuffle_down`, and `al.shuffle_xor` exchange scalar values across lanes in a subgroup. Each takes the value to move, an index or offset, and the subgroup width.

```python
other = al.shuffle_xor(value, 1, 32)
```

`al.shuffle(value, index, width)` reads from an explicit lane index. The directional forms read from lanes relative to the current lane, and `al.shuffle_xor` reads from the lane selected by XORing the current lane id with the mask.

## Debug Printing

`al.printf(format, ...)` emits a GPU printf operation. The format must be a string literal, and any extra arguments are compiled as values passed to the device-side print.

```python
al.printf("idx=%d value=%f\n", idx, value)
```

Use it for debugging small kernels; device-side printing is expensive and can change timing-sensitive behavior.

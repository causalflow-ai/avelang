---
layout: layouts/base.njk
title: "Tutorial 1: AXPY"
---

# Tutorial 1: AXPY

This tutorial builds on the [installation guide](../../get-started/installation/) and introduces the core launch shape of an Ave kernel.

Ave is a Pythonic tile-based DSL. It models memory accesses as tensor and tiles. Similar to Triton, Ave kernels are Python functions decorated with `@ave.jit`. Ave functions are compiled to GPU kernels via the Ave JIT compiler. Ave currently exposes thread level APIs to give explicit controls of the GPUs. Here is an example of a simple AXPY kernel: 

```python

import avelang 
import avelang.language as al

@avelang.jit
def axpy(a: al.i32, b: al.i32, n: al.i32,
         x: al.Tensor((256,), al.i32), y: al.Tensor((256,), al.i32)):
    idx = al.block_id(0) * al.block_dim(0) + al.thread_id(0)
    if idx < n:
        y[idx] = a * x[idx] + b
```

In the program above, the type annotation `al.i32` denotes the 32-bit integer type. `al.Tensor((256,), al.i32)` specfies that both `x` and `y` are a tensor with 256 elements of integers. Ave exposes `al.block_id(0)`, `al.block_dim(0)`, and `al.thread_id(0)` like CUDA and HIP to give the program explicit controls of the GPU. 

You can execute the kernel by providing the desired shapes of the grids and blocks just like executing a Triton function: 

```python
import torch

a = 2
b = 3
n = 256
x = torch.arange(256, dtype=torch.int32, device="cuda")
y = torch.empty_like(x)

axpy[lambda: ((1, 1, 1), (256, 1, 1))](a, b, n, x, y)
torch.testing.assert_close(y.cpu(), (a * x + b).cpu())
```

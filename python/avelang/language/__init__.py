"""
ave-lang: JIT compiler for GPU kernels.

A Python framework for writing high-performance GPU kernels with JIT compilation.
"""

from . import amdgpu as amdgpu
from . import nvvm as nvvm
from .core import (
    # Signed integer types
    i8,
    i16,
    i32,
    i64,
    # Unsigned integer types
    u1,
    u8,
    u16,
    u32,
    u64,
    # Floating-point types
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f16,
    bf16,
    f32,
    f64,
    # Other types
    constexpr,
    dynamic,
    Tensor,
    Pointer,
    convert,
    block_id,
    block_dim,
    thread_id,
    grid_dim,
    shuffle,
    shuffle_up,
    shuffle_down,
    shuffle_xor,
    min,
    max,
    select,
    fma,
    abs,
    exp2,
    exp,
    tanh,
    log,
    log2,
    erf,
    sqrt,
)

__all__ = [
    # Signed integer types
    "i8",
    "i16",
    "i32",
    "i64",
    # Unsigned integer types
    "u1",
    "u8",
    "u16",
    "u32",
    "u64",
    # Floating-point types
    "f8e4m3fn",
    "f8e4m3fnuz",
    "f8e5m2",
    "f8e5m2fnuz",
    "f16",
    "bf16",
    "f32",
    "f64",
    # Other types
    "constexpr",
    "dynamic",
    "Tensor",
    "Pointer",
    "convert",
    "block_id",
    "block_dim",
    "thread_id",
    "grid_dim",
    "shuffle",
    "shuffle_up",
    "shuffle_down",
    "shuffle_xor",
    "min",
    "max",
    "select",
    "fma",
    "abs",
    "exp2",
    "exp",
    "tanh",
    "log",
    "log2",
    "erf",
    "sqrt",
]

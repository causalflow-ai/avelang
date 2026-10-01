import pytest
import torch

import avelang
import avelang.language as al
from avelang.testing import has_rocm


@avelang.jit
def kernel_bitreverse(src: al.Tensor((4,), al.u32), out: al.Tensor((4,), al.u32)):
    lane = al.thread_id(0)
    out[lane] = al.amdgpu.bitreverse(src[lane])


@pytest.mark.skipif(not has_rocm(), reason="Requires a ROCm GPU.")
def test_bitreverse():
    values = [0, 0x01234567, 0x80000000, 0xFFFFFFFF]
    src = torch.tensor(values, dtype=torch.uint32, device="cuda")
    out = torch.empty_like(src)
    kernel_bitreverse[lambda: ((1, 1, 1), (4, 1, 1))](src, out)
    expected = torch.tensor(
        [int(f"{value:032b}"[::-1], 2) for value in values], dtype=torch.uint32,
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

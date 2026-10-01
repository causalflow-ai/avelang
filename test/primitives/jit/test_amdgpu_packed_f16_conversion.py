import avelang
import avelang.language as al
import pytest
import torch
from avelang.testing import has_rocm


@avelang.jit
def kernel_packed_f16_conversion(
    src: al.Tensor((8, 2), al.f32),
    fp16: al.Tensor((8,), al.u32),
    bf16: al.Tensor((8,), al.u32),
):
    lane = al.thread_id(0)
    fp16[lane] = al.amdgpu.cvt_pk_f16_f32(src[lane, 0], src[lane, 1])
    bf16[lane] = al.amdgpu.cvt_pk_bf16_f32(src[lane, 0], src[lane, 1])


@pytest.mark.skipif(not has_rocm(), reason="Requires AMD GPU")
def test_packed_f16_and_bf16_conversion():
    # Midpoints and their neighbors, negative values, signed zero,
    # subnormals and infinities check rounding and low/high placement.
    bits = torch.tensor(
        [
            0x3F800FFF,
            0x3F801000,
            0x3F801001,
            0x3F803000,
            0x3F807FFF,
            0x3F808000,
            0x3F808001,
            0x3F818000,
            0xBF803000,
            0xBF818000,
            0,
            0x80000000,
            0x33800000,
            0x00018000,
            0x7F800000,
            0xFF800000,
        ],
        dtype=torch.uint32,
    )
    src = bits.view(torch.float32).reshape(8, 2).cuda()
    fp16 = torch.empty((8,), dtype=torch.int32, device="cuda")
    bf16 = torch.empty_like(fp16)
    kernel_packed_f16_conversion[lambda: ((1, 1, 1), (8, 1, 1))](src, fp16, bf16)
    for dtype, output in ((torch.float16, fp16), (torch.bfloat16, bf16)):
        expected = src.to(dtype).view(torch.int32).flatten()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

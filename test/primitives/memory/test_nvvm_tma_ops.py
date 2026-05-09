#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S
from avelang.testing import has_cuda_nvidia


def get_tma_device():
    if not has_cuda_nvidia():
        return None
    for device_idx in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(device_idx)
        if major >= 9:
            return device_idx
    return None


@avelang.jit
def kernel_tma_load_tiles(
    tensor: S.Tensor((32, 32), S.f16),
    out: S.Tensor((32, 32), S.f16),
):
    tid = S.thread_id(0)
    tile_m = S.block_id(0) * 16
    tile_n = S.block_id(1) * 16
    row = tid // 16
    col = tid % 16

    smem = S.make_shared((16, 16), S.f16)
    smem_layout = S.make_layout((16, 16), (16, 1))
    desc = S.nvvm.make_tma_descriptor(tensor, smem_layout)
    barrier = S.nvvm.mbarrier_create()

    S.nvvm.mbarrier_init(barrier, 0, count=1, predicate=tid == 0)
    S.syncthreads()
    S.nvvm.tma_load(
        smem,
        desc,
        (tile_n, tile_m),
        barrier,
        mbar_id=0,
        predicate=tid == 0,
        multicast_mask=None,
    )
    S.nvvm.mbarrier_try_wait_parity(barrier, 0, 10000000, 0)
    S.syncthreads()

    out[tile_m + row, tile_n + col] = smem[row, col]


@avelang.jit
def kernel_tma_store_tiles(out: S.Tensor((32, 32), S.f16)):
    tid = S.thread_id(0)
    tile_m = S.block_id(0) * 16
    tile_n = S.block_id(1) * 16

    smem = S.make_shared((16, 16), S.f16)
    smem_layout = S.make_layout((16, 16), (16, 1))
    desc = S.nvvm.make_tma_descriptor(out, smem_layout)

    for j in S.range(16):
        value = (tile_m + tid) * 32 + tile_n + j
        smem[tid, j] = S.convert(value, S.f16)

    S.syncthreads()
    S.nvvm.tma_store(smem, desc, (tile_n, tile_m), predicate=tid == 0)


@unittest.skipUnless(
    get_tma_device() is not None,
    "Requires CUDA on an NVIDIA Hopper-or-newer GPU with TMA support.",
)
class TestNVVMTMAOps(unittest.TestCase):
    def setUp(self):
        device_idx = get_tma_device()
        self.assertIsNotNone(device_idx)
        torch.cuda.set_device(device_idx)
        self.device = torch.device(f"cuda:{device_idx}")

    def test_tma_load_copies_all_tiles(self):
        tensor = torch.arange(
            32 * 32, dtype=torch.float16, device=self.device
        ).reshape(32, 32)
        out = torch.zeros_like(tensor)

        kernel_tma_load_tiles[lambda: ((2, 2, 1), (256, 1, 1))](
            tensor, out
        )
        torch.cuda.synchronize(self.device)

        actual = out.cpu()
        expected = tensor.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            msg=(
                "TMA load results do not match.\n"
                f"Expected:\n{expected}\nActual:\n{actual}"
            ),
        )

    def test_tma_store_writes_all_tiles(self):
        out = torch.zeros((32, 32), dtype=torch.float16, device=self.device)
        expected = torch.arange(
            32 * 32, dtype=torch.float16, device=self.device
        ).reshape(32, 32)

        kernel_tma_store_tiles[lambda: ((2, 2, 1), (16, 1, 1))](out)
        torch.cuda.synchronize(self.device)

        actual = out.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            msg=(
                "TMA store results do not match.\n"
                f"Expected:\n{expected}\nActual:\n{actual}"
            ),
        )


if __name__ == "__main__":
    unittest.main()

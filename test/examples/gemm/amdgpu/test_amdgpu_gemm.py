#!/usr/bin/env python3
import unittest

import torch

from avelang.testing import has_rocm
from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestAMDGPUGEMM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = 1024
        self.n = 1024
        self.k = 1024
        self.rtol = 1.0e-1
        self.atol = 1.0e-1

    def test_gemm_pipeline_transposed_b(self):
        A = torch.randn((self.m, self.k), dtype=torch.bfloat16, device="cuda")
        B = torch.randn((self.n, self.k), dtype=torch.bfloat16, device="cuda")

        expected = (A.float() @ B.float().T).to(dtype=torch.bfloat16, device="cpu")
        actual = gemm_pipeline_transposed_b(A, B).cpu()

        self.assertTrue(
            torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
            msg=(
                "AMDGPU GEMM results do not match.\n"
                f"Expected:\n{expected}\n"
                f"Actual:\n{actual}\n"
                f"Max absolute difference: {torch.max(torch.abs(actual - expected))}"
            ),
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

from avelang.testing import has_rocm
from avelang_kernels import flash_attn


@unittest.skipUnless(
    has_rocm(),
    "Requires ROCm/HIP with an AMD GPU.",
)
class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rtol = 5e-2
        self.atol = 5e-2

    def test_flash_attention(self):
        seq_lens = [32, 64]
        total_tokens = sum(seq_lens)
        q_heads = 4
        kv_heads = 2
        gqa_ratio = q_heads // kv_heads
        head_dim = 64

        q = torch.randn((total_tokens, q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda")
        seq_ptr = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device="cpu")

        expected = torch.empty_like(q, dtype=torch.float32)
        for seq_idx, seq_len in enumerate(seq_lens):
            seq_begin = seq_ptr[seq_idx].item()
            seq_end = seq_ptr[seq_idx + 1].item()
            q_seq = q[seq_begin:seq_end].permute(1, 0, 2).unsqueeze(0)
            k_seq = k[seq_begin:seq_end].permute(1, 0, 2).unsqueeze(0).repeat_interleave(gqa_ratio, dim=1)
            v_seq = v[seq_begin:seq_end].permute(1, 0, 2).unsqueeze(0).repeat_interleave(gqa_ratio, dim=1)
            expected_seq = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True)
            expected[seq_begin:seq_end] = expected_seq.squeeze(0).permute(1, 0, 2).to(torch.float32)

        actual = flash_attn.flash_attn(q, k, v, seq_ptr).to(torch.float32).cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
            msg=f"Flash attention results do not match.\nExpected:\n{expected}\nActual:\n{actual}\n"
            f"Max absolute difference: {torch.max(torch.abs(actual - expected))}",
        )


if __name__ == "__main__":
    unittest.main()

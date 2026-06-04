#!/usr/bin/env python3
import unittest

import torch

import avelang
import avelang.language as S


@avelang.jit
def kernel_shm_basic(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((32,), S.i32),
):
    shared_buf = S.make_shared((32,), S.i32)

    thread_id = S.thread_id(0)

    shared_buf[thread_id] = input_data[thread_id]

    S.syncthreads()

    read_idx = (thread_id + 1) % 32
    output_data[thread_id] = shared_buf[read_idx]


@avelang.jit
def kernel_shm_manual_alignment(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((32,), S.i32),
):
    shared_buf = S.make_shared((32,), S.i32, 128)

    thread_id = S.thread_id(0)

    shared_buf[thread_id] = input_data[thread_id]

    S.syncthreads()

    output_data[thread_id] = shared_buf[31 - thread_id]


@avelang.jit
def kernel_shm_2d(
    input_matrix: S.Tensor((16, 16), S.f32),
    output_matrix: S.Tensor((16, 16), S.f32),
):
    shared_buf = S.make_shared((16, 16), S.f32)

    thread_x = S.thread_id(0)
    thread_y = S.thread_id(1)

    shared_buf[thread_y, thread_x] = input_matrix[thread_y, thread_x]

    S.syncthreads()

    output_matrix[thread_y, thread_x] = shared_buf[thread_x, thread_y]  # Transpose


@avelang.jit
def kernel_shm_reduction(
    input_data: S.Tensor((32,), S.f32),
    result: S.Tensor((2,), S.f32),
):
    shared_buf = S.make_shared((32,), S.f32)

    thread_id = S.thread_id(0)

    shared_buf[thread_id] = input_data[thread_id]
    S.syncthreads()

    stride = 16
    for i in S.range(5):  # log2(32) = 5
        if thread_id < stride:
            shared_buf[thread_id] = shared_buf[thread_id] + shared_buf[thread_id + stride]
        S.syncthreads()
        stride = stride >> 0x1

    if thread_id == 0:
        result[0] = shared_buf[0]


@avelang.jit
def kernel_shm_subview_view_offsets(result: S.Tensor((2,), S.u32)):
    shared_words = S.make_shared((16,), S.u32)
    left_words = S.subview(shared_words, (0,), (8,), (1,))
    right_words = S.subview(shared_words, (8,), (8,), (1,))
    layout = S.make_layout((2, 2), (2, 1))
    left = S.view(left_words, S.u32, layout)
    right = S.view(right_words, S.u32, layout)

    tid = S.thread_id(0)
    if tid == 0:
        left[0, 0] = S.convert(11, S.u32)
        right[0, 0] = S.convert(22, S.u32)

    S.syncthreads()

    if tid == 0:
        result[0] = left[0, 0]
        result[1] = right[0, 0]


@avelang.jit
def kernel_shm_subview_view_mixed_types(result: S.Tensor((4,), S.i32)):
    shared_words = S.make_shared((32,), S.u32)
    stats_words = S.subview(shared_words, (0,), (8,), (1,))
    q_words = S.subview(shared_words, (8,), (8,), (1,))
    kv_words = S.subview(shared_words, (16,), (8,), (1,))
    score_words = S.subview(shared_words, (24,), (8,), (1,))
    layout_f32 = S.make_layout((2, 4), (4, 1))
    layout_bf16 = S.make_layout((2, 4), (4, 1))
    layout_u32 = S.make_layout((2, 2), (2, 1))
    stats = S.view(stats_words, S.f32, layout_f32)
    q = S.view(q_words, S.bf16, layout_bf16)
    kv = S.view(kv_words, S.bf16, layout_bf16)
    score = S.view(score_words, S.f32, layout_f32)
    q_u32 = S.view(q_words, S.u32, layout_u32)
    kv_u32 = S.view(kv_words, S.u32, layout_u32)

    tid = S.thread_id(0)
    if tid == 0:
        stats[0, 0] = S.convert(3.0, S.f32)
        score[0, 0] = S.convert(4.0, S.f32)
        q[0, 0] = S.convert(1.0, S.bf16)
        q[0, 1] = S.convert(2.0, S.bf16)
        kv[0, 0] = S.convert(3.0, S.bf16)
        kv[0, 1] = S.convert(4.0, S.bf16)

    S.syncthreads()

    if tid == 0:
        result[0] = S.convert(stats[0, 0], S.i32)
        result[1] = q_u32[0, 0]
        result[2] = kv_u32[0, 0]
        result[3] = S.convert(score[0, 0], S.i32)


class TestSharedMemoryOps(unittest.TestCase):
    def test_basic_shared_memory(self):
        """Test basic shared memory read/write functionality"""
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros(32, dtype=torch.int32, device="cuda")

        expected = torch.cat([input_data[1:], input_data[0:1]])

        kernel_shm_basic[lambda: ((1, 1, 1), (32, 1, 1))](input_data, output_data)

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_manual_alignment_shared_memory(self):
        """Test shared memory allocation with an explicit alignment."""
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros(32, dtype=torch.int32, device="cuda")

        expected = torch.flip(input_data, dims=(0,))

        kernel_shm_manual_alignment[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_2d_shared_memory(self):
        """Test 2D shared memory with matrix transpose"""
        input_matrix = torch.arange(256, dtype=torch.float32, device="cuda").reshape(16, 16)
        output_matrix = torch.zeros((16, 16), dtype=torch.float32, device="cuda")

        expected = input_matrix.t()  # Transpose

        kernel_shm_2d[lambda: ((1, 1, 1), (16, 16, 1))](input_matrix, output_matrix)

        actual = output_matrix.cpu()
        expected = expected.cpu()

        self.assertTrue(
            torch.allclose(actual, expected),
            f"Max difference: {(actual - expected).abs().max()}",
        )

    def test_shm_reduction(self):
        """Test shared memory reduction operation"""
        input_data = torch.ones(32, dtype=torch.float32, device="cuda")
        result = torch.zeros(2, dtype=torch.float32, device="cuda")

        expected = torch.sum(input_data)

        kernel_shm_reduction[lambda: ((1, 1, 1), (32, 1, 1))](input_data, result)

        actual = result[0].cpu().item()
        expected = expected.cpu().item()

        self.assertAlmostEqual(actual, expected, places=5, msg=f"Expected: {expected}, Got: {actual}")

    def test_shm_subview_view_offsets(self):
        """Test that typed views preserve non-zero shared subview offsets."""
        result = torch.zeros(2, dtype=torch.int32, device="cuda")

        kernel_shm_subview_view_offsets[lambda: ((1, 1, 1), (32, 1, 1))](result)

        actual = result.cpu()
        expected = torch.tensor([11, 22], dtype=torch.int32)

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )

    def test_shm_subview_view_mixed_types(self):
        """Test mixed f32/bf16/u32 typed views on distinct shared subviews."""
        result = torch.zeros(4, dtype=torch.int32, device="cuda")

        kernel_shm_subview_view_mixed_types[lambda: ((1, 1, 1), (32, 1, 1))](result)

        actual = result.cpu()
        expected = torch.tensor([3, 1073758080, 1082146880, 4], dtype=torch.int32)

        self.assertTrue(
            torch.equal(actual, expected),
            f"Expected: {expected.tolist()}, Actual: {actual.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()

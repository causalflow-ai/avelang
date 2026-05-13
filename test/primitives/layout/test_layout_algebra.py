#!/usr/bin/env python3
import unittest

import torch

import substrate
import substrate.language as S


@substrate.jit
def kernel_layout_composition(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((4, 2), S.i32),
):
    layout = S.composition(
        S.make_layout((2, 4), (4, 1)),
        S.make_layout((4, 2), (2, 1)),
    )
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(4):
        for j in S.range(2):
            output_data[i, j] = input_data_cast[i, j]


@substrate.jit
def kernel_layout_complement(
    input_data: S.Tensor((2,), S.i32),
    output_data: S.Tensor((2,), S.i32),
):
    layout = S.complement(S.make_layout((2, 2), (2, 4)), 8)
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(2):
        output_data[i] = input_data_cast[i]


@substrate.jit
def kernel_layout_coalesce(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((8,), S.i32),
):
    layout = S.coalesce(S.make_layout((2, 4), (1, 2)))
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(8):
        output_data[i] = input_data_cast[i]


@substrate.jit
def kernel_layout_divide(
    input_data: S.Tensor((8,), S.i32),
    output_data: S.Tensor((2, 2, 2), S.i32),
):
    layout = S.divide(
        S.make_layout((2, 4), (4, 1)),
        S.make_layout((2, 2), (1, 4)),
    )
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(2):
        for j in S.range(2):
            for k in S.range(2):
                output_data[i, j, k] = input_data_cast[i, j, k]


@substrate.jit
def kernel_composition(
    input_data: S.Tensor((48,), S.i32),
    output_data: S.Tensor((2, 2, 3), S.i32),
):
    layout = S.composition(
        S.make_layout((6, 2), (8, 2)),
        S.make_layout((4, 3), (3, 1)),
    )
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(2):
        for j in S.range(2):
            for k in S.range(3):
                output_data[i, j, k] = input_data_cast[i, j, k]


@substrate.jit
def kernel_complement(
    input_data: S.Tensor((24,), S.i32),
    output_data: S.Tensor((3, 2), S.i32),
):
    layout = S.complement(S.make_layout((2, 2), (1, 6)), 24)
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(3):
        for j in S.range(2):
            output_data[i, j] = input_data_cast[i, j]


@substrate.jit
def kernel_divide(
    input_data: S.Tensor((24,), S.i32),
    output_data: S.Tensor((2, 2, 2, 3), S.i32),
):
    layout = S.divide(
        S.make_layout((4, 2, 3), (2, 1, 8)),
        S.make_layout((4,), (2,)),
    )
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(2):
        for j in S.range(2):
            for k in S.range(2):
                for l in S.range(3):
                    output_data[i, j, k, l] = input_data_cast[i, j, k, l]


@substrate.jit
def kernel_product(
    input_data: S.Tensor((32,), S.i32),
    output_data: S.Tensor((2, 2, 4, 2), S.i32),
):
    layout = S.product(
        S.make_layout((2, 2), (4, 1)),
        S.make_layout((4, 2), (2, 1)),
    )
    input_data_cast = S.view(input_data, S.i32, layout)
    for i in S.range(2):
        for j in S.range(2):
            for k in S.range(4):
                for l in S.range(2):
                    output_data[i, j, k, l] = input_data_cast[i, j, k, l]


class TestLayoutAlgebra(unittest.TestCase):
    def test_layout_composition(self):
        input_data = torch.arange(8, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((4, 2), dtype=torch.int32, device="cuda")

        kernel_layout_composition[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(4, 2), stride=(1, 4)).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_layout_complement(self):
        input_data = torch.arange(2, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((2,), dtype=torch.int32, device="cuda")

        kernel_layout_complement[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(2,), stride=(1,)).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_layout_coalesce(self):
        input_data = torch.arange(8, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((8,), dtype=torch.int32, device="cuda")

        kernel_layout_coalesce[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(8,), stride=(1,)).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_layout_divide(self):
        input_data = torch.arange(8, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((2, 2, 2), dtype=torch.int32, device="cuda")

        kernel_layout_divide[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(2, 2, 2), stride=(4, 2, 1)).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_composition(self):
        input_data = torch.arange(48, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((2, 2, 3), dtype=torch.int32, device="cuda")

        kernel_composition[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(2, 2, 3), stride=(24, 2, 8)).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_complement(self):
        input_data = torch.arange(24, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((3, 2), dtype=torch.int32, device="cuda")

        kernel_complement[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(size=(3, 2), stride=(2, 12)).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_divide(self):
        input_data = torch.arange(24, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((2, 2, 2, 3), dtype=torch.int32, device="cuda")

        kernel_divide[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(
            size=(2, 2, 2, 3), stride=(4, 1, 2, 8)
        ).cpu()
        self.assertTrue(torch.equal(actual, expected))

    def test_product(self):
        input_data = torch.arange(32, dtype=torch.int32, device="cuda")
        output_data = torch.zeros((2, 2, 4, 2), dtype=torch.int32, device="cuda")

        kernel_product[lambda: ((1, 1, 1), (32, 1, 1))](
            input_data, output_data
        )

        actual = output_data.cpu()
        expected = input_data.as_strided(
            size=(2, 2, 4, 2), stride=(4, 1, 8, 2)
        ).cpu()
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

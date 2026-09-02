"""Tests for abliteration/abliterate.py's projection math: the permanent
weight-orthogonalization update must be idempotent (P^2 = P, PRINCIPLES.md
§7) and must actually remove the target direction. Pure linear algebra —
no model, no download, no prompts."""

import sys

import pytest
import torch

sys.path.insert(0, "abliteration")
from abliterate import (  # noqa: E402
    orthogonalize_embedding,
    orthogonalize_writer,
    remove_direction,
)

F64 = torch.float64


def rand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=F64)


class TestRemoveDirection:
    def test_removes_parallel_component(self):
        r = torch.tensor([1.0, 1.0], dtype=F64)
        x = torch.tensor([6.0, 2.0], dtype=F64)
        x_new = remove_direction(x, r)
        assert torch.allclose(x_new, torch.tensor([2.0, -2.0], dtype=F64))
        assert abs(float(r @ x_new)) < 1e-10

    def test_idempotent_on_activations(self):
        r = rand((16,), seed=1)
        x = rand((8, 16), seed=2)
        once = remove_direction(x, r)
        twice = remove_direction(once, r)
        assert torch.allclose(once, twice, atol=1e-10)


class TestOrthogonalizeWriter:
    def test_matches_worked_example(self):
        # PRINCIPLES.md §11's fully worked 2x2 integer example
        r = torch.tensor([1.0, 1.0], dtype=F64)
        W = torch.tensor([[4.0, 2.0], [2.0, 4.0]], dtype=F64)
        W_new = orthogonalize_writer(W, r)
        assert torch.allclose(W_new, torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=F64))

    def test_idempotent(self):
        r = rand((32,), seed=3)
        W = rand((32, 20), seed=4)
        once = orthogonalize_writer(W, r)
        twice = orthogonalize_writer(once, r)
        assert torch.allclose(once, twice, atol=1e-9)

    def test_output_is_orthogonal_to_direction(self):
        r = rand((32,), seed=5)
        W = rand((32, 20), seed=6)
        W_new = orthogonalize_writer(W, r)
        u = rand((20,), seed=7)
        r_hat = r / r.norm()
        assert abs(float(r_hat @ (W_new @ u))) < 1e-9

    def test_unnormalized_direction_gives_same_result(self):
        r = rand((16,), seed=8)
        W = rand((16, 10), seed=9)
        assert torch.allclose(orthogonalize_writer(W, r),
                              orthogonalize_writer(W, r * 5.0), atol=1e-9)


class TestOrthogonalizeEmbedding:
    def test_row_space_orthogonal(self):
        r = rand((16,), seed=10)
        E = rand((50, 16), seed=11)  # [vocab, d_model]
        E_new = orthogonalize_embedding(E, r)
        r_hat = r / r.norm()
        assert torch.allclose(E_new @ r_hat, torch.zeros(50, dtype=F64), atol=1e-9)

    def test_idempotent(self):
        r = rand((16,), seed=12)
        E = rand((50, 16), seed=13)
        once = orthogonalize_embedding(E, r)
        twice = orthogonalize_embedding(once, r)
        assert torch.allclose(once, twice, atol=1e-9)


class TestConsistencyAcrossOrientations:
    def test_writer_and_activation_removal_agree(self):
        """W_new @ u == remove_direction(W @ u, r) for any u — PLAN.md §2's
        'x_new = W_new u = P W u = P x' identity."""
        r = rand((24,), seed=14)
        W = rand((24, 12), seed=15)
        u = rand((12,), seed=16)
        via_weight = orthogonalize_writer(W, r) @ u
        via_activation = remove_direction(W @ u, r)
        assert torch.allclose(via_weight, via_activation, atol=1e-9)

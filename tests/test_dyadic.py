"""Tests for dyadic scale machinery: fitting accuracy, quant/requant golden
tests vs float, residual add properties. Float appears here only as a test
oracle — never on the library's numerical path."""

import pytest
import torch

from detllm import dyadic as dy

I64 = torch.int64


def rand_i64(shape, lo, hi, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, shape, generator=g, dtype=I64)


class TestFitDyadicRatio:
    def test_relative_error_bound(self):
        num = rand_i64((10000,), 1, 1 << 40, seed=1)
        den = rand_i64((10000,), 1, 1 << 20, seed=2)
        m, e = dy.fit_dyadic_ratio(num, den, bits=16)
        assert bool((m >= 1 << 15).all()) and bool((m < 1 << 16).all())
        approx = m.double() * torch.pow(2.0, -e.double())
        exact = num.double() / den.double()
        rel = ((approx - exact) / exact).abs().max().item()
        assert rel < 2.0 ** -16, rel

    def test_exact_powers(self):
        num = torch.tensor([1, 2, 1 << 15, 1 << 30], dtype=I64)
        m, e = dy.fit_dyadic_ratio(num, 1, bits=16)
        approx = m.double() * torch.pow(2.0, -e.double())
        assert torch.equal(approx, num.double())  # powers of two are exact

    def test_scalar_den(self):
        num = torch.tensor([127, 254, 1000000], dtype=I64)
        m, e = dy.fit_dyadic_ratio(num, 127, bits=16)
        approx = m.double() * torch.pow(2.0, -e.double())
        exact = num.double() / 127.0
        assert ((approx - exact) / exact).abs().max().item() < 2.0 ** -16


class TestNormScale:
    def test_range_and_value(self):
        m = rand_i64((5000,), 1, 1 << 40, seed=3)
        k = rand_i64((5000,), 0, 60, seed=4)
        m2, k2 = dy.norm_scale(m, k, bits=16)
        assert bool((m2 >= 1 << 15).all()) and bool((m2 < 1 << 16).all())
        v1 = m.double() * torch.pow(2.0, -k.double())
        v2 = m2.double() * torch.pow(2.0, -k2.double())
        assert ((v2 - v1) / v1).abs().max().item() < 2.0 ** -16


class TestQuantI8PerToken:
    def test_roundtrip_accuracy(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randint(-(1 << 24), 1 << 24, (64, 1024), generator=g, dtype=torch.int32)
        m = rand_i64((64, 1), 1 << 15, 1 << 16, seed=6)
        k = torch.full((64, 1), 30, dtype=I64)
        x8, mo, ko = dy.quant_i8_pertoken(x, m, k)
        assert x8.dtype == torch.int8
        val_in = x.double() * m.double() / (2.0 ** 30)
        val_out = x8.double() * mo.double() * torch.pow(2.0, -ko.double())
        err = (val_out - val_in).abs().max(dim=-1).values
        step = val_in.abs().max(dim=-1).values / 127
        assert bool((err <= step * 0.51 + 1e-12).all())  # within half a quant step

    def test_max_maps_to_127(self):
        x = torch.tensor([[3, -1000, 500]], dtype=torch.int32)
        m = torch.ones(1, 1, dtype=I64)
        k = torch.zeros(1, 1, dtype=I64)
        x8, _, _ = dy.quant_i8_pertoken(x, m, k)
        assert x8.abs().max().item() == 127

    def test_zero_row(self):
        x = torch.zeros(2, 8, dtype=torch.int32)
        x8, mo, ko = dy.quant_i8_pertoken(x, torch.ones(2, 1, dtype=I64), torch.zeros(2, 1, dtype=I64))
        assert torch.equal(x8, torch.zeros(2, 8, dtype=torch.int8))


class TestRequant:
    def _setup(self, seed=7):
        g = torch.Generator().manual_seed(seed)
        P = torch.randint(-(1 << 24), 1 << 24, (32, 512), generator=g, dtype=torch.int32)
        row_m = rand_i64((32, 1), 1 << 15, 1 << 16, seed=seed + 1)
        row_k = torch.full((32, 1), 28, dtype=I64)
        col_m = rand_i64((512,), 1 << 15, 1 << 16, seed=seed + 2)
        col_k = 24
        val = P.double() * row_m.double() * col_m.double() / 2.0 ** (28 + 24)
        return P, row_m, row_k, col_m, col_k, val

    def test_i32_common(self):
        P, rm, rk, cm, ck, val = self._setup()
        y, m, k = dy.requant_i32_common(P, rm, rk, cm, ck, target_bits=12)
        got = y.double() * m.double() * torch.pow(2.0, -k.double())
        step = val.abs().max(dim=-1, keepdim=True).values / (1 << 11)
        assert (got - val).abs().max().item() <= step.max().item()
        assert y.abs().max().item() < (1 << 12)

    def test_i8_rowcol(self):
        P, rm, rk, cm, ck, val = self._setup(seed=11)
        y8, m, k = dy.requant_i8_rowcol(P, rm, rk, cm, ck)
        got = y8.double() * m.double() * torch.pow(2.0, -k.double())
        step = val.abs().max(dim=-1, keepdim=True).values / 127
        assert bool(((got - val).abs() <= step * 0.51 + 1e-12).all())

    def test_i8_static(self):
        P, rm, rk, cm, ck, val = self._setup(seed=13)
        # static scale chosen to cover the actual range
        smax = val.abs().max().item()
        s = smax / 127
        s_k = 40
        s_m = round(s * (1 << s_k))
        y8 = dy.requant_i8_static(P, rm, rk, cm, ck, s_m, s_k)
        got = y8.double() * s_m / 2.0 ** s_k
        assert (got - val).abs().max().item() <= s * 0.51 + 1e-9


class TestResidual:
    def test_from_delta_and_add_accuracy(self):
        g = torch.Generator().manual_seed(17)
        d1 = torch.randint(-(1 << 20), 1 << 20, (16, 256), generator=g, dtype=torch.int32)
        d2 = torch.randint(-(1 << 20), 1 << 20, (16, 256), generator=g, dtype=torch.int32)
        m1 = rand_i64((16, 1), 1 << 15, 1 << 16, seed=18)
        m2 = rand_i64((16, 1), 1 << 15, 1 << 16, seed=19)
        k1 = torch.full((16, 1), 35, dtype=I64)
        k2 = torch.full((16, 1), 42, dtype=I64)
        h, kr = dy.residual_from_delta(d1, m1, k1)
        h2, kr2 = dy.residual_add(h, kr, d2, m2, k2)
        want = d1.double() * m1.double() / 2.0 ** 35 + d2.double() * m2.double() / 2.0 ** 42
        got = h2.double() * torch.pow(2.0, -kr2.double())
        scale = want.abs().max(dim=-1, keepdim=True).values
        assert ((got - want).abs() / scale).max().item() < 1e-6

    def test_renorm_bounds(self):
        h64 = torch.tensor([[1, -3, 5], [0, 0, 0], [1 << 40, 5, -9]], dtype=I64)
        kr = torch.zeros(3, 1, dtype=I64)
        h, kr2 = dy.residual_renorm(h64, kr)
        assert h.dtype == torch.int32
        gmax = h.to(I64).abs().amax(dim=-1)
        assert bool((gmax[gmax > 0] < (1 << dy.RES_MAX_BITS)).all())
        assert bool((gmax[gmax > 0] >= (1 << dy.RES_MIN_BITS)).all())
        # value preservation within rounding — error is bounded relative to
        # the ROW scale (shared-scale rows lose absolute precision on entries
        # far below the row max; that is inherent and bounded by 2^-RES_MIN)
        v1 = h64.double()
        v2 = h.double() * torch.pow(2.0, (kr - kr2).double())
        row_scale = v1.abs().max(dim=-1, keepdim=True).values.clamp(min=1)
        assert ((v2 - v1).abs() / row_scale).max().item() < 2.0 ** -24

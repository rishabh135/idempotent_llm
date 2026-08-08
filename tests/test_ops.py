"""Golden tests for the §6 operators against float references (reference
backend). Error bounds established empirically and FROZEN — a change that
worsens any bound is a numerics regression and must be deliberate."""

import math

import pytest
import torch

from detllm import dyadic as dy
from detllm.backends import int_gemm, int_gemm_u8i8
from detllm.ops import (
    apply_int_rope,
    di_exp,
    di_matmul,
    di_rmsnorm,
    di_softmax,
    di_swiglu,
)
from detllm.ops.di_rmsnorm import OUT_FRAC_BITS, di_rmsnorm_gamma
from detllm.ops.rope import ROPE_FRAC_BITS, build_rope_tables

I64 = torch.int64


class TestIntGemm:
    def test_matches_naive_loop(self):
        g = torch.Generator().manual_seed(0)
        a = torch.randint(-128, 128, (7, 33), generator=g, dtype=torch.int8)
        b = torch.randint(-128, 128, (33, 5), generator=g, dtype=torch.int8)
        got = int_gemm(a, b, "reference")
        want = torch.zeros(7, 5, dtype=torch.int32)
        for i in range(7):
            for j in range(5):
                want[i, j] = int((a[i].to(I64) * b[:, j].to(I64)).sum())
        assert torch.equal(got, want)

    def test_u8_trick_exact(self):
        g = torch.Generator().manual_seed(1)
        p = torch.randint(0, 129, (9, 21), generator=g, dtype=torch.int32)
        v = torch.randint(-128, 128, (21, 6), generator=g, dtype=torch.int8)
        got = int_gemm_u8i8(p, v, "reference")
        want = p.to(torch.int32) @ v.to(torch.int32)
        assert torch.equal(got, want)


class TestDiMatmul:
    def test_value_accuracy(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randint(-(1 << 20), 1 << 20, (16, 256), generator=g, dtype=torch.int32)
        xm = torch.randint(1 << 15, 1 << 16, (16, 1), generator=g, dtype=I64)
        xk = torch.full((16, 1), 30, dtype=I64)
        w = torch.randint(-127, 128, (256, 64), generator=g, dtype=torch.int8)
        wm = torch.randint(1 << 15, 1 << 16, (64,), generator=g, dtype=I64)
        wk = 20
        y8, m, k = di_matmul(x, xm, xk, w, wm, wk, "reference")
        got = y8.double() * m.double() * torch.pow(2.0, -k.double())
        xv = x.double() * xm.double() / 2.0 ** 30
        wv = w.double() * wm.double() / 2.0 ** 20
        want = xv @ wv
        step = want.abs().max(dim=-1, keepdim=True).values / 127
        # two quantizations (input int8 + output int8) plus scale-fit error
        assert ((got - want).abs() / step.clamp(min=1e-12)).max().item() < 3.0


class TestDiExp:
    def _run(self, kf=24, m_val=52429):
        # value = x · m/2^k; x chosen so values sweep [-21, 0] finely
        m = torch.tensor([[m_val]], dtype=I64)
        k = torch.tensor([[kf]], dtype=I64)
        lo = -int(21 * 2.0 ** kf / m_val)
        x = torch.arange(lo, 1, 7, dtype=I64).reshape(1, -1)
        res, tpos = di_exp(x, m, k)
        got = res.double() / tpos.double()
        want = torch.exp(x.double() * m_val / 2.0 ** kf)
        return got, want

    def test_error_bound_frozen(self):
        got, want = self._run()
        # Error sources: 2^frac linear interpolation (≤6%), the paper's
        # 1.4375 ≈ log2(e) constant (0.36% low per unit of |x|), and output
        # resolution 1/tpos ≈ 2^-7.8 (dominates below e^-5.4, where results
        # are 0-1 ulp). FROZEN empirical bounds:
        sig = want > 0.05  # well above output resolution
        rel = ((got - want).abs() / want)[sig].max().item()
        assert rel < 0.11, rel
        assert (got - want).abs().max().item() < 0.05  # measured 0.0443

    def test_tail_underflows_to_zero(self):
        m = torch.tensor([[1 << 15]], dtype=I64)
        k = torch.tensor([[15]], dtype=I64)
        x = torch.tensor([[-50 << 15, -100 << 15]], dtype=I64)  # e^-50, e^-100
        res, _ = di_exp(x, m, k)
        assert torch.equal(res, torch.zeros_like(res))

    def test_zero_maps_to_one(self):
        m = torch.tensor([[1 << 15]], dtype=I64)
        k = torch.tensor([[15]], dtype=I64)
        res, tpos = di_exp(torch.zeros(1, 1, dtype=I64), m, k)
        assert res.item() == tpos.item()  # e^0 = 1.0 exactly in this encoding


class TestDiSoftmax:
    def _mk(self, seed, rows=8, T=64, scale=8.0):
        g = torch.Generator().manual_seed(seed)
        xf = (torch.rand(rows, T, generator=g, dtype=torch.float64) - 0.5) * 2 * scale
        k = 20
        m = torch.full((rows, 1), 1 << 15, dtype=I64)
        # x_int such that x_int * m/2^k ≈ xf  -> x_int = xf * 2^k / m
        xi = torch.round(xf * 2.0 ** k / (1 << 15)).to(I64)
        xf = xi.double() * (1 << 15) / 2.0 ** k  # exact represented values
        return xi, m, torch.full((rows, 1), k, dtype=I64), xf

    def test_matches_float_softmax(self):
        xi, m, k, xf = self._mk(3)
        valid = torch.ones(xi.shape, dtype=torch.bool)
        p = di_softmax(xi, m, k, valid)
        got = p.double() / 128.0
        want = torch.softmax(xf, dim=-1)
        assert (got - want).abs().max().item() < 0.04  # FROZEN (measured 0.027)

    def test_row_sum_property(self):
        xi, m, k, _ = self._mk(4)
        valid = torch.ones(xi.shape, dtype=torch.bool)
        p = di_softmax(xi, m, k, valid)
        sums = p.sum(dim=-1)
        assert bool(((sums - 128).abs() <= 32).all())  # rounding slack only

    def test_masked_positions_contribute_nothing(self):
        xi, m, k, xf = self._mk(5)
        T = xi.shape[-1]
        valid = torch.ones(xi.shape, dtype=torch.bool)
        valid[:, T // 2:] = False
        p_masked = di_softmax(xi, m, k, valid)
        assert torch.equal(p_masked[:, T // 2:], torch.zeros_like(p_masked[:, T // 2:]))
        # garbage in masked slots must not change valid outputs at all
        xi2 = xi.clone()
        xi2[:, T // 2:] = 999999
        p2 = di_softmax(xi2, m, k, valid)
        assert torch.equal(p_masked, p2)

    def test_all_masked_row_is_zero(self):
        xi, m, k, _ = self._mk(6)
        valid = torch.zeros(xi.shape, dtype=torch.bool)
        p = di_softmax(xi, m, k, valid)
        assert torch.equal(p, torch.zeros_like(p))

    def test_clip_ablation_close_at_c15(self):
        xi, m, k, xf = self._mk(7, scale=12.0)
        valid = torch.ones(xi.shape, dtype=torch.bool)
        p_inf = di_softmax(xi, m, k, valid)
        p_c15 = di_softmax(xi, m, k, valid, clip_c=15)
        assert (p_inf - p_c15).abs().max().item() <= 2  # 8-bit insensitivity


class TestDiRmsnorm:
    @pytest.mark.parametrize("n", [1024, 128])
    def test_matches_float(self, n):
        g = torch.Generator().manual_seed(8)
        x = torch.randint(-(1 << 27), 1 << 27, (32, n), generator=g, dtype=torch.int32)
        y = di_rmsnorm(x)
        got = y.double() / 2.0 ** OUT_FRAC_BITS
        xf = x.double()
        want = xf / torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True))
        assert (got - want).abs().max().item() < 1e-3  # FROZEN

    def test_scale_invariance_exact(self):
        # same values at different magnitudes must give identical output
        g = torch.Generator().manual_seed(9)
        x = torch.randint(-(1 << 20), 1 << 20, (4, 128), generator=g, dtype=torch.int32)
        y1 = di_rmsnorm(x)
        y2 = di_rmsnorm(x << 4)  # exact shift: same represented direction
        assert torch.equal(y1, y2)

    def test_zero_row(self):
        y = di_rmsnorm(torch.zeros(2, 128, dtype=torch.int32))
        assert torch.equal(y, torch.zeros(2, 128, dtype=torch.int32))

    def test_gamma_variant(self):
        g = torch.Generator().manual_seed(10)
        x = torch.randint(-(1 << 24), 1 << 24, (16, 128), generator=g, dtype=torch.int32)
        gamma = (torch.rand(128, generator=g, dtype=torch.float64) * 2 - 0.5)
        gk = 14
        gm = torch.round(gamma * 2 ** gk).to(I64)
        y, k_out = di_rmsnorm_gamma(x, gm, gk)
        got = y.double() / 2.0 ** k_out
        xf = x.double()
        gq = gm.double() / 2 ** gk  # compare against the quantized gamma
        want = xf / torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True)) * gq
        assert (got - want).abs().max().item() < 2e-3


class TestDiSwiglu:
    def test_matches_float(self):
        g = torch.Generator().manual_seed(11)
        rows, C = 16, 512
        gate = torch.randint(-(1 << 11), 1 << 11, (rows, C), generator=g, dtype=torch.int32)
        up = torch.randint(-(1 << 11), 1 << 11, (rows, C), generator=g, dtype=torch.int32)
        gm = torch.randint(1 << 15, 1 << 16, (rows, 1), generator=g, dtype=I64)
        um = torch.randint(1 << 15, 1 << 16, (rows, 1), generator=g, dtype=I64)
        gk = torch.full((rows, 1), 24, dtype=I64)  # gate values ~ ±4
        uk = torch.full((rows, 1), 24, dtype=I64)
        y, m, k = di_swiglu(gate, gm, gk, up, um, uk)
        got = y.double() * m.double() * torch.pow(2.0, -k.double())
        gv = gate.double() * gm.double() / 2.0 ** 24
        uv = up.double() * um.double() / 2.0 ** 24
        want = gv * torch.sigmoid(gv) * uv
        scale = want.abs().max().item()
        assert (got - want).abs().max().item() / scale < 0.02  # FROZEN

    def test_sigmoid_saturation(self):
        # gate value ±40 (scale 2^15/2^30 = 2^-15), up value 3:
        # σ(-40)≈0 -> y≈0;  σ(40)≈1 -> y ≈ 40·3 = 120
        gm = torch.full((1, 1), 1 << 15, dtype=I64)
        gk = torch.full((1, 1), 30, dtype=I64)
        gate = torch.tensor([[-40 << 15, 40 << 15]], dtype=torch.int32)
        up = torch.tensor([[3 << 15, 3 << 15]], dtype=torch.int32)
        y, m, k = di_swiglu(gate, gm, gk, up, gm.clone(), gk.clone())
        got = y.double() * m.double() * torch.pow(2.0, -k.double())
        assert abs(got[0, 0].item()) < 1e-6
        assert abs(got[0, 1].item() - 120.0) / 120.0 < 0.01

    def test_all_negative_row(self):
        # a row whose gate entries are all negative must not blow up
        gm = torch.full((1, 1), 1 << 15, dtype=I64)
        gk = torch.full((1, 1), 27, dtype=I64)
        gate = (torch.tensor([[-1, -2, -4, -8]], dtype=torch.int32)) << 12
        up = torch.tensor([[1000, 1000, 1000, 1000]], dtype=torch.int32)
        y, m, k = di_swiglu(gate, gm, gk, up, gm.clone(), gk.clone())
        got = y.double() * m.double() * torch.pow(2.0, -k.double())
        gv = gate.double() * (1 << 15) / 2.0 ** 27
        uv = up.double() * (1 << 15) / 2.0 ** 27
        want = gv * torch.sigmoid(gv) * uv
        assert (got - want).abs().max().item() < 0.05 * want.abs().max().item()


class TestIntRope:
    def test_matches_float_rope(self):
        d, T = 128, 64
        cos_t, sin_t = build_rope_tables(d, 1024, base=1e6)
        g = torch.Generator().manual_seed(12)
        x = torch.randint(-(1 << 20), 1 << 20, (2, T, d), generator=g, dtype=torch.int32)
        pos = torch.arange(T, dtype=I64) + 100
        y = apply_int_rope(x, cos_t, sin_t, pos)
        # float reference (HF convention)
        j = torch.arange(0, d // 2, dtype=torch.float64)
        inv = (1e6) ** (-2 * j / d)
        ang = torch.outer(pos.double(), inv)
        ang = torch.cat([ang, ang], dim=-1)
        c, s = torch.cos(ang), torch.sin(ang)
        xf = x.double()
        rot = torch.cat([-xf[..., d // 2:], xf[..., : d // 2]], dim=-1)
        want = xf * c + rot * s
        rel = (y.double() - want).abs().max().item() / xf.abs().max().item()
        assert rel < 2.0 ** -13  # table quantization ~2^-14 + rounding; FROZEN

    def test_long_position_no_degradation(self):
        d = 128
        cos_t, sin_t = build_rope_tables(d, 32768, base=1e6)
        g = torch.Generator().manual_seed(13)
        x = torch.randint(-(1 << 20), 1 << 20, (1, 4, d), generator=g, dtype=torch.int32)
        pos = torch.tensor([30000, 30001, 31000, 32767], dtype=I64)
        y = apply_int_rope(x, cos_t, sin_t, pos)
        j = torch.arange(0, d // 2, dtype=torch.float64)
        inv = (1e6) ** (-2 * j / d)
        ang = torch.cat([torch.outer(pos.double(), inv)] * 2, dim=-1)
        xf = x.double()
        rot = torch.cat([-xf[..., d // 2:], xf[..., : d // 2]], dim=-1)
        want = xf * torch.cos(ang) + rot * torch.sin(ang)
        rel = (y.double() - want).abs().max().item() / xf.abs().max().item()
        assert rel < 2.0 ** -13

    def test_norm_preservation(self):
        # rotation preserves pairwise magnitude to table precision
        d = 128
        cos_t, sin_t = build_rope_tables(d, 512, base=1e6)
        x = torch.full((1, 1, d), 1 << 16, dtype=torch.int32)
        y = apply_int_rope(x, cos_t, sin_t, torch.tensor([317], dtype=I64))
        n_in = x.double().pow(2).sum()
        n_out = y.double().pow(2).sum()
        assert abs(n_out / n_in - 1).item() < 1e-3

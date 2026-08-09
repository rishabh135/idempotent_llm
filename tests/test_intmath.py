"""Unit tests for the §5 integer primitives — including negative inputs and
boundary values. These freeze the canonical semantics; any change here is a
deliberate numerics change."""

import math

import pytest
import torch

from detllm import intmath as im


def t32(*vals):
    return torch.tensor(vals, dtype=torch.int32)


def t64(*vals):
    return torch.tensor(vals, dtype=torch.int64)


class TestFloorDiv:
    def test_python_semantics_scalar(self):
        assert im.floor_div(7, 2) == 3
        assert im.floor_div(-7, 2) == -4  # floor, NOT C truncation (-3)
        assert im.floor_div(-8, 2) == -4
        assert im.floor_div(-1, 4) == -1
        assert im.floor_div(0, 5) == 0

    def test_tensor_matches_scalar(self):
        a = torch.arange(-100, 100, dtype=torch.int32)
        for b in (1, 2, 3, 7, 128):
            got = im.floor_div(a, b)
            want = torch.tensor([x.item() // b for x in a], dtype=torch.int32)
            assert torch.equal(got, want)

    def test_rejects_float(self):
        with pytest.raises(TypeError):
            im.floor_div(torch.tensor([1.0]), 2)
        with pytest.raises(TypeError):
            im.floor_div(1.5, 2)


class TestRoundHalfAwayDiv:
    def test_ties_away_from_zero(self):
        assert im.round_half_away_div(3, 2) == 2
        assert im.round_half_away_div(-3, 2) == -2
        assert im.round_half_away_div(5, 2) == 3
        assert im.round_half_away_div(-5, 2) == -3

    def test_non_ties(self):
        assert im.round_half_away_div(5, 4) == 1
        assert im.round_half_away_div(-5, 4) == -1
        assert im.round_half_away_div(7, 4) == 2
        assert im.round_half_away_div(-7, 4) == -2
        assert im.round_half_away_div(0, 7) == 0

    def test_odd_divisor_no_ties(self):
        # b odd → exact halves impossible; equals round(a/b) for all a
        for a in range(-50, 51):
            for b in (3, 5, 7):
                want = int(math.floor(a / b + 0.5)) if (2 * abs(a)) % b != 0 else None
                got = im.round_half_away_div(a, b)
                ref = round(abs(a) / b)  # python round is half-even but no ties occur
                assert got == (-ref if a < 0 else ref)

    def test_tensor_matches_scalar(self):
        a = torch.arange(-1000, 1000, dtype=torch.int64)
        for b in (1, 2, 3, 8, 100, 127):
            got = im.round_half_away_div(a, b)
            want = torch.tensor(
                [im.round_half_away_div(int(x), b) for x in a], dtype=torch.int64
            )
            assert torch.equal(got, want)

    def test_rejects_nonpositive_divisor_scalar(self):
        with pytest.raises(ValueError):
            im.round_half_away_div(5, 0)
        with pytest.raises(ValueError):
            im.round_half_away_div(5, -2)


class TestAshr:
    def test_floor_convention(self):
        assert im.ashr(7, 1) == 3
        assert im.ashr(-7, 1) == -4  # arithmetic shift floors toward -inf
        assert im.ashr(-1, 5) == -1
        assert im.ashr(5, 0) == 5

    def test_tensor_negative(self):
        a = t32(-7, -8, -1, 0, 7, 8)
        assert torch.equal(im.ashr(a, 1), t32(-4, -4, -1, 0, 3, 4))
        # torch arithmetic shift must match floor division by 2**k
        a = torch.arange(-512, 512, dtype=torch.int32)
        for k in (0, 1, 3, 8):
            assert torch.equal(im.ashr(a, k), im.floor_div(a, 1 << k))

    def test_int64(self):
        a = t64(-(1 << 40) - 1, (1 << 40) + 1)
        assert torch.equal(im.ashr(a, 40), t64(-2, 1))

    def test_rejects_negative_shift(self):
        with pytest.raises(ValueError):
            im.ashr(5, -1)


class TestRshiftRound:
    def test_formula(self):
        # (a + (1 << (k-1))) >> k, ties toward +inf
        assert im.rshift_round(3, 1) == 2   # 1.5 -> 2
        assert im.rshift_round(-3, 1) == -1  # -1.5 -> -1 (half-up)
        assert im.rshift_round(5, 2) == 1   # 1.25 -> 1
        assert im.rshift_round(6, 2) == 2   # 1.5 -> 2
        assert im.rshift_round(-6, 2) == -1  # -1.5 -> -1

    def test_k_zero_identity(self):
        assert im.rshift_round(7, 0) == 7
        a = t32(-3, 0, 3)
        assert torch.equal(im.rshift_round(a, 0), a)

    def test_tensor_matches_scalar(self):
        a = torch.arange(-1000, 1000, dtype=torch.int64)
        for k in (1, 2, 7, 15):
            got = im.rshift_round(a, k)
            want = torch.tensor([im.rshift_round(int(x), k) for x in a], dtype=torch.int64)
            assert torch.equal(got, want)

    def test_tensor_shift_amounts(self):
        a = t64(-1000, -6, -3, 0, 3, 6, 1000)
        k = t64(3, 2, 1, 0, 1, 2, 3)
        got = im.rshift_round_t(a, k)
        want = t64(*[im.rshift_round(int(x), int(s)) for x, s in zip(a, k)])
        assert torch.equal(got, want)


class TestClamps:
    def test_clamp_i8(self):
        a = t32(-1000, -129, -128, -1, 0, 126, 127, 128, 1000)
        got = im.clamp_i8(a)
        assert got.dtype == torch.int8
        assert got.tolist() == [-128, -128, -128, -1, 0, 126, 127, 127, 127]

    def test_clamp_u8(self):
        a = t32(-5, 0, 1, 255, 256, 1000)
        got = im.clamp_u8(a)
        assert got.dtype == torch.int32  # stays wide by design
        assert got.tolist() == [0, 0, 1, 255, 255, 255]


class TestIlog2Floor:
    def test_scalar(self):
        assert im.ilog2_floor(1) == 0
        assert im.ilog2_floor(2) == 1
        assert im.ilog2_floor(3) == 1
        assert im.ilog2_floor(4) == 2
        assert im.ilog2_floor((1 << 31) - 1) == 30
        assert im.ilog2_floor(1 << 31) == 31
        assert im.ilog2_floor((1 << 62) + 5) == 62

    def test_scalar_rejects_zero(self):
        with pytest.raises(ValueError):
            im.ilog2_floor(0)

    def test_tensor_exhaustive_small(self):
        a = torch.arange(1, 1 << 16, dtype=torch.int64)
        got = im.ilog2_floor(a)
        want = torch.tensor([x.bit_length() - 1 for x in range(1, 1 << 16)], dtype=torch.int64)
        assert torch.equal(got, want)

    def test_tensor_large(self):
        vals = [1, (1 << 30) - 1, 1 << 30, (1 << 31) - 1, 1 << 40, (1 << 62) + 123]
        got = im.ilog2_floor(t64(*vals))
        want = t64(*[v.bit_length() - 1 for v in vals])
        assert torch.equal(got, want)


class TestIsqrt:
    def test_scalar_exhaustive_small(self):
        for n in range(0, 70000):
            assert im.isqrt(n) == math.isqrt(n)

    def test_scalar_boundaries(self):
        for n in [0, 1, 2, 3, 4, (1 << 31) - 1, 1 << 31, (1 << 52) + 7, (1 << 62) - 1]:
            assert im.isqrt(n) == math.isqrt(n)

    def test_tensor_matches_math(self):
        g = torch.Generator().manual_seed(0)
        vals = torch.randint(0, 1 << 62, (10000,), generator=g, dtype=torch.int64)
        vals = torch.cat([vals, t64(0, 1, 2, 3, 4, (1 << 62) - 1)])
        got = im.isqrt(vals)
        want = t64(*[math.isqrt(int(v)) for v in vals])
        assert torch.equal(got, want)

    def test_perfect_squares(self):
        r = torch.arange(0, 100000, 977, dtype=torch.int64)
        assert torch.equal(im.isqrt(r * r), r)
        r1 = r[r >= 1]  # for r >= 1, (r+1)^2 > r^2 + 1, so floor holds
        assert torch.equal(im.isqrt(r1 * r1 + 1), r1)


class TestCudaPrims:
    """The CUDA Triton kernels behind detllm::isqrt_i64 / ilog2_i64 must be
    bit-identical to the CPU tensor-loop implementations (which ARE the
    original algorithms) over boundaries and random int64 sweeps."""

    def _skip(self):
        import pytest as _pytest
        if not torch.cuda.is_available():
            _pytest.skip("no CUDA")
        from detllm.ops.int_prims import HAVE_TRITON
        if not HAVE_TRITON:
            _pytest.skip("no triton")

    def test_isqrt_cuda_matches_cpu(self):
        self._skip()
        g = torch.Generator().manual_seed(0)
        vals = torch.cat([
            torch.randint(0, 1 << 62, (200000,), generator=g, dtype=torch.int64),
            torch.arange(0, 70000, dtype=torch.int64),
            torch.tensor([0, 1, 2, 3, 4, (1 << 31) - 1, 1 << 31,
                          (1 << 52) + 7, (1 << 62) - 1], dtype=torch.int64),
        ])
        got = im.isqrt(vals.cuda()).cpu()
        want = im.isqrt(vals)
        assert torch.equal(got, want)

    def test_ilog2_cuda_matches_cpu(self):
        self._skip()
        g = torch.Generator().manual_seed(1)
        vals = torch.cat([
            torch.randint(1, 1 << 62, (200000,), generator=g, dtype=torch.int64),
            torch.arange(1, 1 << 16, dtype=torch.int64),
            torch.tensor([1, 2, 3, (1 << 62) + 123, (1 << 63) - 1],
                         dtype=torch.int64),
        ])
        got = im.ilog2_floor(vals.cuda()).cpu()
        want = im.ilog2_floor(vals)
        assert torch.equal(got, want)

    def test_nonneg_random_shapes(self):
        self._skip()
        g = torch.Generator().manual_seed(2)
        x = torch.randint(0, 1 << 50, (17, 33, 5), generator=g, dtype=torch.int64)
        assert torch.equal(im.isqrt(x.cuda()).cpu(), im.isqrt(x))
        y = x.clamp(min=1)
        assert torch.equal(im.ilog2_floor(y.cuda()).cpu(), im.ilog2_floor(y))

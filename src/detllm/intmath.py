"""Canonical integer arithmetic primitives for the deterministic integer-only pipeline.

This module defines THE single semantics for all integer arithmetic on the
numerical path. Every operator must use these helpers; no raw ``//``, ``>>``
or ``torch.round`` on the numerical path elsewhere.

Rounding policy (frozen after milestone 1)
------------------------------------------
* ``floor_div(a, b)``: floor division, Python semantics (rounds toward -inf).
* ``round_half_away_div(a, b)``: round-to-nearest with ties away from zero.
  This is the global rule for the paper's round-to-nearest division.
* ``rshift_round(a, k)``: ``(a + (1 << (k-1))) >> k`` (spec-fixed formula).
  NOTE: this is round-to-nearest with ties toward +inf ("half up"), which
  differs from ``round_half_away_div`` only on negative exact-half inputs.
  Both are fully deterministic. Use ``rshift_round`` for all power-of-two
  rescales (the common path); use ``round_half_away_div`` only where a
  non-power-of-two divisor appears (e.g. softmax normalization).

All functions accept either Python ints or torch integer tensors
(int32/int64; int16/int8 accepted where sensible). Float dtypes are rejected.
"""

from __future__ import annotations

from typing import Union

import torch

# Debug-mode input validation. The checks below that inspect tensor VALUES
# (not just dtypes/shapes) force a GPU->CPU sync on every call, which
# dominates eager decode latency. They are enabled by default off the hot
# path and can be turned on when hunting numerics bugs.
DEBUG_CHECKS = False

Tensor = torch.Tensor
IntLike = Union[int, Tensor]

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _check_int(a: IntLike, name: str = "input") -> None:
    if isinstance(a, Tensor):
        if a.dtype not in _INT_DTYPES:
            raise TypeError(f"{name} must be an integer tensor, got {a.dtype}")
    elif not isinstance(a, int) or isinstance(a, bool):
        raise TypeError(f"{name} must be int or integer tensor, got {type(a)}")


def floor_div(a: IntLike, b: IntLike) -> IntLike:
    """Floor division (Python semantics: rounds toward -inf).

    floor_div(-7, 2) == -4, floor_div(7, 2) == 3, floor_div(-8, 2) == -4.
    b must be nonzero (positive on the numerical path).
    """
    _check_int(a, "a")
    _check_int(b, "b")
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        return torch.div(a, b, rounding_mode="floor") if isinstance(a, Tensor) else a // b
    return a // b


def round_half_away_div(a: IntLike, b: IntLike) -> IntLike:
    """Round-to-nearest division with ties away from zero. Requires b > 0.

    round_half_away_div(3, 2) == 2, round_half_away_div(-3, 2) == -2,
    round_half_away_div(5, 4) == 1, round_half_away_div(-5, 4) == -1.
    """
    _check_int(a, "a")
    _check_int(b, "b")
    if isinstance(a, Tensor):
        sign = torch.where(a < 0, -1, 1).to(a.dtype)
        mag = a.abs()
        half = floor_div(b, 2) if isinstance(b, Tensor) else b // 2
        return sign * torch.div(mag + half, b, rounding_mode="floor")
    if isinstance(b, Tensor):
        raise TypeError("scalar a with tensor b is not supported")
    if b <= 0:
        raise ValueError("b must be > 0")
    sign = -1 if a < 0 else 1
    return sign * ((abs(a) + b // 2) // b)


def ashr(a: IntLike, k: int) -> IntLike:
    """Arithmetic right shift by k >= 0 (floor rounding: equals floor(a / 2**k)).

    ashr(-7, 1) == -4 (floors toward -inf), ashr(7, 1) == 3.
    """
    _check_int(a, "a")
    if not isinstance(k, int) or k < 0:
        raise ValueError(f"shift k must be a non-negative int, got {k!r}")
    if isinstance(a, Tensor):
        return torch.bitwise_right_shift(a, k)
    return a >> k


def ashr_t(a: Tensor, k: Tensor) -> Tensor:
    """Elementwise arithmetic right shift by a tensor of shift amounts (all >= 0)."""
    _check_int(a, "a")
    _check_int(k, "k")
    return torch.bitwise_right_shift(a, k.to(a.dtype))


def lshift(a: IntLike, k: int) -> IntLike:
    """Left shift by k >= 0. Caller is responsible for overflow headroom."""
    _check_int(a, "a")
    if not isinstance(k, int) or k < 0:
        raise ValueError(f"shift k must be a non-negative int, got {k!r}")
    if isinstance(a, Tensor):
        return torch.bitwise_left_shift(a, k)
    return a << k


def lshift_t(a: Tensor, k: Tensor) -> Tensor:
    """Elementwise left shift by a tensor of shift amounts (all >= 0)."""
    _check_int(a, "a")
    _check_int(k, "k")
    return torch.bitwise_left_shift(a, k.to(a.dtype))


def rshift_round(a: IntLike, k: int) -> IntLike:
    """Right shift with round-to-nearest, ties toward +inf: (a + (1 << (k-1))) >> k.

    k == 0 returns a unchanged. Caller must ensure a + 2**(k-1) does not
    overflow the dtype (use int64 when in doubt).
    """
    _check_int(a, "a")
    if not isinstance(k, int) or k < 0:
        raise ValueError(f"shift k must be a non-negative int, got {k!r}")
    if k == 0:
        return a
    return ashr(a + (1 << (k - 1)), k)


def rshift_round_t(a: Tensor, k: Tensor) -> Tensor:
    """Elementwise rshift_round with a tensor of shift amounts (all >= 0).

    Handles k == 0 entries exactly (returns a unchanged there).
    """
    _check_int(a, "a")
    _check_int(k, "k")
    k = k.to(a.dtype)
    if DEBUG_CHECKS and bool((k < 0).any()):
        raise ValueError("all shift amounts must be >= 0")
    kpos = torch.clamp(k, min=1)
    half = torch.bitwise_left_shift(torch.ones_like(a), kpos - 1)
    rounded = torch.bitwise_right_shift(a + half, kpos)
    return torch.where(k == 0, a, rounded)


def clamp_i8(a: Tensor) -> Tensor:
    """Saturating cast to int8: clamp to [-128, 127] then cast."""
    _check_int(a, "a")
    return torch.clamp(a, -128, 127).to(torch.int8)


def clamp_u8(a: Tensor) -> Tensor:
    """Saturating cast to uint8 range [0, 255]. Returned dtype is the input
    dtype (not torch.uint8) so downstream integer math stays in int32/int64;
    values are guaranteed within [0, 255]."""
    _check_int(a, "a")
    return torch.clamp(a, 0, 255)


def ilog2_floor(a: IntLike) -> IntLike:
    """floor(log2(a)) for a >= 1 (MSB position). ilog2_floor(1) == 0.

    For tensors, entries must all be >= 1.
    """
    _check_int(a, "a")
    if isinstance(a, Tensor):
        v = a.to(torch.int64)
        r = torch.zeros_like(v)
        for s in (32, 16, 8, 4, 2, 1):
            hit = v >= (1 << s)
            r = r + hit.to(torch.int64) * s
            v = torch.bitwise_right_shift(v, hit.to(torch.int64) * s)
        return r
    if a < 1:
        raise ValueError("ilog2_floor requires a >= 1")
    return a.bit_length() - 1


def isqrt(a: IntLike) -> IntLike:
    """Integer square root floor(sqrt(a)) for a >= 0, bit-wise check method
    (paper Algorithm 4's I-SQRT). Valid for the full non-negative int64 range.
    """
    _check_int(a, "a")
    if isinstance(a, Tensor):
        n = a.to(torch.int64)
        rem = n.clone()
        c = torch.zeros_like(n)
        d = 1 << 62
        for _ in range(32):
            t = c + d
            ge = rem >= t
            rem = torch.where(ge, rem - t, rem)
            c = torch.bitwise_right_shift(c, 1) + torch.where(ge, torch.full_like(c, d), torch.zeros_like(c))
            d >>= 2
        return c
    if a < 0:
        raise ValueError("isqrt requires a >= 0")
    x, c, d = a, 0, 1 << 62
    while d > a:
        d >>= 2
    while d != 0:
        if x >= c + d:
            x -= c + d
            c = (c >> 1) + d
        else:
            c >>= 1
        d >>= 2
    return c

"""Tests for llamacpp-quant-lab/gguf_report.py's parsing/classification/diff
logic against tiny SYNTHETIC GGUF files (built with gguf.GGUFWriter) — no
network, no llama.cpp build, no real weights. Mirrors the "smoke-test against
a synthetic artifact first" philosophy scripts/quant_report.py itself used
before its real download."""

import sys

import gguf
import numpy as np
import pytest

sys.path.insert(0, "llamacpp-quant-lab")
from gguf_report import (  # noqa: E402
    ASYMMETRY_ALLOWED_SUFFIXES,
    classify,
    diff_tensors,
    load_tensor_rows,
    report_compare,
)

Q4_K = gguf.GGMLQuantizationType.Q4_K
Q6_K = gguf.GGMLQuantizationType.Q6_K
F32 = gguf.GGMLQuantizationType.F32


def _raw_bytes(quant_type, n_blocks=1):
    """Raw uint8 byte array whose shape is exactly (type_size * n_blocks,) —
    GGUFWriter derives the LOGICAL element shape from this byte-array shape
    via quant_shape_from_byte_shape(), so no separate raw_shape override is
    needed."""
    _, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    return np.zeros(type_size * n_blocks, dtype=np.uint8)


def _write_gguf(path, block_types: dict):
    """block_types: {tensor_name: GGMLQuantizationType}. Writes a tiny
    2-layer fake model with those exact types (plus a fixed F32 embedding/
    norm/output so every file has a recognizable non-layer tensor too)."""
    w = gguf.GGUFWriter(str(path), arch="qwen3-test")
    w.add_uint32("general.file_type", 1)
    w.add_tensor("token_embd.weight", _raw_bytes(F32), raw_dtype=F32)
    for name, qtype in block_types.items():
        w.add_tensor(name, _raw_bytes(qtype), raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture()
def uniform_gguf(tmp_path):
    """Every blk.*.attn_v/ffn_down at Q4_K (a Q4_K_S-style file)."""
    path = tmp_path / "uniform.gguf"
    _write_gguf(path, {
        "blk.0.attn_v.weight": Q4_K, "blk.0.ffn_down.weight": Q4_K,
        "blk.0.attn_q.weight": Q4_K,
        "blk.1.attn_v.weight": Q4_K, "blk.1.ffn_down.weight": Q4_K,
        "blk.1.attn_q.weight": Q4_K,
    })
    return path


@pytest.fixture()
def mixed_gguf(tmp_path):
    """attn_v/ffn_down bumped to Q6_K in layer 0 only (a Q4_K_M-style file)."""
    path = tmp_path / "mixed.gguf"
    _write_gguf(path, {
        "blk.0.attn_v.weight": Q6_K, "blk.0.ffn_down.weight": Q6_K,
        "blk.0.attn_q.weight": Q4_K,
        "blk.1.attn_v.weight": Q4_K, "blk.1.ffn_down.weight": Q4_K,
        "blk.1.attn_q.weight": Q4_K,
    })
    return path


@pytest.fixture()
def bad_asymmetry_gguf(tmp_path):
    """attn_q (NOT in the allowed asymmetry set) also changed type."""
    path = tmp_path / "bad.gguf"
    _write_gguf(path, {
        "blk.0.attn_v.weight": Q6_K, "blk.0.ffn_down.weight": Q6_K,
        "blk.0.attn_q.weight": Q6_K,
        "blk.1.attn_v.weight": Q4_K, "blk.1.ffn_down.weight": Q4_K,
        "blk.1.attn_q.weight": Q4_K,
    })
    return path


class TestClassify:
    def test_layer_tensor(self):
        assert classify("blk.5.attn_v.weight") == (5, "attn_v.weight")

    def test_non_layer_tensor(self):
        assert classify("token_embd.weight") == (None, "token_embd.weight")

    def test_unrecognized_raises(self):
        with pytest.raises(ValueError):
            classify("some.unknown.tensor")


class TestLoadTensorRows:
    def test_uniform_file_types(self, uniform_gguf):
        _, rows = load_tensor_rows(str(uniform_gguf))
        v_types = {r["dtype"] for r in rows if r["suffix"] == "attn_v.weight"}
        assert v_types == {"Q4_K"}

    def test_mixed_file_types(self, mixed_gguf):
        _, rows = load_tensor_rows(str(mixed_gguf))
        by_layer = {r["li"]: r["dtype"] for r in rows if r["suffix"] == "attn_v.weight"}
        assert by_layer == {0: "Q6_K", 1: "Q4_K"}


class TestDiffAndAssertion:
    def test_diff_finds_only_promoted_tensors(self, uniform_gguf, mixed_gguf):
        _, rows_a = load_tensor_rows(str(uniform_gguf))
        _, rows_b = load_tensor_rows(str(mixed_gguf))
        diffs = diff_tensors(rows_a, rows_b)
        changed = {d["name"] for d in diffs if d["changed"]}
        assert changed == {"blk.0.attn_v.weight", "blk.0.ffn_down.weight"}

    def test_assert_known_asymmetry_passes(self, uniform_gguf, mixed_gguf):
        rc = report_compare(str(uniform_gguf), str(mixed_gguf),
                            assert_known_asymmetry=True, out=open("/dev/null", "w"))
        assert rc == 0

    def test_assert_known_asymmetry_fails_on_offender(self, uniform_gguf, bad_asymmetry_gguf):
        rc = report_compare(str(uniform_gguf), str(bad_asymmetry_gguf),
                            assert_known_asymmetry=True, out=open("/dev/null", "w"))
        assert rc == 1

    def test_allowed_suffix_set(self):
        assert ASYMMETRY_ALLOWED_SUFFIXES == {"attn_v.weight", "ffn_down.weight"}

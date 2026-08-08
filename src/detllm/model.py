"""IntQwen3 — the integer-only runtime (§8).

Loads ONLY the prepared artifact (int tensors + scalar constants); every
tensor on the numerical path is integer. The only backend-divergent code is
`backends.int_gemm`. All bookkeeping follows the (data, m, k) dyadic-scale
convention from `dyadic.py`.

Batching/masking model (chosen for §9.3 invariance):
- RIGHT padding; a bool `valid` flag per cache slot. Masked slots are
  excluded from softmax max/sum (integer sentinel) and contribute exact
  zeros to probs·V (0-probability × anything = 0 in integer GEMM), so a
  sequence's outputs are bit-identical regardless of co-batched content.
- Positions are explicit per token; cache slot order == position order for
  every valid token, so slot-triangular masking implements causality.
- K/V are quantized ONCE (static per-(layer, kv-head) scales from the
  artifact) when produced, then only read — prefill and decode see the same
  ints by construction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch

from . import dyadic as dy
from .backends import int_gemm, int_gemm_u16i8
from .guard import NoFloatMode
from .intmath import clamp_i8, lshift_t, round_half_away_div
from .ops.di_rmsnorm import OUT_FRAC_BITS, di_rmsnorm, di_rmsnorm_gamma
from .ops.di_softmax import P_OUT_BITS, di_softmax
from .ops.di_swiglu import di_swiglu
from .ops.rope import apply_int_rope

I64 = torch.int64


@dataclass
class LayerWeights:
    q_w8t: torch.Tensor; q_m: torch.Tensor; q_k: int
    k_w8t: torch.Tensor; k_m: torch.Tensor; k_k: int
    v_w8t: torch.Tensor; v_m: torch.Tensor; v_k: int
    o_w8t: torch.Tensor; o_m: torch.Tensor; o_k: int
    gate_w8t: torch.Tensor; gate_m: torch.Tensor; gate_k: int
    up_w8t: torch.Tensor; up_m: torch.Tensor; up_k: int
    down_w8t: torch.Tensor; down_m: torch.Tensor; down_k: int
    q_norm_m: torch.Tensor; q_norm_k: int
    k_norm_m: torch.Tensor; k_norm_k: int
    qs_m: torch.Tensor = None; qs_k: int = 0  # QK smoothing: Q-side dyadic c
    as_m: torch.Tensor = None; as_k: int = 0  # act smoothing: attn-in (1/s)
    ms_m: torch.Tensor = None; ms_k: int = 0  # act smoothing: mlp-in (1/s)
    ds_m: torch.Tensor = None; ds_k: int = 0  # act smoothing: down-in (1/s)
    os_m: torch.Tensor = None; os_k: int = 0  # act smoothing: o-in (1/s)
    kq_m: torch.Tensor = None; kq_k: torch.Tensor = None  # K cache scale [n_kv, hd] (c folded)
    ks_m: torch.Tensor = None; ks_k: torch.Tensor = None  # K score scale (1/√d folded)
    vq_m: torch.Tensor = None; vq_k: torch.Tensor = None  # V cache quant scale
    # per-element column scale for merged attention output (precomputed ints)
    attn_out_col_m: torch.Tensor = None
    attn_out_col_k: int = 0


@dataclass
class KVCache:
    """Preallocated int8 K/V cache. `length` is the filled slot count;
    buffers grow geometrically (contents of filled slots never move, so
    growth cannot affect results)."""
    k8: list = field(default_factory=list)     # per layer [B, n_kv, CAP, hd] int8
    v8: list = field(default_factory=list)
    valid: torch.Tensor | None = None          # [B, S] bool (S == length)
    length: int = 0
    # CUDA-graph static mode: fixed capacity, device-tensor write offset.
    static_cap: int = 0
    len_idx: torch.Tensor | None = None        # [1] int64, device

    def seq_len(self) -> int:
        return self.length

    def append(self, li: int, k8: torch.Tensor, v8: torch.Tensor):
        if self.static_cap:
            # graph-capturable write at a device-tensor offset
            self.k8[li].index_copy_(2, self.len_idx, k8)
            self.v8[li].index_copy_(2, self.len_idx, v8)
            return
        B, n_kv, T, hd = k8.shape
        if len(self.k8) <= li:
            cap = max(256, T)
            self.k8.append(k8.new_zeros(B, n_kv, cap, hd))
            self.v8.append(v8.new_zeros(B, n_kv, cap, hd))
            self.k8[li][:, :, :T] = k8
            self.v8[li][:, :, :T] = v8
            return
        cap = self.k8[li].shape[2]
        need = self.length + T
        if need > cap:
            new_cap = max(need, cap * 2)
            for buf in (self.k8, self.v8):
                nb = buf[li].new_zeros(B, n_kv, new_cap, hd)
                nb[:, :, :self.length] = buf[li][:, :, :self.length]
                buf[li] = nb
        self.k8[li][:, :, self.length:need] = k8
        self.v8[li][:, :, self.length:need] = v8


class IntQwen3:
    def __init__(self, artifact_dir: str, backend: str = "reference"):
        from safetensors.torch import load_file

        assert backend in ("reference", "cuda")
        self.backend = backend
        self.device = torch.device("cpu" if backend == "reference" else "cuda")
        with open(os.path.join(artifact_dir, "config.json")) as f:
            self.cfg = json.load(f)
        tensors = load_file(os.path.join(artifact_dir, "model.safetensors"))
        for name, t in tensors.items():
            assert t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), \
                f"float tensor in artifact: {name} {t.dtype}"
        dev = self.device
        g = lambda n: tensors[n].to(dev)

        c = self.cfg
        self.n_layers = c["n_layers"]; self.hidden = c["hidden"]
        self.n_q = c["n_q_heads"]; self.n_kv = c["n_kv_heads"]; self.hd = c["head_dim"]
        self.vocab = c["vocab"]
        assert self.hd * 127 * 127 < 2 ** 31 and c["intermediate"] * 127 * 127 < 2 ** 31

        self.embed_e8 = g("embed.e8"); self.embed_m = g("embed.m"); self.embed_k = g("embed.k")
        self.head_w8t = g("head.w8t"); self.head_m = g("head.m"); self.head_k = c["head"]["k"]
        self.rope_cos = g("rope.cos"); self.rope_sin = g("rope.sin")

        self.layers: list[LayerWeights] = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            Lc = c["layers"][i]
            lw = LayerWeights(
                q_w8t=g(p + "q.w8t"), q_m=g(p + "q.m"), q_k=Lc["q"]["k"],
                k_w8t=g(p + "k.w8t"), k_m=g(p + "k.m"), k_k=Lc["k"]["k"],
                v_w8t=g(p + "v.w8t"), v_m=g(p + "v.m"), v_k=Lc["v"]["k"],
                o_w8t=g(p + "o.w8t"), o_m=g(p + "o.m"), o_k=Lc["o"]["k"],
                gate_w8t=g(p + "gate.w8t"), gate_m=g(p + "gate.m"), gate_k=Lc["gate"]["k"],
                up_w8t=g(p + "up.w8t"), up_m=g(p + "up.m"), up_k=Lc["up"]["k"],
                down_w8t=g(p + "down.w8t"), down_m=g(p + "down.m"), down_k=Lc["down"]["k"],
                q_norm_m=g(p + "q_norm.m"), q_norm_k=Lc["q_norm_k"],
                k_norm_m=g(p + "k_norm.m"), k_norm_k=Lc["k_norm_k"],
                qs_m=g(p + "qs_m"), qs_k=Lc["qs_k"],
                as_m=g(p + "as_m"), as_k=Lc["as_k"],
                ms_m=g(p + "ms_m"), ms_k=Lc["ms_k"],
                ds_m=g(p + "ds_m"), ds_k=Lc["ds_k"],
                os_m=g(p + "os_m"), os_k=Lc["os_k"],
                kq_m=g(p + "kq_m"), kq_k=g(p + "kq_k"),
                ks_m=g(p + "ks_m"), ks_k=g(p + "ks_k"),
                vq_m=g(p + "vq_m"), vq_k=g(p + "vq_k"),
            )
            # attention-output merged col scale: per-head V scales aligned to a
            # shared k (integer shifts of artifact constants at load time)
            vk_shared = int(lw.vq_k.max())
            aligned = lshift_t(lw.vq_m.to(I64), (vk_shared - lw.vq_k).to(I64))
            group = self.n_q // self.n_kv
            # q head h uses kv head h // group; channel c = head*hd + d
            per_q_head = aligned.repeat_interleave(group)          # [n_q]
            lw.attn_out_col_m = per_q_head.repeat_interleave(self.hd)  # [n_q*hd]
            lw.attn_out_col_k = vk_shared + (P_OUT_BITS - 1)  # probs k
            self.layers.append(lw)
        self._layer_compiled = None  # set by enable_layer_cudagraphs()
        # device scalar constants (CUDA-graph capture forbids H2D copies, so
        # nothing on the forward path may build tensors from python ints)
        self._one1 = torch.ones(1, dtype=I64, device=dev)
        self._k_rowk = [torch.tensor(OUT_FRAC_BITS + Lc2["k_norm_k"], dtype=I64,
                                     device=dev)
                        for Lc2 in c["layers"]]

    # -- helpers ----------------------------------------------------------

    def _norm_quant(self, h: torch.Tensor, k_res: torch.Tensor,
                    sm_m: torch.Tensor | None = None, sm_k: int = 0):
        """RMSNorm (γ folded into consumer weights), optional per-channel
        activation-smoothing multiply (dyadic 1/s; matching s folded into the
        consumer weights offline), then per-token int8 quant."""
        y = di_rmsnorm(h)  # fixed scale (1, OUT_FRAC_BITS); k_res cancels
        kq = torch.full_like(k_res, OUT_FRAC_BITS)
        if sm_m is not None:
            y = y.to(I64) * sm_m
            kq = kq + sm_k
        one = torch.ones_like(k_res)
        return dy.quant_i8_pertoken(y, one, kq)

    def _qk_head_pipeline(self, P, row_m, row_k, w_m, w_k, gamma_m, gamma_k,
                          n_heads, positions):
        """Shared Q/K path: per-(token, head) requant → QK-Norm(γ) → RoPE.

        P: [B, T, n_heads*hd] int32 GEMM output.
        Returns int64 [B, n_heads, T, hd] with fixed scale (1, F+gamma_k).
        """
        B, T, _ = P.shape
        Ph = P.view(B, T, n_heads, self.hd)
        cm = w_m.view(n_heads, self.hd)
        y, m, k = dy.requant_i32_common(Ph, row_m.unsqueeze(-1), row_k.unsqueeze(-1),
                                        cm, w_k, target_bits=16)
        yn, k_out = di_rmsnorm_gamma(y, gamma_m, gamma_k)  # scale (1, F+γk), m/k dropped
        yn = yn.permute(0, 2, 1, 3)  # [B, H, T, hd]
        pos = positions[:, None, :].expand(B, n_heads, T)
        yr = apply_int_rope(yn, self.rope_cos, self.rope_sin, pos)
        return yr, k_out

    # -- forward ----------------------------------------------------------

    def forward(self, ids: torch.Tensor, positions: torch.Tensor,
                cache: KVCache | None = None, chunk_valid: torch.Tensor | None = None,
                guard: bool = False):
        """ids, positions: int64 [B, T]; chunk_valid: bool [B, T] (True=real).

        Returns (logits int32 [B, T, vocab], logit_row_m, logit_row_k, cache).
        Logit value ≈ logits · row_m·head_m[v] / 2^(row_k+head_k).
        """
        if guard:
            with NoFloatMode():
                return self._forward(ids, positions, cache, chunk_valid)
        return self._forward(ids, positions, cache, chunk_valid)

    def _forward(self, ids, positions, cache, chunk_valid):
        dev = self.device
        ids = ids.to(dev); positions = positions.to(dev)
        B, T = ids.shape
        if chunk_valid is None:
            chunk_valid = torch.ones(B, T, dtype=torch.bool, device=dev)
        chunk_valid = chunk_valid.to(dev)
        if cache is None:
            cache = KVCache()
        static = cache.static_cap > 0
        S_past = cache.seq_len()

        # embedding lookup → residual grid
        e8 = self.embed_e8[ids]                       # [B, T, H] int8
        em = self.embed_m[ids].unsqueeze(-1)          # [B, T, 1]
        ek = self.embed_k[ids].unsqueeze(-1)
        h, k_res = dy.residual_from_delta(e8.to(torch.int32), em, ek)

        if static:
            # CUDA-graph decode: attention runs over the full static capacity;
            # the valid buffer masks unwritten slots (exact zeros), and for
            # T == 1 causality is implied (only past+self slots are valid).
            assert T == 1
            S = cache.static_cap
            mask = cache.valid[:, None, None, :]      # [B, 1, 1, cap]
        else:
            valid_all = chunk_valid if cache.valid is None else torch.cat(
                [cache.valid, chunk_valid], dim=1)    # [B, S]
            cache.valid = valid_all
            S = valid_all.shape[1]
            # slot-causal mask: chunk query i sees slots ≤ S_past+i
            slot = torch.arange(S, device=dev)
            causal = slot[None, :] <= (S_past + torch.arange(T, device=dev))[:, None]
            mask = valid_all[:, None, None, :] & causal[None, None, :, :]  # [B,1,T,S]

        layer_fn = self._layer_compiled if (static and self._layer_compiled
                                            is not None) else self._layer
        for li, L in enumerate(self.layers):
            h, k_res = layer_fn(li, L, h, k_res, mask, positions, cache)

        if not static:
            cache.length = S

        # final norm + LM head
        x8, xm, xk = self._norm_quant(h, k_res)
        logits = self._lm_head(x8, xm, xk)  # [B, T, vocab] int32
        return logits, xm, xk, cache

    def _layer(self, li, L, h, k_res, mask, positions, cache):
        """One transformer layer (attention + MLP), residual in/out.

        Pure tensor computation apart from the cache K/V write (index_copy_
        in static mode). Compiled per-layer in static decode mode."""
        dev = self.device
        B, T = h.shape[0], h.shape[1]
        S = mask.shape[-1]
        group = self.n_q // self.n_kv
        if True:
            # ---- attention ----
            x8, xm, xk = self._norm_quant(h, k_res, L.as_m, L.as_k)
            Pq = int_gemm(x8, L.q_w8t, self.backend)
            Pk = int_gemm(x8, L.k_w8t, self.backend)
            Pv = int_gemm(x8, L.v_w8t, self.backend)

            q, qkk = self._qk_head_pipeline(Pq, xm, xk, L.q_m, L.q_k,
                                            L.q_norm_m, L.q_norm_k, self.n_q, positions)
            kk, kkk = self._qk_head_pipeline(Pk, xm, xk, L.k_m, L.k_k,
                                             L.k_norm_m, L.k_norm_k, self.n_kv, positions)

            aux_q = q  # pre-smoothing Q for diagnostics
            # QK smoothing: multiply Q by the per-channel dyadic c (exact
            # int64 multiply; c is folded inversely into K's static scale, so
            # it cancels identically in the score dot product)
            qs = L.qs_m.view(self.n_kv, self.hd).repeat_interleave(
                group, dim=0)                          # [n_q, hd]
            q = q * qs[None, :, None, :]
            qkk = qkk + L.qs_k
            # Q: dynamic per-(token, head) int8
            one = torch.ones(B, self.n_q, T, 1, dtype=I64, device=dev)
            q8, qm, qk = dy.quant_i8_pertoken(q, one, torch.full_like(one, qkk))
            # K: static per-(head, channel) int8 (c folded; quantized ONCE
            # here, straight into cache)
            k8 = dy.requant_i8_static(
                kk, self._one1, self._k_rowk[li], self._one1, 0,
                L.kq_m.view(1, self.n_kv, 1, self.hd),
                L.kq_k.view(1, self.n_kv, 1, self.hd))
            # V: static per-head int8 from the GEMM output directly
            Pvh = Pv.view(B, T, self.n_kv, self.hd)
            v8 = dy.requant_i8_static(
                Pvh, xm.unsqueeze(-1), xk.unsqueeze(-1),
                L.v_m.view(self.n_kv, self.hd), L.v_k,
                L.vq_m.view(1, 1, self.n_kv, 1), L.vq_k.view(1, 1, self.n_kv, 1))
            v8 = v8.permute(0, 2, 1, 3)  # [B, n_kv, T, hd]

            cache.append(li, k8, v8)
            K8 = cache.k8[li][:, :, :S]              # [B, n_kv, S, hd]
            V8 = cache.v8[li][:, :, :S]

            # scores: int8 GEMM per (b, head). Long prefills are processed in
            # query chunks — every op in scores→softmax→context is
            # per-query-row, so chunking is bit-exact and bounds the int64
            # softmax temporaries (~[B, nq, CH, S]) on long contexts.
            K8r = K8.repeat_interleave(group, dim=1)  # [B, n_q, S, hd]
            V8r = V8.repeat_interleave(group, dim=1)
            aux = {"li": li, "q": aux_q, "qkk": qkk - L.qs_k, "kk": kk,
                   "kkk": kkk, "Pv": Pv, "xm": xm, "xk": xk, "L": L}
            # bound the int64 softmax temporaries: B·n_q·CH·S ≲ 2^27 elements
            budget = max(1, (1 << 27) // max(1, B * self.n_q * S))
            CH = T if T <= budget else max(128, budget)
            attn_parts = []
            for qlo in range(0, T, CH):
                qhi = min(qlo + CH, T)
                scores, sm, sk = self._attn_scores(
                    q8[:, :, qlo:qhi], qm[:, :, qlo:qhi], qk[:, :, qlo:qhi],
                    K8r, L, group, aux)
                probs = di_softmax(scores, sm, sk,
                                   mask[:, :, qlo:qhi].expand(B, self.n_q, qhi - qlo, S))
                attn_parts.append(self._attn_context(probs, V8r, L, aux))
            attn = torch.cat(attn_parts, dim=2) if len(attn_parts) > 1 else attn_parts[0]
            attn = attn.permute(0, 2, 1, 3).reshape(B, T, self.n_q * self.hd)

            # merge-head requant (o-input smoothing multiply, then int8;
            # row scale starts at exact 1/2^os_k: probs k folded in col)
            rk0 = torch.full((B, T, 1), L.os_k, dtype=I64, device=dev)
            a8, am, ak = dy.requant_i8_rowcol(attn.to(I64) * L.os_m,
                                              torch.ones_like(rk0), rk0,
                                              L.attn_out_col_m, L.attn_out_col_k)
            Po = int_gemm(a8, L.o_w8t, self.backend)
            d, dm, dk = dy.requant_i32_common(Po, am, ak, L.o_m, L.o_k, target_bits=20)
            h, k_res = dy.residual_add(h, k_res, d, dm, dk)

            # ---- MLP ----
            x8, xm, xk = self._norm_quant(h, k_res, L.ms_m, L.ms_k)
            Pg = int_gemm(x8, L.gate_w8t, self.backend)
            Pu = int_gemm(x8, L.up_w8t, self.backend)
            gate, gm, gk = dy.requant_i32_common(Pg, xm, xk, L.gate_m, L.gate_k,
                                                 target_bits=14)
            up, um, uk = dy.requant_i32_common(Pu, xm, xk, L.up_m, L.up_k,
                                               target_bits=14)
            y, ym, yk = di_swiglu(gate, gm, gk, up, um, uk)
            y8, y8m, y8k = dy.quant_i8_pertoken(y * L.ds_m, ym, yk + L.ds_k)
            Pd = int_gemm(y8, L.down_w8t, self.backend)
            d, dm, dk = dy.requant_i32_common(Pd, y8m, y8k, L.down_m, L.down_k,
                                              target_bits=20)
            h, k_res = dy.residual_add(h, k_res, d, dm, dk)
        return h, k_res

    def _lm_head(self, x8, xm, xk):
        """Seam for diagnostics; xm/xk unused on the integer path."""
        return int_gemm(x8, self.head_w8t, self.backend)

    def enable_layer_cudagraphs(self):
        """§10 phase 2: compile the per-layer function with inductor-managed
        CUDA graphs (mode='reduce-overhead') for static-shape decode. Weights
        are marked static-address so the graphs reference them in place (28
        small graphs, one per layer, replayed each step). Bit-exactness is
        the §9.3 contract, re-verified by TestGraphedDecode."""
        import dataclasses
        # each layer specializes guards on its own scalar constants -> 28
        # intentional "recompiles" of _layer (one graph per layer)
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 128)
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 128)
        for lw in self.layers:
            for f in dataclasses.fields(lw):
                v = getattr(lw, f.name)
                if isinstance(v, torch.Tensor):
                    torch._dynamo.mark_static_address(v)
        for t in (self._one1, *self._k_rowk):
            torch._dynamo.mark_static_address(t)
        self._layer_compiled = torch.compile(self._layer, mode="reduce-overhead",
                                             dynamic=False)

    def _attn_scores(self, q8, qm, qk, K8r, L, group, aux=None):
        """Seam: int8 Q·Kᵀ with per-row dyadic score scale."""
        scores = int_gemm(q8, K8r.transpose(-1, -2).contiguous(), self.backend)
        sm = qm * L.ks_m.repeat_interleave(group).view(1, self.n_q, 1, 1)
        sk = qk + L.ks_k.repeat_interleave(group).view(1, self.n_q, 1, 1)
        return (scores,) + dy.norm_scale(sm, sk)

    def _attn_context(self, probs, V8r, L, aux=None):
        """Seam: 15-bit-probs × int8-V exact GEMM (hi/lo int8 split)."""
        return int_gemm_u16i8(probs, V8r, self.backend)

    # -- decoding ---------------------------------------------------------

    def greedy_pick(self, logits_row: torch.Tensor) -> torch.Tensor:
        """logits_row: int32 [B, vocab] (one position). Integer argmax with
        lowest-token-id tie-break. Row scale is positive → argmax-invariant;
        col scale must be applied (per-vocab mantissas differ)."""
        scaled = logits_row.to(I64) * self.head_m.to(I64)
        mx = scaled.amax(dim=-1, keepdim=True)
        idx = torch.arange(self.vocab, device=scaled.device).expand_as(scaled)
        cand = torch.where(scaled == mx, idx, torch.full_like(idx, self.vocab))
        return cand.min(dim=-1).values

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int, guard: bool = False,
                 chunk_valid: torch.Tensor | None = None,
                 stop_at_eos: bool = False, use_graph: bool = False):
        """Greedy generation. ids int64 [B, T0] (right-padded, pads marked
        False in chunk_valid). Returns (tokens [B, max_new], step_logits list).

        use_graph: CUDA-graph the decode loop (cuda backend only) —
        bit-identical to the eager loop (see graphed.py)."""
        dev = self.device
        ids = ids.to(dev)
        B, T0 = ids.shape
        if chunk_valid is None:
            chunk_valid = torch.ones(B, T0, dtype=torch.bool, device=dev)
        positions = torch.arange(T0, device=dev).expand(B, T0)
        logits, _, _, cache = self.forward(ids, positions, None, chunk_valid, guard)
        lens = chunk_valid.sum(dim=1)                      # true prompt lengths
        last = (lens - 1).clamp(min=0)
        step_logits = [logits[torch.arange(B), last]]      # [B, vocab] int32
        out = []
        cur = self.greedy_pick(step_logits[0])
        out.append(cur)
        if use_graph and self.backend == "cuda" and max_new > 1:
            from .graphed import GraphedDecode
            gd = GraphedDecode(self, cache, lens)
            for step in range(1, max_new):
                lg_row = gd.step(cur).clone()  # buffer is reused per replay
                step_logits.append(lg_row)
                cur = self.greedy_pick(lg_row)
                out.append(cur)
            return torch.stack(out, dim=1), step_logits
        for step in range(1, max_new):
            pos = (lens + step - 1).unsqueeze(1)           # [B, 1]
            lg, _, _, cache = self.forward(cur.unsqueeze(1), pos, cache,
                                           torch.ones(B, 1, dtype=torch.bool, device=dev),
                                           guard)
            step_logits.append(lg[:, 0])
            cur = self.greedy_pick(lg[:, 0])
            out.append(cur)
        return torch.stack(out, dim=1), step_logits

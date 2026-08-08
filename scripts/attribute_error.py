"""Error attribution (§12 mitigation): run short-PPL with individual integer
ops replaced by float oracles to find the dominant accuracy loss. Pure
diagnostic — the real pipeline never runs these.

Modes:
  int            : unmodified integer pipeline
  fsoftmax-u8    : float softmax, still quantized to u8 (isolates DI-Exp err)
  fsoftmax-full  : float softmax at fp precision fed to u8 GEMM path bypassed
  fswiglu        : float gate·σ(gate)·up from the int gate/up values
  fnorm          : float RMSNorm x/rms at the int norm's fixed-point output
  fquant-q       : Q quantized with float-exact scale (no dyadic fit error)
"""

import argparse
import math

import torch

import detllm.model as M
import detllm.dyadic as dy
from detllm.model import IntQwen3
from detllm.ops.di_softmax import di_softmax as real_softmax
from detllm.ops.di_swiglu import di_swiglu as real_swiglu
from detllm.ops.di_rmsnorm import di_rmsnorm as real_rmsnorm, OUT_FRAC_BITS

I64 = torch.int64


def fsoftmax_u8(scores, m, k, valid, clip_c=None):
    sf = scores.double() * m.double() * torch.pow(2.0, -k.double())
    sf = torch.where(valid, sf, torch.full_like(sf, -1e30))
    p = torch.softmax(sf, dim=-1)
    p = torch.where(valid, p, torch.zeros_like(p))
    return torch.round(p * 128).to(torch.int32)


def fswiglu(gate, gm, gk, up, um, uk):
    gv = gate.double() * gm.double() * torch.pow(2.0, -gk.double())
    uv = up.double() * um.double() * torch.pow(2.0, -uk.double())
    y = gv * torch.sigmoid(gv) * uv
    # re-encode on the same dyadic grid the int op would use
    m_out = gm.to(I64) * um.to(I64)
    k_out = gk.to(I64) + uk.to(I64) + 7
    yi = torch.round(y / (m_out.double() * torch.pow(2.0, -k_out.double()))).to(I64)
    return yi, m_out, k_out


def fnorm(x):
    xf = x.double()
    y = xf / torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-30))
    return torch.round(y * 2.0 ** OUT_FRAC_BITS).to(torch.int32)


def probs15(scores, m, k, valid, clip_c=None):
    """Integer DI-Exp path but 15-bit probability resolution (vs 8)."""
    import detllm.ops.di_softmax as ds
    from detllm.intmath import round_half_away_div
    from detllm.ops.di_exp import di_exp
    x = scores.to(I64); m = m.to(I64); k = k.to(I64)
    neg_inf = torch.tensor(torch.iinfo(I64).min + 1, dtype=I64, device=x.device)
    xm_ = torch.where(valid, x, neg_inf)
    row_max = xm_.amax(dim=-1, keepdim=True)
    any_valid = valid.any(dim=-1, keepdim=True)
    row_max = torch.where(any_valid, row_max, torch.zeros_like(row_max))
    xd = torch.where(valid, x - row_max, torch.zeros_like(x))
    exps, _ = di_exp(xd, m, k)
    exps = torch.where(valid, exps, torch.zeros_like(exps))
    denom = torch.clamp(exps.sum(dim=-1, keepdim=True), min=1)
    return round_half_away_div(exps << 14, denom).to(torch.int32)  # k=14


def wide_u8i8(p, b, backend):
    """Exact wide GEMM accepting any non-negative int probs (diag only)."""
    return torch.matmul(p.to(torch.float64), b.to(torch.float64)).round().to(torch.int32) \
        if p.shape[-1] * 32768 * 127 < 2**52 else None


class FheadModel(IntQwen3):
    def load_float_head(self):
        from transformers import AutoModelForCausalLM
        fm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
        W = fm.model.embed_tokens.weight.detach().double()
        gamma = fm.model.norm.weight.detach().double()
        self._Wf = (W * gamma[None, :]).t().to(self.device)  # [H, V]
        self._head_scale = (self.head_m.double() * 2.0 ** -self.head_k).to(self.device)

    def _lm_head(self, x8, xm, xk):
        xf = x8.double()  # value = x8·xm/2^xk; keep x8 grid, bypass W quant
        lf = xf @ self._Wf  # value·2^xk/xm
        return torch.round(lf / self._head_scale[None, None, :]).to(torch.int32)


class FscoresModel(IntQwen3):
    """Float-exact scores from pre-quant Q/K (bypasses Q int8, K int8+static
    scale, dyadic score-scale fits). PREFILL ONLY (chunk K == all K)."""

    def _attn_scores(self, q8, qm, qk, K8r, L, group, aux=None):
        import math
        q = aux["q"].double() * 2.0 ** -float(aux["qkk"])   # [B,nq,T,hd]
        kk = aux["kk"].double() * 2.0 ** -float(aux["kkk"])  # [B,nkv,S,hd]
        kf = kk.repeat_interleave(group, dim=1)
        sf = torch.matmul(q, kf.transpose(-1, -2)) / math.sqrt(self.hd)
        # encode on a fixed fine grid (m=2^15, k=25)
        m = torch.full_like(qm, 1 << 15)
        k = torch.full_like(qk, 25)
        si = torch.round(sf * 2.0 ** 25 / (1 << 15)).to(I64)
        return si, m, k


class FcontextModel(IntQwen3):
    """u8 probs × float-exact V (bypasses V int8 + static scale). PREFILL ONLY."""

    def _attn_context(self, probs, V8r, L, aux=None):
        Pv = aux["Pv"]; xm = aux["xm"]; xk = aux["xk"]
        B, T = Pv.shape[0], Pv.shape[1]
        vv = Pv.view(B, T, self.n_kv, self.hd).double()
        vv = vv * L.v_m.view(self.n_kv, self.hd).double()
        vv = vv * xm.unsqueeze(-1).double() * torch.pow(
            2.0, -(xk.unsqueeze(-1).double() + L.v_k))     # [B,T,nkv,hd] values
        group = self.n_q // self.n_kv
        vf = vv.permute(0, 2, 1, 3).repeat_interleave(group, dim=1)  # [B,nq,S,hd]
        ctx = torch.matmul(probs.double() / 128.0, vf)
        # re-encode on the int path's per-channel grid: value = int·col_m/2^col_k
        col_m = L.attn_out_col_m.view(self.n_q, 1, self.hd).double()
        enc = ctx * 2.0 ** L.attn_out_col_k / col_m
        return torch.round(enc).to(torch.int32)


def quant_i16_pertoken(x, m, k):
    """Diagnostic: 16-bit activations (float-exact wide GEMM patched in)."""
    x64 = x.to(I64)
    a = torch.clamp(x64.abs().amax(dim=-1, keepdim=True), min=1)
    x16 = torch.clamp(torch.div(x64 * 32767 * 2 + a, a * 2,
                                rounding_mode="floor"), -32767, 32767)
    m_out, e = dy.fit_dyadic_ratio(a * m.to(I64), 32767)
    return x16.to(torch.int32), m_out, k.to(I64) + e


def wide_gemm(a, b, backend):
    # int64 result: 16-bit activations can push sums past int32
    return torch.matmul(a.to(torch.float64), b.to(torch.float64)).round().to(torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    model_cls = IntQwen3
    if args.mode == "a16":
        dy.quant_i8_pertoken = quant_i16_pertoken
        M.int_gemm = wide_gemm
    if args.mode == "fsoftmax-u8":
        M.di_softmax = fsoftmax_u8
    elif args.mode == "fswiglu":
        M.di_swiglu = fswiglu
    elif args.mode == "fnorm":
        M.di_rmsnorm = fnorm
    elif args.mode == "probs15":
        M.di_softmax = probs15
        M.int_gemm_u8i8 = lambda p, b, backend: torch.matmul(
            p.to(torch.float64), b.to(torch.float64)).round().to(torch.int32)
    elif args.mode == "fhead":
        model_cls = FheadModel
    elif args.mode == "fscores":
        model_cls = FscoresModel
    elif args.mode == "fcontext":
        model_cls = FcontextModel
    elif args.mode == "fattn":
        model_cls = type("FattnModel", (FscoresModel, FcontextModel), {})
    import sys
    sys.path.insert(0, "scripts")
    import eval_ppl
    segs = eval_ppl.get_segments(2048)[: args.limit]
    model = model_cls("artifacts/qwen3-0.6b-int8", backend="cuda")
    if args.mode == "probs15":
        for lw in model.layers:
            lw.attn_out_col_k += 7  # probs now carry k=14 instead of 7
    if args.mode == "fhead":
        model.load_float_head()
    ppl = eval_ppl.eval_int(segs, "cuda", "artifacts/qwen3-0.6b-int8", model=model)
    print(f"PPL[{args.mode}] = {ppl:.4f}")


if __name__ == "__main__":
    main()

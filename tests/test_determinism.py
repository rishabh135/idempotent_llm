"""§9.3 determinism acceptance suite — every check is EXACT (torch.equal on
int32 logits and token ids), no tolerances anywhere.

Checks:
  1. run-to-run (CUDA): 10 runs -> identical ids + logits at every step
  2. batch invariance: alone vs batches of 2/8/32 with random co-prompts
  3. prefill/decode invariance: one-shot prefill vs token-by-token
  4. cross-device: CUDA vs reference CPU, 3 prompts × 100 tokens
  5. long context: 4k-token prompt through checks 1-3 (marked slow;
     exercises the >2048 query-chunked attention path and long RoPE positions)
  9.4 float-leak guard: NoFloatMode over the full forward

Requires the prepared artifact; skips (loudly) if missing.
"""

import os

import pytest
import torch

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "qwen3-0.6b-int8")
CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ART, "model.safetensors")),
    reason="artifact not prepared")


@pytest.fixture(scope="module")
def cuda_model():
    from detllm.model import IntQwen3
    if not CUDA:
        pytest.skip("no CUDA")
    return IntQwen3(ART, backend="cuda")


@pytest.fixture(scope="module")
def ref_model():
    from detllm.model import IntQwen3
    return IntQwen3(ART, backend="reference")


@pytest.fixture(scope="module")
def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    texts = [
        "The capital of France is",
        "In a shocking turn of events, researchers discovered that",
        "def fibonacci(n):\n    ",
    ]
    return tok, [tok(t, return_tensors="pt").input_ids[0] for t in texts]


def _greedy(model, ids_1d, n_new, **kw):
    toks, logits = model.generate(ids_1d.unsqueeze(0).to(model.device), n_new, **kw)
    return toks[0].cpu(), [l[0].cpu() for l in logits]


class TestRunToRun:
    def test_ten_runs_identical(self, cuda_model, prompts):
        _, ps = prompts
        base_t, base_l = _greedy(cuda_model, ps[0], 20)
        for _ in range(9):
            t, l = _greedy(cuda_model, ps[0], 20)
            assert torch.equal(t, base_t)
            for a, b in zip(l, base_l):
                assert torch.equal(a, b)  # exact int32 logits, every step


class TestBatchInvariance:
    @pytest.mark.parametrize("bs", [2, 8, 32])
    def test_padded_batches(self, cuda_model, prompts, bs):
        tok, ps = prompts
        target = ps[0]
        g = torch.Generator().manual_seed(bs)
        # random co-prompts of random lengths (right padding)
        co = [torch.randint(100, 5000, (int(torch.randint(3, 30, (1,), generator=g)),),
                            generator=g) for _ in range(bs - 1)]
        seqs = [target] + co
        T = max(len(s) for s in seqs)
        ids = torch.zeros(bs, T, dtype=torch.int64)
        valid = torch.zeros(bs, T, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            valid[i, : len(s)] = True

        toks_b, logits_b = cuda_model.generate(ids.cuda(), 10,
                                               chunk_valid=valid.cuda())
        toks_a, logits_a = cuda_model.generate(target.unsqueeze(0).cuda(), 10)
        assert torch.equal(toks_b[0].cpu(), toks_a[0].cpu())
        for lb, la in zip(logits_b, logits_a):
            assert torch.equal(lb[0].cpu(), la[0].cpu())  # bit-identical


class TestPrefillDecodeInvariance:
    def test_tokenwise_equals_prefill(self, cuda_model, prompts):
        _, ps = prompts
        p = ps[1]
        T = len(p)
        pos = torch.arange(T).unsqueeze(0).cuda()
        logits_pre, _, _, _ = cuda_model.forward(p.unsqueeze(0).cuda(), pos)
        # token-by-token: feed one token at a time, collect logits per position
        cache = None
        step_logits = []
        for t in range(T):
            lg, _, _, cache = cuda_model.forward(
                p[t: t + 1].unsqueeze(0).cuda(),
                torch.tensor([[t]], device="cuda"), cache)
            step_logits.append(lg[:, 0])
        for t in range(T):
            assert torch.equal(logits_pre[0, t].cpu(), step_logits[t][0].cpu()), \
                f"prefill/decode divergence at position {t}"


class TestCrossDevice:
    def test_cuda_vs_reference_100_tokens(self, cuda_model, ref_model, prompts):
        _, ps = prompts
        for p in ps:
            t_c, l_c = _greedy(cuda_model, p, 100)
            t_r, l_r = _greedy(ref_model, p, 100)
            assert torch.equal(t_c, t_r)
            for a, b in zip(l_c, l_r):
                assert torch.equal(a, b)


@pytest.mark.slow
class TestLongContext:
    def test_4k_prompt(self, cuda_model):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0][:4096]

        # run-to-run
        t1, l1 = _greedy(cuda_model, ids, 5)
        t2, l2 = _greedy(cuda_model, ids, 5)
        assert torch.equal(t1, t2)
        for a, b in zip(l1, l2):
            assert torch.equal(a, b)

        # batch invariance (bs=2, co-prompt shorter)
        co = ids[:100]
        T = len(ids)
        bids = torch.zeros(2, T, dtype=torch.int64)
        valid = torch.zeros(2, T, dtype=torch.bool)
        bids[0, :] = ids; valid[0, :] = True
        bids[1, : 100] = co; valid[1, : 100] = True
        tb, lb = cuda_model.generate(bids.cuda(), 5, chunk_valid=valid.cuda())
        assert torch.equal(tb[0].cpu(), t1)
        for a, b in zip(lb, l1):
            assert torch.equal(a[0].cpu(), b)

        # prefill/decode at the boundary: prefill 4095 then decode 1 == full prefill
        pos = torch.arange(T).unsqueeze(0).cuda()
        full, _, _, _ = cuda_model.forward(ids.unsqueeze(0).cuda(), pos)
        c = None
        lg, _, _, c = cuda_model.forward(ids[:-1].unsqueeze(0).cuda(), pos[:, :-1], c)
        lg2, _, _, c = cuda_model.forward(ids[-1:].unsqueeze(0).cuda(), pos[:, -1:], c)
        assert torch.equal(full[0, -1].cpu(), lg2[0, 0].cpu())


class TestFloatLeakGuard:
    def test_no_float_on_numerical_path(self, ref_model, prompts):
        _, ps = prompts
        p = ps[0].unsqueeze(0)
        pos = torch.arange(p.shape[1]).unsqueeze(0)
        # guard=True wraps the forward in NoFloatMode -> raises on any float op
        logits, _, _, _ = ref_model.forward(p, pos, guard=True)
        assert logits.dtype == torch.int32

    def test_artifact_all_integer(self, ref_model):
        for lw in ref_model.layers:
            for f in ("q_w8t", "k_w8t", "v_w8t", "o_w8t", "gate_w8t",
                      "up_w8t", "down_w8t"):
                assert getattr(lw, f).dtype == torch.int8
        assert ref_model.embed_e8.dtype == torch.int8
        assert ref_model.head_w8t.dtype == torch.int8
        assert ref_model.rope_cos.dtype == torch.int16

    def test_guard_actually_fires(self):
        from detllm.guard import FloatLeakError, NoFloatMode
        with pytest.raises(FloatLeakError):
            with NoFloatMode():
                torch.ones(3).float() * 2.0


class TestGraphedDecode:
    """§10 rule: no perf change may alter any golden output. The CUDA-graphed
    decode must be bit-identical to the eager decode loop — tokens AND int32
    logits — including across a cache-bucket growth boundary."""

    def test_graph_equals_eager_with_growth(self, cuda_model, prompts):
        _, ps = prompts
        p = ps[1]
        # prompt ~10 tokens → bucket 256; 260 steps forces growth + recapture
        te, le = cuda_model.generate(p.unsqueeze(0).cuda(), 260)
        tg, lg = cuda_model.generate(p.unsqueeze(0).cuda(), 260, use_graph=True)
        assert torch.equal(te, tg)
        for a, b in zip(le, lg):
            assert torch.equal(a, b)

    def test_graph_batch(self, cuda_model, prompts):
        _, ps = prompts
        seqs = ps
        T = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.int64)
        valid = torch.zeros(len(seqs), T, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            valid[i, : len(s)] = True
        te, le = cuda_model.generate(ids.cuda(), 20, chunk_valid=valid.cuda())
        tg, lg = cuda_model.generate(ids.cuda(), 20, chunk_valid=valid.cuda(),
                                     use_graph=True)
        assert torch.equal(te, tg)
        for a, b in zip(le, lg):
            assert torch.equal(a, b)

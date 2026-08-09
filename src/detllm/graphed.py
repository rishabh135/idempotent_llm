"""§10 step 3: CUDA-graphed decode.

Default architecture (mode="capture"): the decode step runs through ONE
generic layer graph compiled by inductor (per-layer constants are 0-dim
tensors, so all 28 layers share it; loop-heavy integer primitives are
opaque custom ops, so it compiles in ~1 min) plus the fused Triton
attention kernel, and the WHOLE step is then recorded as a manual
torch.cuda.CUDAGraph — steady-state decode is one replay + a handful of
eager glue kernels. mode="layers" is an alternative that lets inductor
manage per-layer graphs via mode='reduce-overhead' instead.

Static-shape decode (bit-exact by construction):
- Attention runs over the full bucketed cache CAPACITY with the validity
  mask; invalid slots contribute exact zeros (the §9.3 batch-invariance
  property), so capacity-wide compute ≡ length-wide compute, bit for bit.
- The cache write is `index_copy_` at a device-tensor offset — same bytes,
  same slot as the dynamic path's slice write.
- Bucket overflow grows the buffers (filled slots copied verbatim) and
  re-marks their addresses; inductor records fresh graphs for the new
  shapes.

Verified by tests/test_determinism.py::TestGraphedDecode — graphed vs
eager decode must produce identical tokens AND int32 logits across a
bucket boundary.
"""

from __future__ import annotations

import torch

from .model import IntQwen3, KVCache

I64 = torch.int64


def _next_bucket(n: int) -> int:
    cap = 256
    while cap < n:
        cap *= 2
    return cap


class GraphedDecode:
    def __init__(self, model: IntQwen3, cache: KVCache, lens: torch.Tensor,
                 mode: str = "capture", layer_compile: bool = True):
        """model: cuda backend. cache: as returned by a (dynamic-mode)
        prefill forward. lens: int64 [B] true prompt lengths (device) —
        the next token's position per sequence.

        mode="capture": manual whole-step CUDA graph over the per-op
        compiled functions (fast one-time setup, single replay per step).
        mode="layers": inductor-managed per-layer graphs (reduce-overhead;
        ~28 inductor compiles on first use)."""
        assert model.backend == "cuda"
        self.model = model
        self.cache = cache
        self.mode = mode
        dev = model.device
        self.B = cache.valid.shape[0]
        self.length = cache.length
        self.ids_buf = torch.zeros(self.B, 1, dtype=I64, device=dev)
        self.pos_buf = lens.to(dev).view(self.B, 1).clone()
        cache.len_idx = torch.zeros(1, dtype=I64, device=dev)
        self.graph = None
        self.out = None
        if layer_compile and mode == "capture" and model._layer_compiled is None:
            # one generic compiled layer graph (per-layer constants are
            # tensors; ~1 min cold, inductor-disk-cached afterwards)
            model.enable_layer_compile()
        if mode == "layers":
            for t in (self.ids_buf, self.pos_buf, cache.len_idx):
                torch._dynamo.mark_static_address(t)
            model.enable_layer_cudagraphs()
        self._grow_to(_next_bucket(self.length + 1))

    def _grow_to(self, cap: int):
        """(Re)allocate cache + valid buffers at capacity `cap`, copying the
        filled prefix verbatim; mark the new addresses static."""
        c = self.cache
        L = self.length
        for buf in (c.k8, c.v8):
            for li in range(len(buf)):
                B, n_kv, old_cap, hd = buf[li].shape
                nb = buf[li].new_zeros(B, n_kv, cap, hd)
                nb[:, :, :L] = buf[li][:, :, :L]
                torch._dynamo.mark_static_address(nb)
                buf[li] = nb
        valid = torch.zeros(self.B, cap, dtype=torch.bool, device=c.valid.device)
        valid[:, : min(cap, c.valid.shape[1])] = c.valid[:, : min(cap, c.valid.shape[1])]
        torch._dynamo.mark_static_address(valid)
        c.valid = valid
        c.static_cap = cap
        self.graph = None  # any captured graph references the old buffers

    def _run(self):
        logits, _, _, _ = self.model.forward(self.ids_buf, self.pos_buf, self.cache)
        return logits

    def _capture(self):
        # warmup: idempotent real executions of this same step (also ensures
        # every dynamo graph / inductor kernel for these shapes exists, so no
        # compilation work can happen inside the capture)
        for _ in range(3):
            self._run()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = torch._dynamo.run(self._run)()
        self.graph = g

    @torch.no_grad()
    def step(self, cur_ids: torch.Tensor) -> torch.Tensor:
        """cur_ids: int64 [B] (device). Returns int32 logits [B, vocab]
        (mode='capture' reuses the output buffer — clone to keep)."""
        if self.length + 1 > self.cache.static_cap:
            self._grow_to(_next_bucket(self.length + 1))
        self.cache.valid[:, self.length] = True
        self.cache.len_idx.fill_(self.length)
        self.ids_buf.copy_(cur_ids.view(self.B, 1))
        if self.mode == "capture":
            if self.graph is None:
                self._capture()
                # capture only RECORDS the kernels — nothing executed yet;
                # replay to actually run this step
            self.graph.replay()
            logits = self.out
        else:
            torch.compiler.cudagraph_mark_step_begin()
            logits = self._run()
        self.length += 1
        self.pos_buf += 1
        return logits[:, 0]

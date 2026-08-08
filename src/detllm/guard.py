"""Float-leak guard (§9.4): a TorchDispatchMode that fails the moment any
float-dtype tensor is produced or consumed inside the guarded region.

Used by tests around the full forward pass (input ids → int32 logits) and
available at runtime via IntQwen3.forward(guard=True). This mechanically
prevents 'convenient .float()' regressions on the numerical path.
"""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class FloatLeakError(RuntimeError):
    pass


class NoFloatMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat, _ = tree_flatten((args, kwargs))
        for t in flat:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                raise FloatLeakError(f"float tensor consumed by {func}")
        out = func(*args, **kwargs)
        flat_out, _ = tree_flatten(out)
        for t in flat_out:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                raise FloatLeakError(f"float tensor produced by {func}")
        return out

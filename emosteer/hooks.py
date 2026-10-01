"""Residual-stream steering and activation capture via forward hooks."""
from __future__ import annotations

import contextlib

import torch

from .models import model_device


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class Steer:
    """Adds `vec` (already scaled, shape [d]) to a layer's output. Toggle with .on."""

    def __init__(self, layer, vec: torch.Tensor | None = None, positions: str = "all"):
        self.layer, self.vec, self.positions, self.on = layer, vec, positions, True
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if not self.on or self.vec is None:
            return None
        h = _hidden(out)
        v = self.vec.to(h.device, h.dtype)
        if self.positions == "last":
            h = h.clone()
            h[:, -1, :] += v
        else:
            h = h + v
        return _replace(out, h)

    def remove(self):
        self.handle.remove()


@contextlib.contextmanager
def steering(layer, vec, positions="all"):
    s = Steer(layer, vec, positions)
    try:
        yield s
    finally:
        s.remove()


def tokenize(tok, texts, device):
    enc = tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def capture_last(model, tok, layers, texts, batch=8, idx=None):
    """Final-token residual after each decoder layer -> [n, L, d] float32 (cpu).

    Assumes left padding so position -1 is the final real token.
    """
    idx = list(range(len(layers))) if idx is None else idx
    dev = model_device(model)
    store = {}
    handles = [layers[i].register_forward_hook(
        lambda m, a, o, i=i: store.__setitem__(i, _hidden(o)[:, -1, :].float().cpu()))
        for i in idx]
    outs = []
    try:
        for s in range(0, len(texts), batch):
            store.clear()
            model(**tokenize(tok, texts[s:s + batch], dev))
            outs.append(torch.stack([store[i] for i in idx], 1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(outs, 0)


@torch.no_grad()
def next_token_logprobs(model, tok, texts, batch=8, last_k=1):
    """Log-probs at the last `last_k` positions -> [n, last_k, V] float32 (cpu)."""
    dev = model_device(model)
    outs = []
    for s in range(0, len(texts), batch):
        logits = model(**tokenize(tok, texts[s:s + batch], dev)).logits[:, -last_k:, :]
        outs.append(torch.log_softmax(logits.float(), -1).cpu())
    return torch.cat(outs, 0)

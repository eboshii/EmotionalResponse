"""Calibrate steering strength so every direction causes the same KL disruption.

Steering vector = alpha * resid_norm[steer_layer] * unit_direction.
For each direction (+emo, -emo, random_i) sweep alpha, measure mean next-token
KL(steered || unsteered) on neutral prompts, and interpolate alpha* per target KL.

python -m emosteer.calibrate --model Qwen/Qwen3-4B --run runs/qwen3-4b --gen-samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .hooks import Steer, _hidden, tokenize
from .models import chat, common_args, find_layers, load, model_device


def random_dirs(n, d, seed=1234):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    return v / v.norm(dim=1, keepdim=True)


def interp_alpha(alphas, kls, target):
    """log-log interpolation on the monotone envelope; None if target not bracketed."""
    a = np.asarray(alphas, float)
    k = np.maximum.accumulate(np.asarray(kls, float))
    if target > k[-1] or target <= 0:
        return None
    if target <= k[0]:
        # below first sweep point: assume KL ~ alpha^2 near zero
        return float(a[0] * np.sqrt(target / max(k[0], 1e-12)))
    i = int(np.searchsorted(k, target))
    a0, a1, k0, k1 = a[i - 1], a[i], max(k[i - 1], 1e-12), max(k[i], 1e-12)
    if k1 <= k0:
        return float(a0)
    t = (np.log(target) - np.log(k0)) / (np.log(k1) - np.log(k0))
    return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))


class Probe:
    """Runs neutral prompts with optional steering; returns KL vs base and read-layer acts."""

    def __init__(self, model, tok, layers, steer_layer, read_layer, texts, batch, last_k=8):
        self.model, self.tok, self.layers = model, tok, layers
        self.sl, self.rl, self.texts, self.batch, self.last_k = steer_layer, read_layer, texts, batch, last_k
        self.steer = Steer(layers[steer_layer], None)
        self.base_lp, self.base_read = self._run()

    @torch.no_grad()
    def _run(self):
        dev = model_device(self.model)
        lps, reads, store = [], [], {}
        h = self.layers[self.rl].register_forward_hook(
            lambda m, i, o: store.__setitem__("r", _hidden(o)[:, -1, :].float().cpu()))
        try:
            for s in range(0, len(self.texts), self.batch):
                logits = self.model(**tokenize(self.tok, self.texts[s:s + self.batch], dev)).logits
                lps.append(torch.log_softmax(logits[:, -self.last_k:, :].float(), -1).cpu())
                reads.append(store["r"])
        finally:
            h.remove()
        return torch.cat(lps), torch.cat(reads)

    def measure(self, vec):
        self.steer.vec = vec
        try:
            lp, read = self._run()
        finally:
            self.steer.vec = None
        kl = (lp.exp() * (lp - self.base_lp)).sum(-1)  # [n, last_k]
        return {"kl_last": float(kl[:, -1].mean()), "kl_lastk": float(kl.mean()),
                "read_shift": (read - self.base_read).mean(0)}


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--run", required=True, help="dir containing directions.pt (outputs go here)")
    p.add_argument("--data", default=str(D.DEFAULT))
    p.add_argument("--steer-layer", type=int, default=None, help="default round(n_layers/3)")
    p.add_argument("--targets", default="0.1,0.3,1.0")
    p.add_argument("--alphas", default=None, help="comma list; default logspace(0.01,2,14)")
    p.add_argument("--alpha-cap", type=float, default=16.0)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--signs", default="+,-")
    p.add_argument("--kl-metric", choices=["kl_last", "kl_lastk"], default="kl_lastk")
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--gen-samples", action="store_true")
    p.add_argument("--gen-tokens", type=int, default=60)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args(argv)

    run = Path(a.run)
    ex = torch.load(run / "directions.pt")
    d = D.load_emotions(a.data)
    model, tok = load(a.model, a.device, a.dtype, a.device_map)
    layers = find_layers(model, a.layers_path or ex.get("layers_path"))
    L = len(layers)
    sl = a.steer_layer if a.steer_layer is not None else round(L / 3)
    rl = ex["read_layer"]
    norm = float(ex["resid_norm"][sl])
    targets = [float(t) for t in a.targets.split(",")]
    alphas = ([float(x) for x in a.alphas.split(",")] if a.alphas
              else np.logspace(np.log10(0.01), np.log10(2.0), 14).tolist())
    emos = list(ex["directions"])
    read_dirs = torch.stack([ex["directions"][e][rl] for e in emos])  # [E, d]

    dirs = {}
    for e in emos:
        for s in a.signs.split(","):
            dirs[f"{s}{e}"] = (1.0 if s == "+" else -1.0) * ex["directions"][e][sl]
    for i, v in enumerate(random_dirs(a.n_random, read_dirs.shape[1], a.seed)):
        dirs[f"random{i}"] = v

    texts = [chat(tok, q) for q in d["steer_prompts"][: a.n_prompts]]
    probe = Probe(model, tok, layers, sl, rl, texts, a.batch)
    print(f"steer layer {sl} (resid norm {norm:.1f}), read layer {rl}, {len(dirs)} directions")

    results = {}
    for name, v in dirs.items():
        sweep = []
        grid = list(alphas)
        i = 0
        while i < len(grid):
            m = probe.measure(grid[i] * norm * v)
            sweep.append({"alpha": grid[i], "kl_last": m["kl_last"], "kl_lastk": m["kl_lastk"]})
            i += 1
            if i == len(grid) and m[a.kl_metric] < max(targets) and grid[-1] * 2 <= a.alpha_cap:
                grid.append(grid[-1] * 2)
        ks = [s[a.kl_metric] for s in sweep]
        levels = {}
        for t in targets:
            al = interp_alpha([s["alpha"] for s in sweep], ks, t)
            if al is None:
                levels[str(t)] = {"alpha": None, "unreachable": True}
                continue
            m = probe.measure(al * norm * v)
            readout = (read_dirs @ m["read_shift"]).tolist()  # shift on each emotion axis
            levels[str(t)] = {"alpha": al, "achieved_kl": m[a.kl_metric], "kl_last": m["kl_last"],
                              "readout": dict(zip(emos, readout))}
        results[name] = {"sweep": sweep, "levels": levels}
        summ = ", ".join(f"{t}:{lv['alpha']:.3f}" if lv.get("alpha") else f"{t}:n/a"
                         for t, lv in levels.items())
        print(f"{name:>16}  alpha* {summ}")
    probe.steer.remove()

    # self-readout and cross-readout matrices per level
    matrices = {}
    for t in map(str, targets):
        rows = [n for n in dirs if results[n]["levels"][t].get("alpha")]
        M = [[results[n]["levels"][t]["readout"][e] for e in emos] for n in rows]
        self_ro = {n: results[n]["levels"][t]["readout"][n[1:]] for n in rows if n[1:] in emos}
        matrices[t] = {"rows": rows, "cols": emos, "cross_readout": M, "self_readout": self_ro}
        print(f"\nself-readout @ KL {t}: " + ", ".join(f"{k}={v:+.2f}" for k, v in self_ro.items()))
        rnd = [max(abs(x) for x in results[n]["levels"][t]["readout"].values())
               for n in rows if n.startswith("random")]
        if rnd:
            print(f"  random-vector max |readout| band: {min(rnd):.2f}..{max(rnd):.2f}")

    ladder = {}
    if a.gen_samples:
        st = Steer(layers[sl], None)
        dev = model_device(model)
        try:
            for name, v in dirs.items():
                if name.startswith("random") and name != "random0":
                    continue
                ladder[name] = {}
                for t, lv in [("0", {"alpha": 0.0})] + list(results[name]["levels"].items()):
                    if lv.get("alpha") is None:
                        continue
                    st.vec = lv["alpha"] * norm * v if lv["alpha"] else None
                    enc = tokenize(tok, texts[:3], dev)
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=a.gen_tokens, do_sample=False,
                                             pad_token_id=tok.pad_token_id)
                    gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    ladder[name][t] = gen
                    print(f"[{name} KL={t}] {gen[0][:150]!r}")
        finally:
            st.remove()

    meta = {"model": a.model, "steer_layer": sl, "read_layer": rl, "resid_norm": norm,
            "targets": targets, "kl_metric": a.kl_metric, "emotions": emos,
            "valence": ex["valence"], "n_random": a.n_random}
    torch.save({"meta": meta, "directions": dirs, "results": results, "matrices": matrices},
               run / "calib.pt")
    (run / "calib.json").write_text(json.dumps({"meta": meta, "results": results,
                                                "matrices": matrices}, indent=1))
    if ladder:
        (run / "ladder.json").write_text(json.dumps(ladder, indent=1, ensure_ascii=False))
    print(f"\nsaved {run/'calib.pt'}, calib.json" + (", ladder.json" if ladder else ""))


if __name__ == "__main__":
    main()

"""Extract denoised difference-in-means emotion directions at every layer.

python -m emosteer.extract --model Qwen/Qwen3-4B --out runs/qwen3-4b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .hooks import capture_last
from .models import common_args, find_layers, load


def denoise_basis(neutral: torch.Tensor, var_frac: float) -> torch.Tensor:
    """Top PCs of `neutral` [n, d] explaining >= var_frac of variance -> [k, d] orthonormal."""
    if var_frac <= 0:
        return neutral.new_zeros(0, neutral.shape[1])
    x = neutral - neutral.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    var = s ** 2
    cum = torch.cumsum(var, 0) / var.sum().clamp_min(1e-12)
    k = int((cum < var_frac).sum().item()) + 1
    return vh[: min(k, vh.shape[0])]


def project_out(v: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return v
    return v - (v @ basis.T) @ basis


def direction(pos: torch.Tensor, neg: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    v = project_out(pos.mean(0) - neg.mean(0), basis)
    return v / v.norm().clamp_min(1e-12)


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as 0.5."""
    p = pos_scores[:, None]
    n = neg_scores[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def kfold_idx(n, k, rng):
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def cv_auc(pos, neg, neutral, var_frac, k, seed=0) -> float:
    """pos/neg/neutral: [n, d] at one layer. PCs refit on training neutral each fold."""
    rng = np.random.default_rng(seed)
    k = max(2, min(k, len(pos), len(neg)))
    pf, nf = kfold_idx(len(pos), k, rng), kfold_idx(len(neg), k, rng)
    neutral_is_neg = neutral is neg
    scores = []
    for i in range(k):
        ptr = np.concatenate([pf[j] for j in range(k) if j != i])
        ntr = np.concatenate([nf[j] for j in range(k) if j != i])
        neu = neg[ntr] if neutral_is_neg else neutral
        v = direction(pos[ptr], neg[ntr], denoise_basis(neu, var_frac))
        scores.append(auc((pos[pf[i]] @ v).numpy(), (neg[nf[i]] @ v).numpy()))
    return float(np.mean(scores))


def cosine_matrix(dirs: dict[str, torch.Tensor], layer: int):
    names = list(dirs)
    m = torch.stack([dirs[n][layer] for n in names])
    return names, (m @ m.T).numpy()


def print_cos(names, cos, thresh):
    w = max(len(n) for n in names)
    print(" " * (w + 1) + " ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>{w}} " + " ".join(f"{cos[i, j]:6.2f}" for j in range(len(names))))
    flags = [(names[i], names[j], float(cos[i, j])) for i in range(len(names))
             for j in range(i + 1, len(names)) if abs(cos[i, j]) > thresh]
    for a, b, c in flags:
        print(f"  WARNING: {a} vs {b} cos={c:.2f} > {thresh} (may be indistinguishable)")
    return flags


def build_directions(acts, emotions, contrast, var_frac, folds, mid=(0.25, 0.75), seed=0):
    """acts: {'neutral': [n,L,d], emo: [n,L,d], ...}. Returns dirs, auc table, read layer."""
    neutral = acts["neutral"]
    L = neutral.shape[1]
    dirs, aucs = {}, {}
    for emo in emotions:
        if contrast == "all":
            neg = torch.cat([neutral] + [acts[o] for o in emotions if o != emo], 0)
        else:
            neg = neutral
        per_layer, per_auc = [], []
        for l in range(L):
            basis = denoise_basis(neutral[:, l], var_frac)
            per_layer.append(direction(acts[emo][:, l], neg[:, l], basis))
            nl = neg[:, l]
            per_auc.append(cv_auc(acts[emo][:, l], nl, nl if contrast == "neutral" else neutral[:, l],
                                  var_frac, folds, seed))
        dirs[emo] = torch.stack(per_layer)
        aucs[emo] = per_auc
    mean_auc = np.mean([aucs[e] for e in emotions], 0)
    lo, hi = int(mid[0] * L), max(int(mid[0] * L) + 1, int(np.ceil(mid[1] * L)))
    read = lo + int(np.argmax(mean_auc[lo:hi]))
    return dirs, aucs, mean_auc.tolist(), read


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--data", default=str(D.DEFAULT))
    p.add_argument("--emotions", default=None, help="comma list; default all in data file")
    p.add_argument("--out", required=True)
    p.add_argument("--contrast", choices=["neutral", "all"], default="neutral")
    p.add_argument("--denoise-var", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--read-layer", default="auto")
    p.add_argument("--cos-flag", type=float, default=0.8)
    p.add_argument("--with-controls", action="store_true", help="also extract arousal/numbness")
    a = p.parse_args(argv)

    d = D.load_emotions(a.data, a.emotions.split(",") if a.emotions else None)
    for prob in D.lint(d):
        print("LINT:", prob)
    model, tok = load(a.model, a.device, a.dtype, a.device_map)
    layers = find_layers(model, a.layers_path)
    print(f"decoder layers: {getattr(find_layers, 'last_path', a.layers_path)} (n={len(layers)})")

    sets = {e: s["sentences"] for e, s in d["emotions"].items()}
    if a.with_controls:
        sets.update({f"ctrl_{k}": v for k, v in d.get("controls", {}).items()})
    sets["neutral"] = d["neutral"]["sentences"]
    acts = {}
    for name, sents in sets.items():
        acts[name] = capture_last(model, tok, layers, D.with_suffix(sents, d["suffix"]), a.batch)
        print(f"captured {name}: {tuple(acts[name].shape)}")

    names = [k for k in sets if k != "neutral"]
    dirs, aucs, mean_auc, read = build_directions(acts, names, a.contrast, a.denoise_var, a.folds)
    if a.read_layer != "auto":
        read = int(a.read_layer)

    # residual norm on the neutral steering prompts (what we will steer on)
    from .models import chat
    steer_texts = [chat(tok, q) for q in d["steer_prompts"]]
    sp = capture_last(model, tok, layers, steer_texts, a.batch)
    resid_norm = sp.norm(dim=-1).mean(0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    valence = {e: d["emotions"][e]["valence"] if e in d["emotions"] else 0 for e in names}
    torch.save({"model": a.model, "layers_path": a.layers_path, "n_layers": len(layers),
                "directions": dirs, "resid_norm": resid_norm, "read_layer": read,
                "valence": valence, "contrast": a.contrast, "auc": aucs,
                "neutral_mean": acts["neutral"].mean(0)}, out / "directions.pt")
    steer_default = round(len(layers) / 3)
    print(f"\nread layer {read} (mean CV AUC {mean_auc[read]:.3f}); default steer layer {steer_default}")
    report = {"model": a.model, "n_layers": len(layers), "read_layer": read, "contrast": a.contrast,
              "mean_auc": mean_auc, "auc": aucs, "resid_norm": resid_norm.tolist(),
              "valence": valence, "cos": {}, "flags": {}}
    for tag, l in (("read", read), ("steer", steer_default)):
        print(f"\ncosine similarity @ {tag} layer {l}:")
        nm, cos = cosine_matrix(dirs, l)
        report["flags"][tag] = print_cos(nm, cos, a.cos_flag)
        report["cos"][tag] = {"layer": l, "names": nm, "matrix": cos.round(4).tolist()}
    (out / "extract.json").write_text(json.dumps(report, indent=1))
    print(f"\nsaved {out/'directions.pt'} and extract.json")


if __name__ == "__main__":
    main()

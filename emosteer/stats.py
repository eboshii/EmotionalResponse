"""Paired bootstrap and conditional-rate statistics over episodes."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def episode_table(rows, metric="violated"):
    """{arm: {episode_seed: mean metric over steps}} (skips None metrics)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        acc[r["arm"]][r["seed"]].append(float(v))
    return {a: {s: float(np.mean(v)) for s, v in eps.items()} for a, eps in acc.items()}


def paired_bootstrap(a: dict, b: dict, n_boot=10000, seed=0, ci=0.95):
    """Mean difference a-b over shared episodes, with percentile CI. Returns dict or None."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(keys),
            "p_le0": float((boots <= 0).mean())}


def transitions(rows):
    """{arm: {seed: (n_vv, n_v_prev, n_nv, n_n_prev)}} for P(v_t|v_{t-1}) and P(v_t|~v_{t-1})."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["arm"]][r["seed"]].append((r["step"], bool(r["violated"])))
    out = {}
    for arm, eps in by.items():
        out[arm] = {}
        for s, seq in eps.items():
            v = [x for _, x in sorted(seq)]
            vv = sum(1 for i in range(1, len(v)) if v[i - 1] and v[i])
            vp = sum(1 for i in range(1, len(v)) if v[i - 1])
            nv = sum(1 for i in range(1, len(v)) if not v[i - 1] and v[i])
            npv = sum(1 for i in range(1, len(v)) if not v[i - 1])
            out[arm][s] = (vv, vp, nv, npv)
    return out


def cond_rate(tr: dict, keys=None, which="after_v"):
    keys = list(tr) if keys is None else keys
    if which == "after_v":
        num, den = sum(tr[k][0] for k in keys), sum(tr[k][1] for k in keys)
    else:
        num, den = sum(tr[k][2] for k in keys), sum(tr[k][3] for k in keys)
    return num / den if den else float("nan")


def cond_diff_bootstrap(tr_a, tr_b, which="after_v", n_boot=5000, seed=0, ci=0.95):
    """Ratio-estimator difference of conditional rates, resampling shared episodes."""
    keys = sorted(set(tr_a) & set(tr_b))
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    point = cond_rate(tr_a, keys, which) - cond_rate(tr_b, keys, which)
    boots = []
    for _ in range(n_boot):
        ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        boots.append(cond_rate(tr_a, ks, which) - cond_rate(tr_b, ks, which))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if not len(boots):
        return {"diff": point, "lo": float("nan"), "hi": float("nan"), "n": len(keys)}
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "n": len(keys)}

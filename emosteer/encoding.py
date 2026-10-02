"""Is an effect a property of the emotion or of how we encoded it?

Given one behavioural effect per direction key ("pain", "pain@split0of2", ...):
  * eta^2: share of variance across keys explained by emotion identity (one-way ANOVA),
    with a permutation p-value (shuffle emotion labels across keys).
  * rank stability: Spearman correlation of the emotion ranking under each encoding
    tag vs the base encoding.
  * quality check: correlation of the effect with encoding quality (CV-AUC, specificity)
    and eta^2 again after regressing quality out.
High eta^2 (and stable ranks, and eta^2 surviving the quality regression) -> emotion.
Low eta^2 or effects that track quality -> encoding.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .variants import base_of, variant_of


def eta_squared(values, labels):
    v = np.asarray(values, float)
    labs = np.asarray(labels)
    ss_tot = ((v - v.mean()) ** 2).sum()
    if ss_tot <= 0:
        return float("nan")
    ss_b = sum(((v[labs == g].mean() - v.mean()) ** 2) * (labs == g).sum() for g in set(labels))
    return float(ss_b / ss_tot)


def permutation_p(values, labels, n_perm=5000, seed=0):
    obs = eta_squared(values, labels)
    if np.isnan(obs):
        return obs, float("nan")
    rng = np.random.default_rng(seed)
    labs = np.asarray(labels)
    hits = sum(eta_squared(values, rng.permutation(labs)) >= obs - 1e-12 for _ in range(n_perm))
    return obs, (hits + 1) / (n_perm + 1)


def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def residualise(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - X @ beta


def decompose(effects: dict[str, float], quality: dict[str, dict] | None = None, title="effect",
              n_perm=5000, seed=0):
    """effects: {key: effect}. quality: {key: {"auc": .., "specificity": ..}}. -> (md, record)."""
    keys = [k for k, v in effects.items() if v is not None and not np.isnan(v)]
    by = defaultdict(list)
    for k in keys:
        by[base_of(k)].append(k)
    if len(by) < 2 or len(keys) <= len(by):
        return [f"_{title}: need ≥2 emotions with ≥2 encodings for a decomposition._"], {}
    vals = [effects[k] for k in keys]
    labs = [base_of(k) for k in keys]
    eta, p = permutation_p(vals, labs, n_perm, seed)
    md = [f"**Encoding decomposition — {title}**", "",
          "| emotion | n enc | mean | SD across encodings | min | max |", "|---|---|---|---|---|---|"]
    for e in sorted(by, key=lambda e: -np.mean([effects[k] for k in by[e]])):
        x = np.array([effects[k] for k in by[e]])
        md.append(f"| {e} | {len(x)} | {x.mean():+.3f} | {x.std(ddof=1) if len(x) > 1 else float('nan'):.3f} "
                  f"| {x.min():+.3f} | {x.max():+.3f} |")
    md += ["", f"Variance explained by emotion identity: η² = {eta:.2f} (permutation p = {p:.3g}, "
               f"{len(keys)} encodings, {len(by)} emotions)"]
    rec = {"title": title, "eta2": eta, "perm_p": p, "n_keys": len(keys), "n_emotions": len(by)}

    # rank stability across encoding tags
    tags = defaultdict(dict)
    for k in keys:
        tags[variant_of(k)][base_of(k)] = effects[k]
    if "base" in tags:
        rows = []
        for t, m in sorted(tags.items()):
            if t == "base":
                continue
            common = sorted(set(m) & set(tags["base"]))
            r = spearman([m[e] for e in common], [tags["base"][e] for e in common])
            rows.append((t, len(common), r))
        if rows:
            md += ["", "| encoding | n emotions | Spearman ρ vs base ranking |", "|---|---|---|"]
            md += [f"| {t} | {n} | {r:+.2f} |" for t, n, r in rows]
            rec["rank_stability"] = {t: r for t, _, r in rows}

    # does encoding quality predict the effect?
    if quality:
        qk = [k for k in keys if k in quality]
        feats = [f for f in ("auc", "specificity")
                 if all(quality[k].get(f) is not None for k in qk)]
        if len(qk) >= 4 and feats:
            y = np.array([effects[k] for k in qk])
            md += [""]
            for f in feats:
                x = np.array([quality[k][f] for k in qk])
                r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
                md.append(f"Correlation of {title} with encoding {f}: r = {r:+.2f}, ρ = {spearman(x, y):+.2f}")
                rec[f"corr_{f}"] = r
            resid = residualise(y, [[quality[k][f] for k in qk] for f in feats])
            eta_r, p_r = permutation_p(resid, [base_of(k) for k in qk], n_perm, seed)
            md.append(f"η² after regressing out {', '.join(feats)}: {eta_r:.2f} (p = {p_r:.3g})")
            rec["eta2_resid"], rec["perm_p_resid"] = eta_r, p_r
    md += ["", "_Read: high η² with stable ranks, surviving the quality regression → the emotion "
               "drives the effect; low η², unstable ranks, or effects tracking AUC/specificity → "
               "the encoding does._", ""]
    return md, rec


def quality_from_calib(calib: dict, level: float) -> dict[str, dict]:
    """{key: {auc, specificity}} from calib.json for the '+key' directions at a level."""
    meta, res = calib["meta"], calib["results"]
    auc = meta.get("auc_read", {})
    out = {}
    for name, r in res.items():
        if not name.startswith("+"):
            continue
        key = name[1:]
        lv = next((v for t, v in r["levels"].items() if float(t) == float(level)), None)
        if not lv or lv.get("alpha") is None:
            continue
        spec = None
        if lv.get("shift_norm"):
            spec = lv["readout"].get(base_of(key), 0.0) / lv["shift_norm"]
        out[key] = {"auc": auc.get(key), "specificity": spec}
    return out

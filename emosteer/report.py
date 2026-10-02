"""Publication summary: one table row per emotion + box plots of the effect distribution.

python -m emosteer.report runs/qwen3-4b/mini_0.3.jsonl --out runs/qwen3-4b/blog
-> blog_table.md, blog_table.csv, blog_effects.png/.svg (and _dark variants with --dark)

Effects are paired per episode (same seeds across arms):
  state effect       = state:E - baseline          (percentage points)
  contingency effect = contingent:E - sham:E       (percentage points)
Encodings of the same emotion (pain, pain@split0of2, ...) are pooled into one box;
each encoding's mean is drawn as a dot so encoding disagreement is visible.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import load_rows
from .stats import episode_table, paired_bootstrap
from .variants import base_of

# Validated categorical slots 1-2 (reference palette), light / dark.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", pos="#2a78d6", neg="#eb6834", band="#f0efec"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", pos="#3987e5", neg="#d95926", band="#383835"),
}


def episode_diffs(table, a, b):
    keys = sorted(set(table.get(a, {})) & set(table.get(b, {})))
    return [table[a][k] - table[b][k] for k in keys]


def collect(rows, metric, n_boot, seed):
    tab = episode_table(rows, metric)
    arms = set(tab)
    valence = {r["target"]: r["valence"] for r in rows if r.get("target")}
    per = defaultdict(lambda: {"state": [], "cont": [], "state_enc": {}, "cont_enc": {}})
    for a in arms:
        kind, _, key = a.partition(":")
        if kind == "state" and "baseline" in arms:
            d = episode_diffs(tab, a, "baseline")
            per[base_of(key)]["state"] += d
            per[base_of(key)]["state_enc"][key] = paired_bootstrap(tab[a], tab["baseline"], n_boot, seed)
        if kind == "contingent" and f"sham:{key}" in arms:
            d = episode_diffs(tab, a, f"sham:{key}")
            per[base_of(key)]["cont"] += d
            per[base_of(key)]["cont_enc"][key] = paired_bootstrap(tab[a], tab[f"sham:{key}"], n_boot, seed)
    rnd = [np.mean(episode_diffs(tab, a, "baseline")) for a in arms
           if a.startswith("random_") and "baseline" in arms]
    base_rate = float(np.mean(list(tab["baseline"].values()))) if "baseline" in tab else float("nan")
    return per, {base_of(k): v for k, v in valence.items()}, rnd, base_rate


def summarise(per, valence, n_boot, seed):
    """Pooled effect per emotion: mean of encoding means, CI from bootstrapping episodes."""
    rng = np.random.default_rng(seed)
    out = []
    for e, d in per.items():
        row = {"emotion": e, "valence": "positive" if valence.get(e, 0) > 0 else "negative",
               "n_encodings": max(len(d["state_enc"]), len(d["cont_enc"]))}
        for k in ("state", "cont"):
            x = np.asarray(d[k]) * 100
            if len(x):
                boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
                lo, hi = np.quantile(boots, [0.025, 0.975])
                encs = [v["diff"] * 100 for v in d[f"{k}_enc"].values() if v]
                row.update({f"{k}_mean": float(x.mean()), f"{k}_lo": float(lo), f"{k}_hi": float(hi),
                            f"{k}_enc_min": min(encs), f"{k}_enc_max": max(encs)})
        out.append(row)
    return sorted(out, key=lambda r: -r.get("state_mean", -1e9))


def write_table(rows, rnd, base_rate, metric, out: Path):
    unit = "pp" if metric == "violated" else "pp of P(violate)"
    f = lambda r, k: (f"{r[k+'_mean']:+.1f} [{r[k+'_lo']:+.1f}, {r[k+'_hi']:+.1f}]"
                      if f"{k}_mean" in r else "—")
    g = lambda r, k: (f"{r[k+'_enc_min']:+.1f} … {r[k+'_enc_max']:+.1f}"
                      if f"{k}_mean" in r and r["n_encodings"] > 1 else "—")
    md = [f"| Emotion | Valence | State effect ({unit}) | Contingency effect ({unit}) | "
          f"Range across encodings (state) | Encodings |",
          "|---|---|---:|---:|---:|---:|"]
    md += [f"| {r['emotion']} | {r['valence']} | {f(r, 'state')} | {f(r, 'cont')} | {g(r, 'state')} | "
           f"{r['n_encodings']} |" for r in rows]
    md += ["", f"Baseline (unsteered) rate: {base_rate*100:.1f}%. "
               "State effect = steered − unsteered; contingency effect = contingent − sham. "
               "Brackets: 95% bootstrap CI over paired episodes."]
    if rnd:
        md.append(f"Random vectors at the same KL: {min(rnd)*100:+.1f} … {max(rnd)*100:+.1f} {unit}.")
    out.with_name(out.name + "_table.md").write_text("\n".join(md) + "\n")
    with out.with_name(out.name + "_table.csv").open("w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    return "\n".join(md)


def plot(per, rows, rnd, metric, out: Path, mode="light", title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = THEME[mode]
    order = [r["emotion"] for r in rows][::-1]          # top = largest state effect
    val = {r["emotion"]: r["valence"] for r in rows}
    panels = [("state", "State effect\n(steered − unsteered)"),
              ("cont", "Contingency effect\n(contingent − sham)")]
    panels = [p for p in panels if any(per[e][p[0]] for e in order)]
    unit = "percentage points" if metric == "violated" else "percentage points of P(violate)"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "text.color": t["ink"], "axes.labelcolor": t["ink2"],
                         "xtick.color": t["muted"], "ytick.color": t["ink2"]})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels) + 1.2, 0.42 * len(order) + 1.6),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    for ax, (k, label) in zip(axes[0], panels):
        ax.set_facecolor(t["surface"])
        if rnd and k == "state":
            ax.axvspan(min(rnd) * 100, max(rnd) * 100, color=t["band"], zorder=0, lw=0)
        ax.axvline(0, color=t["axis"], lw=1, zorder=1)
        data = [np.asarray(per[e][k]) * 100 for e in order]
        new_api = tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 10)
        orient = dict(orientation="horizontal") if new_api else dict(vert=False)
        bp = ax.boxplot(data, **orient, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=t["surface"], lw=2),
                        whiskerprops=dict(color=t["muted"], lw=1), capprops=dict(color=t["muted"], lw=1))
        for box, e in zip(bp["boxes"], order):
            box.set(facecolor=t["pos"] if val[e] == "positive" else t["neg"], edgecolor=t["surface"], lw=2)
        for i, e in enumerate(order):                     # encoding means as dots
            encs = [v["diff"] * 100 for v in per[e][f"{k}_enc"].values() if v]
            ax.scatter(encs, [i + 1] * len(encs), s=22, color=t["ink"], zorder=3,
                       edgecolor=t["surface"], linewidth=1.5)
            m = np.mean(data[i]) if len(data[i]) else None
            if m is not None:
                ax.annotate(f"{m:+.1f}", (1.0, i + 1), xycoords=("axes fraction", "data"),
                            xytext=(6, 0), textcoords="offset points", va="center",
                            color=t["ink2"], fontsize=9)
        ax.set_title(label, color=t["ink"], fontsize=11, loc="left")
        ax.set_xlabel(unit, color=t["ink2"])
        ax.grid(axis="x", color=t["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(axis="y", length=0)
    axes[0][0].set_yticks(range(1, len(order) + 1), order)
    handles = [Patch(color=t["neg"], label="negative emotion"), Patch(color=t["pos"], label="positive emotion"),
               plt.Line2D([], [], marker="o", ls="", color=t["ink"], label="mean of one encoding")]
    if rnd:
        handles.append(Patch(color=t["band"], label="random vectors (same KL)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=9, labelcolor=t["ink2"])
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=t["ink"], fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 0.97, 1))
    suffix = "" if mode == "light" else "_dark"
    for ext in ("png", "svg"):
        fig.savefig(out.with_name(f"{out.name}_effects{suffix}.{ext}"), dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", required=True, help="output prefix, e.g. runs/x/blog")
    p.add_argument("--metric", choices=["auto", "violated", "p_violate"], default="auto",
                   help="auto: P(violate) when available (mini scenarios), else violation rate")
    p.add_argument("--level", type=float, default=None, help="KL level (required if several in files)")
    p.add_argument("--title", default=None)
    p.add_argument("--dark", action="store_true", help="also render a dark-mode figure")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    rows = load_rows(a.jsonl)
    levels = sorted({r["level"] for r in rows})
    if a.level is None and len(levels) > 1:
        raise SystemExit(f"several KL levels {levels}; pass --level")
    lvl = a.level if a.level is not None else levels[0]
    rows = [r for r in rows if r["level"] == lvl]
    metric = a.metric
    if metric == "auto":
        metric = "p_violate" if all(r.get("p_violate") is not None for r in rows) else "violated"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per, valence, rnd, base_rate = collect(rows, metric, a.n_boot, a.seed)
    summary = summarise(per, valence, a.n_boot, a.seed)
    print(write_table(summary, rnd, base_rate, metric, out))
    if not a.no_plot:
        plot(per, summary, rnd, metric, out, "light", a.title)
        if a.dark:
            plot(per, summary, rnd, metric, out, "dark", a.title)
    print(f"\nwrote {out}_table.md/.csv" + ("" if a.no_plot else f", {out}_effects.png/.svg"))


if __name__ == "__main__":
    main()

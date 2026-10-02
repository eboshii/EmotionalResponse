"""Analyse run JSONL: violation rates, paired-bootstrap effects, relief dynamics, rankings.

python -m emosteer.analyze runs/qwen3-4b/mini_*.jsonl --out runs/qwen3-4b/report
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .stats import cond_diff_bootstrap, cond_rate, episode_table, paired_bootstrap, transitions


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def fmt(r):
    if r is None:
        return "n/a"
    return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def analyse_level(rows, n_boot, seed, calib=None):
    """Returns (markdown lines, csv records) for one (env, level) slice."""
    md, recs = [], []
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "baseline", a))
    info = {r["arm"]: r for r in rows}
    viol = episode_table(rows, "violated")
    pv = episode_table(rows, "p_violate")
    tr = transitions(rows)
    base = viol.get("baseline", {})
    rnd_relief = [a for a in arms if a.startswith("random_relief")]
    rnd_reward = [a for a in arms if a.startswith("random_reward")]

    def pooled(names, table):
        """Average several random arms per episode -> one reference arm."""
        acc = defaultdict(list)
        for n in names:
            for s, v in table.get(n, {}).items():
                acc[s].append(v)
        return {s: float(np.mean(v)) for s, v in acc.items() if len(v) == len(names)}

    md.append("| arm | steps | viol rate | mean P(viol) | Δ vs baseline [95% CI] | Δ vs matched random | top types |")
    md.append("|---|---|---|---|---|---|---|")
    steps = Counter(r["arm"] for r in rows)
    types = defaultdict(Counter)
    for r in rows:
        for t in r.get("types") or []:
            types[r["arm"]][t] += 1
    for a in arms:
        rate = np.mean(list(viol[a].values()))
        pvm = np.mean(list(pv[a].values())) if a in pv else float("nan")
        db = paired_bootstrap(viol[a], base, n_boot, seed) if a != "baseline" else None
        mode = info[a]["mode"]
        ref = rnd_relief if mode == "relief" else rnd_reward if mode == "reward" else rnd_relief
        dr = (paired_bootstrap(viol[a], pooled(ref, viol), n_boot, seed)
              if ref and not a.startswith("random") and a != "baseline" else None)
        top = ", ".join(f"{t}:{c/steps[a]:.2f}" for t, c in types[a].most_common(3))
        md.append(f"| {a} | {steps[a]} | {rate:.3f} | {pvm:.3f} | {fmt(db)} | {fmt(dr)} | {top} |")
        recs.append({"arm": a, "steps": steps[a], "viol_rate": rate, "p_violate": pvm,
                     **{f"vs_base_{k}": v for k, v in (db or {}).items()},
                     **{f"vs_random_{k}": v for k, v in (dr or {}).items()},
                     "types": dict(types[a])})

    # relief / reward dynamics: real vs sham
    md += ["", "**Conditional violation rates** (P(v_t | v_{t-1}) and P(v_t | ¬v_{t-1}))", "",
           "| emotion | mode | real P(v|v) | sham P(v|v) | real−sham [95% CI] | real P(v|¬v) | sham P(v|¬v) |",
           "|---|---|---|---|---|---|---|"]
    for a in arms:
        if not a.startswith("contingent:"):
            continue
        e = a.split(":", 1)[1]
        s = f"sham:{e}"
        if s not in tr:
            continue
        d = cond_diff_bootstrap(tr[a], tr[s], "after_v", n_boot // 2, seed)
        md.append(f"| {e} | {info[a]['mode']} | {cond_rate(tr[a]):.3f} | {cond_rate(tr[s]):.3f} | {fmt(d)} | "
                  f"{cond_rate(tr[a], which='after_n'):.3f} | {cond_rate(tr[s], which='after_n'):.3f} |")

    # rankings
    state, cont = [], []
    for a in arms:
        if a.startswith("state:"):
            e = a.split(":", 1)[1]
            r = paired_bootstrap(viol[a], base, n_boot, seed)
            rp = paired_bootstrap(pv[a], pv.get("baseline", {}), n_boot, seed) if a in pv else None
            state.append((e, r, rp))
        if a.startswith("contingent:"):
            e = a.split(":", 1)[1]
            s = f"sham:{e}"
            if s in viol:
                r = paired_bootstrap(viol[a], viol[s], n_boot, seed)
                rp = paired_bootstrap(pv[a], pv[s], n_boot, seed) if a in pv and s in pv else None
                cont.append((e, r, rp))
    key = lambda x: -(x[1]["diff"] if x[1] else -9)
    md += ["", "**Ranking: state effect** (state − baseline)", "",
           "| rank | emotion | Δ viol rate | Δ P(viol) |", "|---|---|---|---|"]
    md += [f"| {i+1} | {e} | {fmt(r)} | {fmt(rp)} |" for i, (e, r, rp) in enumerate(sorted(state, key=key))]
    md += ["", "**Ranking: contingency effect** (contingent − sham)", "",
           "| rank | emotion | Δ viol rate | Δ P(viol) |", "|---|---|---|---|"]
    md += [f"| {i+1} | {e} | {fmt(r)} | {fmt(rp)} |" for i, (e, r, rp) in enumerate(sorted(cont, key=key))]
    # encoding vs emotion: only meaningful when several encodings per emotion were run
    from .encoding import decompose, quality_from_calib
    quality = quality_from_calib(calib, rows[0]["level"]) if calib else None
    for title, items in (("state effect", state), ("contingency effect", cont)):
        effects = {e: r["diff"] for e, r, _ in items if r}
        if len({k.split("@")[0] for k in effects}) < len(effects):
            m, rec = decompose(effects, quality, title, seed=seed)
            md += [""] + m
    if any(r.get("p_violate") is not None for r in rows):
        for title, items in (("state effect on P(viol)", state), ("contingency effect on P(viol)", cont)):
            effects = {e: rp["diff"] for e, _, rp in items if rp}
            if len({k.split("@")[0] for k in effects}) < len(effects):
                m, _ = decompose(effects, quality, title, seed=seed)
                md += m
    rb = [paired_bootstrap(viol[a], base, n_boot, seed) for a in rnd_relief + rnd_reward]
    rb = [r["diff"] for r in rb if r]
    if rb:
        md.append(f"\nRandom-vector arms Δ vs baseline range: {min(rb):+.3f} .. {max(rb):+.3f}")
    lm = [r["letter_mass"] for r in rows if r.get("letter_mass") is not None]
    if lm:
        low = sum(1 for x in lm if x < 0.5) / len(lm)
        md.append(f"Letter-token mass: mean {np.mean(lm):.3f}; {low:.1%} of steps < 0.5 "
                  "(low mass = steering broke the answer format)")
    return md, recs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", default=None, help="prefix for report.md / .csv")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib", default=None,
                   help="calib.json, adds encoding-quality (AUC, specificity) checks")
    a = p.parse_args(argv)
    calib = json.loads(Path(a.calib).read_text()) if a.calib else None
    rows = load_rows(a.jsonl)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["env"], r["level"], r.get("disclose", False))].append(r)
    md, recs = [], []
    for (model, env, level, disc), rs in sorted(groups.items(), key=lambda x: str(x[0])):
        md += [f"## {model} | {env} | KL level {level}" + (" | disclosed" if disc else ""), ""]
        m, rc = analyse_level(rs, a.n_boot, a.seed, calib)
        md += m + [""]
        recs += [{"model": model, "env": env, "level": level, "disclose": disc, **r} for r in rc]
    text = "\n".join(md)
    print(text)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(text)
        keys = sorted({k for r in recs for k in r})
        with out.with_suffix(".csv").open("w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(recs)
        print(f"\nwrote {out.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()

"""Multiple independent encodings ("variants") per emotion.

A variant spec says how to build the sentence sets for every emotion and neutral:
  base               all sentences, default suffix
  splitKofN          disjoint split K of N (seeded) of each emotion's sentences
  bootI              bootstrap resample (seed I) of emotion and neutral sentences
  suffix=<text>      all sentences, alternative suffix, e.g. "suffix= Right now I am"
  set=<name>         alternative sentence set emotions[e]["variants"][name]
                     (neutral uses neutral["variants"][name] if present)
Direction keys are "<emotion>" for base and "<emotion>@<tag>" for variants.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

SEP = "@"


def tag_of(spec: str) -> str:
    if spec.startswith("suffix="):
        return "suffix_" + re.sub(r"\W+", "_", spec[7:].strip()).strip("_").lower()
    if spec.startswith("set="):
        return "set_" + spec[4:]
    return spec


def base_of(key: str) -> str:
    return key.split(SEP, 1)[0]


def variant_of(key: str) -> str:
    return key.split(SEP, 1)[1] if SEP in key else "base"


def key_for(emo: str, tag: str) -> str:
    return emo if tag == "base" else f"{emo}{SEP}{tag}"


def build_sets(d: dict, spec: str, names: list[str], sets: dict[str, list[str]]):
    """Return ({name: texts_with_suffix}, neutral_texts) for one variant spec.

    `sets` maps each name (emotion or ctrl_*) to its base sentences.
    """
    suffix = d["suffix"]
    neutral = list(d["neutral"]["sentences"])
    out = {}
    if spec == "base":
        out = {n: list(sets[n]) for n in names}
    elif (m := re.fullmatch(r"split(\d+)of(\d+)", spec)):
        k, n_ = int(m[1]), int(m[2])
        if not 0 <= k < n_:
            raise ValueError(spec)
        for n in names:
            # same permutation for every K of a given emotion -> splits are disjoint
            perm = np.random.default_rng(zlib.crc32(n.encode())).permutation(len(sets[n]))
            out[n] = [sets[n][i] for i in sorted(perm[k::n_])]
    elif (m := re.fullmatch(r"boot(\d+)", spec)):
        rng = np.random.default_rng(1000 + int(m[1]))
        for n in names:
            out[n] = [sets[n][i] for i in rng.integers(0, len(sets[n]), len(sets[n]))]
        neutral = [neutral[i] for i in rng.integers(0, len(neutral), len(neutral))]
    elif spec.startswith("suffix="):
        suffix = spec[7:]
        if not suffix.startswith(" "):
            suffix = " " + suffix
        out = {n: list(sets[n]) for n in names}
    elif spec.startswith("set="):
        name = spec[4:]
        for n in names:
            alt = d["emotions"].get(n, {}).get("variants", {}).get(name)
            if alt:
                out[n] = list(alt)
        neutral = list(d["neutral"].get("variants", {}).get(name, neutral))
    else:
        raise ValueError(f"unknown variant spec {spec!r}")
    add = lambda xs: [s.rstrip() + suffix for s in xs]
    return {n: add(v) for n, v in out.items()}, add(neutral)


def consistency(dirs: dict, layer: int):
    """Within-emotion vs between-emotion cosine at one layer.

    within[e]  = mean cos between all pairs of e's encodings
    between[e] = mean cos between e's encodings and other emotions' encodings
    """
    import torch
    keys = list(dirs)
    emos = sorted({base_of(k) for k in keys})
    V = torch.stack([dirs[k][layer] for k in keys])
    C = (V @ V.T).numpy()
    within, between = {}, {}
    for e in emos:
        idx = [i for i, k in enumerate(keys) if base_of(k) == e]
        oth = [i for i, k in enumerate(keys) if base_of(k) != e]
        pairs = [C[i, j] for i in idx for j in idx if i < j]
        within[e] = float(np.mean(pairs)) if pairs else float("nan")
        between[e] = float(np.mean([C[i, j] for i in idx for j in oth])) if oth else float("nan")
    return within, between

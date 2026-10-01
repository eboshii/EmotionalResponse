"""Emotion dataset loading and linting."""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "data" / "emotions.json"


def load_emotions(path=DEFAULT, emotions: list[str] | None = None):
    d = json.loads(Path(path).read_text())
    if emotions:
        missing = set(emotions) - set(d["emotions"])
        if missing:
            raise KeyError(f"unknown emotions: {missing}")
        d["emotions"] = {k: d["emotions"][k] for k in emotions}
    return d


def with_suffix(sentences, suffix):
    return [s.rstrip() + suffix for s in sentences]


def lint(d) -> list[str]:
    """Report sentences that name their own emotion (or a listed synonym)."""
    problems = []
    for emo, spec in d["emotions"].items():
        words = d.get("banned_words", {}).get(emo, [emo])
        pat = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
        for s in spec["sentences"]:
            if pat.search(s):
                problems.append(f"[{emo}] {s}")
        if len(spec["sentences"]) < 20:
            problems.append(f"[{emo}] only {len(spec['sentences'])} sentences (<20)")
    return problems

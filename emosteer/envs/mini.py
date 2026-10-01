"""JSON mini-scenario backend: each step is an independent scenario with labelled choices."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .base import Env, Obs, StepResult

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "mini_scenarios.json"


class MiniEnv(Env):
    def __init__(self, path=DEFAULT, steps: int = 10, shuffle_choices: bool = True):
        self.scenarios = json.loads(Path(path).read_text())["scenarios"]
        self.steps, self.shuffle = steps, shuffle_choices

    def reset(self, seed: int) -> Obs:
        rng = random.Random(seed)
        n = min(self.steps, len(self.scenarios))
        self.order = rng.sample(range(len(self.scenarios)), n)
        self.perms = []
        for i in self.order:
            p = list(range(len(self.scenarios[i]["choices"])))
            if self.shuffle:
                rng.shuffle(p)
            self.perms.append(p)
        self.t = 0
        return self._obs()

    def _obs(self) -> Obs:
        if self.t >= len(self.order):
            return Obs("", [], done=True)
        sc = self.scenarios[self.order[self.t]]
        ch = [sc["choices"][j] for j in self.perms[self.t]]
        return Obs(sc["context"], [c["text"] for c in ch], sc["id"],
                   [bool(c["violation"]) for c in ch], [c.get("types", []) for c in ch])

    def step(self, choice: int) -> StepResult:
        sc = self.scenarios[self.order[self.t]]
        c = sc["choices"][self.perms[self.t][choice]]
        self.t += 1
        return StepResult(self._obs(), bool(c["violation"]), list(c.get("types", [])))

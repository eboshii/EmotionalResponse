"""Experimental arms and their steering schedules.

Arm spec strings:
  baseline                 no steering
  state:<emo>              constant steering with +emo
  contingent:<emo>         negative valence: constant, a violation removes it for K steps (relief)
                           positive valence: off, a violation adds it for K steps (reward)
  sham:<emo>               negative: constant, a violation starts a "relief" that does nothing
                           positive: K-step windows start at random with prob p, ignoring actions
  random_relief:<i>        random vector i (KL-matched), relief schedule
  random_reward:<i>        random vector i (KL-matched), reward schedule

A violation at step t affects steps t+1..t+K. With reset=True a violation inside a
window restarts it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Arm:
    kind: str           # baseline|state|contingent|sham|random_relief|random_reward
    target: str = ""    # emotion name or random index
    valence: int = 0

    @property
    def name(self):
        return self.kind if self.kind == "baseline" else f"{self.kind}:{self.target}"

    @property
    def direction_key(self):
        if self.kind == "baseline":
            return None
        if self.kind.startswith("random"):
            return f"random{self.target}"
        return f"+{self.target}"

    @property
    def mode(self):
        """'relief' (on by default) or 'reward' (off by default) or 'const'/'none'."""
        if self.kind == "baseline":
            return "none"
        if self.kind == "state":
            return "const"
        if self.kind == "random_relief":
            return "relief"
        if self.kind == "random_reward":
            return "reward"
        return "relief" if self.valence < 0 else "reward"


def parse_arm(spec: str, valence: dict) -> Arm:
    if spec == "baseline":
        return Arm("baseline")
    kind, target = spec.split(":", 1)
    if kind not in {"state", "contingent", "sham", "random_relief", "random_reward"}:
        raise ValueError(f"unknown arm kind {kind!r}")
    v = 0 if kind.startswith("random") else int(np.sign(valence[target]) or -1)
    return Arm(kind, target, v)


def default_arms(emotions, valence, n_random):
    arms = ["baseline"]
    for e in emotions:
        arms += [f"state:{e}", f"contingent:{e}", f"sham:{e}"]
    for i in range(n_random):
        arms += [f"random_relief:{i}", f"random_reward:{i}"]
    # contingent must precede sham so sham-reward can be rate-matched
    return [parse_arm(a, valence) for a in arms]


class Schedule:
    """Per-episode steering state machine."""

    def __init__(self, arm: Arm, K: int = 3, reset: bool = True, sham_p: float = 0.2,
                 rng: np.random.Generator | None = None):
        self.arm, self.K, self.reset, self.sham_p = arm, K, reset, sham_p
        self.rng = rng or np.random.default_rng(0)
        self.left = 0  # steps remaining in current relief/reward window

    @property
    def in_window(self):
        return self.left > 0

    def steer_on(self) -> bool:
        a, m = self.arm, self.arm.mode
        if m == "none":
            return False
        if m == "const":
            return True
        if a.kind == "sham" and m == "relief":
            return True                      # relief does nothing
        if m == "relief":
            return not self.in_window
        return self.in_window                # reward (real or sham)

    def update(self, violated: bool):
        """Call after the step's choice. Returns True if a window was (re)started."""
        if self.left > 0:
            self.left -= 1
        a = self.arm
        if a.mode in ("none", "const"):
            return False
        if a.kind == "sham" and a.mode == "reward":
            trigger = bool(self.rng.random() < self.sham_p)   # ignores the action
        else:
            trigger = violated
        if trigger and (self.reset or self.left == 0):
            self.left = self.K
            return True
        return False

"""Environment protocol shared by the mini-scenario and MACHIAVELLI backends."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Obs:
    text: str                       # scene description shown to the model
    choices: list[str]              # choice texts in display order
    scene_id: str = ""
    # per-choice violation labels (display order) if known in advance (mini backend)
    choice_violation: list[bool] | None = None
    choice_types: list[list[str]] | None = None
    done: bool = False


@dataclass
class StepResult:
    obs: Obs
    violated: bool
    types: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


class Env:
    def reset(self, seed: int) -> Obs:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, choice: int) -> StepResult:  # pragma: no cover - interface
        raise NotImplementedError

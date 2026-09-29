"""
controllers/base.py

Shared types and helpers for all Jidenna controllers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from ..jidenna_pose import PoseEstimate


@dataclass
class ControlOutput:
    v: float = 0.0
    w: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class Controller(Protocol):
    def reset(self) -> None:
        ...

    def compute(self, pose: PoseEstimate) -> ControlOutput:
        ...


def wrap_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x
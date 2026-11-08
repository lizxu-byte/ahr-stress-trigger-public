"""Streaming simulator for a physiologically triggered ESM prompt.

The trigger consumes per-window scores in time order and decides at each
window end whether to send a prompt, using only what it has seen so far:

* the score (e.g. AHR) must exceed ``threshold`` for at least
  ``min_duration_s`` seconds: a *usable* window (good quality, no motion
  flag) at or below threshold ends the run,
* unusable windows carry no information about the score, so they neither
  extend nor end the run, unless no usable window above threshold has been
  seen for more than ``max_gap_s`` seconds (``max_gap_s=0``: any unusable
  window ends the run),
* the prompt itself is only sent on a usable, above-threshold window,
* at least ``refractory_s`` seconds must have passed since the previous
  prompt (participant burden).

The scores themselves must be computed causally (see :mod:`ahr.pipeline`);
the trigger adds no look-ahead of its own, which :func:`simulate` checks by
requiring strictly increasing time stamps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriggerConfig:
    threshold: float = 10.0  # bpm above prediction (AHR) or above baseline (raw HR)
    min_duration_s: float = 30.0
    refractory_s: float = 20 * 60.0
    max_gap_s: float = 60.0
    require_quality: bool = True
    block_motion: bool = True


class StreamingTrigger:
    """Stateful trigger; call :meth:`update` once per window, in time order."""

    def __init__(self, config: TriggerConfig):
        self.cfg = config
        self.run_start: float | None = None
        self.last_prompt: float = -np.inf
        self.last_above: float = -np.inf
        self.last_t: float = -np.inf
        self.prompts: list[float] = []

    def _usable(self, score: float, quality_ok: bool, motion: bool) -> bool:
        if not np.isfinite(score):
            return False
        if self.cfg.require_quality and not quality_ok:
            return False
        if self.cfg.block_motion and motion:
            return False
        return True

    def update(self, t: float, score: float, quality_ok: bool = True, motion: bool = False) -> bool:
        """Process the window ending at ``t``; return True if a prompt fires now."""
        if t <= self.last_t:
            raise ValueError(f"windows must arrive in strictly increasing time order ({t} <= {self.last_t})")
        self.last_t = t
        if not self._usable(score, quality_ok, motion):
            if self.run_start is not None and t - self.last_above > self.cfg.max_gap_s:
                self.run_start = None
            return False
        if score <= self.cfg.threshold:
            self.run_start = None
            return False
        self.last_above = t
        if self.run_start is None:
            self.run_start = t
        if t - self.run_start >= self.cfg.min_duration_s and t - self.last_prompt >= self.cfg.refractory_s:
            self.last_prompt = t
            self.prompts.append(t)
            return True
        return False


def simulate(
    df: pd.DataFrame,
    config: TriggerConfig,
    score_col: str = "ahr",
    quality_col: str = "quality_ok",
    motion_col: str = "motion",
) -> np.ndarray:
    """Replay a feature table through :class:`StreamingTrigger`; return prompt times (s)."""
    trig = StreamingTrigger(config)
    t = df["t"].to_numpy(float)
    s = df[score_col].to_numpy(float)
    q = df[quality_col].to_numpy(bool) if quality_col in df else np.ones(len(df), bool)
    m = df[motion_col].to_numpy(bool) if motion_col in df else np.zeros(len(df), bool)
    for i in range(len(df)):
        trig.update(t[i], s[i], q[i], m[i])
    return np.asarray(trig.prompts)

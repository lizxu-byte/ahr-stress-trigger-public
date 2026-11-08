"""Window tables, per-subject calibration and leave-one-subject-out splits for classification.

Shared by the hand-crafted feature models and the 1D-CNN so that every model
sees exactly the same windows and folds.

Calibration
-----------
Per-subject normalisation uses only a *calibration interval*: the first
``calib_s`` seconds of the subject's baseline condition. Statistics are
computed from samples inside that interval only, so data recorded later
(including the stress condition) can never influence them, and windows that
overlap the interval are excluded from training and evaluation, so no
evaluated window has contributed to its own normalisation. In a deployment
this corresponds to a short labelled resting calibration at the start of use.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pipeline import window_labels


@dataclass
class WindowConfig:
    win_s: float = 60.0
    step_s: float = 30.0  # 50 % overlap
    min_purity: float = 0.95  # share of the window inside one condition
    calib_s: float = 300.0  # first 5 min of baseline


def calibration_interval(labels: np.ndarray, label_fs: float, calib_s: float, baseline_label: int = 1) -> tuple[float, float]:
    """``(start, end)`` in s of the first ``calib_s`` seconds of the first baseline segment."""
    labels = np.asarray(labels)
    idx = np.flatnonzero(labels == baseline_label)
    if len(idx) == 0:
        raise ValueError("no baseline samples")
    start = idx[0]
    # end of the first contiguous baseline run
    breaks = np.flatnonzero(np.diff(idx) > 1)
    run_end = idx[breaks[0]] if len(breaks) else idx[-1]
    end = min(run_end + 1, start + int(round(calib_s * label_fs)))
    return start / label_fs, end / label_fs


def make_windows(labels: np.ndarray, label_fs: float, duration_s: float, calib: tuple[float, float], cfg: WindowConfig) -> pd.DataFrame:
    """Candidate windows ``[t0, t0 + win_s)`` with majority label, purity and calibration overlap.

    Returns columns ``t0, t1, label, label_purity, in_calibration, usable``
    where ``usable`` = purity >= ``min_purity`` and no overlap with the
    calibration interval.
    """
    t1 = np.arange(cfg.win_s, duration_s + 1e-9, cfg.step_s)
    wl = window_labels(t1, labels, label_fs, cfg.win_s)
    df = pd.DataFrame({"t0": t1 - cfg.win_s, "t1": t1, "label": wl["label"].to_numpy(), "label_purity": wl["label_purity"].to_numpy()})
    df["in_calibration"] = (df["t0"] < calib[1]) & (df["t1"] > calib[0])
    df["usable"] = (df["label_purity"] >= cfg.min_purity) & ~df["in_calibration"]
    return df


def calibration_stats(x: np.ndarray, fs: float, calib: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD per channel of ``x`` (n,) or (n, c) over the calibration interval only."""
    x = np.asarray(x, float)
    lo, hi = int(np.floor(calib[0] * fs)), int(np.floor(calib[1] * fs))
    seg = x[lo:hi]
    if len(seg) < 2:
        raise ValueError("calibration interval too short")
    mu, sd = np.nanmean(seg, axis=0), np.nanstd(seg, axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def zscore_with_calibration(x: np.ndarray, fs: float, calib: tuple[float, float]) -> np.ndarray:
    """``(x - mean) / sd`` with statistics from the calibration interval only."""
    mu, sd = calibration_stats(x, fs, calib)
    return (np.asarray(x, float) - mu) / sd


def cut_windows(x: np.ndarray, fs: float, t0: np.ndarray, win_s: float) -> np.ndarray:
    """Stack windows ``x[t0*fs : t0*fs + win_s*fs]`` -> (n_windows, n_samples[, c])."""
    n = int(round(win_s * fs))
    starts = np.round(np.asarray(t0) * fs).astype(int)
    x = np.asarray(x)
    if starts.size and starts.max() + n > len(x):
        raise ValueError("window exceeds signal")
    return np.stack([x[s : s + n] for s in starts]) if starts.size else np.empty((0, n) + x.shape[1:])


def loso_splits(subjects: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(test_subject, train_idx, test_idx)`` for every subject."""
    subjects = np.asarray(subjects)
    for s in pd.unique(subjects):
        yield s, np.flatnonzero(subjects != s), np.flatnonzero(subjects == s)


def inner_split(train_subjects: np.ndarray, n_val: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split training indices by subject into inner-train / inner-validation (no subject in both)."""
    train_subjects = np.asarray(train_subjects)
    uniq = pd.unique(train_subjects)
    val_s = rng.choice(uniq, size=min(n_val, len(uniq) - 1), replace=False)
    is_val = np.isin(train_subjects, val_s)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    """Unweighted mean F1 over ``classes`` (classes absent from both y_true and y_pred are skipped)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    """Mean per-class recall over the classes present in ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rec = [np.mean(y_pred[y_true == c] == c) for c in classes if np.any(y_true == c)]
    return float(np.mean(rec)) if rec else float("nan")

"""Accelerometer processing: gravity removal, activity magnitude, motion flags.

Gravity is removed **per axis before** taking the vector magnitude. Taking
the magnitude first and subtracting 1 g (as in the classic ENMO) is only
correct when the dynamic acceleration is parallel to gravity: for motion
orthogonal to gravity, ``|g + a| - |g| ~ |a|^2 / (2|g|)``, so small
movements almost vanish, and any calibration offset of the sensor leaks in
as a constant bias. Per-axis high-pass filtering (or moving-average
subtraction) keeps the dynamic component of every axis, and the magnitude of
that residual is an ENMO-like activity measure in g.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .ppg import moving_average


def remove_gravity(
    acc: np.ndarray, fs: float, method: str = "highpass", cutoff_hz: float = 0.5, window_s: float = 2.0, causal: bool = False
) -> np.ndarray:
    """Remove the gravity (slowly varying) component of each axis separately.

    acc       (n, 3) array in g
    method    ``"highpass"``: 2nd-order Butterworth high-pass at ``cutoff_hz``
              ``"moving_average"``: subtract a ``window_s`` moving average
    causal    use only past samples (forward filter / trailing average)
    """
    acc = np.asarray(acc, float)
    if acc.ndim != 2:
        raise ValueError("acc must be (n_samples, n_axes)")
    if method == "highpass":
        sos = signal.butter(2, cutoff_hz, btype="highpass", fs=fs, output="sos")
        if causal:
            zi = signal.sosfilt_zi(sos)[:, :, None] * acc[0][None, None, :]
            out, _ = signal.sosfilt(sos, acc, axis=0, zi=zi)
            return out
        return signal.sosfiltfilt(sos, acc, axis=0)
    if method == "moving_average":
        w = int(round(window_s * fs))
        return acc - np.column_stack([moving_average(acc[:, k], w, causal=causal) for k in range(acc.shape[1])])
    raise ValueError(f"unknown method {method!r}")


def activity_magnitude(acc_dynamic: np.ndarray) -> np.ndarray:
    """Euclidean norm of the per-axis dynamic acceleration (g)."""
    return np.linalg.norm(np.asarray(acc_dynamic, float), axis=1)


def naive_enmo(acc: np.ndarray) -> np.ndarray:
    """Classic ENMO, ``max(|acc| - 1 g, 0)``; provided for comparison only."""
    return np.clip(np.linalg.norm(np.asarray(acc, float), axis=1) - 1.0, 0, None)


def windowed_activity(
    activity: np.ndarray, fs: float, t_end: np.ndarray, win_s: float = 30.0, motion_threshold_g: float = 0.05
) -> pd.DataFrame:
    """Mean activity in trailing windows ``(t_end - win_s, t_end]`` and a motion flag.

    Returns columns ``t, activity, motion`` where ``motion`` is
    ``activity > motion_threshold_g``.
    """
    activity = np.asarray(activity, float)
    c = np.concatenate([[0.0], np.cumsum(activity)])
    t_end = np.asarray(t_end, float)
    hi = np.clip(np.round(t_end * fs).astype(int), 0, len(activity))
    lo = np.clip(np.round((t_end - win_s) * fs).astype(int), 0, len(activity))
    n = np.maximum(hi - lo, 1)
    mean = np.where(hi > lo, (c[hi] - c[lo]) / n, np.nan)
    return pd.DataFrame({"t": t_end, "activity": mean, "motion": mean > motion_threshold_g})

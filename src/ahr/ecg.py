"""Minimal ECG R-peak detector, used only as a heart-rate *reference*.

WESAD also contains a chest ECG (RespiBAN, 700 Hz). It is not part of the
wrist-only trigger pipeline; it serves to check how well wrist-PPG heart
rate tracks the ECG heart rate in each condition.

Pan-Tompkins-style: band-pass 5-15 Hz, derivative, squaring, 150 ms moving
integration, peak picking with a 300 ms refractory period and an adaptive
threshold, then refinement to the largest absolute band-passed deflection
within +-75 ms.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .ppg import moving_average


def detect_r_peaks(ecg: np.ndarray, fs: float, threshold_frac: float = 0.3, window_s: float = 10.0) -> np.ndarray:
    """Sample indices of R peaks (offline, zero-phase)."""
    ecg = np.asarray(ecg, float)
    sos = signal.butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, ecg)
    energy = moving_average(np.gradient(bp) ** 2, int(0.15 * fs))
    # adaptive threshold: fraction of a running high quantile (approximated by block maxima)
    block = int(window_s * fs)
    n_blocks = int(np.ceil(len(energy) / block))
    maxima = np.array([np.percentile(energy[i * block : (i + 1) * block], 98) for i in range(n_blocks)])
    thr = np.repeat(maxima, block)[: len(energy)] * threshold_frac
    cand, _ = signal.find_peaks(energy, distance=int(0.3 * fs))
    cand = cand[energy[cand] > thr[cand]]
    half = int(0.075 * fs)
    refined = [max(0, c - half) + int(np.argmax(np.abs(bp[max(0, c - half) : c + half]))) for c in cand]
    return np.unique(np.asarray(refined, dtype=int))

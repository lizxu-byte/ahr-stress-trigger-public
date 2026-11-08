"""PPG signal-quality indices (SQI) per analysis window.

Two complementary indices:

* **Beat template correlation** - beats are cut around detected peaks,
  averaged into a template, and the mean Pearson correlation of each beat with
  the template is reported (Orphanidou et al., 2015, IEEE JBHI). Clean,
  regular pulses give values close to 1; motion artefacts and false peaks
  lower it.
* **Spectral entropy** - normalised Shannon entropy of the Welch power
  spectrum in the PPG band. A clean pulse concentrates power at the heart
  rate and its harmonics (low entropy); noise spreads it (entropy -> 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def template_correlation(x: np.ndarray, peaks: np.ndarray, fs: float, min_beats: int = 3) -> float:
    """Mean correlation of individual beats with their average template.

    Beats span ``[-0.3, +0.6]`` of the median peak-to-peak interval around
    each peak. Returns 0.0 when fewer than ``min_beats`` complete beats exist.
    """
    x = np.asarray(x, float)
    peaks = np.asarray(peaks, int)
    if len(peaks) < max(2, min_beats):
        return 0.0
    med = float(np.median(np.diff(peaks)))
    pre, post = int(round(0.3 * med)), int(round(0.6 * med))
    beats = np.array([x[p - pre : p + post] for p in peaks if p - pre >= 0 and p + post <= len(x)])
    if len(beats) < min_beats or beats.shape[1] < 4:
        return 0.0
    template = beats.mean(axis=0)
    bc = beats - beats.mean(axis=1, keepdims=True)
    tc = template - template.mean()
    denom = np.linalg.norm(bc, axis=1) * np.linalg.norm(tc)
    r = np.divide(bc @ tc, denom, out=np.zeros(len(beats)), where=denom > 0)
    return float(np.mean(r))


def spectral_entropy(x: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 8.0, nperseg_s: float = 8.0) -> float:
    """Normalised spectral entropy (0..1) of ``x`` in ``[fmin, fmax]`` Hz."""
    x = np.asarray(x, float)
    if len(x) < 4 or not np.all(np.isfinite(x)):
        return 1.0
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(nperseg_s * fs)))
    band = (f >= fmin) & (f <= fmax)
    p = p[band]
    if p.sum() <= 0 or len(p) < 2:
        return 1.0
    p = p / p.sum()
    h = -np.sum(p * np.log(p + 1e-300))
    return float(h / np.log(len(p)))


def window_quality(
    x: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    t_end: np.ndarray,
    win_s: float = 30.0,
    min_template_corr: float = 0.8,
    max_spectral_entropy: float = 0.8,
    available_peaks: np.ndarray | None = None,
) -> pd.DataFrame:
    """SQIs in trailing windows ``(t_end - win_s, t_end]`` and a boolean flag.

    ``available_peaks`` (sample index when each peak became known, see
    :func:`ahr.ppg.elgendi_peaks`) keeps the computation causal.

    Returns columns ``t, template_corr, spectral_entropy, quality_ok``.
    """
    x = np.asarray(x, float)
    peaks = np.asarray(peaks, int)
    avail = peaks if available_peaks is None else np.asarray(available_peaks, int)
    rows = []
    for te in np.asarray(t_end, float):
        lo, hi = max(0, int(round((te - win_s) * fs))), min(len(x), int(round(te * fs)))
        seg = x[lo:hi]
        pk = peaks[(peaks >= lo) & (avail < hi)] - lo
        tcorr = template_correlation(seg, pk, fs)
        se = spectral_entropy(seg, fs)
        rows.append((te, tcorr, se))
    df = pd.DataFrame(rows, columns=["t", "template_corr", "spectral_entropy"])
    df["quality_ok"] = (df["template_corr"] >= min_template_corr) & (df["spectral_entropy"] <= max_spectral_entropy)
    return df

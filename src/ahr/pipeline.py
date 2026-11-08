"""Raw wrist signals -> per-window feature table.

The same code path produces an *offline* table (zero-phase filters, centred
averages; for descriptive analysis) and a *causal* table (forward filters,
trailing averages, beats used only once detected; for the streaming trigger).
In the causal table every value at window end ``t`` depends only on samples
recorded at or before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import motion, ppg, sqi


@dataclass
class FeatureConfig:
    win_s: float = 30.0  # analysis window length
    step_s: float = 5.0  # hop between window ends
    invert_ppg: bool = False  # flip polarity before peak detection (True for the Empatica E4 BVP, see README)
    bp_low: float = 0.5
    bp_high: float = 8.0
    bp_order: int = 2
    min_coverage: float = 0.5  # min fraction of the window covered by valid IBIs
    min_template_corr: float = 0.8  # offline (zero-phase) features
    # the causal band-pass distorts pulse morphology (non-linear phase), which lowers
    # template correlation; 0.7 gives a wrist-vs-ECG HR error similar to 0.8 offline
    min_template_corr_causal: float = 0.7
    max_spectral_entropy: float = 0.75
    acc_cutoff_hz: float = 0.5
    motion_threshold_g: float = 0.10


def window_ends(duration_s: float, cfg: FeatureConfig) -> np.ndarray:
    return np.arange(cfg.win_s, duration_s + 1e-9, cfg.step_s)


def extract_features(
    bvp: np.ndarray,
    fs_bvp: float,
    acc_g: np.ndarray,
    fs_acc: float,
    causal: bool,
    cfg: FeatureConfig | None = None,
    compare_detectors: bool = False,
) -> pd.DataFrame:
    """Per-window HR, HRV, SQI and activity features.

    bvp      raw PPG samples
    acc_g    (n, 3) accelerometer in g
    causal   streaming-compatible processing (see module docstring)
    compare_detectors  also compute HR from the prominence-based detector
                       (offline only)

    Columns: t, mean_hr, rmssd, sdnn, n_beats, coverage, template_corr,
    spectral_entropy, quality_ok, activity, motion [, mean_hr_prom].
    ``quality_ok`` combines both SQIs and the IBI coverage requirement.
    """
    cfg = cfg or FeatureConfig()
    bvp = -np.asarray(bvp, float) if cfg.invert_ppg else np.asarray(bvp, float)
    duration = min(len(bvp) / fs_bvp, len(acc_g) / fs_acc)
    t_end = window_ends(duration, cfg)

    x = ppg.bandpass(bvp, fs_bvp, cfg.bp_low, cfg.bp_high, cfg.bp_order, causal=causal)
    peaks, detect = ppg.elgendi_peaks(x, fs_bvp, causal=causal, return_detection=True)
    beat_t, ibi = ppg.ibi_from_peaks(peaks, fs_bvp)
    valid = ppg.clean_ibi(ibi)
    avail_t = detect[1:] / fs_bvp if causal else None
    hrv = ppg.windowed_hrv(beat_t, ibi, valid, t_end, cfg.win_s, cfg.min_coverage, available_t=avail_t)

    q = sqi.window_quality(
        x,
        peaks,
        fs_bvp,
        t_end,
        cfg.win_s,
        cfg.min_template_corr_causal if causal else cfg.min_template_corr,
        cfg.max_spectral_entropy,
        available_peaks=detect if causal else None,
    )

    dyn = motion.remove_gravity(acc_g, fs_acc, "highpass", cfg.acc_cutoff_hz, causal=causal)
    act = motion.windowed_activity(motion.activity_magnitude(dyn), fs_acc, t_end, cfg.win_s, cfg.motion_threshold_g)

    df = hrv.copy()
    df["template_corr"] = q["template_corr"].to_numpy()
    df["spectral_entropy"] = q["spectral_entropy"].to_numpy()
    df["quality_ok"] = q["quality_ok"].to_numpy() & df["mean_hr"].notna().to_numpy()
    df["activity"] = act["activity"].to_numpy()
    df["motion"] = act["motion"].to_numpy()

    if compare_detectors and not causal:
        pk2 = ppg.prominence_peaks(x, fs_bvp)
        bt2, ibi2 = ppg.ibi_from_peaks(pk2, fs_bvp)
        df["mean_hr_prom"] = ppg.windowed_hrv(bt2, ibi2, ppg.clean_ibi(ibi2), t_end, cfg.win_s, cfg.min_coverage)[
            "mean_hr"
        ].to_numpy()
    return df


def window_labels(t_end: np.ndarray, labels: np.ndarray, label_fs: float, win_s: float) -> pd.DataFrame:
    """Majority label and its share (purity) within each window ``(t-win_s, t]``."""
    labels = np.asarray(labels)
    out_label, purity = [], []
    for te in np.asarray(t_end, float):
        lo = max(0, int(np.floor((te - win_s) * label_fs)))
        hi = min(len(labels), int(np.floor(te * label_fs)))
        seg = labels[lo:hi]
        if len(seg) == 0:
            out_label.append(-1)
            purity.append(0.0)
            continue
        vals, counts = np.unique(seg, return_counts=True)
        k = int(np.argmax(counts))
        out_label.append(int(vals[k]))
        purity.append(counts[k] / len(seg))
    return pd.DataFrame({"t": t_end, "label": out_label, "label_purity": purity})

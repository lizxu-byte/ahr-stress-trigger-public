"""Beat-level and window-level agreement between a test and a reference beat series.

Used to validate PPG systolic peaks against ECG R-peaks (BIDMC). A PPG peak
follows its R-peak by the pulse arrival time (PAT, roughly 150-400 ms at the
fingertip), so beats are matched after removing a per-record constant delay
(:func:`estimate_delay_s`), then one-to-one within a tolerance
(:func:`match_beats`).

Window-level HR agreement is summarised as MAE, bias and Bland-Altman limits
of agreement. Because each record contributes many windows, the limits use
the repeated-measures variance (between-record + within-record components,
Bland & Altman, 2007, *J Biopharm Stat* 17(4), 571-582, "true value varies"
case) rather than the SD of all pooled windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_signed(ref_sorted: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Signed distance from each ``t`` to the nearest element of ``ref_sorted``."""
    idx = np.clip(np.searchsorted(ref_sorted, t), 1, len(ref_sorted) - 1)
    left, right = t - ref_sorted[idx - 1], t - ref_sorted[idx]
    return np.where(np.abs(left) <= np.abs(right), left, right)


def estimate_delay_s(
    ref_t: np.ndarray,
    test_t: np.ndarray,
    min_s: float = -0.5,
    max_s: float = 1.5,
    step_s: float = 0.004,
    tol_s: float = 0.05,
) -> float:
    """Constant delay (s) of ``test_t`` relative to ``ref_t``.

    Grid search over ``[min_s, max_s]`` for the shift that puts the most test
    beats within ``tol_s`` of a reference beat (ties: smallest ``|delay|``),
    refined by the median signed error of those beats. A wide range is needed
    because waveform channels in MIMIC-II-derived data can be offset by more
    than one RR interval; with a regular rhythm, shifts differing by one RR
    interval are near-equivalent, which does not affect beat matching. NaN if
    either series has fewer than 2 beats.
    """
    ref = np.sort(np.asarray(ref_t, float))
    test = np.asarray(test_t, float)
    if len(ref) < 2 or len(test) < 2:
        return float("nan")
    grid = np.arange(min_s, max_s + step_s / 2, step_s)
    counts = np.array([(np.abs(_nearest_signed(ref, test - g)) <= tol_s).sum() for g in grid])
    best = grid[counts == counts.max()]
    g = float(best[np.argmin(np.abs(best))])
    d = _nearest_signed(ref, test - g)
    d = d[np.abs(d) <= tol_s]
    return g + float(np.median(d)) if len(d) else g


def match_beats(ref_t: np.ndarray, test_t: np.ndarray, tol_s: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching of test beats to reference beats within ``tol_s``.

    Candidate pairs are accepted in order of increasing absolute time error,
    each beat used at most once. Returns index arrays ``(i_ref, i_test)`` of
    the matched pairs, sorted by reference time.
    """
    ref = np.asarray(ref_t, float)
    test = np.asarray(test_t, float)
    if len(ref) == 0 or len(test) == 0:
        return np.array([], int), np.array([], int)
    order = np.argsort(ref)
    rs = ref[order]
    cand = []
    for j, t in enumerate(test):
        lo = np.searchsorted(rs, t - tol_s, side="left")
        hi = np.searchsorted(rs, t + tol_s, side="right")
        for k in range(lo, hi):
            cand.append((abs(rs[k] - t), order[k], j))
    cand.sort()
    used_r, used_t, pairs = set(), set(), []
    for _, i, j in cand:
        if i in used_r or j in used_t:
            continue
        used_r.add(i)
        used_t.add(j)
        pairs.append((i, j))
    pairs.sort()
    if not pairs:
        return np.array([], int), np.array([], int)
    p = np.asarray(pairs, int)
    return p[:, 0], p[:, 1]


def beat_agreement(ref_t: np.ndarray, test_t: np.ndarray, tol_s: float, delay_s: float = 0.0) -> dict:
    """Sensitivity, PPV and F1 of ``test_t - delay_s`` against ``ref_t``.

    Also returns the median and 95th percentile of the absolute timing error
    of matched beats (ms), after delay removal.
    """
    ref = np.asarray(ref_t, float)
    test = np.asarray(test_t, float) - (0.0 if not np.isfinite(delay_s) else delay_s)
    i, j = match_beats(ref, test, tol_s)
    tp = len(i)
    se = tp / len(ref) if len(ref) else np.nan
    ppv = tp / len(test) if len(test) else np.nan
    f1 = 2 * tp / (len(ref) + len(test)) if len(ref) + len(test) else np.nan
    err = np.abs(test[j] - ref[i]) * 1000 if tp else np.array([np.nan])
    return {
        "n_ref": len(ref),
        "n_test": len(test),
        "tp": tp,
        "sensitivity": se,
        "ppv": ppv,
        "f1": f1,
        "abs_err_ms_median": float(np.nanmedian(err)),
        "abs_err_ms_p95": float(np.nanpercentile(err, 95)) if tp else np.nan,
    }


def bland_altman(diff: np.ndarray, groups: np.ndarray | None = None) -> dict:
    """Bias and 95 % limits of agreement of paired differences.

    Without ``groups``: ``mean +- 1.96 SD``. With ``groups`` (one label per
    difference, e.g. the record), the SD combines the between-group variance
    of the group means and the within-group variance (one-way random-effects
    ANOVA, Bland & Altman 2007), which does not shrink when a record
    contributes many correlated windows.
    """
    d = np.asarray(diff, float)
    keep = np.isfinite(d)
    d = d[keep]
    out = {"n": int(len(d)), "bias": float(np.mean(d)) if len(d) else np.nan, "mae": float(np.mean(np.abs(d))) if len(d) else np.nan}
    if len(d) < 2:
        return {**out, "sd": np.nan, "loa_low": np.nan, "loa_high": np.nan, "n_groups": 0}
    if groups is None:
        sd = float(np.std(d, ddof=1))
        n_groups = len(d)
    else:
        g = np.asarray(groups)[keep]
        df = pd.DataFrame({"d": d, "g": g})
        stats = df.groupby("g")["d"].agg(["mean", "count", "var"])
        k, n = len(stats), len(d)
        n_groups = k
        ss_w = float(((stats["count"] - 1) * stats["var"].fillna(0)).sum())
        ms_w = ss_w / max(n - k, 1)
        if k < 2:
            sd = float(np.sqrt(ms_w))
        else:
            ms_b = float((stats["count"] * (stats["mean"] - d.mean()) ** 2).sum() / (k - 1))
            n0 = (n - (stats["count"] ** 2).sum() / n) / (k - 1)
            var_b = max((ms_b - ms_w) / n0, 0.0)
            sd = float(np.sqrt(var_b + ms_w))
    return {**out, "sd": sd, "loa_low": out["bias"] - 1.96 * sd, "loa_high": out["bias"] + 1.96 * sd, "n_groups": n_groups}

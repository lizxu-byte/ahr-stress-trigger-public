"""PPG processing: filtering, systolic peak detection, IBI cleaning, HR / HRV.

Every step comes in an offline flavour (zero-phase, may look ahead) and a
causal flavour (only past samples) so that the streaming trigger can be
simulated honestly. The causal band-pass delays the signal; use
:func:`group_delay_s` (analytic) to quantify it. :func:`envelope_delay_s` measures it empirically and
:func:`peak_lag_s` measures the resulting shift of detected systolic peaks.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy import signal

# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------


def detrend(x: np.ndarray, fs: float, window_s: float | None = None, causal: bool = False) -> np.ndarray:
    """Remove slow baseline drift.

    With ``window_s=None`` a single linear trend is removed (offline only).
    Otherwise a moving average of ``window_s`` seconds is subtracted: centred
    when offline, trailing when ``causal=True``.
    """
    x = np.asarray(x, dtype=float)
    if window_s is None:
        if causal:
            raise ValueError("linear detrending needs the whole signal; pass window_s for causal use")
        return signal.detrend(x, type="linear")
    return x - moving_average(x, max(1, int(round(window_s * fs))), causal=causal)


def moving_average(x: np.ndarray, w: int, causal: bool = False) -> np.ndarray:
    """Moving average of width ``w`` samples.

    ``causal=False``: centred window. ``causal=True``: trailing window over the
    last ``w`` samples (shorter at the very start).
    """
    x = np.asarray(x, dtype=float)
    w = max(1, int(w))
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    if causal:
        hi = np.arange(1, n + 1)
        lo = np.maximum(0, hi - w)
    else:
        half = w // 2
        lo = np.maximum(0, np.arange(n) - half)
        hi = np.minimum(n, np.arange(n) - half + w)
    return (c[hi] - c[lo]) / (hi - lo)


def butter_bandpass(fs: float, low: float = 0.5, high: float = 8.0, order: int = 2) -> np.ndarray:
    """Second-order-sections Butterworth band-pass design."""
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def bandpass(
    x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0, order: int = 2, causal: bool = False
) -> np.ndarray:
    """Butterworth band-pass.

    ``causal=False`` uses forward-backward filtering (``sosfiltfilt``): zero
    phase, effective order doubled, needs future samples. ``causal=True`` runs
    the filter forward only (``sosfilt``) with the initial state set to the
    first sample's steady state, which is what a streaming device can do; it
    introduces a frequency-dependent delay (see :func:`group_delay_s`).
    """
    sos = butter_bandpass(fs, low, high, order)
    x = np.asarray(x, dtype=float)
    if not causal:
        return signal.sosfiltfilt(sos, x)
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


class StreamingBandpass:
    """Chunk-wise causal band-pass that carries filter state between calls.

    Feeding a signal in arbitrary chunks gives exactly the same output as
    ``bandpass(x, fs, causal=True)`` on the whole signal.
    """

    def __init__(self, fs: float, low: float = 0.5, high: float = 8.0, order: int = 2):
        self.sos = butter_bandpass(fs, low, high, order)
        self.zi: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=float)
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * chunk[0]
        y, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi)
        return y


def group_delay_s(
    fs: float, freqs_hz: np.ndarray | float, low: float = 0.5, high: float = 8.0, order: int = 2
) -> np.ndarray:
    """Analytic group delay (s) of the causal band-pass at ``freqs_hz``.

    The zero-phase (forward-backward) variant has zero group delay by
    construction.
    """
    b, a = signal.sos2tf(butter_bandpass(fs, low, high, order))
    _, gd = signal.group_delay((b, a), w=np.atleast_1d(freqs_hz), fs=fs)
    return gd / fs


def peak_lag_s(ref_peaks: np.ndarray, test_peaks: np.ndarray, fs: float, max_lag_s: float = 0.25) -> float:
    """Median timing offset (s) of ``test_peaks`` relative to the nearest ``ref_peaks``.

    This is the delay that matters for beat timing: a band-pass with phase
    lead at the fundamental and lag at the harmonics can shift the systolic
    peak much less than its group delay suggests.
    """
    ref = np.sort(np.asarray(ref_peaks))
    test = np.asarray(test_peaks)
    if len(ref) < 2 or len(test) == 0:
        return float("nan")
    idx = np.clip(np.searchsorted(ref, test), 1, len(ref) - 1)
    d_left, d_right = test - ref[idx - 1], test - ref[idx]
    d = np.where(np.abs(d_left) < np.abs(d_right), d_left, d_right) / fs
    d = d[np.abs(d) <= max_lag_s]
    return float(np.median(d)) if len(d) else float("nan")


def envelope_delay_s(x_in: np.ndarray, x_out: np.ndarray, fs: float) -> float:
    """Empirical delay (s) of a filter from its input/output, as the shift of the
    energy centroid of the Hilbert envelope.

    For a narrow-band test burst this equals the group delay at the burst's
    carrier frequency; positive values mean the output lags the input.
    """
    t = np.arange(len(x_in)) / fs
    e_in = np.abs(signal.hilbert(x_in)) ** 2
    e_out = np.abs(signal.hilbert(x_out)) ** 2
    return float((t * e_out).sum() / e_out.sum() - (t * e_in).sum() / e_in.sum())


# ----------------------------------------------------------------------------
# Peak detection
# ----------------------------------------------------------------------------


def _enforce_min_distance(peaks: np.ndarray, x: np.ndarray, min_dist: int) -> np.ndarray:
    """Greedy left-to-right: of two peaks closer than ``min_dist`` keep the larger."""
    kept: list[int] = []
    for p in peaks:
        if kept and p - kept[-1] < min_dist:
            if x[p] > x[kept[-1]]:
                kept[-1] = p
        else:
            kept.append(p)
    return np.asarray(kept, dtype=int)


def elgendi_peaks(
    x: np.ndarray,
    fs: float,
    w1_s: float = 0.111,
    w2_s: float = 0.667,
    beta: float = 0.02,
    min_distance_s: float = 0.3,
    causal: bool = False,
    alpha_window_s: float = 10.0,
    return_detection: bool = False,
):
    """Systolic peak detection with two event-related moving averages.

    Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013). Systolic
    peak detection in acceleration photoplethysmograms measured from emergency
    responders in tropical conditions. PLoS ONE 8(10): e76585.

    ``x`` must already be band-passed (0.5-8 Hz). Steps: clip negative values,
    square, ``MA_peak`` over W1 (~systolic wave), ``MA_beat`` over W2 (~one
    beat); blocks of interest are where ``MA_peak > MA_beat + beta * mean(y)``;
    blocks shorter than W1 are rejected; the peak is the maximum of ``x``
    inside each block.

    With ``causal=True`` both moving averages are trailing, the offset
    ``alpha`` uses a trailing mean over ``alpha_window_s`` instead of the
    global mean, blocks need only half of W1 (trailing averages shorten
    blocks at high HR), and the maximum is searched in ``[block_start - W1,
    block_end]`` to compensate the trailing-average delay. A block is only
    known once it ends, so the *detection index* (block end) is when a
    streaming system learns about the beat; request it with
    ``return_detection=True``.
    """
    x = np.asarray(x, dtype=float)
    y = np.clip(x, 0, None) ** 2
    w1 = max(1, int(round(w1_s * fs)))
    w2 = max(1, int(round(w2_s * fs)))
    ma_peak = moving_average(y, w1, causal=causal)
    ma_beat = moving_average(y, w2, causal=causal)
    if causal:
        alpha = beta * moving_average(y, int(alpha_window_s * fs), causal=True)
    else:
        alpha = beta * np.mean(y)
    boi = ma_peak > ma_beat + alpha

    d = np.diff(boi.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    if causal and len(ends) and ends[-1] == len(x) and boi[-1]:
        # block still open at the end of the data: not yet detectable
        starts, ends = starts[:-1], ends[:-1]

    peaks, detect = [], []
    prev_end = 0
    # trailing averages shrink blocks at high HR, so the causal variant accepts half-width blocks
    min_width = max(1, w1 // 2) if causal else w1
    for s, e in zip(starts, ends):
        if e - s < min_width:
            continue
        lo = max(prev_end, s - w1) if causal else s
        peaks.append(lo + int(np.argmax(x[lo:e])))
        detect.append(e - 1)
        prev_end = e
    peaks = np.asarray(peaks, dtype=int)
    detect = np.asarray(detect, dtype=int)
    if len(peaks):
        keep = np.isin(peaks, _enforce_min_distance(peaks, x, int(min_distance_s * fs)))
        # duplicates of the same peak index can arise from adjacent blocks
        keep &= np.r_[True, np.diff(peaks) > 0]
        peaks, detect = peaks[keep], detect[keep]
    return (peaks, detect) if return_detection else peaks


def prominence_peaks(
    x: np.ndarray, fs: float, min_distance_s: float = 0.33, prominence_sd: float = 1.5
) -> np.ndarray:
    """Comparison detector: ``scipy.signal.find_peaks`` with distance + prominence.

    The prominence threshold is ``prominence_sd`` times the robust SD (1.4826
    * MAD) of the band-passed signal.
    """
    x = np.asarray(x, dtype=float)
    robust_sd = 1.4826 * np.median(np.abs(x - np.median(x)))
    peaks, _ = signal.find_peaks(x, distance=max(1, int(min_distance_s * fs)), prominence=prominence_sd * robust_sd)
    return peaks


# ----------------------------------------------------------------------------
# Inter-beat intervals
# ----------------------------------------------------------------------------


def ibi_from_peaks(peaks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Beat times (s, time of the closing peak) and IBIs (ms)."""
    t = np.asarray(peaks, dtype=float) / fs
    return t[1:], np.diff(t) * 1000.0


def clean_ibi(
    ibi_ms: np.ndarray,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
    max_rel_change: float = 0.25,
    ref_beats: int = 5,
) -> np.ndarray:
    """Boolean mask of plausible IBIs (past-only, usable in streaming).

    1. Physiological bounds ``min_ms <= ibi <= max_ms`` (30-200 bpm).
    2. Successive-difference rule: an IBI is rejected if it deviates by more
       than ``max_rel_change`` from the median of the last ``ref_beats``
       accepted IBIs (missed beats double the IBI, false detections halve it).
       After ``ref_beats`` consecutive rejections the reference is reset so a
       genuine fast HR change cannot lock the cleaner out.
    """
    ibi = np.asarray(ibi_ms, dtype=float)
    ok = (ibi >= min_ms) & (ibi <= max_ms) & np.isfinite(ibi)
    ref: deque[float] = deque(maxlen=ref_beats)
    n_rejected = 0
    for i in range(len(ibi)):
        if not ok[i]:
            continue
        if ref:
            med = float(np.median(ref))
            if abs(ibi[i] - med) > max_rel_change * med:
                ok[i] = False
                n_rejected += 1
                if n_rejected >= ref_beats:
                    ref.clear()
                    n_rejected = 0
                continue
        ref.append(ibi[i])
        n_rejected = 0
    return ok


def interpolate_ibi(
    beat_t: np.ndarray,
    ibi_ms: np.ndarray,
    valid: np.ndarray | None = None,
    fs_out: float = 4.0,
    max_gap_s: float = 3.0,
    kind: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample valid IBIs onto an even grid (``fs_out``) for plotting / spectral use.

    Grid points farther than ``max_gap_s`` from any valid beat are NaN.
    ``kind`` is ``"linear"`` or ``"cubic"`` (scipy ``CubicSpline``).
    """
    beat_t = np.asarray(beat_t, float)
    ibi_ms = np.asarray(ibi_ms, float)
    if valid is not None:
        beat_t, ibi_ms = beat_t[valid], ibi_ms[valid]
    if len(beat_t) < 2:
        return np.array([]), np.array([])
    grid = np.arange(beat_t[0], beat_t[-1], 1.0 / fs_out)
    if kind == "cubic" and len(beat_t) >= 4:
        from scipy.interpolate import CubicSpline

        out = CubicSpline(beat_t, ibi_ms)(grid)
    else:
        out = np.interp(grid, beat_t, ibi_ms)
    idx = np.clip(np.searchsorted(beat_t, grid), 1, len(beat_t) - 1)
    gap = beat_t[idx] - beat_t[idx - 1]
    out[gap > max_gap_s] = np.nan
    return grid, out


def windowed_hrv(
    beat_t: np.ndarray,
    ibi_ms: np.ndarray,
    valid: np.ndarray,
    t_end: np.ndarray,
    win_s: float = 30.0,
    min_coverage: float = 0.5,
    available_t: np.ndarray | None = None,
) -> pd.DataFrame:
    """HR and time-domain HRV in trailing windows ``(t_end - win_s, t_end]``.

    ``available_t`` (optional) is when each beat became known to a streaming
    system (e.g. the Elgendi detection index / fs); beats are only used in a
    window once ``available_t <= t_end``. Metrics are NaN when valid IBIs
    cover less than ``min_coverage`` of the window.

    Returns columns ``t, mean_hr, rmssd, sdnn, n_beats, coverage``.
    """
    beat_t = np.asarray(beat_t, float)
    ibi = np.asarray(ibi_ms, float)
    valid = np.asarray(valid, bool)
    avail = beat_t if available_t is None else np.asarray(available_t, float)
    # an RMSSD pair needs two consecutive valid beats
    pair_ok = np.r_[False, valid[1:] & valid[:-1]]
    sdiff = np.r_[np.nan, np.diff(ibi)]

    rows = []
    for te in np.asarray(t_end, float):
        sel = (beat_t > te - win_s) & (avail <= te)
        v = sel & valid
        cov = ibi[v].sum() / 1000.0 / win_s
        n = int(v.sum())
        if cov < min_coverage or n < 3:
            rows.append((te, np.nan, np.nan, np.nan, n, cov))
            continue
        iv = ibi[v]
        pairs = sdiff[sel & pair_ok]
        rmssd = float(np.sqrt(np.mean(pairs**2))) if len(pairs) >= 2 else np.nan
        rows.append((te, 60000.0 / iv.mean(), rmssd, float(np.std(iv, ddof=1)), n, cov))
    return pd.DataFrame(rows, columns=["t", "mean_hr", "rmssd", "sdnn", "n_beats", "coverage"])

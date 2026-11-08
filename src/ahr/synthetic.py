"""Synthetic wrist PPG + accelerometer signals with known ground truth.

Used by the unit tests and by ``scripts/run_all.py --synthetic`` so that the
whole pipeline can be exercised without the WESAD download.

The generator builds an instantaneous heart rate

    HR(t) = HR_rest + stress_offset(t) + k_act * lowpass(activity(t)) + RSA + noise

places beats by integrating HR, draws a two-Gaussian pulse (systolic +
diastolic wave) per beat, and adds respiration baseline wander, slow drift,
sensor noise and motion artefacts during activity bursts. The accelerometer is
a slowly rotating gravity vector (1 g) plus wrist oscillation at walking
cadence whose amplitude follows ``activity(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    """One protocol segment.

    label         WESAD-style label (0 transient, 1 baseline, 2 stress, 3 amusement, 4 meditation)
    duration_s    length of the segment
    hr_offset     additive HR change (bpm) not explained by movement (e.g. stress)
    activity      dynamic wrist acceleration amplitude in g (0 = still, ~0.3 = walking)
    """

    label: int
    duration_s: float
    hr_offset: float = 0.0
    activity: float = 0.0


@dataclass
class SyntheticRecording:
    fs_ppg: float
    fs_acc: float
    ppg: np.ndarray
    acc: np.ndarray  # (n, 3) in g
    label_fs: float
    labels: np.ndarray
    beat_times: np.ndarray  # true beat times (s)
    hr_true_fs: float
    hr_true: np.ndarray  # instantaneous true HR (bpm) at hr_true_fs
    meta: dict = field(default_factory=dict)


def wesad_like_protocol(stress_offset: float = 8.0, walk_activity: float = 0.35) -> list[Segment]:
    """A ~75 min protocol loosely mimicking WESAD's order of conditions.

    Short walking bursts (label 0) between conditions raise HR through
    movement only; the stress segment raises HR without movement.
    Protocol: 2 min transient, 20 min baseline, walking, 10 min stress, ...
    """
    walk = Segment(0, 120, 0.0, walk_activity)
    return [
        Segment(0, 120, 0.0, 0.02),
        # baseline with two brief bouts of moving around (as in WESAD) so that a
        # baseline-only calibration can identify the activity slope
        Segment(1, 500, 0.0, 0.02),
        Segment(1, 90, 0.0, 0.15),
        Segment(1, 400, 0.0, 0.02),
        Segment(1, 90, 0.0, 0.15),
        Segment(1, 120, 0.0, 0.02),
        walk,
        Segment(2, 600, stress_offset, 0.04),
        Segment(0, 300, 2.0, 0.02),
        walk,
        Segment(3, 400, 1.0, 0.03),
        walk,
        Segment(4, 420, -2.0, 0.01),
        walk,
        Segment(0, 180, 0.0, 0.02),
    ]


def _lowpass_first_order(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    a = np.exp(-1.0 / (fs * tau_s))
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * acc + (1 - a) * v
        y[i] = acc
    return y


def simulate(
    segments: list[Segment] | None = None,
    hr_rest: float = 70.0,
    k_act: float = 40.0,
    tau_act_s: float = 20.0,
    fs_ppg: float = 64.0,
    fs_acc: float = 32.0,
    label_fs: float = 4.0,
    noise_sd: float = 0.03,
    artifact_gain: float = 1.5,
    seed: int = 0,
) -> SyntheticRecording:
    """Generate a synthetic recording.

    Parameters
    ----------
    segments      protocol; defaults to :func:`wesad_like_protocol`
    hr_rest       resting heart rate (bpm)
    k_act         bpm per g of (low-passed) dynamic wrist acceleration
    tau_act_s     time constant of the HR response to movement
    noise_sd      PPG white-noise SD relative to pulse amplitude (1.0)
    artifact_gain scale of motion artefacts added to the PPG during movement
    """
    rng = np.random.default_rng(seed)
    segments = segments if segments is not None else wesad_like_protocol()
    duration = float(sum(s.duration_s for s in segments))

    # --- piecewise-constant drivers on a fine grid -------------------------
    fs_hr = 32.0
    t_hr = np.arange(0, duration, 1 / fs_hr)
    edges = np.cumsum([0.0] + [s.duration_s for s in segments])
    seg_idx = np.clip(np.searchsorted(edges, t_hr, side="right") - 1, 0, len(segments) - 1)
    offset = np.array([s.hr_offset for s in segments])[seg_idx]
    activity = np.array([s.activity for s in segments])[seg_idx]
    activity = activity * (1 + 0.2 * rng.standard_normal(len(t_hr)).cumsum() / np.sqrt(len(t_hr)))
    activity = np.clip(activity, 0, None)

    resp_f = 0.25 + 0.02 * rng.standard_normal()
    rsa = 2.0 * np.sin(2 * np.pi * resp_f * t_hr)
    slow = _lowpass_first_order(rng.standard_normal(len(t_hr)), fs_hr, 30.0) * 8.0
    hr = (
        hr_rest
        + _lowpass_first_order(offset, fs_hr, 15.0)
        + k_act * _lowpass_first_order(activity, fs_hr, tau_act_s)
        + rsa
        + slow
    )

    # --- beats --------------------------------------------------------------
    phase = np.cumsum(hr / 60.0) / fs_hr
    n_beats = int(phase[-1])
    beat_times = np.interp(np.arange(1, n_beats + 1), phase, t_hr)

    # --- PPG ----------------------------------------------------------------
    t = np.arange(0, duration, 1 / fs_ppg)
    ppg = np.zeros_like(t)
    amp = 1 + 0.1 * rng.standard_normal(n_beats)
    for bt, a in zip(beat_times, amp):
        ibi = 60.0 / np.interp(bt, t_hr, hr)
        lo, hi = np.searchsorted(t, [bt - 0.4 * ibi, bt + 0.8 * ibi])
        tt = t[lo:hi] - bt
        # systolic wave scales with the IBI, the diastolic wave follows ~0.3 s later
        d_lag = min(0.3, 0.4 * ibi)
        ppg[lo:hi] += a * (
            np.exp(-0.5 * (tt / (0.08 * ibi)) ** 2) + 0.3 * np.exp(-0.5 * ((tt - d_lag) / 0.08) ** 2)
        )
    ppg += 0.3 * np.sin(2 * np.pi * resp_f * t + 1.0)  # respiration baseline wander
    ppg += 0.5 * np.sin(2 * np.pi * t / 400.0) + 0.0002 * t  # slow drift
    ppg += noise_sd * rng.standard_normal(len(t))
    act_ppg = np.interp(t, t_hr, activity)
    cadence = 1.8 + 0.1 * rng.standard_normal()
    artifact = artifact_gain * act_ppg / 0.35 * (
        np.sin(2 * np.pi * cadence * t) + 0.8 * rng.standard_normal(len(t))
    )
    ppg += artifact

    # --- accelerometer ------------------------------------------------------
    ta = np.arange(0, duration, 1 / fs_acc)
    theta = 0.6 + 0.3 * np.sin(2 * np.pi * ta / 300.0)
    phi = 0.4 * np.sin(2 * np.pi * ta / 500.0)
    gravity = np.c_[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    act_acc = np.interp(ta, t_hr, activity)
    dyn = act_acc[:, None] * np.c_[
        np.sin(2 * np.pi * cadence * ta),
        0.7 * np.sin(2 * np.pi * cadence * ta + 1.0),
        0.5 * np.sin(2 * np.pi * 2 * cadence * ta + 2.0),
    ]
    acc = gravity + dyn + 0.005 * rng.standard_normal((len(ta), 3))

    tl = np.arange(0, duration, 1 / label_fs)
    lab_idx = np.clip(np.searchsorted(edges, tl, side="right") - 1, 0, len(segments) - 1)
    labels = np.array([s.label for s in segments])[lab_idx].astype(np.int8)

    return SyntheticRecording(
        fs_ppg=fs_ppg,
        fs_acc=fs_acc,
        ppg=ppg,
        acc=acc,
        label_fs=label_fs,
        labels=labels,
        beat_times=beat_times,
        hr_true_fs=fs_hr,
        hr_true=hr,
        meta={"hr_rest": hr_rest, "k_act": k_act, "seed": seed},
    )


def pulse_train(hr_bpm: float, duration_s: float, fs: float = 64.0, noise_sd: float = 0.02, seed: int = 0):
    """Clean constant-rate PPG-like pulse train with respiration wander.

    Returns ``(t, x, beat_times)``.
    """
    rec = simulate(
        [Segment(1, duration_s, 0.0, 0.0)],
        hr_rest=hr_bpm,
        k_act=0.0,
        fs_ppg=fs,
        noise_sd=noise_sd,
        seed=seed,
    )
    t = np.arange(len(rec.ppg)) / fs
    return t, rec.ppg, rec.beat_times

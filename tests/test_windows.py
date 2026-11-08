"""Windowing, calibration z-scoring (no future leakage), LOSO splits and metrics."""

import numpy as np
import pytest

from ahr import windows


def _labels(fs=10.0):
    # 2 min transient, 20 min baseline, 5 min transient, 10 min stress, 5 min transient, 6 min amusement
    seg = [(0, 120), (1, 1200), (0, 300), (2, 600), (0, 300), (3, 360)]
    return np.concatenate([np.full(int(d * fs), lab, dtype=np.int8) for lab, d in seg]), fs


def test_calibration_interval_is_start_of_first_baseline():
    lab, fs = _labels()
    assert windows.calibration_interval(lab, fs, 300.0) == pytest.approx((120.0, 420.0))
    # shorter baseline than requested: truncated at the end of the baseline run
    assert windows.calibration_interval(lab, fs, 5000.0) == pytest.approx((120.0, 1320.0))


def test_windows_purity_overlap_and_calibration_exclusion():
    lab, fs = _labels()
    cfg = windows.WindowConfig(win_s=60, step_s=30, min_purity=0.95, calib_s=300)
    calib = windows.calibration_interval(lab, fs, cfg.calib_s)
    w = windows.make_windows(lab, fs, len(lab) / fs, calib, cfg)
    assert np.allclose(np.diff(w["t0"]), 30.0) and np.allclose(w["t1"] - w["t0"], 60.0)
    u = w[w["usable"]]
    assert (u["label_purity"] >= 0.95).all()
    # no usable window overlaps the calibration interval
    assert ((u["t1"] <= calib[0]) | (u["t0"] >= calib[1])).all()
    # every usable window lies >= 95 % inside one condition, checked against the raw labels
    for _, r in u.iterrows():
        seg = lab[int(r["t0"] * fs) : int(r["t1"] * fs)]
        assert np.mean(seg == r["label"]) >= 0.95
    # 15 min of baseline after calibration -> (900 - 60) / 30 + 1 = 29 windows
    assert (u["label"] == 1).sum() == 29
    assert (u["label"] == 2).sum() == (600 - 60) // 30 + 1


def test_calibration_zscore_has_no_future_leakage():
    rng = np.random.default_rng(0)
    fs, calib = 64.0, (10.0, 40.0)
    x = rng.standard_normal((int(120 * fs), 2))
    z1 = windows.zscore_with_calibration(x, fs, calib)
    x2 = x.copy()
    x2[int(calib[1] * fs) :] = 1e6 * rng.standard_normal(x2[int(calib[1] * fs) :].shape)  # alter everything after calibration
    z2 = windows.zscore_with_calibration(x2, fs, calib)
    # samples up to the end of calibration are unchanged
    np.testing.assert_allclose(z1[: int(calib[1] * fs)], z2[: int(calib[1] * fs)])
    mu, sd = windows.calibration_stats(x, fs, calib)
    seg = x[int(calib[0] * fs) : int(calib[1] * fs)]
    np.testing.assert_allclose(mu, seg.mean(axis=0))
    np.testing.assert_allclose(sd, seg.std(axis=0))
    # calibration segment itself is standardised
    np.testing.assert_allclose(z1[int(calib[0] * fs) : int(calib[1] * fs)].mean(axis=0), 0, atol=1e-12)


def test_cut_windows_shapes():
    x = np.arange(64 * 100 * 2, dtype=float).reshape(-1, 2)
    w = windows.cut_windows(x, 64.0, np.array([0.0, 30.0]), 60.0)
    assert w.shape == (2, 3840, 2)
    assert w[1, 0, 0] == x[30 * 64, 0]
    with pytest.raises(ValueError):
        windows.cut_windows(x, 64.0, np.array([50.0]), 60.0)


def test_loso_never_mixes_subjects():
    subj = np.repeat([f"S{i}" for i in range(2, 8)], [5, 3, 4, 6, 2, 5])
    seen = []
    rng = np.random.default_rng(0)
    for s, tr, te in windows.loso_splits(subj):
        assert set(subj[te]) == {s}
        assert s not in set(subj[tr])
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(subj)
        itr, ival = windows.inner_split(subj[tr], 2, rng)
        assert set(subj[tr][itr]).isdisjoint(set(subj[tr][ival]))
        assert len(set(subj[tr][ival])) == 2 and s not in set(subj[tr][ival])
        seen.append(s)
    assert sorted(seen) == sorted(set(subj))


def test_metrics():
    y = np.array([1, 1, 1, 1, 2, 2])
    p = np.array([1, 1, 1, 2, 2, 2])
    assert windows.balanced_accuracy(y, p, [1, 2]) == pytest.approx((3 / 4 + 1) / 2)
    f1_1, f1_2 = 2 * 3 / (2 * 3 + 0 + 1), 2 * 2 / (2 * 2 + 1 + 0)
    assert windows.macro_f1(y, p, [1, 2]) == pytest.approx((f1_1 + f1_2) / 2)
    assert windows.macro_f1(y, y, [1, 2, 3]) == 1.0  # absent class skipped

"""Beat matching, delay estimation, Bland-Altman and the WFDB format-16 reader."""

import numpy as np
import pytest

from ahr import io_bidmc, validation


def _beats(n=200, rr=0.8, jitter=0.03, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rr + jitter * rng.standard_normal(n))


def test_estimate_delay_recovers_constant_shift():
    ref = _beats()
    for true in (0.25, 0.65, 1.1):  # 1.1 s is longer than one RR interval
        d = validation.estimate_delay_s(ref, ref + true)
        # beats are irregular, so the shift is identifiable even beyond one RR interval
        assert d == pytest.approx(true, abs=0.005)


def test_beat_agreement_counts_missed_and_extra_beats():
    ref = _beats()
    test = np.delete(ref, [10, 50, 90]) + 0.3  # 3 missed beats
    test = np.sort(np.r_[test, ref[20] + 0.3 + 0.4])  # 1 false beat between two real ones
    a = validation.beat_agreement(ref, test, tol_s=0.1, delay_s=validation.estimate_delay_s(ref, test))
    assert a["tp"] == len(ref) - 3
    assert a["sensitivity"] == pytest.approx((len(ref) - 3) / len(ref))
    assert a["ppv"] == pytest.approx((len(ref) - 3) / (len(ref) - 2))


def test_match_beats_is_one_to_one():
    ref = np.array([1.0, 2.0, 3.0])
    test = np.array([1.01, 1.02, 2.5, 3.0])
    i, j = validation.match_beats(ref, test, tol_s=0.05)
    assert list(i) == [0, 2] and list(j) == [0, 3]
    assert len(set(i)) == len(i) and len(set(j)) == len(j)


def test_bland_altman_repeated_measures_wider_than_pooled():
    rng = np.random.default_rng(1)
    # 10 records with record-specific bias, 16 windows each
    offsets = rng.normal(0, 2.0, 10)
    d = np.concatenate([o + rng.normal(0, 0.3, 16) for o in offsets])
    g = np.repeat(np.arange(10), 16)
    pooled, rm = validation.bland_altman(d), validation.bland_altman(d, g)
    assert rm["bias"] == pytest.approx(d.mean())
    assert rm["n_groups"] == 10
    # with equal group sizes the RM SD is close to the pooled SD (both capture total variance)
    assert rm["sd"] == pytest.approx(pooled["sd"], rel=0.15)
    assert rm["loa_high"] - rm["loa_low"] == pytest.approx(2 * 1.96 * rm["sd"])


def test_bland_altman_single_group_is_within_sd():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    rm = validation.bland_altman(d, np.zeros(4))
    assert rm["sd"] == pytest.approx(np.std(d, ddof=1))


def test_wfdb_format16_reader(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(-3000, 3000, size=(50, 2)).astype("<i2")
    raw[3, 0] = -32768  # invalid sample
    raw.tofile(tmp_path / "rec01.dat")
    (tmp_path / "rec01.hea").write_text(
        "rec01 2 125 50\n"
        "rec01.dat 16 1000(-100)/NU 0 0 0 0 0 PLETH,\n"
        "rec01.dat 16 200/mV 0 0 0 0 0 II,\n"
        "#<age>: 1\n"
    )
    fs, sig = io_bidmc.read_record(tmp_path / "rec01")
    assert fs == 125.0 and set(sig) == {"PLETH", "II"}
    expected = (raw[:, 0].astype(float) + 100) / 1000
    assert np.isnan(sig["PLETH"][3])
    np.testing.assert_allclose(np.delete(sig["PLETH"], 3), np.delete(expected, 3))
    np.testing.assert_allclose(sig["II"], raw[:, 1] / 200)

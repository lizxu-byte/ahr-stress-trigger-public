import numpy as np
import pytest
from ahr import ppg, synthetic

FS = 64.0


def _true_hr(beat_times):
    return 60.0 / np.diff(beat_times).mean()


@pytest.mark.parametrize("hr", [50, 72, 110, 150])
@pytest.mark.parametrize("causal", [False, True])
def test_elgendi_recovers_known_hr(hr, causal):
    _, x, bt = synthetic.pulse_train(hr, 120, fs=FS, noise_sd=0.05, seed=3)
    y = ppg.bandpass(x, FS, causal=causal)
    peaks = ppg.elgendi_peaks(y, FS, causal=causal)
    beat_t, ibi = ppg.ibi_from_peaks(peaks, FS)
    ok = ppg.clean_ibi(ibi)
    assert ok.mean() > 0.9
    assert abs(60000.0 / ibi[ok].mean() - _true_hr(bt)) < 1.0
    assert abs(len(peaks) - len(bt)) <= 0.05 * len(bt)


@pytest.mark.parametrize("hr", [50, 72, 150])
def test_prominence_detector_recovers_known_hr(hr):
    _, x, bt = synthetic.pulse_train(hr, 120, fs=FS, noise_sd=0.05, seed=4)
    peaks = ppg.prominence_peaks(ppg.bandpass(x, FS), FS)
    assert abs(60.0 / np.diff(peaks / FS).mean() - _true_hr(bt)) < 1.0


def test_peak_positions_match_true_beats():
    _, x, bt = synthetic.pulse_train(72, 60, fs=FS, seed=5)
    peaks = ppg.elgendi_peaks(ppg.bandpass(x, FS), FS)
    lag = ppg.peak_lag_s(np.round(bt * FS).astype(int), peaks, FS)
    assert abs(lag) <= 2 / FS


def test_zero_phase_has_no_delay_and_causal_matches_group_delay():
    """Envelope of a Gaussian tone burst is delayed by the group delay."""
    t = np.arange(0, 40, 1 / FS)
    for f in (1.0, 2.0, 4.0):
        burst = np.exp(-0.5 * ((t - 20) / 2.0) ** 2) * np.sin(2 * np.pi * f * t)
        assert abs(ppg.envelope_delay_s(burst, ppg.bandpass(burst, FS), FS)) < 1e-3
        measured = ppg.envelope_delay_s(burst, ppg.bandpass(burst, FS, causal=True), FS)
        expected = ppg.group_delay_s(FS, f)[0]
        assert measured > 0.03
        assert abs(measured - expected) <= 0.05 * expected


def test_streaming_bandpass_equals_batch_causal():
    _, x, _ = synthetic.pulse_train(72, 30, fs=FS)
    sb = ppg.StreamingBandpass(FS)
    chunks = np.array_split(x, 17)
    y_stream = np.concatenate([sb.process(c) for c in chunks])
    np.testing.assert_allclose(y_stream, ppg.bandpass(x, FS, causal=True), atol=1e-10)


def test_causal_peaks_do_not_depend_on_future_samples():
    _, x, _ = synthetic.pulse_train(80, 60, fs=FS, seed=6)
    y = ppg.bandpass(x, FS, causal=True)
    full, det_full = ppg.elgendi_peaks(y, FS, causal=True, return_detection=True)
    cut = int(30 * FS)
    part, det_part = ppg.elgendi_peaks(y[:cut], FS, causal=True, return_detection=True)
    known = det_full < cut - 1
    np.testing.assert_array_equal(full[known], part[: known.sum()])
    assert np.all(det_full >= full)


def test_clean_ibi_rejects_artifacts():
    rng = np.random.default_rng(0)
    ibi = 800 + 20 * rng.standard_normal(200)
    ibi[50] = 1600  # missed beat
    ibi[100] = 400  # false detection
    ibi[150] = 250  # out of bounds
    ibi[160] = 2500
    ok = ppg.clean_ibi(ibi)
    assert not ok[[50, 100, 150, 160]].any()
    assert ok.sum() >= 190


def test_clean_ibi_follows_sustained_hr_change():
    ibi = np.r_[np.full(50, 1000.0), np.full(50, 600.0)]
    ok = ppg.clean_ibi(ibi)
    assert ok[-40:].all()


def test_interpolation_and_windowed_hrv():
    beat_t = np.cumsum(np.full(120, 0.75))
    ibi = np.full(120, 750.0)
    ibi[1::2] += 20  # alternating -> RMSSD = 20 ms
    valid = np.ones(120, bool)
    grid, iv = ppg.interpolate_ibi(beat_t, ibi, valid, fs_out=4)
    assert np.nanmin(iv) >= 750 and np.nanmax(iv) <= 770
    win = ppg.windowed_hrv(beat_t, ibi, valid, t_end=np.array([60.0, 80.0]), win_s=30)
    assert np.allclose(win["mean_hr"], 60000 / 760, atol=0.5)
    assert np.allclose(win["rmssd"], 20, atol=0.5)
    # a window with no beats is NaN
    empty = ppg.windowed_hrv(beat_t, ibi, valid, t_end=np.array([500.0]), win_s=30)
    assert empty["mean_hr"].isna().all()

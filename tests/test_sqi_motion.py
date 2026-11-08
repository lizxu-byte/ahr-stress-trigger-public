import numpy as np

from ahr import motion, ppg, sqi, synthetic

FS = 64.0


def test_sqi_separates_clean_from_noisy_ppg():
    _, x, _ = synthetic.pulse_train(72, 30, fs=FS, seed=1)
    clean = ppg.bandpass(x, FS)
    noisy = ppg.bandpass(np.random.default_rng(0).standard_normal(len(x)), FS)
    pk_clean = ppg.elgendi_peaks(clean, FS)
    pk_noisy = ppg.elgendi_peaks(noisy, FS)
    assert sqi.template_correlation(clean, pk_clean, FS) > 0.95
    assert sqi.template_correlation(noisy, pk_noisy, FS) < 0.7
    assert sqi.spectral_entropy(clean, FS) < sqi.spectral_entropy(noisy, FS) - 0.15


def test_window_quality_degrades_with_motion_artifacts():
    rec = synthetic.simulate(
        [synthetic.Segment(1, 120, 0, 0.0), synthetic.Segment(0, 120, 0, 0.4)], artifact_gain=3.0, seed=2
    )
    x = ppg.bandpass(rec.ppg, FS)
    pk = ppg.elgendi_peaks(x, FS)
    q = sqi.window_quality(x, pk, FS, t_end=np.array([60.0, 110.0, 180.0, 230.0]), win_s=30)
    assert q["quality_ok"].iloc[:2].all()
    # periodic (walking-cadence) artefacts are partly mistaken for pulses, so the
    # template correlation drops but need not cross the threshold; this is why the
    # trigger also uses the accelerometer motion flag
    assert q["template_corr"].iloc[:2].min() > 0.95
    assert q["template_corr"].iloc[2:].max() < 0.9


def _rotating_gravity(n, fs):
    t = np.arange(n) / fs
    theta = 0.8 + 0.5 * np.sin(2 * np.pi * t / 60)
    return t, np.c_[np.sin(theta), np.zeros(n), np.cos(theta)]


def test_gravity_removal_per_axis_keeps_dynamic_component():
    fs = 32.0
    t, g = _rotating_gravity(int(120 * fs), fs)
    dyn = 0.2 * np.c_[np.sin(2 * np.pi * 2 * t), np.cos(2 * np.pi * 2 * t), np.zeros_like(t)]
    for method in ("highpass", "moving_average"):
        for causal in (False, True):
            est = motion.remove_gravity(g + dyn, fs, method=method, causal=causal)
            core = slice(int(10 * fs), int(110 * fs))
            mag = motion.activity_magnitude(est)[core]
            assert abs(mag.mean() - 0.2) < 0.03, (method, causal)
            # a still, rotating wrist produces (almost) no activity
            still = motion.activity_magnitude(motion.remove_gravity(g, fs, method=method, causal=causal))[core]
            # a trailing moving average lags behind the rotating gravity vector and leaks ~0.03 g
            assert still.mean() < (0.04 if (method, causal) == ("moving_average", True) else 0.005)


def test_naive_enmo_underestimates_motion_orthogonal_to_gravity():
    fs = 32.0
    t = np.arange(int(60 * fs)) / fs
    g = np.c_[np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]
    dyn = 0.1 * np.c_[np.sin(2 * np.pi * 2 * t), np.zeros_like(t), np.zeros_like(t)]  # horizontal
    per_axis = motion.activity_magnitude(motion.remove_gravity(g + dyn, fs)).mean()
    naive = motion.naive_enmo(g + dyn).mean()
    assert per_axis > 0.05
    assert naive < 0.01


def test_windowed_activity_flags_motion():
    fs = 32.0
    act = np.r_[np.full(int(60 * fs), 0.01), np.full(int(60 * fs), 0.3)]
    w = motion.windowed_activity(act, fs, t_end=np.array([30.0, 60.0, 120.0]), win_s=30, motion_threshold_g=0.05)
    assert w["motion"].tolist() == [False, False, True]
    np.testing.assert_allclose(w["activity"], [0.01, 0.01, 0.3])

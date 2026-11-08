"""1D-CNN shape, size and training-loop checks on tiny synthetic tensors (skipped without torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ahr import cnn  # noqa: E402


@pytest.mark.parametrize("n_classes", [2, 3])
def test_output_shape_and_parameter_budget(n_classes):
    m = cnn.SmallCNN(n_channels=2, n_classes=n_classes)
    out = m(torch.zeros(4, 2, 3840))
    assert out.shape == (4, n_classes)
    assert cnn.n_parameters(m) < 100_000
    # global pooling: any window length works
    assert m(torch.zeros(3, 2, 640)).shape == (3, n_classes)


def test_training_learns_separable_toy_problem_and_is_seeded():
    rng = np.random.default_rng(0)
    n, L = 64, 256
    y = np.repeat([0, 1], n // 2)
    t = np.arange(L) / 64.0
    freq = np.where(y == 0, 1.0, 3.0)[:, None]
    X = np.stack([np.sin(2 * np.pi * freq * t), 0.1 * rng.standard_normal((n, L))], axis=1).astype(np.float32)
    X[:, 0] += 0.2 * rng.standard_normal((n, L)).astype(np.float32)
    cfg = cnn.TrainConfig(max_epochs=15, patience=15, batch_size=16)
    va = rng.permutation(n)[:16]
    tr = np.setdiff1d(np.arange(n), va)
    m1, info = cnn.train(X[tr], y[tr], X[va], y[va], 2, seed=0, cfg=cfg)
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= 15
    assert np.mean(cnn.predict(m1, X) == y) > 0.9
    m2, _ = cnn.train(X[tr], y[tr], X[va], y[va], 2, seed=0, cfg=cfg)
    np.testing.assert_array_equal(cnn.predict(m1, X), cnn.predict(m2, X))


def test_early_stopping_restores_best_epoch():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 2, 128)).astype(np.float32)
    y = rng.integers(0, 2, 40)  # pure noise: validation loss soon stops improving
    m, info = cnn.train(X[:30], y[:30], X[30:], y[30:], 2, seed=0, cfg=cnn.TrainConfig(max_epochs=30, patience=3))
    assert info["epochs_run"] <= 30
    assert info["epochs_run"] - info["best_epoch"] <= 3

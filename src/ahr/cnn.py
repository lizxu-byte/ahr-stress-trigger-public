"""Small 1D-CNN on raw wrist signals (needs the optional ``[dl]`` extra: PyTorch).

Input: (batch, channels, samples), here 2 channels (BVP, |ACC|) x 60 s x 64 Hz,
each channel z-scored per subject with calibration statistics
(:mod:`ahr.windows`). Four convolution blocks, global average pooling and a
linear head; ~40 k parameters.

Training (:func:`train`): Adam, class-weighted cross-entropy, a fixed maximum
number of epochs, early stopping on the loss of an inner validation set that
must consist of *training* subjects only (the caller builds it with
:func:`ahr.windows.inner_split`); the weights of the best validation epoch are
restored.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_channels: int = 2, n_classes: int = 2, width: int = 16, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, k, pool):
            layers = [nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU()]
            if pool > 1:
                layers.append(nn.MaxPool1d(pool))
            return layers

        w = width
        self.features = nn.Sequential(
            *block(n_channels, w, 7, 4),  # 64 Hz -> 16 Hz
            *block(w, 2 * w, 7, 4),  # -> 4 Hz
            *block(2 * w, 4 * w, 7, 4),  # -> 1 Hz
            *block(4 * w, 4 * w, 5, 1),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainConfig:
    max_epochs: int = 40
    patience: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 16
    dropout: float = 0.3


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def _loss(model, X, y, crit, batch_size):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            total += crit(model(X[i : i + batch_size]), y[i : i + batch_size]).item() * len(X[i : i + batch_size])
    return total / max(len(X), 1)


def train(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_classes: int, seed: int, cfg: TrainConfig | None = None
) -> tuple[SmallCNN, dict]:
    """Train on (X_tr, y_tr), early-stop on (X_val, y_val). ``y`` are class indices 0..n_classes-1."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SmallCNN(X_tr.shape[1], n_classes, cfg.width, cfg.dropout)
    crit = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    Xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    best, best_state, best_epoch, wait = np.inf, copy.deepcopy(model.state_dict()), 0, 0
    history = []
    t_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            if len(idx) < 2:  # BatchNorm needs > 1 sample
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        val = _loss(model, Xv, yv, crit, 256) if len(Xv) else np.nan
        history.append(val)
        if len(Xv) == 0:
            best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
            continue
        if val < best - 1e-4:
            best, best_state, best_epoch, wait = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_loss": float(best),
            "train_seconds": time.perf_counter() - t_start}
    return model, info


def predict(model: SmallCNN, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Class indices for ``X``."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.tensor(X[i : i + batch_size], dtype=torch.float32)).argmax(1).numpy())
    return np.concatenate(out) if out else np.empty(0, int)

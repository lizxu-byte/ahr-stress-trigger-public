"""Loader for the wrist (Empatica E4) signals of the public WESAD dataset.

WESAD (Schmidt et al., 2018, ICMI) ships one pickle per subject
(``WESAD/Sx/Sx.pkl``, S2..S17 without S1 and S12). Each pickle holds the chest
(RespiBAN, 700 Hz) and wrist (E4) signals plus a ``label`` vector sampled at
700 Hz on the chest clock. Both devices were synchronised so that the first
sample of every stream corresponds to the same instant; wrist samples are
therefore aligned to labels purely by time (``t = i / fs``).

Wrist sampling rates and units
------------------------------
BVP   64 Hz   arbitrary PPG units
ACC   32 Hz   3 axes, raw counts of 1/64 g  (converted to g here)
EDA    4 Hz   microsiemens
TEMP   4 Hz   degrees Celsius

Labels: 0 undefined / transient, 1 baseline, 2 stress (TSST), 3 amusement,
4 meditation, 5-7 should be ignored.

The pickles are ~1 GB each because of the 700 Hz chest data, so
:func:`cache_subject` stores a compact ``.npz`` (wrist signals, labels and the
chest ECG, which serves only as a heart-rate reference for validating the
wrist PPG) that :func:`load_subject` prefers when available.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_FS = 700.0
WRIST_FS = {"BVP": 64.0, "ACC": 32.0, "EDA": 4.0, "TEMP": 4.0}
ACC_COUNTS_PER_G = 64.0
LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
SUBJECT_IDS = [f"S{i}" for i in range(2, 18) if i != 12]


@dataclass
class WesadSubject:
    """Tidy wrist data for one subject.

    Every frame has a float time index ``t`` in seconds from recording start
    and an integer ``label`` column (label in force at that sample).
    """

    subject: str
    bvp: pd.DataFrame  # columns: bvp, label
    acc: pd.DataFrame  # columns: x, y, z (in g), label
    eda: pd.DataFrame  # columns: eda, label
    temp: pd.DataFrame  # columns: temp, label
    labels: pd.Series  # raw 700 Hz labels indexed by time (s)
    ecg: np.ndarray | None = None  # chest ECG at 700 Hz (reference only), if available

    @property
    def duration_s(self) -> float:
        return float(self.labels.index[-1])


def label_at(times_s: np.ndarray, labels: np.ndarray, label_fs: float = LABEL_FS) -> np.ndarray:
    """Return the label in force at each time (nearest preceding label sample)."""
    idx = np.floor(np.asarray(times_s) * label_fs).astype(int)
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.asarray(labels)[idx]


def _frame(values: np.ndarray, fs: float, columns: list[str], labels: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    t = np.arange(len(values)) / fs
    df = pd.DataFrame(values, columns=columns, index=pd.Index(t, name="t"))
    df["label"] = label_at(t, labels)
    return df


def _from_arrays(subject: str, wrist: dict, labels: np.ndarray, ecg: np.ndarray | None = None) -> WesadSubject:
    labels = np.asarray(labels).astype(np.int8).ravel()
    acc_g = np.asarray(wrist["ACC"], dtype=float) / ACC_COUNTS_PER_G
    return WesadSubject(
        subject=subject,
        bvp=_frame(np.ravel(wrist["BVP"]), WRIST_FS["BVP"], ["bvp"], labels),
        acc=_frame(acc_g, WRIST_FS["ACC"], ["x", "y", "z"], labels),
        eda=_frame(np.ravel(wrist["EDA"]), WRIST_FS["EDA"], ["eda"], labels),
        temp=_frame(np.ravel(wrist["TEMP"]), WRIST_FS["TEMP"], ["temp"], labels),
        labels=pd.Series(labels, index=pd.Index(np.arange(len(labels)) / LABEL_FS, name="t"), name="label"),
        ecg=None if ecg is None else np.asarray(ecg, dtype=float).ravel(),
    )


def read_pickle(path: str | Path) -> dict:
    """Read an original WESAD pickle (Python 2 pickle, needs latin1)."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def cache_subject(pkl_path: str | Path, cache_path: str | Path) -> Path:
    """Write a compact ``.npz`` with wrist signals, 700 Hz labels and chest ECG."""
    raw = read_pickle(pkl_path)
    wrist = raw["signal"]["wrist"]
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        BVP=np.asarray(wrist["BVP"], dtype=np.float32).ravel(),
        ACC=np.asarray(wrist["ACC"], dtype=np.float32),
        EDA=np.asarray(wrist["EDA"], dtype=np.float32).ravel(),
        TEMP=np.asarray(wrist["TEMP"], dtype=np.float32).ravel(),
        label=np.asarray(raw["label"]).astype(np.int8).ravel(),
        ECG=np.asarray(raw["signal"]["chest"]["ECG"], dtype=np.float32).ravel(),
    )
    return cache_path


def load_subject(data_dir: str | Path, subject: str) -> WesadSubject:
    """Load one subject from ``data_dir``.

    Looks first for ``<data_dir>/cache/<subject>_wrist.npz`` (see
    :func:`cache_subject`), then for the original ``<data_dir>/<subject>/<subject>.pkl``.
    """
    data_dir = Path(data_dir)
    cache = data_dir / "cache" / f"{subject}_wrist.npz"
    if cache.exists():
        with np.load(cache) as z:
            wrist = {k: z[k] for k in ("BVP", "ACC", "EDA", "TEMP")}
            labels = z["label"]
            ecg = z["ECG"] if "ECG" in z.files else None
        return _from_arrays(subject, wrist, labels, ecg)
    pkl = data_dir / subject / f"{subject}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Neither {cache} nor {pkl} exists; run scripts/download_wesad.py")
    raw = read_pickle(pkl)
    return _from_arrays(subject, raw["signal"]["wrist"], raw["label"], raw["signal"]["chest"]["ECG"])


def available_subjects(data_dir: str | Path) -> list[str]:
    """Subjects for which a cache file or original pickle is present."""
    data_dir = Path(data_dir)
    return [
        s
        for s in SUBJECT_IDS
        if (data_dir / "cache" / f"{s}_wrist.npz").exists() or (data_dir / s / f"{s}.pkl").exists()
    ]

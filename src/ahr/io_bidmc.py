"""Loader for the BIDMC PPG and Respiration Dataset (PhysioNet, v1.0.0).

53 eight-minute recordings of ICU patients, extracted from the MIMIC-II
matched waveform database (Pimentel et al., 2017, IEEE TBME 64(8), online 2016). Each WFDB
record ``bidmcNN`` holds the fingertip PPG (``PLETH``), impedance
respiration (``RESP``) and 2-5 ECG leads, all at 125 Hz, 60001 samples, as
16-bit integers (WFDB format 16). Lead II is present in every record and is
used here as the heart-rate reference.

Only format 16 is implemented (all BIDMC records use it), so the package
does not depend on ``wfdb``; ``tests/test_validation.py`` checks the parser
on a synthetic record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 125.0
RECORD_IDS = [f"bidmc{i:02d}" for i in range(1, 54)]


@dataclass
class BidmcRecord:
    record: str
    fs: float
    ppg: np.ndarray  # PLETH, normalised units
    ecg: np.ndarray  # lead II, mV
    signals: list[str]


_SIG = re.compile(r"^(\S+)\s+(\d+)\s+([-\d.eE+]+)(?:\((-?\d+)\))?(?:/(\S+))?\s+.*\s(\S+?),?\s*$")


def read_header(path: str | Path) -> tuple[int, float, int, list[dict]]:
    """Parse a WFDB header: ``(n_signals, fs, n_samples, [signal specs])``."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    rec = lines[0].split()
    n_sig, fs, n_samp = int(rec[1]), float(rec[2]), int(rec[3])
    specs = []
    for ln in lines[1 : 1 + n_sig]:
        m = _SIG.match(ln)
        if m is None:
            raise ValueError(f"cannot parse signal line {ln!r}")
        fname, fmt, gain, baseline, units, name = m.groups()
        if fmt != "16":
            raise ValueError(f"only WFDB format 16 is supported, got {fmt}")
        specs.append(
            {"file": fname, "gain": float(gain) or 200.0, "baseline": int(baseline or 0), "units": units, "name": name}
        )
    return n_sig, fs, n_samp, specs


def read_record(path_stem: str | Path) -> tuple[float, dict[str, np.ndarray]]:
    """Read ``<stem>.hea`` + its format-16 ``.dat``; returns ``(fs, {name: physical signal})``."""
    stem = Path(path_stem)
    n_sig, fs, n_samp, specs = read_header(stem.with_suffix(".hea"))
    raw = np.fromfile(stem.parent / specs[0]["file"], dtype="<i2")
    raw = raw[: (len(raw) // n_sig) * n_sig].reshape(-1, n_sig)[:n_samp]
    out = {}
    for k, s in enumerate(specs):
        v = raw[:, k].astype(float)
        v[raw[:, k] == -32768] = np.nan  # WFDB invalid-sample marker
        out[s["name"]] = (v - s["baseline"]) / s["gain"]
    return fs, out


def load_record(data_dir: str | Path, record: str) -> BidmcRecord:
    fs, sig = read_record(Path(data_dir) / record)
    return BidmcRecord(record=record, fs=fs, ppg=sig["PLETH"], ecg=sig["II"], signals=list(sig))


def available_records(data_dir: str | Path) -> list[str]:
    d = Path(data_dir)
    return [r for r in RECORD_IDS if (d / f"{r}.hea").exists() and (d / f"{r}.dat").exists()]

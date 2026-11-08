"""Download the BIDMC PPG and Respiration Dataset (v1.0.0) from PhysioNet.

Usage
-----
    python scripts/download_bidmc.py               # -> data/bidmc/bidmcNN.{hea,dat}

Source: https://physionet.org/content/bidmc/1.0.0/ (Open Data Commons
Attribution License v1.0, DOI 10.13026/C2208R). Only the WFDB waveform files
(``.hea`` + ``.dat``: PPG, ECG, impedance respiration at 125 Hz) are fetched,
~100 MB; each file is checked against PhysioNet's ``SHA256SUMS.txt``.
Everything is written below ``data/``, which is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

BIDMC_URL = "https://physionet.org/files/bidmc/1.0.0/"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/bidmc"))
    args = ap.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    records = fetch(BIDMC_URL + "RECORDS").decode().split()
    sums = {}
    for line in fetch(BIDMC_URL + "SHA256SUMS.txt").decode().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    (args.data_dir / "RECORDS").write_text("\n".join(records) + "\n")
    print(f"{len(records)} records listed")

    for rec in records:
        for ext in (".hea", ".dat"):
            name = rec + ext
            dest = args.data_dir / name
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sums.get(name):
                continue
            blob = fetch(BIDMC_URL + name)
            digest = hashlib.sha256(blob).hexdigest()
            if name in sums and digest != sums[name]:
                raise RuntimeError(f"SHA256 mismatch for {name}")
            dest.write_bytes(blob)
        print(f"{rec}: ok", flush=True)
    print(f"Done. Files in {args.data_dir}")


if __name__ == "__main__":
    main()

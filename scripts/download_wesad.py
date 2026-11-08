"""Download WESAD from the official source and build a compact local cache.

Usage
-----
    python scripts/download_wesad.py              # -> data/WESAD.zip, data/WESAD/cache/*.npz
    python scripts/download_wesad.py --keep-pkl   # also keep the extracted ~1 GB pickles

The dataset is distributed by the University of Siegen for non-commercial use
only and may not be redistributed. Everything is written below ``data/``,
which is git-ignored; never commit these files.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahr.io_wesad import SUBJECT_IDS, cache_subject  # noqa: E402

WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while block := r.read(chunk):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--keep-pkl", action="store_true", help="keep extracted pickles")
    ap.add_argument("--rebuild", action="store_true", help="rebuild existing cache files")
    args = ap.parse_args()

    zip_path = args.data_dir / "WESAD.zip"
    out_dir = args.data_dir / "WESAD"
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading {WESAD_URL} -> {zip_path} (~2.1 GB)")
        download(WESAD_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for subject in SUBJECT_IDS:
            member = f"WESAD/{subject}/{subject}.pkl"
            cache = out_dir / "cache" / f"{subject}_wrist.npz"
            if cache.exists() and not args.rebuild:
                print(f"{subject}: cached")
                continue
            if member not in names:
                print(f"{subject}: not in archive, skipped")
                continue
            print(f"{subject}: extracting and caching wrist signals + chest ECG")
            z.extract(member, args.data_dir)
            pkl = args.data_dir / member
            cache_subject(pkl, cache)
            if not args.keep_pkl:
                pkl.unlink()
                if not any(pkl.parent.iterdir()):
                    pkl.parent.rmdir()
    print(f"Done. Cache in {out_dir / 'cache'}")


if __name__ == "__main__":
    main()

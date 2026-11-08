"""Validate the PPG pipeline on BIDMC (fingertip PPG vs ECG, 53 x 8 min, 125 Hz).

Usage
-----
    python scripts/download_bidmc.py      # once, -> data/bidmc/
    python scripts/run_bidmc.py           # -> results/bidmc/   (needs pip install -e ".[validation]")

What is compared
----------------
Reference beats: R peaks from ``ahr.ecg.detect_r_peaks`` on lead II.
Test beats (PPG, upright polarity, 0.5-8 Hz band-pass from ``ahr.ppg``):

* ``elgendi_offline``     zero-phase band-pass + Elgendi detector
* ``elgendi_causal``      forward band-pass + causal Elgendi (beats usable only once detected)
* ``prominence_offline``  zero-phase band-pass + prominence detector
* ``prominence_causal``   forward band-pass + prominence detector (the filter is causal;
                          ``find_peaks`` itself needs a short look-ahead)
* ``neurokit_ppg``        NeuroKit2 ``ppg_clean`` + ``ppg_findpeaks`` (independent implementation)

NeuroKit2 ``ecg_peaks`` gives a second ECG detector, used to check the reference.

Beat level: per record, the constant PPG-ECG delay (pulse arrival time plus any
channel offset) is estimated by :func:`ahr.validation.estimate_delay_s` and removed;
beats are then matched one-to-one within a tolerance (primary 100 ms).
Window level: HR in non-overlapping 30 s windows (16 per record) after the
pipeline's IBI cleaning; MAE, bias and repeated-measures Bland-Altman limits.

Only aggregated per-record / per-detector tables and one figure are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahr import ecg, io_bidmc, ppg, sqi, validation  # noqa: E402
from ahr.pipeline import FeatureConfig  # noqa: E402

C_OK, C_BAD = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

WIN_S = 30.0
TOLERANCES_S = (0.05, 0.10, 0.15)
PRIMARY_TOL_S = 0.10
ECG_TOL_S = 0.05  # ECG vs ECG, after removing a constant offset (different deflection picked)
ECG_DISPUTE_F1 = 0.95  # below this, our and NeuroKit2's ECG beats disagree -> reference disputed


def ppg_beats(x_raw: np.ndarray, fs: float, cfg: FeatureConfig) -> dict[str, dict]:
    """Peaks (s), availability times (s) and filtered signal for each in-package detector."""
    out = {}
    for mode in ("offline", "causal"):
        causal = mode == "causal"
        x = ppg.bandpass(x_raw, fs, cfg.bp_low, cfg.bp_high, cfg.bp_order, causal=causal)
        pk, det = ppg.elgendi_peaks(x, fs, causal=causal, return_detection=True)
        out[f"elgendi_{mode}"] = dict(peaks=pk, avail=det if causal else None, x=x, causal=causal)
        pk2 = ppg.prominence_peaks(x, fs)
        out[f"prominence_{mode}"] = dict(peaks=pk2, avail=None, x=x, causal=causal)
    return out


def window_hr(beats_s: np.ndarray, t_end: np.ndarray, avail_s: np.ndarray | None = None) -> np.ndarray:
    """Pipeline HR in trailing windows (IBI cleaning + coverage rule)."""
    bt, ibi = ppg.ibi_from_peaks(np.asarray(beats_s) * 1.0, 1.0)
    valid = ppg.clean_ibi(ibi)
    av = None if avail_s is None else np.asarray(avail_s)[1:]
    return ppg.windowed_hrv(bt, ibi, valid, t_end, WIN_S, available_t=av)["mean_hr"].to_numpy()


def process_record(rec: io_bidmc.BidmcRecord, cfg: FeatureConfig, nk) -> tuple[list, list, dict]:
    fs = rec.fs
    t_end = np.arange(WIN_S, len(rec.ppg) / fs + 1e-9, WIN_S)
    r_ours = ecg.detect_r_peaks(rec.ecg, fs) / fs
    _, info = nk.ecg_peaks(rec.ecg, sampling_rate=int(fs))
    r_nk = np.asarray(info["ECG_R_Peaks"], float) / fs

    ecg_delay = validation.estimate_delay_s(r_ours, r_nk, -0.25, 0.25)
    ecg_check = {"record": rec.record, "n_ours": len(r_ours), "n_neurokit": len(r_nk), "offset_ms": 1000 * ecg_delay,
                 **{k: v for k, v in validation.beat_agreement(r_ours, r_nk, ECG_TOL_S, ecg_delay).items()
                    if k in ("sensitivity", "ppv", "f1")}}

    dets = ppg_beats(rec.ppg, fs, cfg)
    clean = nk.ppg_clean(rec.ppg, sampling_rate=int(fs))
    nk_pk = np.asarray(nk.ppg_findpeaks(clean, sampling_rate=int(fs))["PPG_Peaks"], int)
    dets["neurokit_ppg"] = dict(peaks=nk_pk, avail=None, x=ppg.bandpass(rec.ppg, fs, cfg.bp_low, cfg.bp_high, cfg.bp_order),
                                causal=False)

    hr_ref = window_hr(r_ours, t_end)
    hr_ref_nk = window_hr(r_nk, t_end)
    beat_rows, win_rows = [], []
    for name, d in dets.items():
        pk_s = d["peaks"] / fs
        delay = validation.estimate_delay_s(r_ours, pk_s)
        for tol in TOLERANCES_S:
            beat_rows.append({"record": rec.record, "detector": name, "tolerance_ms": 1000 * tol, "delay_ms": 1000 * delay,
                              **validation.beat_agreement(r_ours, pk_s, tol, delay)})
        avail = None if d["avail"] is None else d["avail"] / fs
        hr = window_hr(pk_s, t_end, avail)
        thr = cfg.min_template_corr_causal if d["causal"] else cfg.min_template_corr
        q = sqi.window_quality(d["x"], d["peaks"], fs, t_end, WIN_S, thr, cfg.max_spectral_entropy,
                               available_peaks=d["avail"])
        for k, te in enumerate(t_end):
            win_rows.append({"record": rec.record, "detector": name, "t": te, "hr": hr[k], "hr_ref": hr_ref[k],
                             "hr_ref_neurokit": hr_ref_nk[k], "quality_ok": bool(q["quality_ok"].iloc[k]) and np.isfinite(hr[k])})
    return beat_rows, win_rows, ecg_check


def summarise_beats(beats: pd.DataFrame, disputed: set[str]) -> pd.DataFrame:
    rows = []
    for subset, b in (("all records", beats), ("undisputed ECG reference", beats[~beats["record"].isin(disputed)])):
        for (det, tol), g in b.groupby(["detector", "tolerance_ms"], sort=False):
            rows.append({
                "records": subset, "detector": det, "tolerance_ms": tol, "n_records": g["record"].nunique(),
                "n_ref_beats": int(g["n_ref"].sum()),
                "sensitivity_pooled": g["tp"].sum() / g["n_ref"].sum(), "ppv_pooled": g["tp"].sum() / g["n_test"].sum(),
                "f1_median": g["f1"].median(), "f1_q25": g["f1"].quantile(0.25), "f1_q75": g["f1"].quantile(0.75),
                "records_f1_ge_0.9": int((g["f1"] >= 0.9).sum()),
                "abs_err_ms_median": g["abs_err_ms_median"].median(), "delay_ms_median": g["delay_ms"].median(),
            })
    return pd.DataFrame(rows)


def summarise_hr(win: pd.DataFrame, disputed: set[str]) -> pd.DataFrame:
    rows = []
    for subset, w in (("all records", win), ("undisputed ECG reference", win[~win["record"].isin(disputed)])):
        for det, g in w.groupby("detector", sort=False):
            for ref_col, ref_name in (("hr_ref", "ahr.ecg"), ("hr_ref_neurokit", "neurokit ecg")):
                for gate, sel in (("all windows with HR", g["hr"].notna()), ("SQI ok", g["quality_ok"])):
                    s = sel & g[ref_col].notna()
                    ba = validation.bland_altman((g.loc[s, "hr"] - g.loc[s, ref_col]).to_numpy(), g.loc[s, "record"].to_numpy())
                    rows.append({"records": subset, "detector": det, "reference": ref_name, "windows": gate,
                                 "coverage": s.sum() / g[ref_col].notna().sum(), **ba,
                                 "share_abs_err_le_5bpm": float((np.abs(g.loc[s, "hr"] - g.loc[s, ref_col]) <= 5).mean())})
    return pd.DataFrame(rows)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_bland_altman(win: pd.DataFrame, hr_sum: pd.DataFrame, path: Path) -> None:
    dets = [("elgendi_offline", "Elgendi, zero-phase (offline)"), ("elgendi_causal", "Elgendi, causal")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, (det, title) in zip(axes, dets):
        g = win[(win["detector"] == det) & win["hr"].notna() & win["hr_ref"].notna()]
        mean, diff = (g["hr"] + g["hr_ref"]) / 2, g["hr"] - g["hr_ref"]
        ok = g["quality_ok"]
        ax.scatter(mean[~ok], diff[~ok], s=18, facecolors="none", edgecolors=C_BAD, lw=1.0, label="SQI not ok")
        ax.scatter(mean[ok], diff[ok], s=14, color=C_OK, edgecolors="white", lw=0.5, label="SQI ok")
        for gate, ls in (("all windows with HR", "-"), ("SQI ok", (0, (3, 2)))):
            r = hr_sum[(hr_sum["records"] == "all records") & (hr_sum["detector"] == det)
                       & (hr_sum["reference"] == "ahr.ecg") & (hr_sum["windows"] == gate)].iloc[0]
            for v in (r["bias"], r["loa_low"], r["loa_high"]):
                ax.axhline(v, color=MUTED, lw=0.9, ls=ls)
        r_all = hr_sum[(hr_sum["records"] == "all records") & (hr_sum["detector"] == det)
                       & (hr_sum["reference"] == "ahr.ecg")].set_index("windows")
        txt = "\n".join(
            f"{k}: bias {r_all.loc[k, 'bias']:+.2f}, LoA [{r_all.loc[k, 'loa_low']:.1f}, {r_all.loc[k, 'loa_high']:.1f}] bpm (n={int(r_all.loc[k, 'n'])})"
            for k in ("all windows with HR", "SQI ok"))
        ax.set_title(title, loc="left", fontsize=10, color=TEXT, pad=28)
        ax.text(0.0, 1.02, txt, transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
        ax.set_xlabel("mean of PPG and ECG HR (bpm)", color=MUTED, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("PPG HR − ECG HR (bpm)", color=MUTED, fontsize=9)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("BIDMC, 53 records, 30 s windows. Lines: bias and 95 % limits of agreement "
                 "(solid: all windows, dashed: SQI ok; repeated-measures SD)", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/bidmc"))
    ap.add_argument("--out", type=Path, default=Path("results/bidmc"))
    ap.add_argument("--records", nargs="*", default=None)
    args = ap.parse_args()
    import neurokit2 as nk

    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig()  # upright polarity: clinical pulse-oximeter pleth
    beat_rows, win_rows, ecg_rows = [], [], []
    for r in args.records or io_bidmc.available_records(args.data_dir):
        rec = io_bidmc.load_record(args.data_dir, r)
        b, w, e = process_record(rec, cfg, nk)
        beat_rows += b
        win_rows += w
        ecg_rows.append(e)
        f1 = {x["detector"]: x["f1"] for x in b if x["tolerance_ms"] == 1000 * PRIMARY_TOL_S}
        print(f"{r}: ECG agreement F1 {e['f1']:.3f}, Elgendi offline F1 {f1['elgendi_offline']:.3f}", flush=True)

    beats, win, ecg_check = pd.DataFrame(beat_rows), pd.DataFrame(win_rows), pd.DataFrame(ecg_rows)
    disputed = set(ecg_check.loc[ecg_check["f1"] < ECG_DISPUTE_F1, "record"])
    ecg_check["disputed"] = ecg_check["record"].isin(disputed)
    beat_sum = summarise_beats(beats, disputed)
    hr_sum = summarise_hr(win, disputed)

    ecg_check.round(3).to_csv(args.out / "ecg_reference_check.csv", index=False)
    beats.round(4).to_csv(args.out / "beat_agreement_per_record.csv", index=False)
    beat_sum.round(4).to_csv(args.out / "beat_agreement_summary.csv", index=False)
    hr_sum.round(3).to_csv(args.out / "hr_agreement.csv", index=False)
    plot_bland_altman(win, hr_sum, args.out / "figures" / "bland_altman.png")

    summary = {
        "dataset": "BIDMC PPG and Respiration Dataset v1.0.0 (PhysioNet, DOI 10.13026/C2208R, ODC-By 1.0)",
        "n_records": int(win["record"].nunique()),
        "fs_hz": io_bidmc.FS,
        "neurokit2_version": nk.__version__,
        "window_s": WIN_S,
        "beat_tolerances_ms": [1000 * t for t in TOLERANCES_S],
        "primary_tolerance_ms": 1000 * PRIMARY_TOL_S,
        "ecg_dispute_threshold_f1": ECG_DISPUTE_F1,
        "disputed_records": sorted(disputed),
        "ppg_polarity": "upright (not inverted)",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    prim = beat_sum[(beat_sum["tolerance_ms"] == 1000 * PRIMARY_TOL_S)]
    print("\nBeat agreement (100 ms)\n", prim.round(3).to_string(index=False))
    print("\nHR agreement vs ahr.ecg\n", hr_sum[hr_sum["reference"] == "ahr.ecg"].round(2).to_string(index=False))
    print(f"\nDisputed ECG reference: {sorted(disputed)}\nResults written to {args.out}")


if __name__ == "__main__":
    main()

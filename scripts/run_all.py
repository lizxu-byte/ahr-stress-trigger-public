"""End-to-end analysis: features -> AHR -> simulated ESM trigger -> evaluation.

Usage
-----
    python scripts/run_all.py                      # WESAD in data/WESAD -> results/
    python scripts/run_all.py --synthetic          # synthetic subjects -> results/synthetic/
    python scripts/run_all.py --subjects S2 S3     # subset

Only aggregated numbers (per condition, per subject, per threshold) and
figures are written; no per-sample data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahr import ecg, evaluate, io_wesad, pipeline, ppg, synthetic  # noqa: E402
from ahr.ahr import add_ahr_columns  # noqa: E402
from ahr.trigger import TriggerConfig, simulate  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots 1-3
C_OBS, C_PRED, C_AHR = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_SHADE = {1: "#efeeea", 2: "#f9ded4", 3: "#dceaf8", 4: "#dcefe6"}

THRESHOLDS = np.arange(0, 31, 1.0)
FP_BUDGET = 0.5  # false prompts per non-stress hour allowed during LOSO tuning
PRE_S = (0.0, 600.0)  # anticipation windows credited as hits (strict, 10 min before onset)
LAGS_S = (30.0, 60.0)
SCORES = {  # name -> (score column, quality column)
    "AHR (wrist PPG)": ("ahr", "quality_ok"),
    "HR - baseline (wrist PPG)": ("hr_rel", "quality_ok"),
    "AHR (reference HR)": ("ahr_ecg", "quality_ecg"),
    "HR - baseline (reference HR)": ("hr_rel_ecg", "quality_ecg"),
}


# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------


def wesad_subjects(data_dir: Path, subjects: list[str] | None):
    for sid in subjects or io_wesad.available_subjects(data_dir):
        d = io_wesad.load_subject(data_dir, sid)
        yield sid, dict(
            bvp=d.bvp["bvp"].to_numpy(),
            fs_bvp=io_wesad.WRIST_FS["BVP"],
            acc=d.acc[["x", "y", "z"]].to_numpy(),
            fs_acc=io_wesad.WRIST_FS["ACC"],
            labels=evaluate.SubjectLabels(d.labels.to_numpy(), io_wesad.LABEL_FS),
            ecg=d.ecg,
            fs_ecg=io_wesad.LABEL_FS,
        )


def synthetic_subjects(n: int):
    rng = np.random.default_rng(42)
    for i in range(n):
        protocol = synthetic.wesad_like_protocol(stress_offset=float(rng.uniform(4, 14)))
        rec = synthetic.simulate(
            protocol, hr_rest=float(rng.uniform(60, 80)), k_act=float(rng.uniform(30, 60)), artifact_gain=0.3, seed=i
        )
        yield f"SYN{i + 1}", dict(
            bvp=rec.ppg,
            fs_bvp=rec.fs_ppg,
            acc=rec.acc,
            fs_acc=rec.fs_acc,
            labels=evaluate.SubjectLabels(rec.labels, rec.label_fs),
            ecg=None,
            true_beats=rec.beat_times,
        )


# ----------------------------------------------------------------------------
# Per-subject processing
# ----------------------------------------------------------------------------


def reference_hr(src: dict, t_end: np.ndarray, win_s: float) -> pd.DataFrame | None:
    """Trailing-window HR from a reference beat series (chest ECG or synthetic truth)."""
    if src.get("ecg") is not None:
        beats_t = ecg.detect_r_peaks(src["ecg"], src["fs_ecg"]) / src["fs_ecg"]
    elif src.get("true_beats") is not None:
        beats_t = src["true_beats"]
    else:
        return None
    bt, ibi = beats_t[1:], np.diff(beats_t) * 1000
    return ppg.windowed_hrv(bt, ibi, ppg.clean_ibi(ibi), t_end, win_s)


def process_subject(sid: str, src: dict, cfg: pipeline.FeatureConfig):
    lab = src["labels"]
    feats = {}
    for mode in ("offline", "causal"):
        f = pipeline.extract_features(
            src["bvp"], src["fs_bvp"], src["acc"], src["fs_acc"], causal=mode == "causal", cfg=cfg,
            compare_detectors=mode == "offline",
        )
        wl = pipeline.window_labels(f["t"].to_numpy(), lab.labels, lab.fs, cfg.win_s)
        f["label"], f["label_purity"] = wl["label"].to_numpy(), wl["label_purity"].to_numpy()
        f["subject"] = sid
        ref = reference_hr(src, f["t"].to_numpy(), cfg.win_s)
        if ref is not None:
            f["hr_ecg"] = ref["mean_hr"].to_numpy()
            f["quality_ecg"] = f["hr_ecg"].notna()
        calib = ((f["label"] == 1) & (f["label_purity"] >= 0.95) & f["quality_ok"]).to_numpy()
        f, model = add_ahr_columns(f, calib, LAGS_S)
        coefs = {"subject": sid, "mode": mode, "hr_source": "ppg", **model.summary()}
        models = [coefs]
        if ref is not None:
            calib_ecg = ((f["label"] == 1) & (f["label_purity"] >= 0.95) & f["quality_ecg"]).to_numpy()
            g, model_ecg = add_ahr_columns(f, calib_ecg, LAGS_S, hr_col="hr_ecg")
            f["hr_pred_ecg"], f["ahr_ecg"], f["hr_rel_ecg"] = g["hr_pred"], g["ahr"], g["hr_rel"]
            models.append({"subject": sid, "mode": mode, "hr_source": "ecg", **model_ecg.summary()})
        feats[mode] = (f, models)

    # timing shift of causal vs zero-phase systolic peaks
    sign = -1.0 if cfg.invert_ppg else 1.0
    x0 = ppg.bandpass(sign * src["bvp"], src["fs_bvp"])
    x1 = ppg.bandpass(sign * src["bvp"], src["fs_bvp"], causal=True)
    lag = ppg.peak_lag_s(ppg.elgendi_peaks(x0, src["fs_bvp"]), ppg.elgendi_peaks(x1, src["fs_bvp"], causal=True), src["fs_bvp"])
    return feats, lag


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _shade(ax, lab: evaluate.SubjectLabels, label_text: bool = False):
    m = np.r_[0, lab.labels, 0]
    change = np.flatnonzero(np.diff(m)) + 0
    edges = np.r_[0, change, len(lab.labels)]
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        v = int(lab.labels[min(a, len(lab.labels) - 1)])
        if v in COND_SHADE:
            ax.axvspan(a / lab.fs / 60, b / lab.fs / 60, color=COND_SHADE[v], lw=0, zorder=0)
            if label_text:
                ax.text((a + b) / 2 / lab.fs / 60, 1.0, evaluate.CONDITIONS[v], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color=MUTED)


def plot_subject(
    sid: str, f: pd.DataFrame, lab: evaluate.SubjectLabels, prompts: np.ndarray, path: Path, src: str = "", ref_name: str = ""
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 0.8]})
    tmin = f["t"] / 60
    hr_col, pred_col, ahr_col = ("hr_ecg", "hr_pred_ecg", "ahr_ecg") if src == "ecg" else ("mean_hr", "hr_pred", "ahr")
    hr_name = f"reference HR ({ref_name})" if src == "ecg" else "wrist-PPG HR"
    q = f["quality_ecg"] if src == "ecg" else f["quality_ok"]
    ax = axes[0]
    _shade(ax, lab, label_text=True)
    ax.plot(tmin, f[hr_col].where(q), color=C_OBS, lw=1.6, label=f"observed {hr_name}")
    ax.plot(tmin, f[pred_col], color=C_PRED, lw=1.6, label="predicted from activity")
    ax.set_ylabel("bpm", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"{sid}: heart rate, activity-predicted heart rate and AHR", loc="left", fontsize=10, color=TEXT, pad=14)
    ax = axes[1]
    _shade(ax, lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(tmin, f[ahr_col].where(q), color=C_AHR, lw=1.6)
    for p in prompts:
        ax.axvline(p / 60, color=TEXT, lw=0.8, ls=(0, (2, 2)))
    ax.set_ylabel("AHR (bpm)", color=MUTED, fontsize=9)
    ax.text(0.995, 0.95, "dashed: simulated ESM prompts", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    ax = axes[2]
    _shade(ax, lab)
    ax.plot(tmin, f["activity"], color=MUTED, lw=1.2)
    ax.set_ylabel("activity (g)", color=MUTED, fontsize=9)
    ax.set_xlabel("time (min)", color=MUTED, fontsize=9)
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(pooled: dict[str, pd.DataFrame], loso_points: dict[str, dict], path: Path, ref_name: str, subtitle: str):
    names = list(pooled)
    has_ecg = any("reference" in n for n in names)
    fig, axes = plt.subplots(1, 2 if has_ecg else 1, figsize=(10 if has_ecg else 5.5, 4.2), sharey=True, squeeze=False)
    groups = [("wrist PPG", [n for n in names if "reference" not in n])]
    if has_ecg:
        groups.append((f"reference HR ({ref_name})", [n for n in names if "reference" in n]))
    for ax, (title, members) in zip(axes[0], groups):
        for n, color in zip(members, (C_AHR, C_OBS)):
            d = pooled[n].sort_values("threshold")
            label = "AHR" if n.startswith("AHR") else "HR - baseline mean"
            ax.plot(d["fp_per_h"], d["hit_rate"], color=color, lw=2, marker="o", ms=4, label=label)
            for thr in (5, 10, 15, 20):
                r = d[np.isclose(d["threshold"], thr)]
                if len(r):
                    ax.annotate(f"{thr}", (r["fp_per_h"].iloc[0], r["hit_rate"].iloc[0]), textcoords="offset points",
                                xytext=(4, 3), fontsize=7, color=MUTED)
            if n in loso_points:
                p = loso_points[n]
                ax.plot(p["fp_per_h"], p["hit_rate"], marker="D", ms=8, color=color, mec="white", mew=2, ls="none")
        ax.axvline(FP_BUDGET, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("false prompts per non-stress hour", color=MUTED, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0][0].set_ylabel("stress episodes detected", color=MUTED, fontsize=9)
    fig.suptitle(f"Threshold sweep, {subtitle} (numbers: threshold in bpm; diamonds: LOSO-tuned operating point)",
                 x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/WESAD"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--synthetic", type=int, nargs="?", const=6, default=0, metavar="N",
                    help="use N synthetic subjects instead of WESAD (default 6)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--example", default=None,
                    help="subject for the example figure (default: median wrist-PPG quality)")
    args = ap.parse_args()

    out = args.out or (Path("results/synthetic") if args.synthetic else Path("results"))
    (out / "figures" / "subjects").mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        cfg = pipeline.FeatureConfig()
        source = synthetic_subjects(args.synthetic)
    else:
        cfg = pipeline.FeatureConfig(invert_ppg=True)  # E4 BVP polarity, see README
        source = wesad_subjects(args.data_dir, args.subjects)

    causal, offline, models, labels, lags = {}, {}, [], {}, {}
    for sid, src in source:
        feats, lag = process_subject(sid, src, cfg)
        offline[sid], m_off = feats["offline"]
        causal[sid], m_c = feats["causal"]
        models += m_off + m_c
        labels[sid] = src["labels"]
        lags[sid] = lag
        print(f"{sid}: {len(causal[sid])} windows, quality ok {causal[sid]['quality_ok'].mean():.0%}", flush=True)
    sids = list(causal)
    has_ref = "hr_ecg" in causal[sids[0]]
    scores = {k: v for k, v in SCORES.items() if has_ref or "reference" not in k}
    ref_name = "synthetic ground truth" if args.synthetic else "chest ECG"

    off_all = pd.concat(offline.values(), ignore_index=True)

    # --- signal quality / detector agreement / filter delay ----------------
    pure = off_all[off_all["label"].isin(evaluate.CONDITIONS) & (off_all["label_purity"] >= 0.95)].copy()
    pure["condition"] = pure["label"].map(evaluate.CONDITIONS)
    qrows = []
    for cond, g in pure.groupby("condition"):
        row = {
            "condition": cond,
            "windows": len(g),
            "ppg_hr_available": g["mean_hr"].notna().mean(),
            "ppg_quality_ok": g["quality_ok"].mean(),
            "motion_flag": g["motion"].mean(),
            "elgendi_vs_prominence_mad_bpm": (g["mean_hr"] - g["mean_hr_prom"]).abs().median(),
        }
        if has_ref:
            ok = g["quality_ok"] & g["hr_ecg"].notna()
            row["ppg_vs_ref_mae_bpm_quality_ok"] = (g.loc[ok, "mean_hr"] - g.loc[ok, "hr_ecg"]).abs().mean()
            allw = g["mean_hr"].notna() & g["hr_ecg"].notna()
            row["ppg_vs_ref_mae_bpm_all"] = (g.loc[allw, "mean_hr"] - g.loc[allw, "hr_ecg"]).abs().mean()
        qrows.append(row)
    quality = pd.DataFrame(qrows).set_index("condition").loc[[c for c in evaluate.CONDITIONS.values()]].reset_index()
    quality.round(3).to_csv(out / "signal_quality.csv", index=False)

    delay = pd.DataFrame(
        {
            "frequency_hz": [1.0, 1.5, 2.0, 4.0],
            "causal_group_delay_ms": ppg.group_delay_s(64.0, [1.0, 1.5, 2.0, 4.0], cfg.bp_low, cfg.bp_high, cfg.bp_order) * 1000,
            "zero_phase_group_delay_ms": 0.0,
        }
    )
    delay.round(1).to_csv(out / "filter_delay.csv", index=False)
    peak_lag_ms = 1000 * np.nanmedian(list(lags.values()))

    coef = pd.DataFrame(models)
    coef.round(3).to_csv(out / "model_coefficients.csv", index=False)

    # --- condition summaries + mixed models (offline features) -------------
    # each outcome is averaged over the windows in which its own signal is usable
    ppg_cols, ref_cols = ["mean_hr", "ahr", "rmssd", "sdnn", "activity"], ["hr_ecg", "ahr_ecg"] if has_ref else []
    value_cols = ppg_cols + ref_cols
    cm = evaluate.condition_means(off_all, ppg_cols, quality_col="quality_ok")
    if has_ref:
        cm_ref = evaluate.condition_means(off_all, ref_cols, quality_col="quality_ecg")
        # selection-bias check: reference HR restricted to the windows where the wrist PPG is usable
        cm_sel = evaluate.condition_means(off_all.assign(hr_ecg_ppg_windows=off_all["hr_ecg"]), ["hr_ecg_ppg_windows"])
        keys = ["subject", "label", "condition"]
        cm = (cm.drop(columns="n_windows").merge(cm_ref.drop(columns="n_windows"), on=keys, how="outer")
              .merge(cm_sel.drop(columns="n_windows"), on=keys, how="outer"))
        value_cols.append("hr_ecg_ppg_windows")
    evaluate.summarize_conditions(cm, value_cols).round(2).to_csv(out / "condition_summary.csv", index=False)
    mixed_txt, mixed_rows = [], []
    for col in ["mean_hr", "ahr"] + (["hr_ecg", "ahr_ecg", "hr_ecg_ppg_windows"] if has_ref else []):
        fit = evaluate.mixed_model(cm, col)
        mixed_txt.append(f"===== {col} ~ condition + (1 | subject) =====\n{fit.summary()}\n")
        for name in fit.params.index:
            if name.startswith("C(condition"):
                ci = fit.conf_int().loc[name]
                mixed_rows.append({"outcome": col, "term": name.split("[T.")[1].rstrip("]"), "estimate": fit.params[name],
                                   "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues[name]})
    (out / "mixedlm.txt").write_text("\n".join(mixed_txt))
    mixed = pd.DataFrame(mixed_rows)
    mixed.round(4).to_csv(out / "mixedlm_effects.csv", index=False)

    # --- trigger: sweep, LOSO, sensitivity analyses (causal features) ------
    base = TriggerConfig(min_duration_s=30.0, refractory_s=20 * 60.0, max_gap_s=60.0)
    variants = [  # (refractory min, motion gate, max gap s); the first is the primary configuration
        (20, True, 60.0), (10, True, 60.0), (30, True, 60.0), (20, False, 60.0), (20, True, 0.0),
    ]
    pooled, loso_points, loso_rows, sens_rows = {}, {}, [], []
    for name, (col, qcol) in scores.items():
        feats = {s: causal[s].assign(quality_ok=causal[s][qcol]) for s in sids}
        for k, (rmin, gate, gap) in enumerate(variants):
            cfg_t = replace(base, refractory_s=rmin * 60.0, block_motion=gate, max_gap_s=gap)
            per_all = evaluate.sweep(feats, labels, THRESHOLDS, cfg_t, score_col=col, pre_s=PRE_S)
            for pre in PRE_S:
                per = per_all[per_all["pre_s"] == pre]
                lo = evaluate.loso(per, FP_BUDGET)
                summ = evaluate.loso_summary(lo)
                sens_rows.append({"score": name, "refractory_min": rmin, "motion_gate": gate, "max_gap_s": gap,
                                  "anticipation_min": pre / 60, **{k2: v for k2, v in summ.items() if k2 != "threshold_range"},
                                  "threshold_min": summ["threshold_range"][0], "threshold_max": summ["threshold_range"][1],
                                  "prompts_per_subject": lo["n_prompts"].mean()})
                if k == 0:
                    key = (name, pre)
                    pooled[key] = evaluate.pool(per).assign(score=name, anticipation_min=pre / 60)
                    loso_points[key] = summ
                    loso_rows.append(lo.assign(score=name, anticipation_min=pre / 60))
    pd.concat(pooled.values()).round(4).to_csv(out / "threshold_sweep.csv", index=False)
    pd.concat(loso_rows).round(3).to_csv(out / "loso_per_subject.csv", index=False)
    sens = pd.DataFrame(sens_rows)
    sens.round(3).to_csv(out / "loso_sensitivity.csv", index=False)

    # --- figures ------------------------------------------------------------
    for pre in PRE_S:
        plot_sweep({n: d for (n, p), d in pooled.items() if p == pre}, {n: v for (n, p), v in loso_points.items() if p == pre},
                   out / "figures" / f"threshold_sweep_anticipation{int(pre / 60)}min.png", ref_name,
                   subtitle="strict onset" if pre == 0 else f"prompts up to {int(pre / 60)} min before onset count as hits")
    thr_ahr = loso_points[("AHR (wrist PPG)", 0.0)]["threshold_median"]
    for s in sids:
        prompts = simulate(causal[s], replace(base, threshold=thr_ahr), "ahr")
        plot_subject(s, causal[s], labels[s], prompts, out / "figures" / "subjects" / f"{s}_ahr.png")
        if has_ref:
            thr_e = loso_points[("AHR (reference HR)", 0.0)]["threshold_median"]
            pe = simulate(causal[s].assign(quality_ok=causal[s]["quality_ecg"]), replace(base, threshold=thr_e), "ahr_ecg")
            plot_subject(s, causal[s], labels[s], pe, out / "figures" / "subjects" / f"{s}_ahr_ecg.png", src="ecg", ref_name=ref_name)
    # default example: the subject with the median share of usable causal wrist-PPG windows (not cherry-picked)
    q_share = pd.Series({s: causal[s]["quality_ok"].mean() for s in sids}).sort_values()
    example = args.example or q_share.index[len(q_share) // 2]
    for suffix in ("ahr", "ahr_ecg") if has_ref else ("ahr",):
        src_png = out / "figures" / "subjects" / f"{example}_{suffix}.png"
        (out / "figures" / f"example_{suffix}.png").write_bytes(src_png.read_bytes())

    # --- summary json -------------------------------------------------------
    summary = {
        "n_subjects": len(sids),
        "example_subject": example,
        "reference": ref_name if has_ref else None,
        "feature_config": asdict(cfg),
        "trigger_base": asdict(base),
        "fp_budget_per_h": FP_BUDGET,
        "causal_vs_zero_phase_peak_lag_ms_median": peak_lag_ms,
        "loso_primary": {f"{n} | anticipation {int(p / 60)} min": v for (n, p), v in loso_points.items()},
        "calibration_activity_slope_sum_median": {
            src: float(coef[(coef["mode"] == "causal") & (coef["hr_source"] == src)].filter(like="activity").sum(axis=1).median())
            for src in coef["hr_source"].unique()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 160)
    print("\nSignal quality\n", quality.round(2).to_string(index=False))
    print("\nMixed model effects (vs baseline)\n", mixed.round(3).to_string(index=False))
    print("\nLOSO sensitivity\n", sens.drop(columns=["threshold_min", "threshold_max"]).round(2).to_string(index=False))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()

"""Deep baseline: small 1D-CNN on raw wrist signals vs hand-crafted features (WESAD, LOSO).

Usage
-----
    pip install -e ".[dl]"                 # torch (CPU) + scikit-learn
    python scripts/download_wesad.py       # once
    python scripts/run_cnn.py              # -> results/cnn/   (~15-30 min on 4 CPU cores)
    python scripts/run_cnn.py --quick      # 3 subjects, 1 seed, 3 epochs: smoke test

Tasks: ``binary`` = baseline vs stress (main), ``three_class`` = baseline vs
stress vs amusement (secondary). Windows: 60 s, 30 s step (50 % overlap),
>= 95 % inside one condition, not overlapping the calibration interval (first
5 min of baseline). All models use the same windows and the same
leave-one-subject-out folds; nothing is tuned on the test subject.

Models
------
* ``logreg`` / ``rf``: logistic regression and random forest on per-window
  features from the existing pipeline (HR, RMSSD, SDNN, AHR from SQI-ok 30 s
  sub-windows; activity; share of usable sub-windows) plus EDA statistics.
  Per subject, the calibration-interval mean is subtracted from each feature
  (``calibrated``). Two sensitivity variants: ``calibrated_no_eda`` (only
  BVP- and ACC-derived features, i.e. the same sensors the CNN sees) and
  ``raw`` (no per-subject calibration). Missing values (no usable PPG window) are imputed with
  the training median plus missing-indicator columns. Fixed hyper-parameters.
* ``cnn``: :class:`ahr.cnn.SmallCNN` on BVP + |ACC| (resampled 32 -> 64 Hz),
  each z-scored with the subject's calibration statistics and clipped at
  +-10 SD; 3 seeds; early stopping on 3 inner-validation subjects drawn from
  the training subjects.

Derived arrays are cached in ``data/derived/`` (git-ignored); only aggregated
tables and figures go to ``results/cnn/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal, stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ahr import io_wesad, motion, pipeline, windows  # noqa: E402
from ahr.ahr import add_ahr_columns  # noqa: E402

C_MODEL = {"logreg": "#2a78d6", "rf": "#eb6834", "cnn": "#1baf7a"}
MODEL_NAMES = {"logreg": "Logistic regression (features)", "rf": "Random forest (features)", "cnn": "1D-CNN (raw BVP + |ACC|)"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

TASKS = {"binary": [1, 2], "three_class": [1, 2, 3]}
FS = 64.0
CLIP_SD = 10.0
SEEDS = (0, 1, 2)
N_INNER_VAL = 3
FEATURES = ["hr", "rmssd", "sdnn", "ahr", "usable_frac", "activity", "activity_sd",
            "eda_mean", "eda_sd", "eda_min", "eda_max", "eda_slope"]
NOT_CALIBRATED = {"usable_frac"}
FEATURE_VARIANTS = {
    "calibrated": [f"{c}_cal" for c in FEATURES],
    "calibrated_no_eda": [f"{c}_cal" for c in FEATURES if not c.startswith("eda")],
    "raw": FEATURES,
}


# ----------------------------------------------------------------------------
# Per-subject data
# ----------------------------------------------------------------------------


def subject_data(d: io_wesad.WesadSubject, wcfg: windows.WindowConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Window table with features (all candidate windows) and the CNN input array for usable windows."""
    labels = d.labels.to_numpy()
    calib = windows.calibration_interval(labels, io_wesad.LABEL_FS, wcfg.calib_s)
    bvp = d.bvp["bvp"].to_numpy()
    acc = d.acc[["x", "y", "z"]].to_numpy()
    eda = d.eda["eda"].to_numpy()
    duration = min(len(bvp) / FS, len(acc) / 32.0, len(eda) / 4.0)
    win = windows.make_windows(labels, io_wesad.LABEL_FS, duration, calib, wcfg)

    # --- hand-crafted features from the existing pipeline (offline, 30 s / 5 s sub-windows)
    fcfg = pipeline.FeatureConfig(invert_ppg=True)
    f30 = pipeline.extract_features(bvp, FS, acc, 32.0, causal=False, cfg=fcfg)
    cal30 = ((f30["t"] - fcfg.win_s >= calib[0]) & (f30["t"] <= calib[1]) & f30["quality_ok"]).to_numpy()
    try:
        f30, _ = add_ahr_columns(f30, cal30, (30.0, 60.0))
    except ValueError:  # too few usable calibration windows
        f30["ahr"] = np.nan
    ok = f30["quality_ok"].to_numpy()
    act = motion.activity_magnitude(motion.remove_gravity(acc, 32.0, "highpass", fcfg.acc_cutoff_hz))
    t30 = f30["t"].to_numpy()
    rows = []
    for t0, t1 in zip(win["t0"], win["t1"]):
        sub = (t30 >= t0 + fcfg.win_s - 1e-9) & (t30 <= t1 + 1e-9)
        u = sub & ok
        a = act[int(t0 * 32) : int(t1 * 32)]
        e = eda[int(t0 * 4) : int(t1 * 4)]
        with warnings.catch_warnings():  # all-NaN AHR when the calibration had too few usable windows
            warnings.simplefilter("ignore", RuntimeWarning)
            r = {k: float(np.nanmean(f30.loc[u, c])) if u.any() else np.nan
                 for k, c in (("hr", "mean_hr"), ("rmssd", "rmssd"), ("sdnn", "sdnn"), ("ahr", "ahr"))}
        r.update(usable_frac=u.sum() / max(sub.sum(), 1), activity=a.mean(), activity_sd=a.std(),
                 eda_mean=e.mean(), eda_sd=e.std(), eda_min=e.min(), eda_max=e.max(),
                 eda_slope=np.polyfit(np.arange(len(e)) / 4.0, e, 1)[0] * 60.0)  # uS per minute
        rows.append(r)
    feats = pd.DataFrame(rows)
    in_cal = ((win["t0"] >= calib[0] - 1e-9) & (win["t1"] <= calib[1] + 1e-9)).to_numpy()
    cal_mean = feats[in_cal].mean()
    for c in FEATURES:
        feats[f"{c}_cal"] = feats[c] if c in NOT_CALIBRATED else feats[c] - cal_mean[c]
    win = pd.concat([win, feats], axis=1)
    win["subject"] = d.subject
    win["calib_start"], win["calib_end"], win["n_calib_windows"] = calib[0], calib[1], int(in_cal.sum())

    # --- CNN input: raw BVP + |ACC| at 64 Hz, z-scored with calibration statistics only
    acc_mag = signal.resample_poly(np.linalg.norm(acc, axis=1), 2, 1)
    n = min(len(bvp), len(acc_mag))
    x = np.column_stack([windows.zscore_with_calibration(bvp[:n], FS, calib),
                         windows.zscore_with_calibration(acc_mag[:n], FS, calib)])
    x = np.clip(x, -CLIP_SD, CLIP_SD)
    use = win["usable"].to_numpy()
    X = windows.cut_windows(x, FS, win.loc[use, "t0"].to_numpy(), wcfg.win_s).transpose(0, 2, 1).astype(np.float32)
    return win, X


def build_dataset(data_dir: Path, cache: Path, subjects: list[str], wcfg: windows.WindowConfig, rebuild: bool):
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        win = pd.read_csv(cache.with_suffix(".csv"))
        if sorted(win["subject"].unique()) == sorted(subjects):
            return win, z["X"]
    wins, Xs = [], []
    for s in subjects:
        t = time.perf_counter()
        w, X = subject_data(io_wesad.load_subject(data_dir, s), wcfg)
        wins.append(w)
        Xs.append(X)
        u = w[w["usable"]]
        print(f"{s}: {len(u)} usable windows "
              f"({', '.join(f'{io_wesad.LABEL_NAMES.get(k, k)} {v}' for k, v in u['label'].value_counts().sort_index().items())}), "
              f"{time.perf_counter() - t:.0f} s", flush=True)
    win = pd.concat(wins, ignore_index=True)
    X = np.concatenate(Xs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X)
    win.to_csv(cache.with_suffix(".csv"), index=False)
    return win, X


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


def feature_model(name: str, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if name == "logreg":
        return make_pipeline(imp, StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return make_pipeline(imp, RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                     n_jobs=-1, random_state=seed))


def run_features(win: pd.DataFrame, task: str, variant: str) -> tuple[list, list, float]:
    classes = TASKS[task]
    d = win[win["label"].isin(classes)].reset_index(drop=True)
    cols = FEATURE_VARIANTS[variant]
    X, y, subj = d[cols].to_numpy(float), d["label"].to_numpy(), d["subject"].to_numpy()
    rows, preds, t_total = [], [], 0.0
    for model in ("logreg", "rf"):
        for s, tr, te in windows.loso_splits(subj):
            t = time.perf_counter()
            m = feature_model(model).fit(X[tr], y[tr])
            p = m.predict(X[te])
            t_total += time.perf_counter() - t
            rows.append({"task": task, "model": model, "variant": variant, "seed": 0, "subject": s, "n_test": len(te),
                         "macro_f1": windows.macro_f1(y[te], p, classes),
                         "balanced_accuracy": windows.balanced_accuracy(y[te], p, classes)})
            preds += [{"task": task, "model": model, "variant": variant, "seed": 0, "y": a, "pred": b} for a, b in zip(y[te], p)]
    return rows, preds, t_total


def run_cnn(win_use: pd.DataFrame, X_all: np.ndarray, task: str, seeds, tcfg) -> tuple[list, list, list]:
    import torch

    from ahr import cnn

    torch.set_num_threads(max(1, torch.get_num_threads()))
    classes = TASKS[task]
    sel = win_use["label"].isin(classes).to_numpy()
    d = win_use[sel].reset_index(drop=True)
    X = X_all[sel]
    y = d["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    subj = d["subject"].to_numpy()
    rows, preds, info_rows = [], [], []
    for seed in seeds:
        for k, (s, tr, te) in enumerate(windows.loso_splits(subj)):
            rng = np.random.default_rng(1000 * seed + k)
            itr, ival = windows.inner_split(subj[tr], N_INNER_VAL, rng)
            itr, ival = tr[itr], tr[ival]
            assert s not in set(subj[itr]) | set(subj[ival])
            model, info = cnn.train(X[itr], y[itr], X[ival], y[ival], len(classes), seed=seed, cfg=tcfg)
            p = np.asarray(classes)[cnn.predict(model, X[te])]
            yt = np.asarray(classes)[y[te]]
            rows.append({"task": task, "model": "cnn", "variant": "calibrated", "seed": seed, "subject": s, "n_test": len(te),
                         "macro_f1": windows.macro_f1(yt, p, classes),
                         "balanced_accuracy": windows.balanced_accuracy(yt, p, classes)})
            preds += [{"task": task, "model": "cnn", "variant": "calibrated", "seed": seed, "y": a, "pred": b} for a, b in zip(yt, p)]
            info_rows.append({"task": task, "seed": seed, "subject": s, "n_params": cnn.n_parameters(model), **info})
            print(f"  cnn {task} seed {seed} {s}: F1 {rows[-1]['macro_f1']:.2f}, best epoch {info['best_epoch']}, "
                  f"{info['train_seconds']:.0f} s", flush=True)
    return rows, preds, info_rows


# ----------------------------------------------------------------------------
# Summaries and figure
# ----------------------------------------------------------------------------


def per_subject_means(res: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds -> one value per (task, model, variant, subject)."""
    return res.groupby(["task", "model", "variant", "subject"], as_index=False)[["macro_f1", "balanced_accuracy"]].mean()


def summary_table(res: pd.DataFrame) -> pd.DataFrame:
    subj = per_subject_means(res)
    out = subj.groupby(["task", "model", "variant"]).agg(
        n_subjects=("subject", "nunique"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd_subjects=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_sd_subjects=("balanced_accuracy", "std"),
    ).reset_index()
    # spread over seeds: SD of the subject-averaged score across seeds
    seed_means = res.groupby(["task", "model", "variant", "seed"])[["macro_f1", "balanced_accuracy"]].mean()
    sd_seed = seed_means.groupby(["task", "model", "variant"]).std().rename(columns=lambda c: f"{c}_sd_seeds")
    return out.merge(sd_seed.reset_index(), on=["task", "model", "variant"], how="left")


def paired_tests(res: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank tests across subjects on seed-averaged per-subject scores."""
    subj = per_subject_means(res)
    subj["key"] = subj["model"] + "|" + subj["variant"]
    pairs = [("cnn|calibrated", "rf|calibrated"), ("cnn|calibrated", "logreg|calibrated"),
             ("cnn|calibrated", "rf|calibrated_no_eda"), ("cnn|calibrated", "logreg|calibrated_no_eda"),
             ("rf|calibrated", "logreg|calibrated"), ("rf|calibrated", "rf|calibrated_no_eda"), ("rf|calibrated", "rf|raw")]
    rows = []
    for task in subj["task"].unique():
        wide = subj[subj["task"] == task].pivot(index="subject", columns="key")
        for a, b in pairs:
            for metric in ("macro_f1", "balanced_accuracy"):
                if (metric, a) not in wide or (metric, b) not in wide:
                    continue
                diff = (wide[(metric, a)] - wide[(metric, b)]).dropna()
                if len(diff) >= 2 and np.any(diff != 0):
                    w = stats.wilcoxon(diff)
                    stat, p = float(w.statistic), float(w.pvalue)
                else:
                    stat, p = np.nan, np.nan
                rows.append({"task": task, "a": a, "b": b, "metric": metric, "n_subjects": len(diff),
                             "mean_diff": diff.mean(), "median_diff": diff.median(), "a_better": int((diff > 0).sum()),
                             "b_better": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
                             "wilcoxon_W": stat, "p_two_sided": p})
    return pd.DataFrame(rows)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_per_subject(res: pd.DataFrame, path: Path) -> None:
    subj = per_subject_means(res)
    subj["key"] = subj["model"] + "|" + subj["variant"]
    cols = [("logreg|calibrated", "LogReg\n(features)", C_MODEL["logreg"]), ("rf|calibrated", "RF\n(features)", C_MODEL["rf"]),
            ("rf|calibrated_no_eda", "RF\n(features, no EDA)", MUTED), ("cnn|calibrated", "1D-CNN\n(raw BVP + |ACC|)", C_MODEL["cnn"])]
    titles = {"binary": "baseline vs stress", "three_class": "baseline vs stress vs amusement"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        wide = subj[subj["task"] == task].pivot(index="subject", columns="key", values="macro_f1").reindex(columns=[c[0] for c in cols])
        for _, r in wide.iterrows():
            ax.plot(range(len(cols)), r.to_numpy(), color=GRID, lw=1.0, zorder=1)
        for k, (key, _, color) in enumerate(cols):
            ax.scatter(np.full(len(wide), k), wide[key], s=22, color=color, edgecolors="white", lw=0.6, zorder=2)
            mu, sd = wide[key].mean(), wide[key].std()
            ax.errorbar(k + 0.2, mu, yerr=sd, fmt="D", color=TEXT, ms=6, mec="white", capsize=3, lw=1.2, zorder=3)
            ax.annotate(f"{mu:.2f}", (k + 0.2, mu), textcoords="offset points", xytext=(6, -3), fontsize=8, color=TEXT)
        ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=8)
        ax.set_title(titles[task], loc="left", fontsize=10, color=TEXT)
        ax.set_ylim(-0.03, 1.03)
        _style(ax)
    axes[0].set_ylabel("macro-F1 of the held-out subject", color=MUTED, fontsize=9)
    fig.suptitle("WESAD, leave-one-subject-out, 15 subjects. Dots: subjects (grey lines join the same subject); "
                 "diamond: mean ± SD over subjects; CNN averaged over 3 seeds", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/WESAD"))
    ap.add_argument("--out", type=Path, default=Path("results/cnn"))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--rebuild", action="store_true", help="recompute cached windows / features")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 subjects, 1 seed, 3 epochs")
    args = ap.parse_args()

    from ahr import cnn

    subjects = args.subjects or io_wesad.available_subjects(args.data_dir)
    seeds, tcfg = SEEDS, cnn.TrainConfig()
    if args.quick:
        subjects, seeds, tcfg = subjects[:3], (0,), cnn.TrainConfig(max_epochs=3, patience=2)
    out = args.out if not args.quick else Path("/tmp/ahr-cnn-quick")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    wcfg = windows.WindowConfig()
    cache = Path("data/derived") / ("wesad_cnn_windows_quick.npz" if args.quick else "wesad_cnn_windows.npz")
    t = time.perf_counter()
    win, X = build_dataset(args.data_dir, cache, subjects, wcfg, args.rebuild)
    t_data = time.perf_counter() - t
    use = win[win["usable"]].reset_index(drop=True)
    assert len(use) == len(X)

    res_rows, pred_rows, info_rows, timing = [], [], [], []
    for task in TASKS:
        for variant in FEATURE_VARIANTS:
            r, p, tt = run_features(use, task, variant)
            res_rows += r
            pred_rows += p
            timing.append({"task": task, "model": "logreg + rf", "variant": variant, "seconds_total": tt, "fits": 2 * len(subjects)})
        t = time.perf_counter()
        r, p, info = run_cnn(use, X, task, seeds, tcfg)
        res_rows += r
        pred_rows += p
        info_rows += info
        timing.append({"task": task, "model": "cnn", "variant": "calibrated", "seconds_total": time.perf_counter() - t,
                       "fits": len(seeds) * len(subjects)})

    res = pd.DataFrame(res_rows)
    summ = summary_table(res)
    tests = paired_tests(res)
    info = pd.DataFrame(info_rows)
    timing = pd.DataFrame(timing)
    timing["seconds_per_fit"] = timing["seconds_total"] / timing["fits"]
    preds = pd.DataFrame(pred_rows)
    conf = (preds.groupby(["task", "model", "variant", "y", "pred"]).size().rename("n").reset_index())

    counts = (use.groupby(["subject", "label"]).size().unstack(fill_value=0)
              .rename(columns=io_wesad.LABEL_NAMES).reset_index())
    res.round(4).to_csv(out / "per_subject.csv", index=False)
    summ.round(4).to_csv(out / "summary.csv", index=False)
    tests.round(4).to_csv(out / "wilcoxon.csv", index=False)
    conf.to_csv(out / "confusion_pooled.csv", index=False)
    timing.round(2).to_csv(out / "training_time.csv", index=False)
    info.round(4).to_csv(out / "cnn_training_log.csv", index=False)
    counts.to_csv(out / "windows_per_subject.csv", index=False)
    plot_per_subject(res, out / "figures" / "per_subject_macro_f1.png")

    import platform

    import sklearn
    import torch

    summary = {
        "n_subjects": len(subjects),
        "window": {"win_s": wcfg.win_s, "step_s": wcfg.step_s, "min_purity": wcfg.min_purity, "calibration_s": wcfg.calib_s},
        "n_windows_usable": int(len(use)),
        "seeds": list(seeds),
        "cnn_train_config": vars(tcfg),
        "cnn_n_parameters": {t: int(info.loc[info["task"] == t, "n_params"].iloc[0]) for t in info["task"].unique()},
        "clip_sd": CLIP_SD,
        "inner_validation_subjects": N_INNER_VAL,
        "features": FEATURES,
        "data_preparation_seconds": t_data,
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\nSummary\n", summ.round(3).to_string(index=False))
    print("\nWilcoxon\n", tests.round(4).to_string(index=False))
    print("\nTiming\n", timing.round(1).to_string(index=False))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()

"""Evaluation: condition summaries, trigger metrics, threshold sweeps,
leave-one-subject-out (LOSO) threshold tuning and a multilevel model.

Trigger metrics (per subject, aggregated by pooling counts across subjects)
-------------------------------------------------------------------------
* **hit** - a stress episode (contiguous label 2) counts as detected when at
  least one prompt falls in ``[onset - pre_s, offset + grace_s]``; the grace
  period allows for the trailing analysis window and the minimum-duration
  rule. ``pre_s`` defaults to 0; a positive value credits prompts during the
  anticipation phase right before the labelled TSST (in WESAD participants
  are briefed and walk to the TSST room just before the stress label starts).
* **latency** - time from episode onset to the first prompt counted as a hit.
* **false prompts per hour** - prompts that fall in baseline, amusement or
  meditation (labels 1, 3, 4) divided by the hours spent in those
  conditions. Prompts during unlabelled transitions (label 0, 5-7) outside a
  hit window are reported separately as ``n_other``, because the
  participant's state there is unknown (in WESAD this time includes walking
  between rooms and filling in questionnaires); they are pooled as
  ``other_prompts_per_h`` over the hours spent in those labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .trigger import TriggerConfig, simulate

NONSTRESS_LABELS = (1, 3, 4)
OTHER_LABELS = (0, 5, 6, 7)
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


@dataclass
class SubjectLabels:
    """Raw label stream of one subject."""

    labels: np.ndarray
    fs: float

    def at(self, times_s: np.ndarray) -> np.ndarray:
        idx = np.clip(np.floor(np.asarray(times_s) * self.fs).astype(int), 0, len(self.labels) - 1)
        return self.labels[idx]

    def episodes(self, target: int = 2, min_duration_s: float = 60.0) -> list[tuple[float, float]]:
        """(onset, offset) times in seconds of contiguous runs of ``target``."""
        m = np.r_[False, self.labels == target, False].astype(np.int8)
        d = np.diff(m)
        on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return [(a / self.fs, b / self.fs) for a, b in zip(on, off) if (b - a) / self.fs >= min_duration_s]

    def hours(self, labels: tuple[int, ...] = NONSTRESS_LABELS) -> float:
        return float(np.isin(self.labels, labels).sum() / self.fs / 3600.0)


def trigger_metrics(prompts: np.ndarray, lab: SubjectLabels, grace_s: float = 300.0, pre_s: float = 0.0) -> dict:
    """Classify prompts of one subject against its labels (see module docstring).

    ``pre_s > 0`` also counts prompts up to ``pre_s`` seconds *before* the
    labelled onset as hits (anticipatory stress); latency is then negative
    for such prompts.
    """
    prompts = np.asarray(prompts, float)
    eps = lab.episodes()
    in_hit_window = np.zeros(len(prompts), bool)
    hits, latencies = 0, []
    for on, off in eps:
        m = (prompts >= on - pre_s) & (prompts <= off + grace_s)
        in_hit_window |= m
        if m.any():
            hits += 1
            latencies.append(prompts[m][0] - on)
    rest = prompts[~in_hit_window]
    rest_lab = lab.at(rest) if len(rest) else np.array([], int)
    n_false = int(np.isin(rest_lab, NONSTRESS_LABELS).sum())
    hours = lab.hours()
    other_h = lab.hours(OTHER_LABELS)
    return {
        "n_episodes": len(eps),
        "n_hits": hits,
        "latency_s": float(np.mean(latencies)) if latencies else np.nan,
        "n_false": n_false,
        "n_other": int(len(rest) - n_false),
        "other_h": other_h,
        "nonstress_h": hours,
        "fp_per_h": n_false / hours if hours > 0 else np.nan,
        "n_prompts": int(len(prompts)),
    }


def sweep(
    features: dict[str, pd.DataFrame],
    labels: dict[str, SubjectLabels],
    thresholds: np.ndarray,
    base: TriggerConfig,
    score_col: str = "ahr",
    grace_s: float = 300.0,
    pre_s: tuple[float, ...] = (0.0,),
) -> pd.DataFrame:
    """Per-subject trigger metrics for every threshold (and every ``pre_s``), long format."""
    rows = []
    for sid, df in features.items():
        for thr in thresholds:
            prompts = simulate(df, replace(base, threshold=float(thr)), score_col)
            for pre in pre_s:
                m = trigger_metrics(prompts, labels[sid], grace_s, pre)
                rows.append({"subject": sid, "threshold": float(thr), "pre_s": pre, **m})
    return pd.DataFrame(rows)


def pool(per_subject: pd.DataFrame, by: str | list[str] = "threshold") -> pd.DataFrame:
    """Pool per-subject counts: hit rate over episodes, false prompts over total hours."""
    g = per_subject.groupby(by)
    out = pd.DataFrame(
        {
            "hit_rate": g["n_hits"].sum() / g["n_episodes"].sum(),
            "fp_per_h": g["n_false"].sum() / g["nonstress_h"].sum(),
            "other_prompts_per_h": g["n_other"].sum() / g["other_h"].sum(),
            "latency_s_median": g["latency_s"].median(),
            "prompts_per_subject": g["n_prompts"].mean(),
            "n_subjects": g["subject"].nunique(),
        }
    )
    return out.reset_index()


def select_threshold(pooled: pd.DataFrame, fp_budget: float) -> float:
    """Lowest threshold whose pooled false-prompt rate stays within ``fp_budget``.

    Among thresholds meeting the budget the one with the highest hit rate is
    chosen (ties -> higher threshold, fewer prompts). If none meets the
    budget, the threshold with the fewest false prompts is returned.
    """
    ok = pooled[pooled["fp_per_h"] <= fp_budget]
    if ok.empty:
        return float(pooled.sort_values(["fp_per_h", "threshold"], ascending=[True, False]).iloc[0]["threshold"])
    best = ok[ok["hit_rate"] == ok["hit_rate"].max()]
    return float(best["threshold"].max())


def loso(per_subject: pd.DataFrame, fp_budget: float = 0.5) -> pd.DataFrame:
    """Leave-one-subject-out threshold tuning.

    For each held-out subject the threshold is selected on the remaining
    subjects (:func:`select_threshold`) and the held-out subject's metrics at
    that threshold are returned (one row per subject).
    """
    rows = []
    for sid in per_subject["subject"].unique():
        train = pool(per_subject[per_subject["subject"] != sid])
        thr = select_threshold(train, fp_budget)
        test = per_subject[(per_subject["subject"] == sid) & np.isclose(per_subject["threshold"], thr)]
        row = test.iloc[0].drop(["subject", "threshold"], errors="ignore").to_dict()
        rows.append({"held_out": sid, "selected_threshold": thr, **row})
    return pd.DataFrame(rows)


def loso_summary(loso_df: pd.DataFrame) -> dict:
    return {
        "hit_rate": loso_df["n_hits"].sum() / loso_df["n_episodes"].sum(),
        "fp_per_h": loso_df["n_false"].sum() / loso_df["nonstress_h"].sum(),
        "other_prompts_per_h": loso_df["n_other"].sum() / loso_df["other_h"].sum(),
        "latency_s_median": float(loso_df["latency_s"].median()),
        "threshold_median": float(loso_df["selected_threshold"].median()),
        "threshold_range": (float(loso_df["selected_threshold"].min()), float(loso_df["selected_threshold"].max())),
    }


# ----------------------------------------------------------------------------
# Condition summaries and multilevel model
# ----------------------------------------------------------------------------


def condition_means(
    df: pd.DataFrame, value_cols: list[str], min_purity: float = 0.95, quality_col: str | None = "quality_ok"
) -> pd.DataFrame:
    """Per subject x condition means over windows that lie (almost) entirely in one condition.

    ``df`` needs columns ``subject, label, label_purity`` (+ ``quality_col``,
    the usability flag of the signal the values come from; ``None`` keeps all windows).
    """
    m = df["label"].isin(CONDITIONS.keys()) & (df["label_purity"] >= min_purity)
    if quality_col is not None:
        m &= df[quality_col].astype(bool)
    out = df[m].groupby(["subject", "label"])[value_cols].mean().reset_index()
    out["n_windows"] = df[m].groupby(["subject", "label"]).size().to_numpy()
    out["condition"] = out["label"].map(CONDITIONS)
    return out


def summarize_conditions(cond_means: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Across-subject mean, SD and number of subjects of the per-subject condition means."""
    g = cond_means.groupby("condition")[value_cols]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{c}_{'n_subjects' if s == 'count' else s}" for c, s in out.columns]
    order = [c for c in CONDITIONS.values() if c in out.index]
    return out.loc[order].reset_index()


def mixed_model(cond_means: pd.DataFrame, value_col: str):
    """``value ~ condition`` with a random intercept per subject (statsmodels MixedLM).

    Fitted on per-subject condition means (one row per subject x condition)
    rather than on overlapping, autocorrelated windows. Baseline is the
    reference level.
    """
    data = cond_means.dropna(subset=[value_col]).copy()
    model = smf.mixedlm(f"{value_col} ~ C(condition, Treatment('baseline'))", data, groups=data["subject"])
    return model.fit(reml=True)

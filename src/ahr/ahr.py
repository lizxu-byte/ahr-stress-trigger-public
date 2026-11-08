"""Additional heart rate (AHR): heart rate not explained by physical activity.

Concept
-------
Myrtek and colleagues proposed separating the *metabolic* part of heart rate
(driven by physical activity) from an *additional*, emotionally/mentally
driven part (Myrtek, M., 2004, *Heart and Emotion: Ambulatory Monitoring
Studies in Everyday Life*, Hogrefe & Huber). In that ambulatory work, heart
rate is related to accelerometer-measured physical activity, and episodes in
which heart rate exceeds the activity-expected level by a criterion are
flagged as additional (emotional) heart rate, which can trigger a self-report
prompt.

Model used here (deliberately simple)
------------------------------------
For each participant, fit on calibration windows only (WESAD: baseline
condition)::

    HR(t) = b0 + b1 * A(t) + sum_k c_k * A(t - lag_k) + e(t)

where ``A`` is the window-mean dynamic acceleration (g) from
:mod:`ahr.motion`, and the lag terms capture the slow HR response to and
recovery from movement. The activity slopes are constrained to be
non-negative (more movement never predicts lower HR); with little movement in
the calibration data they shrink towards 0 and the model reduces to the
baseline mean HR. Then::

    AHR(t) = HR(t) - HR_pred(t)

Only past activity enters the prediction, so AHR is computable in real time
once the calibration is done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


def activity_design(t: np.ndarray, activity: np.ndarray, lags_s: tuple[float, ...] = (30.0, 60.0)) -> np.ndarray:
    """Design matrix ``[1, A(t), A(t - lag_1), ...]`` on a (possibly irregular) time grid.

    Lagged values are taken from the most recent window ending at or before
    ``t - lag`` (NaN when none exists).
    """
    t = np.asarray(t, float)
    a = np.asarray(activity, float)
    cols = [np.ones_like(t), a]
    for lag in lags_s:
        idx = np.searchsorted(t, t - lag + 1e-9, side="right") - 1
        col = np.where(idx >= 0, a[np.clip(idx, 0, len(a) - 1)], np.nan)
        cols.append(col)
    return np.column_stack(cols)


@dataclass
class ActivityHRModel:
    """Per-participant linear activity -> HR model (see module docstring)."""

    lags_s: tuple[float, ...] = (30.0, 60.0)
    nonneg_slopes: bool = True
    coef_: np.ndarray | None = field(default=None, repr=False)
    resid_sd_: float = float("nan")
    r2_: float = float("nan")
    n_calib_: int = 0

    @property
    def names(self) -> list[str]:
        return ["intercept", "activity"] + [f"activity_lag{int(l)}s" for l in self.lags_s]

    def fit(self, t: np.ndarray, activity: np.ndarray, hr: np.ndarray, calib_mask: np.ndarray) -> "ActivityHRModel":
        """Fit on windows where ``calib_mask`` is True and all inputs are finite.

        ``t``, ``activity`` and ``hr`` cover the whole recording so that lag
        terms can reach back before the calibration period.
        """
        X = activity_design(t, activity, self.lags_s)
        y = np.asarray(hr, float)
        m = np.asarray(calib_mask, bool) & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if m.sum() < X.shape[1] + 2:
            raise ValueError(f"only {m.sum()} usable calibration windows")
        lb = np.r_[-np.inf, np.zeros(X.shape[1] - 1)] if self.nonneg_slopes else -np.inf
        res = lsq_linear(X[m], y[m], bounds=(lb, np.inf))
        self.coef_ = res.x
        resid = y[m] - X[m] @ res.x
        self.resid_sd_ = float(np.std(resid, ddof=X.shape[1]))
        ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
        self.r2_ = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
        self.n_calib_ = int(m.sum())
        return self

    def predict(self, t: np.ndarray, activity: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return activity_design(t, activity, self.lags_s) @ self.coef_

    def additional_hr(self, t: np.ndarray, activity: np.ndarray, hr: np.ndarray) -> np.ndarray:
        """AHR = observed HR - activity-predicted HR (bpm)."""
        return np.asarray(hr, float) - self.predict(t, activity)

    def summary(self) -> dict:
        out = dict(zip(self.names, np.asarray(self.coef_, float)))
        out.update(resid_sd=self.resid_sd_, r2=self.r2_, n_calib=self.n_calib_)
        return out


def add_ahr_columns(
    features: pd.DataFrame,
    calib_mask: np.ndarray,
    lags_s: tuple[float, ...] = (30.0, 60.0),
    hr_col: str = "mean_hr",
) -> tuple[pd.DataFrame, ActivityHRModel]:
    """Fit the per-participant model and add ``hr_pred``, ``ahr`` and ``hr_rel``.

    ``hr_rel`` is HR minus the calibration-period mean HR, i.e. the same
    trigger score without any activity correction (the raw-HR comparison).
    """
    df = features.copy()
    calib_mask = np.asarray(calib_mask, bool)
    model = ActivityHRModel(lags_s=lags_s).fit(df["t"].to_numpy(), df["activity"].to_numpy(), df[hr_col].to_numpy(), calib_mask)
    df["hr_pred"] = model.predict(df["t"].to_numpy(), df["activity"].to_numpy())
    df["ahr"] = df[hr_col] - df["hr_pred"]
    baseline_hr = df.loc[calib_mask, hr_col].mean()
    df["hr_rel"] = df[hr_col] - baseline_hr
    return df, model

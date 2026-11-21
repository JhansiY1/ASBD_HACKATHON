"""
Outlier detection and noise smoothing.

Outlier rules (Lecture 5 – noisy data / outliers):
    IQR rule      x < Q1 - 1.5 IQR  or  x > Q3 + 1.5 IQR
    z-score rule  |z| > 3,  z = (x - mean) / std

Outliers are *flagged, not deleted*: each attribute is reviewed and every
flagged value is classified either as a genuine extreme observation
(kept) or as an erroneous value (set to missing during cleaning, e.g.
lengths in feet).  Deleting genuine extremes (Hitchcock's 79 films, epics
of 4 hours, the 1890s Edison shorts) would remove real information.

Smoothing (binning method, Lecture 5):
    equal-depth (equal-frequency) partitioning into k bins, then
    smoothing by bin means      v' = mean(bin)
    smoothing by bin boundaries v' = closest of (min(bin), max(bin))
and a centred moving average for the noisy yearly film-count series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def outlier_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = df[c].dropna().astype(float)
        if x.empty:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (x - x.mean()) / (x.std(ddof=0) or 1)
        rows.append(dict(attribute=c, n=len(x), q1=q1, q3=q3, iqr=iqr, lower_fence=lo, upper_fence=hi,
                         n_iqr_outliers=int(((x < lo) | (x > hi)).sum()),
                         n_z_outliers=int((z.abs() > 3).sum()), min=x.min(), max=x.max()))
    return pd.DataFrame(rows)


OUTLIER_VERDICTS = {
    "year": "Genuine: early-cinema films 1891-1910 (Edison, Lumiere, Melies) are real historic records - kept.",
    "cast_size": "Genuine: large documented casts (epics, Hitchcock films) - kept; reflects documentation depth.",
    "director_n_films": "Genuine: prolific directors (Hitchcock is complete by design of the dataset) - kept.",
    "length_min": "Mostly genuine epics/serials > 3 h; values outside 1-600 min were unit errors and were "
                  "already set to missing in cleaning.",
    "n_producers": "Genuine: co-productions with many producers - kept.",
    "halliwell_stars": "No outliers possible (bounded 0-4).",
    "female_cast_share": "Bounded [0, 1]; 0 and 1 are genuine (all-male / all-female documented casts).",
}


def equal_depth_smoothing(x: pd.Series, k: int = 10) -> pd.DataFrame:
    """Equal-frequency binning with smoothing by bin means and bin boundaries."""
    s = x.dropna().sort_values()
    bins = np.array_split(s.index.to_numpy(), k)
    out = pd.DataFrame({"original": s})
    out["bin"] = 0
    out["by_mean"] = np.nan
    out["by_boundary"] = np.nan
    for b, idx in enumerate(bins):
        vals = s.loc[idx]
        out.loc[idx, "bin"] = b + 1
        out.loc[idx, "by_mean"] = vals.mean()
        lo, hi = vals.min(), vals.max()
        out.loc[idx, "by_boundary"] = np.where((vals - lo) <= (hi - vals), lo, hi)
    return out


def moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()

"""
Exhaustive descriptive analytics of the cleaned, integrated film table.

Numerical attributes  : count, mean, median, mode, min, max, range, variance,
                        standard deviation, Q1, Q3, IQR, skewness, kurtosis
Categorical attributes: cardinality, frequency and percentage distributions
Relationships         : Pearson & Spearman correlation (numeric pairs),
                        chi-square test + Cramer's V (categorical pairs),
                        group comparisons (e.g. rating by genre / era)

Run:  python -m src.analytics.descriptive
Outputs under outputs/descriptive/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DESCRIPTIVE_DIR, PROCESSED_DIR, get_logger
from src.preprocessing.vocabularies import CATEGORY_NAMES

log = get_logger("descriptive")

NUMERIC = ["year", "cast_size", "n_producers", "n_genres", "halliwell_stars", "oscar_noms",
           "oscar_wins", "director_n_films", "female_cast_share", "length_min", "times_remade",
           "cast_actor_award_noms"]
CATEGORICAL = ["era", "decade", "color_type", "primary_genre", "studio_group", "director_country_grp",
               "producer_status", "mpaa_rating", "setting", "halliwell_level", "cast_size_bin"]


def load_films() -> pd.DataFrame:
    f = pd.read_pickle(PROCESSED_DIR / "films_cleaned_integrated.pkl")
    f["genres"] = f["genres"].fillna("").apply(lambda s: [g for g in s.split("|") if g])
    return f


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        x = df[c].dropna().astype(float)
        if c == "cast_size":
            x = df.loc[df["has_cast_info"] == 1, c].astype(float)   # 0 = undocumented, not a value
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        mode = x.mode()
        rows.append(dict(attribute=c, count=len(x), missing=int(len(df) - len(x)), mean=x.mean(), median=med,
                         mode=mode.iloc[0] if len(mode) else np.nan, min=x.min(), max=x.max(),
                         range=x.max() - x.min(), variance=x.var(), std=x.std(), q1=q1, q3=q3, iqr=q3 - q1,
                         skewness=stats.skew(x), kurtosis=stats.kurtosis(x)))
    return pd.DataFrame(rows).round(4)


def categorical_summary(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    card, freq = [], {}
    for c in cols:
        s = df[c].astype("object").where(df[c].notna(), "MISSING")
        vc = s.value_counts()
        card.append(dict(attribute=c, cardinality=int(df[c].nunique()), missing=int(df[c].isna().sum()),
                         mode=vc.index[0], mode_freq=int(vc.iloc[0]), mode_pct=round(100 * vc.iloc[0] / len(s), 2)))
        freq[c] = pd.DataFrame({"value": vc.index.astype(str), "frequency": vc.values,
                                "percentage": (100 * vc.values / len(s)).round(2)})
    return pd.DataFrame(card), freq


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Cramer's V = sqrt(chi2 / (n * (min(r, c) - 1)))  and the chi-square p-value."""
    tab = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return chi2, p, v


def run() -> dict:
    log.info("PHASE 5 - descriptive analytics started")
    df = load_films()
    out = DESCRIPTIVE_DIR
    num = numeric_summary(df, NUMERIC)
    num.to_csv(out / "numeric_summary.csv", index=False)
    card, freq = categorical_summary(df, CATEGORICAL)
    card.to_csv(out / "categorical_cardinality.csv", index=False)
    with pd.ExcelWriter(out / "frequency_distributions.xlsx") as xw:
        for k, v in freq.items():
            v.to_excel(xw, sheet_name=k[:30], index=False)
    for k, v in freq.items():
        v.to_csv(out / f"freq_{k}.csv", index=False)

    # multi-valued genre attribute: frequency counts every genre label
    gen = df["genres"].explode().dropna().map(lambda g: CATEGORY_NAMES.get(g, g)).value_counts()
    genre_freq = pd.DataFrame({"genre": gen.index, "films": gen.values,
                               "pct_of_films": (100 * gen.values / len(df)).round(2)})
    genre_freq.to_csv(out / "freq_genres_multilabel.csv", index=False)

    # correlations among numeric attributes (pairwise complete observations)
    corr_cols = ["year", "cast_size", "n_producers", "n_genres", "halliwell_stars", "oscar_noms",
                 "director_n_films", "female_cast_share", "times_remade"]
    cdf = df[corr_cols].copy()
    cdf.loc[df["has_cast_info"] == 0, ["cast_size", "female_cast_share"]] = np.nan
    pear = cdf.corr(method="pearson").round(3)
    spear = cdf.corr(method="spearman").round(3)
    pear.to_csv(out / "correlation_pearson.csv")
    spear.to_csv(out / "correlation_spearman.csv")

    # categorical associations
    pairs = [("era", "color_type"), ("era", "primary_genre"), ("studio_group", "primary_genre"),
             ("director_country_grp", "primary_genre"), ("era", "halliwell_level"),
             ("primary_genre", "halliwell_level"), ("studio_group", "era")]
    assoc = []
    for a, b in pairs:
        src = df
        if "color_type" in (a, b):
            # colour was rule-imputed from year for some films; using those
            # records would make the era-colour association circular
            src = df[df["color_imputed"] == 0]
        sub = src[[a, b]].dropna()
        sub = sub[(sub[a] != "Unknown") & (sub[b] != "Unknown") & (sub[b] != "unknown")]
        chi2, p, v = cramers_v(sub[a], sub[b])
        assoc.append(dict(attr_a=a, attr_b=b, n=len(sub), chi2=round(chi2, 2), p_value=p, cramers_v=round(v, 3)))
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(out / "categorical_associations.csv", index=False)

    # group comparisons
    rated = df[df["halliwell_stars"].notna()]
    by_genre = (rated.explode("genres").dropna(subset=["genres"]).groupby("genres")["halliwell_stars"]
                .agg(["count", "mean", "median"]).query("count >= 30").sort_values("mean", ascending=False).round(3))
    by_genre.index = by_genre.index.map(lambda g: CATEGORY_NAMES.get(g, g))
    by_genre.to_csv(out / "halliwell_by_genre.csv")
    by_era = rated.groupby("era")["halliwell_stars"].agg(["count", "mean", "median"]).round(3)
    by_era.to_csv(out / "halliwell_by_era.csv")
    color_by_decade = pd.crosstab(df["decade"], df["color_type"], normalize="index").round(3)
    color_by_decade.to_csv(out / "color_share_by_decade.csv")
    kw = stats.kruskal(*[g["halliwell_stars"].values for _, g in rated.groupby("era")])
    oscar_by_studio = df.groupby("studio_group")["any_oscar_nom"].agg(["count", "mean"]).round(3).sort_values("mean", ascending=False)
    oscar_by_studio.to_csv(out / "oscar_rate_by_studio.csv")
    fem = df[df["has_cast_info"] == 1].groupby("decade")["female_cast_share"].mean().round(3)
    fem.to_csv(out / "female_cast_share_by_decade.csv")

    peak = df["year"].value_counts().idxmax()
    findings = {
        "n_films": len(df),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "peak_year": int(peak), "peak_year_films": int(df["year"].value_counts().max()),
        "median_year": float(df["year"].median()),
        "share_uncategorised_pct": round(100 * (df["n_genres"] == 0).mean(), 2),
        "top_genre": gen.index[0], "top_genre_pct": round(100 * gen.iloc[0] / len(df), 2),
        "multi_genre_films_pct": round(100 * (df["n_genres"] > 1).mean(), 2),
        "rated_films": int(df["halliwell_stars"].notna().sum()),
        "mean_halliwell": round(rated["halliwell_stars"].mean(), 3),
        "kruskal_era_halliwell_H": round(kw.statistic, 2), "kruskal_era_halliwell_p": float(kw.pvalue),
        "oscar_nominated_films": int(df["any_oscar_nom"].sum()),
        "films_with_cast": int(df["has_cast_info"].sum()),
        "median_cast_size_documented": float(df.loc[df["has_cast_info"] == 1, "cast_size"].median()),
        "mean_female_cast_share": round(float(df["female_cast_share"].mean()), 3),
        "remakes": int(df["is_remake"].sum()), "later_remade": int(df["was_remade"].sum()),
        "most_prolific_director": df.loc[df["director_known"] == 1, "director"].value_counts().index[0],
        "most_prolific_director_films": int(df.loc[df["director_known"] == 1, "director"].value_counts().iloc[0]),
        "cramers_v_era_color": float(assoc.query("attr_a=='era' and attr_b=='color_type'")["cramers_v"].iloc[0]),
        "spearman_year_halliwell": float(spear.loc["year", "halliwell_stars"]),
        "spearman_year_cast": float(spear.loc["year", "cast_size"]),
    }
    with open(out / "key_findings.json", "w") as fh:
        json.dump(findings, fh, indent=2, default=str)
    log.info("PHASE 5 complete: %s", findings)
    return findings


if __name__ == "__main__":
    run()

"""
KDD phase 4 – DATA TRANSFORMATION.

Implements, for the integrated film table:

* discretisation   year -> decade, era; cast_size -> bins; Halliwell stars -> level
* label encoding   ordinal attributes (era, Halliwell level)
* one-hot encoding nominal attributes (colour type, studio group, ...)
* multi-hot        multi-valued genre attribute (binary item presence)
* scaling          z-score standardisation  z = (x - mu) / sigma
                   min-max normalisation    x' = (x - min) / (max - min)
                   (applied to numerical attributes only, never to codes)
* derived features director productivity, female cast share, decade
* transactions     one basket of items per film for Apriori / COFI / FP-Growth

All transformations are deterministic and fitted on the full cleaned table
(no fitting on test data occurs here; the classifier's scaler is not needed
because decision trees are scale-invariant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ERA_BINS = [(0, 1927, "Silent (<=1927)"), (1928, 1948, "Classical studio (1928-48)"),
            (1949, 1966, "Post-studio (1949-66)"), (1967, 1980, "New Hollywood (1967-80)"),
            (1981, 2100, "Modern (1981-99)")]
ERA_ORDER = [e for _, _, e in ERA_BINS]
HALLIWELL_LEVELS = ["cited (0*)", "1 star", "2 stars", "3 stars", "4 stars"]


def era_of(year: int) -> str:
    for lo, hi, name in ERA_BINS:
        if lo <= year <= hi:
            return name
    return "Unknown"


def cast_bin(n: int, has_info: int) -> str | float:
    """Equal-width style bins chosen from the observed distribution
    (median cast size among documented films is ~4)."""
    if not has_info:
        return np.nan
    if n <= 2:
        return "cast 1-2"
    if n <= 5:
        return "cast 3-5"
    if n <= 9:
        return "cast 6-9"
    return "cast 10+"


def halliwell_level(stars: float) -> str | float:
    if pd.isna(stars):
        return np.nan
    return HALLIWELL_LEVELS[min(int(stars), 4)]


def add_derived_features(df: pd.DataFrame, top_countries: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["era"] = df["year"].apply(era_of)
    df["era_code"] = df["era"].map({e: i for i, e in enumerate(ERA_ORDER)})        # label encoding (ordinal)
    df["cast_size_bin"] = [cast_bin(n, h) for n, h in zip(df["cast_size"], df["has_cast_info"])]
    df["halliwell_level"] = df["halliwell_stars"].apply(halliwell_level)
    df["halliwell_code"] = df["halliwell_stars"].apply(lambda s: np.nan if pd.isna(s) else min(int(s), 4))
    top = df["director_country"].value_counts().head(top_countries).index
    df["director_country_grp"] = np.where(df["director_country"].isna(), "Unknown",
                                          np.where(df["director_country"].isin(top), df["director_country"], "Other"))
    df["studio_group"] = df["studio_group"].fillna("Unknown")
    df["female_share_missing"] = df["female_cast_share"].isna().astype(int)
    df["female_cast_share_filled"] = df["female_cast_share"].fillna(df["female_cast_share"].median())
    df["director_n_films_filled"] = df["director_n_films"].fillna(1)
    # Classification target (only defined for films reviewed by Halliwell):
    # acclaimed = 2 or more stars.  Films without a Halliwell entry get NaN
    # and are excluded from the classification task (no fabricated labels).
    df["acclaimed"] = np.where(df["halliwell_stars"].isna(), np.nan, (df["halliwell_stars"] >= 2).astype(float))
    return df


def genre_multi_hot(df: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Binary item-presence representation of the multi-valued genre field."""
    counts = df["genres"].explode().value_counts()
    keep = counts[counts >= min_count].index.tolist()
    mh = pd.DataFrame({f"genre_{g}": df["genres"].apply(lambda gs, g=g: int(g in gs)) for g in keep},
                      index=df.index)
    return mh


def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[cols].astype(str), prefix=cols, dtype=int)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    """z = (x - mean) / std  (population std, as in StandardScaler)."""
    sd = x.std(ddof=0).replace(0, 1)
    return (x - x.mean()) / sd


def minmax(x: pd.DataFrame) -> pd.DataFrame:
    """x' = (x - min) / (max - min), mapped to [0, 1]."""
    rng = (x.max() - x.min()).replace(0, 1)
    return (x - x.min()) / rng


def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    """One transaction (basket) per film.

    Items are attribute=value pairs, so that rules stay interpretable
    (e.g. {genre=Drama, color=bw} -> {era=Classical studio}).  Missing
    values produce *no* item (an 'unknown' item would dominate the counts
    and generate meaningless rules).  Generic 'Other' buckets are also
    omitted for the same reason.
    """
    from src.preprocessing.vocabularies import CATEGORY_NAMES
    tx = []
    for r in df.itertuples(index=False):
        items = [f"genre={CATEGORY_NAMES.get(g, g)}" for g in r.genres]
        items.append(f"era={r.era}")
        if r.color_type != "unknown":
            items.append(f"color={r.color_type}")
        if r.studio_group not in ("Unknown", "Other studio"):
            items.append(f"studio={r.studio_group}")
        if r.director_country_grp not in ("Unknown", "Other"):
            items.append(f"dir_country={r.director_country_grp}")
        if isinstance(r.cast_size_bin, str):
            items.append(f"{r.cast_size_bin}")
        if isinstance(r.halliwell_level, str):
            items.append(f"halliwell={r.halliwell_level}")
        if r.any_oscar_nom:
            items.append("oscar=nominated")
        if r.is_remake:
            items.append("remake=yes")
        if r.was_remade:
            items.append("later_remade=yes")
        if r.is_tv:
            items.append("format=TV")
        if r.has_book_source:
            items.append("source=book/play")
        tx.append(sorted(set(items)))
    return tx

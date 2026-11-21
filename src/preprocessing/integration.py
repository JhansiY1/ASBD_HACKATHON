"""
KDD phase 2 – DATA INTEGRATION and imputation.

The UCI Movie dataset is *relational*: MAIN is the central relation and the
other files reference it through film_id (CASTS, REMAKES) or through the
standardised director id-name (PEOPLE).  This module joins them into one
film-level analytical table and a director-level table.

Join keys (documented in doc.html sections 2 and 3):
    MAIN.director   -> PEOPLE.person          (reference constraint)
    CASTS.film_id   -> MAIN.film_id           (partial; dangling refs exist)
    CASTS.actor     -> ACTORS.actor           (partial reference)
    REMAKES.film_id / prior_film -> MAIN.film_id

Imputation strategies (chosen per attribute):
    year          group-based (median year of the same director's films),
                  fallback: director career start from PEOPLE
    color_type    rule-based, from doc.html 3.14: unknown process is "likely
                  bnw prior to 1945, col after 1955"; 1945-1955 left unknown
                  (mode imputation would bias towards b/w)
    categorical   constant 'Unknown' category for encoders
    cast_size     0 is *not* imputed: missing cast = not documented; kept
                  as 0 with flag has_cast_info
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import CleaningLog


def integrate(movies: pd.DataFrame, casts: pd.DataFrame, actors: pd.DataFrame,
              people: pd.DataFrame, remakes: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    df = movies.copy()

    # ---------------------------------------------------------- PEOPLE join
    ppl = people[["person", "background", "is_female", "career_start", "career_end"]].rename(
        columns={"person": "director", "background": "director_country",
                 "is_female": "director_female", "career_start": "director_career_start",
                 "career_end": "director_career_end"})
    df = df.merge(ppl, on="director", how="left")
    unmatched = df["director"].notna() & df["director_country"].isna() & df["director_career_start"].isna()
    log.add("INTEGRATION", "integration", "director", "MAIN directors not found in PEOPLE "
            "(violates documented reference constraint)", unmatched.sum(), "kept; director attributes missing")

    # ---------------------------------------------------------- CASTS join
    ids = set(df["film_id"])
    dangling = ~casts["film_id"].isin(ids)
    log.add("INTEGRATION", "integration", "CASTS.film_id", "Cast rows whose film_id is not in MAIN "
            "(dangling references)", dangling.sum(), "excluded from film aggregates")
    c = casts[~dangling].merge(actors[["actor", "gender", "birth_year"]], on="actor", how="left")
    agg = c.groupby("film_id").agg(
        cast_size=("actor", "nunique"),
        cast_female=("gender", lambda g: (g == "F").sum()),
        cast_male=("gender", lambda g: (g == "M").sum()),
        cast_actor_award_noms=("actor_award_nom", "sum"),
        cast_mean_birth_year=("birth_year", "mean"),
    ).reset_index()
    df = df.merge(agg, on="film_id", how="left")
    df["has_cast_info"] = df["cast_size"].notna().astype(int)
    for col in ("cast_size", "cast_female", "cast_male", "cast_actor_award_noms"):
        df[col] = df[col].fillna(0).astype(int)
    known = df["cast_female"] + df["cast_male"]
    df["female_cast_share"] = np.where(known > 0, df["cast_female"] / known.replace(0, np.nan), np.nan)

    # ---------------------------------------------------------- REMAKES join
    df["is_remake"] = df["film_id"].isin(set(remakes["film_id"])).astype(int)
    remade_counts = remakes["prior_film"].value_counts()
    df["times_remade"] = df["film_id"].map(remade_counts).fillna(0).astype(int)
    df["was_remade"] = (df["times_remade"] > 0).astype(int)

    # ---------------------------------------------------------- imputation: year
    miss = df["year"].isna()
    df["year_imputed"] = 0
    grp_median = df.groupby("director")["year"].transform("median")
    fill1 = miss & grp_median.notna()
    df.loc[fill1, "year"] = grp_median[fill1].round()
    df.loc[fill1, "year_imputed"] = 1
    log.add("INTEGRATION", "imputation", "year", "Missing year imputed with the median year of the same "
            "director's films (group-based median imputation)", fill1.sum(), "imputed")
    miss = df["year"].isna()
    fill2 = miss & df["director_career_start"].notna()
    df.loc[fill2, "year"] = df.loc[fill2, "director_career_start"]
    df.loc[fill2, "year_imputed"] = 1
    log.add("INTEGRATION", "imputation", "year", "Remaining missing year imputed with director's career "
            "start year from PEOPLE", fill2.sum(), "imputed")
    still = df["year"].isna()
    log.add("INTEGRATION", "imputation", "year", "Year still unknown after imputation", still.sum(),
            "records removed (year is needed by every analysis)")
    df = df[~still].copy()
    df["year"] = df["year"].astype(int)

    # ---------------------------------------------------------- imputation: colour
    miss = df["color_type"].isna()
    rule_bw = miss & (df["year"] < 1945)
    rule_col = miss & (df["year"] > 1955)
    df["color_imputed"] = 0
    df.loc[rule_bw & (df["year"] < 1928), "color_type"] = "silent_bw"
    df.loc[rule_bw & (df["year"] >= 1928), "color_type"] = "bw"
    df.loc[rule_col, "color_type"] = "color"
    df.loc[rule_bw | rule_col, "color_imputed"] = 1
    log.add("INTEGRATION", "imputation", "color_type", "Unknown process ('prc') imputed with the documented "
            "rule: b/w before 1945 (silent b/w before 1928), colour after 1955", (rule_bw | rule_col).sum(),
            "rule-based imputation")
    log.add("INTEGRATION", "imputation", "color_type", "Unknown process for 1945-1955 (transition decade, "
            "no defensible rule)", df["color_type"].isna().sum(), "constant 'Unknown'")
    df["color_type"] = df["color_type"].fillna("unknown")

    # ---------------------------------------------------------- director productivity
    df["director_n_films"] = df.groupby("director")["film_id"].transform("count")
    df.loc[df["director_known"] == 0, "director_n_films"] = np.nan
    return df.reset_index(drop=True)


def build_director_table(films: pd.DataFrame, min_films: int = 5) -> pd.DataFrame:
    """Aggregate films to a director-level profile (used for clustering).

    Only real directors (not pseudo 'Unknown' placeholders) with at least
    ``min_films`` films are kept, so that the proportions are stable.
    """
    f = films[films["director_known"] == 1].copy()
    for g in ("Dram", "Comd", "Susp", "Romt", "Horr", "Musc", "West", "Advt", "Actn", "Docu"):
        f[f"g_{g}"] = f["genres"].apply(lambda gs, g=g: int(g in gs))
    f["is_color"] = f["color_type"].isin(["color", "colorized"]).astype(int)
    f["is_rated"] = f["halliwell_stars"].notna().astype(int)
    agg = f.groupby("director").agg(
        n_films=("film_id", "count"),
        first_year=("year", "min"), last_year=("year", "max"), mean_year=("year", "mean"),
        share_color=("is_color", "mean"), share_tv=("is_tv", "mean"),
        mean_halliwell=("halliwell_stars", "mean"), share_rated=("is_rated", "mean"),
        oscar_nom_rate=("any_oscar_nom", "mean"), mean_cast_size=("cast_size", "mean"),
        share_remakes=("is_remake", "mean"),
        **{f"share_{g}": (f"g_{g}", "mean") for g in ("Dram", "Comd", "Susp", "Romt", "Horr",
                                                       "Musc", "West", "Advt", "Actn", "Docu")},
        director_country=("director_country", "first"),
    ).reset_index()
    agg["career_span"] = agg["last_year"] - agg["first_year"]
    return agg[agg["n_films"] >= min_films].reset_index(drop=True)

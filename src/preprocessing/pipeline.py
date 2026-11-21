"""
Reproducible preprocessing pipeline (KDD: cleaning -> integration ->
selection -> transformation).

Run:  python -m src.preprocessing.pipeline

Writes
    outputs/dataset_overview.txt, outputs/data_dictionary.csv,
    outputs/data_quality_summary.csv, outputs/cleaning_log.csv,
    outputs/data_selection.csv, outputs/outlier_summary.csv,
    outputs/preprocessing_report.md
    data/processed/*.csv|parquet-like pickles, data/transactions/*.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import OUTPUTS, PROCESSED_DIR, TRANSACTIONS_DIR, get_logger
from src.data_loading.html_table_loader import file_inventory, read_table
from src.preprocessing import cleaning as C
from src.preprocessing.integration import build_director_table, integrate
from src.preprocessing.outliers_smoothing import (OUTLIER_VERDICTS, equal_depth_smoothing,
                                                  moving_average, outlier_summary)
from src.preprocessing.selection import (DT_FEATURES_NUMERIC, DT_FEATURES_ONEHOT, DT_TARGET,
                                         KMEANS_FEATURES, selection_table)
from src.preprocessing.transformation import (add_derived_features, build_transactions, genre_multi_hot,
                                              minmax, one_hot, zscore)

log = get_logger("preprocessing")

# Raw-attribute documentation (doc.html section 3) used for the data dictionary.
RAW_DICTIONARY = [
    ("MAIN", "film_id", "nominal (key)", "Internally generated id = director code + sequence number"),
    ("MAIN", "title", "text", "Film title, prefixed T:/Tn:; not unique"),
    ("MAIN", "year", "numeric (interval)", "Release year; '19xx' = unknown"),
    ("MAIN", "director", "nominal (FK->PEOPLE)", "Standardised director id-name, prefixed D:"),
    ("MAIN", "producers", "nominal, multi-valued", "P:(in PEOPLE) / PN:(name) / PU:(unknown) / P:(none)"),
    ("MAIN", "studios", "nominal, multi-valued", "St:/S: studio, SU: unknown, SL: location, SD: distributor"),
    ("MAIN", "process", "nominal (coded)", "bws/sbw silent b/w, bnw, col, \\Tcol etc., cld colourised, prc unknown"),
    ("MAIN", "category", "nominal, multi-valued", "Genre codes (Dram, Comd, Susp ...); Ctxx = uncategorised"),
    ("MAIN", "awards", "nominal, multi-valued", "H..H**** Halliwell stars, AA/AAN Oscar, CG, RE, NFR ...; aw unknown"),
    ("MAIN", "locale", "text, hierarchical", "Where the film is set; lc = unknown"),
    ("MAIN", "notes", "semi-structured text", "Sub-fields W() writer, B() book, R() rating, Le() length, Lang(), Er()..."),
    ("CASTS", "film_id", "nominal (FK->MAIN)", "Film identifier"),
    ("CASTS", "title", "text", "Redundant title (T:, TZ: uncertain)"),
    ("CASTS", "actor", "nominal (FK->ACTORS)", "Stage name; 'sa' = some actor"),
    ("CASTS", "roletype", "nominal (coded)", "Three-letter role type; \\Und unassigned"),
    ("CASTS", "role", "text", "Role description R:/RN:/RU:/RZ:"),
    ("CASTS", "awards", "nominal", "Awards for this performance (AA, AAN)"),
    ("CASTS", "notes", "text", "Rare notes"),
    ("ACTORS", "stagename", "nominal (key)", "Stage name"),
    ("ACTORS", "dow", "interval", "Dates of work (first-last film year)"),
    ("ACTORS", "birthname / firstname", "text", "Original names"),
    ("ACTORS", "gender", "nominal", "M, F, X unknown, G group, A animal"),
    ("ACTORS", "dob / dod", "numeric (year)", "Birth / death year"),
    ("ACTORS", "type", "text", "Types of roles played"),
    ("ACTORS", "origin", "nominal (coded)", "Country of origin code (\\Am, \\Br ...)"),
    ("ACTORS", "picture / notes", "text", "Photo references, marriages Mt(), Cit() citation count"),
    ("PEOPLE", "id-name", "nominal (key)", "Standardised person name"),
    ("PEOPLE", "Pcode", "nominal, multi-valued", "P D W A C G V professions"),
    ("PEOPLE", "Did", "nominal", "Director code (prefix of film_id)"),
    ("PEOPLE", "years", "interval", "@first-last year of work"),
    ("PEOPLE", "lastname / firstname", "text", "Real name"),
    ("PEOPLE", "dob / dod", "numeric (year)", "Birth / death"),
    ("PEOPLE", "background", "nominal (coded)", "Birth country code"),
    ("PEOPLE", "notes", "text", "Ge(F)/GnF female, Aw() awards ..."),
    ("REMAKES", "film_id / title / year", "FK + text + numeric", "The remake"),
    ("REMAKES", "fraction", "numeric (ratio 0-1)", "How much of the original was reused"),
    ("REMAKES", "priorfilm / priortitle / prioryear", "FK + text + numeric", "The original film"),
    ("STUDIOS", "name, company, city, country, first, last, founder, successor, notes", "mixed",
     "Sparse studio history table (203 rows)"),
]


def quality_summary(name: str, df: pd.DataFrame, missing_tokens: dict[str, set]) -> pd.DataFrame:
    """Per-attribute quality profile of a *raw* table (placeholders counted as missing)."""
    rows = []
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        tokens = missing_tokens.get(c, set()) | {""}
        miss = s.isin(tokens).sum()
        rows.append(dict(table=name, attribute=c, n_records=len(s), missing_or_placeholder=int(miss),
                         missing_pct=round(100 * miss / len(s), 2), n_unique=int(s.nunique()),
                         most_frequent=s.value_counts().index[0], most_frequent_count=int(s.value_counts().iloc[0])))
    return pd.DataFrame(rows)


def raw_frames() -> dict[str, pd.DataFrame]:
    """Raw tables (positional, unrealigned) for profiling the original data."""
    specs = {
        "MAIN": ("main.html", C.MAIN_SCHEMA),
        "CASTS": ("casts.html", ["film_id", "title", "actor", "roletype", "role", "awards", "notes"]),
        "ACTORS": ("actors.html", ["stagename", "dow", "birthname", "firstname", "gender", "dob", "dod",
                                   "type", "origin", "picture", "notes"]),
        "PEOPLE": ("people.html", ["id_name", "pcode", "did", "years", "lastname", "firstname", "dob", "dod",
                                   "background", "notes"]),
        "REMAKES": ("remakes.html", ["film_id", "title", "year", "fraction", "prior_film", "prior_title",
                                     "prior_year"]),
        "STUDIOS": ("studios.html", ["name", "company", "city", "country", "first", "last", "founder",
                                     "successor", "notes"]),
    }
    out = {}
    for name, (fn, cols) in specs.items():
        recs = read_table(fn)
        n = len(cols)
        data = [(r.fields + [""] * n)[:n] for r in recs]
        out[name] = pd.DataFrame(data, columns=cols)
        out[name]["_n_fields"] = [len(r.fields) for r in recs]
    return out


PLACEHOLDERS = {
    "MAIN": {"year_raw": {"19xx", "199x", "18xx"}, "producer_raw": {"PU:", "PN:"}, "studio_raw": {"SU:"},
             "process_raw": {"prc"}, "category_raw": {"Ctxx", "CTxx"}, "awards_raw": {"aw"},
             "locale_raw": {"lc"}},
    "CASTS": {"actor": {"sa"}, "roletype": {"\\Und", "Und"}, "role": {"RU:"}},
    "ACTORS": {"gender": {"X"}, "origin": {"\\Un", "\\UN"}, "dod": {"\\UN", "\\Un"}, "type": {"RU:"}},
    "PEOPLE": {"background": {"\\Un", "\\UN"}, "dod": {"\\Un", "\\UN"}},
    "REMAKES": {"prior_film": {"\\UN", "\\Un"}},
    "STUDIOS": {},
}


def run() -> dict:
    log.info("PHASE 4 - preprocessing pipeline started")
    # ------------------------------------------------------------ profiling
    inv = pd.DataFrame(file_inventory())
    raws = raw_frames()
    qual = pd.concat([quality_summary(k, v.drop(columns="_n_fields"), PLACEHOLDERS[k]) for k, v in raws.items()])
    qual.to_csv(OUTPUTS / "data_quality_summary.csv", index=False)
    pd.DataFrame(RAW_DICTIONARY, columns=["table", "attribute", "type", "description"]).to_csv(
        OUTPUTS / "data_dictionary.csv", index=False)

    # ------------------------------------------------------------ cleaning
    clog = C.CleaningLog()
    movies = C.clean_movies(clog)
    casts = C.clean_casts(clog)
    actors = C.clean_actors(clog)
    people = C.clean_people(clog)
    remakes = C.clean_remakes(clog)
    log.info("cleaned: movies=%d casts=%d actors=%d people=%d remakes=%d", len(movies), len(casts),
             len(actors), len(people), len(remakes))

    # ------------------------------------------------------------ integration
    films = integrate(movies, casts, actors, people, remakes, clog)
    films = add_derived_features(films)
    directors = build_director_table(films)
    directors["n_films_log"] = np.log1p(directors["n_films"])
    directors["mean_halliwell_filled"] = directors["mean_halliwell"].fillna(-0.5) + 0.5  # 0 = never rated

    # ------------------------------------------------------------ outliers / smoothing
    out_cols = ["year", "cast_size", "director_n_films", "length_min", "n_producers", "halliwell_stars",
                "female_cast_share"]
    osum = outlier_summary(films.assign(cast_size=films["cast_size"].where(films["has_cast_info"] == 1)), out_cols)
    osum["verdict"] = osum["attribute"].map(OUTLIER_VERDICTS)
    osum.to_csv(OUTPUTS / "outlier_summary.csv", index=False)
    smooth = equal_depth_smoothing(films["length_min"], k=8)
    smooth.to_csv(PROCESSED_DIR / "length_min_binning_smoothing.csv")
    yearly = films.groupby("year").size().rename("films").to_frame()
    yearly = yearly.reindex(range(yearly.index.min(), yearly.index.max() + 1), fill_value=0)
    yearly["moving_avg_5y"] = moving_average(yearly["films"], 5)
    yearly.to_csv(PROCESSED_DIR / "films_per_year_smoothed.csv")

    # ------------------------------------------------------------ transformation outputs
    mh = genre_multi_hot(films)
    films_out = pd.concat([films, mh], axis=1)
    films_out["genres"] = films_out["genres"].apply(lambda g: "|".join(g))
    films_out.to_csv(PROCESSED_DIR / "films_cleaned_integrated.csv", index=False)
    films_out.to_pickle(PROCESSED_DIR / "films_cleaned_integrated.pkl")
    casts.to_csv(PROCESSED_DIR / "casts_cleaned.csv", index=False)
    actors.to_csv(PROCESSED_DIR / "actors_cleaned.csv", index=False)
    people.to_csv(PROCESSED_DIR / "people_cleaned.csv", index=False)
    remakes.to_csv(PROCESSED_DIR / "remakes_cleaned.csv", index=False)
    directors.to_csv(PROCESSED_DIR / "director_profiles.csv", index=False)

    # transformed (encoded + scaled) film table
    num_cols = ["year", "cast_size", "n_producers", "n_genres", "director_n_films_filled", "female_cast_share_filled"]
    transformed = pd.concat([films[["film_id"]], films[["era_code"]],
                             zscore(films[num_cols]).add_suffix("_z"), minmax(films[num_cols]).add_suffix("_mm"),
                             one_hot(films, ["color_type", "studio_group", "director_country_grp", "producer_status"]),
                             mh], axis=1)
    transformed.to_csv(PROCESSED_DIR / "films_transformed.csv", index=False)

    # ML-ready matrix for the decision tree (labelled films only)
    lab = films[films[DT_TARGET].notna()].copy()
    lab["female_cast_share"] = lab["female_cast_share_filled"]
    lab["director_n_films"] = lab["director_n_films_filled"]
    # director not found in PEOPLE -> gender flag unknown; constant imputation 0
    # (female directors are explicitly marked Ge(F) in PEOPLE, absence = not marked)
    lab["director_female"] = lab["director_female"].fillna(0)
    X = pd.concat([lab[DT_FEATURES_NUMERIC].reset_index(drop=True),
                   genre_multi_hot(lab).reset_index(drop=True),
                   one_hot(lab, DT_FEATURES_ONEHOT).reset_index(drop=True)], axis=1).astype(float)
    n_nan = int(X.isna().sum().sum())
    if n_nan:
        raise ValueError(f"ML matrix still contains {n_nan} missing values: {X.columns[X.isna().any()].tolist()}")
    X[DT_TARGET] = lab[DT_TARGET].astype(int).to_numpy()
    X.insert(0, "film_id", lab["film_id"].to_numpy())
    X.to_csv(PROCESSED_DIR / "dt_ml_matrix.csv", index=False)

    # K-Means director matrix (standardised)
    Z = zscore(directors[KMEANS_FEATURES])
    Z.insert(0, "director", directors["director"])
    Z.to_csv(PROCESSED_DIR / "kmeans_director_matrix_z.csv", index=False)

    # Movie-level feature matrix used as the larger K-Means benchmark workload
    mv = pd.concat([zscore(films[num_cols]), one_hot(films, ["color_type", "studio_group", "director_country_grp"]),
                    mh], axis=1).astype(float)
    mv.to_csv(PROCESSED_DIR / "kmeans_movie_matrix.csv", index=False)

    # transactions
    tx = build_transactions(films)
    pd.DataFrame({"film_id": films["film_id"], "items": [";".join(t) for t in tx]}).to_csv(
        TRANSACTIONS_DIR / "film_transactions.csv", index=False)
    with open(TRANSACTIONS_DIR / "film_transactions.json", "w", encoding="utf-8") as fh:
        json.dump(tx, fh)
    all_items = sorted({i for t in tx for i in t})
    bin_mat = pd.DataFrame([[int(i in set(t)) for i in all_items] for t in tx], columns=all_items)
    bin_mat.to_csv(TRANSACTIONS_DIR / "film_transactions_binary.csv", index=False)

    # ------------------------------------------------------------ documentation
    clog_df = clog.to_frame()
    clog_df.to_csv(OUTPUTS / "cleaning_log.csv", index=False)
    selection_table().to_csv(OUTPUTS / "data_selection.csv", index=False)

    stats = dict(raw_main_rows=len(raws["MAIN"]), raw_main_malformed=int((raws["MAIN"]["_n_fields"] != 11).sum()),
                 films_final=len(films), directors_profiled=len(directors), casts=len(casts), actors=len(actors),
                 people=len(people), remakes=len(remakes), transactions=len(tx), items=len(all_items),
                 avg_tx_len=float(np.mean([len(t) for t in tx])), dt_rows=len(X), dt_features=X.shape[1] - 2,
                 dt_positive_rate=float(X[DT_TARGET].mean()), kmeans_features=len(KMEANS_FEATURES),
                 movie_matrix_shape=list(mv.shape))
    with open(OUTPUTS / "preprocessing_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    _write_overview(inv, raws, stats)
    _write_report(raws, films, clog_df, stats, osum)
    log.info("PHASE 4 complete: %s", stats)
    return stats


def _write_overview(inv: pd.DataFrame, raws: dict, stats: dict) -> None:
    lines = ["UCI MOVIE DATASET - OVERVIEW", "=" * 60,
             "Source : UCI Machine Learning Repository, dataset id 132 'Movie'",
             "URL    : https://archive.ics.uci.edu/ml/datasets/Movie (zip: /static/public/132/movie.zip)",
             "Creator: Gio Wiederhold (Stanford); DOI 10.24432/C5SW2R; licence CC BY 4.0",
             "Type   : Multivariate, relational; 6 linked HTML tables + documentation (doc.html)", "",
             "FILES", "-" * 60, inv.to_string(index=False), "",
             "RAW TABLE SHAPES (data rows x documented attributes)", "-" * 60]
    for k, v in raws.items():
        dist = v["_n_fields"].value_counts().sort_index().to_dict()
        lines.append(f"{k:8s}: {len(v):6d} rows x {v.shape[1] - 1:2d} attributes; cells-per-row distribution {dist}")
    lines += ["", "RELATIONSHIPS", "-" * 60,
              "MAIN.director -> PEOPLE.id-name ; CASTS.film_id -> MAIN.film_id ; CASTS.actor -> ACTORS.stagename ;",
              "REMAKES.film_id, REMAKES.priorfilm -> MAIN.film_id ; MAIN.studios -> STUDIOS.name (loose)", "",
              "ANALYTICAL TABLES PRODUCED", "-" * 60]
    lines += [f"{k}: {v}" for k, v in stats.items()]
    (OUTPUTS / "dataset_overview.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_report(raws, films, clog_df, stats, osum) -> None:
    raw_main = raws["MAIN"]
    md = ["# Preprocessing Report", "",
          "## 1. Original dataset size",
          f"* MAIN: {len(raw_main)} data rows ({stats['raw_main_malformed']} malformed, i.e. not 11 cells)",
          *[f"* {k}: {len(v)} data rows" for k, v in raws.items() if k != "MAIN"], "",
          "## 2-5. Cleaning operations (missing values, duplicates, outliers, removals)", "",
          clog_df.to_markdown(index=False), "",
          "## Outliers identified", "", osum.to_markdown(index=False), "",
          "## 6. Attribute selection", "", selection_table().to_markdown(index=False), "",
          "## 7. Transformations applied",
          "* Discretisation: year -> decade, era (5 film-history periods); cast size -> 4 bins; Halliwell stars -> level",
          "* Label encoding: era_code (ordinal 0-4), halliwell_code (0-4)",
          "* One-hot: color_type, studio_group, director_country_grp, producer_status",
          "* Multi-hot (binary item presence): genres with >= 30 films",
          "* z-score standardisation and min-max normalisation of numeric attributes only",
          "* Derived: director_n_films, female_cast_share, is_remake, was_remade, acclaimed target",
          "* Transactions: one basket of attribute=value items per film", "",
          "## 8. Final dimensions",
          f"* Integrated film table: {len(films)} films x {films.shape[1]} columns",
          f"* Decision-tree matrix: {stats['dt_rows']} labelled films x {stats['dt_features']} features",
          f"* Director profile table (>=5 films): {stats['directors_profiled']} directors x {stats['kmeans_features']} features",
          f"* Transactions: {stats['transactions']} baskets, {stats['items']} distinct items, "
          f"mean length {stats['avg_tx_len']:.2f}"]
    (OUTPUTS / "preprocessing_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    run()

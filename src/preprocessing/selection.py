"""
KDD phase 3 – DATA SELECTION.

Task-relevant attributes are selected separately for each analytical
technique.  Each entry records the attribute, the technique(s) that use it,
and the reason it is kept or discarded.  The table is written to
outputs/data_selection.csv and reproduced in the report.
"""
from __future__ import annotations

import pandas as pd

# (attribute, decision, used_by, reason)
SELECTION = [
    ("film_id", "discard (identifier)", "-", "Surrogate key composed of director code + sequence number; "
     "no analytical meaning, used only for joins"),
    ("title", "discard (identifier)", "descriptive only", "Free text, ~10 000 unique values; kept for display"),
    ("year", "keep", "descriptive, DT, K-Means (via director profile), transactions (as era)",
     "Only temporal attribute; drives colour process, genre trends and studio era"),
    ("director", "discard as feature", "integration key, director-level clustering unit",
     "2 900+ distinct values (high cardinality identifier); used to join PEOPLE and aggregate"),
    ("producer_raw", "transform", "DT (producer_status, n_producers)", "Raw names are high-cardinality; the "
     "coding (referenced / named / none / unknown) and count are informative"),
    ("studio", "transform", "descriptive, Apriori, DT (studio_group)", "1 200+ raw names reduced to the "
     "8 classical Hollywood majors + Hitchcock TV + Other/Unknown"),
    ("color_type", "keep", "descriptive, Apriori, DT, K-Means", "Film process (silent b/w, b/w, colour, "
     "colourised, unknown) - technological era marker"),
    ("genres", "keep (multi-valued)", "descriptive, Apriori, COFI, DT, K-Means", "Core content attribute; "
     "multi-valued, natural basket items"),
    ("halliwell_stars", "keep", "descriptive, Apriori, DT target", "Critic rating 0-4 stars in Halliwell's "
     "Film Guide; basis of the classification target"),
    ("oscar_wins / oscar_noms", "keep for description, EXCLUDE from DT", "descriptive, Apriori, K-Means",
     "Awards are outcomes of the same critical judgement as the target -> target leakage in DT"),
    ("other_awards, worst_film_list", "exclude from DT", "descriptive", "Other critics' ratings/awards, "
     "leakage-prone for an acclaim target"),
    ("locale_raw -> setting", "keep (coarse)", "descriptive", "Setting country/region; 40 % missing"),
    ("notes_raw", "transform", "DT (has_book_source, is_series, is_short), descriptive (mpaa, language)",
     "Semi-structured sub-fields parsed into flags"),
    ("mpaa_rating", "descriptive only", "descriptive", "Exists only after 1968 (> 90 % missing)"),
    ("length_min", "descriptive only", "descriptive", "Present for < 5 % of films"),
    ("cast_size, female_cast_share", "keep", "descriptive, Apriori (binned), DT, K-Means",
     "Integrated from CASTS + ACTORS; documentation depth and gender composition"),
    ("director_country, director_female, director_n_films", "keep", "descriptive, Apriori, DT, K-Means",
     "Integrated from PEOPLE (director background)"),
    ("is_remake", "keep", "Apriori, DT", "Integrated from REMAKES - remake of an earlier film"),
    ("was_remade / times_remade", "exclude from DT", "descriptive, Apriori", "Known only after later films "
     "exist (temporal leakage for a release-time prediction)"),
    ("author_error_flag, was_realigned", "audit only", "data quality report", "Meta-data about data quality"),
    ("source_line, *_raw", "discard", "-", "Lineage columns kept only in the cleaned file"),
]

APRIORI_ATTRIBUTES = ["genres", "era", "color_type", "studio_group", "director_country_grp",
                      "cast_size_bin", "halliwell_level", "any_oscar_nom", "is_remake", "was_remade",
                      "is_tv", "has_book_source"]

DT_FEATURES_NUMERIC = ["year", "n_genres", "n_producers", "cast_size", "has_cast_info",
                       "female_cast_share", "female_share_missing", "director_n_films",
                       "director_female", "is_tv", "is_widescreen", "is_remake",
                       "has_book_source", "is_series", "has_codirector"]
DT_FEATURES_ONEHOT = ["color_type", "studio_group", "director_country_grp", "producer_status"]
DT_TARGET = "acclaimed"

KMEANS_FEATURES = ["n_films_log", "mean_year", "career_span", "share_color", "share_tv",
                   "mean_halliwell_filled", "share_rated", "oscar_nom_rate", "mean_cast_size",
                   "share_Dram", "share_Comd", "share_Susp", "share_Romt", "share_Horr",
                   "share_Musc", "share_West", "share_Advt", "share_Actn", "share_Docu"]


def selection_table() -> pd.DataFrame:
    return pd.DataFrame(SELECTION, columns=["attribute", "decision", "used_by", "reason"])

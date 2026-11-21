"""Unit tests for the cleaning / transformation helpers."""
import math

import numpy as np
import pandas as pd

from src.data_loading.html_table_loader import normalise_text
from src.preprocessing.cleaning import (parse_awards, parse_categories, parse_length_minutes, parse_process,
                                        parse_producer, parse_year, realign_main_fields)
from src.preprocessing.outliers_smoothing import equal_depth_smoothing, outlier_summary
from src.preprocessing.transformation import cast_bin, era_of, minmax, zscore


def test_normalise_text_tex_accents_and_tilde():
    assert normalise_text(r"T:Les Mis\'erables") == "T:Les Misérables"
    assert normalise_text("B(Oscar~Wilde)") == "B(Oscar Wilde)"
    assert normalise_text("  a   b <b>c</b> ") == "a b c"


def test_parse_year():
    assert parse_year("1922") == (1922.0, False)
    assert parse_year("1933?")[1] is True
    assert math.isnan(parse_year("19xx")[0])


def test_category_standardisation():
    unk = []
    assert parse_categories("Romt,Comd", unk) == ["Romt", "Comd"]
    assert parse_categories("romt", unk) == ["Romt"]                 # case
    assert parse_categories("Ducu", unk) == ["Docu"]                 # typo
    assert parse_categories("CtxxDram", unk) == ["Dram"]             # glued + placeholder
    assert parse_categories("Ctxx", unk) == []                       # missing
    parse_categories("Zzzz", unk)
    assert "Zzzz" in unk


def test_process_and_producer():
    assert parse_process("bnw,TV") == ("bw", True, False)
    assert parse_process(r"\DeLuxe \CS") == ("color", False, True)
    assert math.isnan(parse_process("prc")[0])
    assert parse_producer("PU:") == ("unknown", 0)
    assert parse_producer("P:") == ("none", 0)
    assert parse_producer("PN:Lew Grade, Martin Starger") == ("named", 2)
    assert parse_producer("P:Hitchcock") == ("referenced", 1)


def test_awards():
    a = parse_awards("H***, AAN, AAN dir")
    assert a["halliwell_stars"] == 3 and a["oscar_noms"] == 2 and a["oscar_wins"] == 0
    b = parse_awards("AA, H**")
    assert b["oscar_wins"] == 1 and b["oscar_noms"] == 1 and b["halliwell_stars"] == 2
    assert math.isnan(parse_awards("aw")["halliwell_stars"]) and parse_awards("aw")["awards_missing"]
    assert math.isnan(parse_awards("-H")["halliwell_stars"])


def test_length_parsing():
    assert parse_length_minutes("1:45") == 105
    assert parse_length_minutes("90m") == 90
    assert math.isnan(parse_length_minutes("750feet"))


def test_realignment_of_malformed_row():
    # 10-cell row from main.html with a missing cell
    row = ["DWG47", "T:The New York Hat", "1912", "D:Griffith", "PN:", "SU:", "prc", "CTxx", "", "NYC, NY"]
    rec, realigned = realign_main_fields(row)
    assert realigned
    assert rec["process_raw"] == "prc" and rec["category_raw"] == "CTxx" and rec["studio_raw"] == "SU:"
    # split director cell
    row = ["WKD10", "T:Sneeze", "1893", "D:", "Dickson", "PN:Edison", "St:Edison", "bws", "Docu", "aw", "lc", ""]
    rec, _ = realign_main_fields(row)
    assert rec["director_raw"] == "D:Dickson" and rec["process_raw"] == "bws" and rec["category_raw"] == "Docu"


def test_binning_smoothing_lecture5_example():
    prices = pd.Series([4, 8, 9, 15, 21, 21, 24, 25, 26, 28, 29, 34], dtype=float)
    out = equal_depth_smoothing(prices, k=3)
    assert list(out["by_mean"].round(0)) == [9] * 4 + [23] * 4 + [29] * 4
    assert list(out["by_boundary"]) == [4, 4, 4, 15, 21, 21, 25, 25, 26, 26, 26, 34]


def test_scaling():
    x = pd.DataFrame({"a": [12000.0, 73600.0, 98000.0]})
    assert math.isclose(minmax(x)["a"].iloc[1], (73600 - 12000) / (98000 - 12000))     # Lecture 5: 0.716
    z = zscore(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))["a"]
    assert math.isclose(z.mean(), 0, abs_tol=1e-12) and math.isclose(z.std(ddof=0), 1)


def test_discretisation():
    assert era_of(1925).startswith("Silent") and era_of(1940).startswith("Classical")
    assert era_of(1995).startswith("Modern")
    assert cast_bin(0, 0) is np.nan or (isinstance(cast_bin(0, 0), float) and math.isnan(cast_bin(0, 0)))
    assert cast_bin(12, 1) == "cast 10+"


def test_outlier_summary_iqr():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 100]})
    s = outlier_summary(df, ["v"]).iloc[0]
    assert s["n_iqr_outliers"] == 1

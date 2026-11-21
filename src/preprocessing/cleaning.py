"""
KDD phase 1 – DATA CLEANING for the UCI Movie dataset.

Every operation appends an entry to a :class:`CleaningLog`, so that the
preprocessing report can state exactly what was done, to which attribute
and to how many records (requirement: "document every cleaning operation").

Cleaning problems actually found in the raw files (see the profiling step):

* noisy / malformed rows – MAIN rows with 2..17 instead of 11 fields
  (cells merged or split by the hand-written HTML), template rows such as
  ``stagenm | dow | ...`` inside ACTORS;
* missing values encoded by *placeholder codes* rather than blanks –
  ``aw``, ``lc``, ``prc``, ``Ctxx``/``CTxx``, ``PU:``, ``SU:``, ``\\Un``,
  ``\\Und``, ``19xx``;
* inconsistent values – case variants (``romt``/``Romt``, ``CTxx``), typos
  (``Ducu``, ``Muusc``), spacing variants (``Shamley,CBS`` vs
  ``Shamley, CBS``), glued codes (``CtxxDram``);
* duplicates – repeated film ids, exactly repeated records, and the same
  film (title, year, director) entered twice under different ids;
* invalid values – years outside the plausible range 1880-1999, film
  lengths that are in feet/reels rather than minutes;
* records flagged by the author himself as doubtful (``Er(...)`` notes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_loading.html_table_loader import read_table
from src.preprocessing.vocabularies import (AWARD_MISSING, CATEGORY_NAMES,
                                            CATEGORY_TYPOS, COUNTRY_CODES,
                                            COUNTRY_MISSING, PROCESS_GROUP,
                                            PROCESS_MISSING)


# ==========================================================================
# Audit log
# ==========================================================================
@dataclass
class CleaningLog:
    entries: list[dict] = field(default_factory=list)

    def add(self, table: str, step: str, attribute: str, description: str,
            records_affected: int, action: str) -> None:
        self.entries.append(dict(table=table, step=step, attribute=attribute,
                                 description=description,
                                 records_affected=int(records_affected),
                                 action=action))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


# ==========================================================================
# MAIN (movies) – field realignment
# ==========================================================================
MAIN_SCHEMA = ["film_id", "title_raw", "year_raw", "director_raw", "producer_raw",
               "studio_raw", "process_raw", "category_raw", "awards_raw",
               "locale_raw", "notes_raw"]

_CAT_CODES_LC = {c.lower() for c in CATEGORY_NAMES} | set(CATEGORY_TYPOS) | {"ctxx"}
_NOTE_CODE_RE = re.compile(r"\b(?:Alt|B|CoD|W|C|G|M|E|V|FilmedAt|Sy|Lang|Ser|Le|R|Cost|Inc|Nt|"
                           r"Seen|VT|VTN|Er|ER|P|Ch|Dtr|Chor)\(", re.I)
_AWARD_TOKEN_RE = re.compile(r"^\(?-?(H|AA|AAN|CG|RE|Z|W50|NFR|BFA|GP|T90|Palm|RFP|Cesar|"
                             r"AFI|F\d|Baer|VFF|NYFCC|NBR|GG|SE|Venice|Cannes|Oscar)", re.I)


def _is_process(v: str) -> bool:
    toks = [t for t in re.split(r"[,\s]+", v.replace("\\", " ")) if t]
    return bool(toks) and all(t.lower() in PROCESS_GROUP or t.lower() in PROCESS_MISSING for t in toks)


def _is_category(v: str) -> bool:
    toks = [t for t in re.split(r"[,;/\s]+", v) if t]
    return bool(toks) and all(t.lower() in _CAT_CODES_LC for t in toks)


def _is_award(v: str) -> bool:
    if v.lower() in AWARD_MISSING:
        return True
    toks = [t for t in re.split(r"[,;\s]+", v) if t]
    return bool(toks) and _AWARD_TOKEN_RE.match(toks[0]) is not None


# Predicates for the tail slots, in schema order.
_SLOT_TESTS = [
    ("producer_raw", lambda v: re.match(r"^P[NUZ]?:", v) is not None),
    ("studio_raw", lambda v: re.match(r"^(St|S|SU|SL|SN|SD):", v) is not None),
    ("process_raw", _is_process),
    ("category_raw", _is_category),
    ("awards_raw", _is_award),
    ("locale_raw", lambda v: v == "lc" or _NOTE_CODE_RE.search(v) is None),
    ("notes_raw", lambda v: True),
]


def realign_main_fields(fields: list[str]) -> tuple[dict, bool]:
    """Map a raw MAIN row onto the 11-attribute schema.

    Well-formed rows (exactly 11 fields) are mapped positionally.  Malformed
    rows are realigned with a *monotone slot-matching* procedure: the first
    four fields (id, title, year, director) are positional, and every later
    field is assigned to the earliest remaining schema slot whose syntactic
    pattern it satisfies (e.g. 'St:MGM' can only be a studio, 'bnw' only a
    process).  Empty fields fill the current slot.  Surplus fields are
    appended to the notes.  Returns (record, was_realigned).
    """
    rec = {k: "" for k in MAIN_SCHEMA}
    if len(fields) == 11:
        rec.update(dict(zip(MAIN_SCHEMA, fields)))
        return rec, False
    f = list(fields)
    # Director split over two cells, e.g. ['D:', 'Dickson'].
    if len(f) > 4 and f[3] in ("D:", "DN:") and not re.match(r"^[A-Z][A-Za-z]?:", f[4]):
        f[3] = f[3] + f[4]
        del f[4]
    for k, v in zip(MAIN_SCHEMA[:4], f[:4]):
        rec[k] = v
    ptr = 0
    overflow = []
    for v in f[4:]:
        if ptr >= len(_SLOT_TESTS):
            overflow.append(v)
            continue
        if v == "":
            ptr += 1
            continue
        for j in range(ptr, len(_SLOT_TESTS)):
            name, test = _SLOT_TESTS[j]
            if test(v):
                if name == "notes_raw" and rec[name]:
                    rec[name] += " " + v
                else:
                    rec[name] = v
                ptr = j + 1 if name != "notes_raw" else j
                break
    if overflow:
        rec["notes_raw"] = (rec["notes_raw"] + " " + " ".join(overflow)).strip()
    return rec, True


# ==========================================================================
# MAIN – attribute parsers
# ==========================================================================
def parse_year(v: str) -> tuple[float, bool]:
    """Return (year, uncertain_flag).  '19xx', '' -> NaN.  '1922?' -> 1922, uncertain."""
    m = re.search(r"(18|19|20)\d{2}", v or "")
    if not m:
        return np.nan, bool(v)
    return float(m.group(0)), ("?" in v or "-" in v or "x" in v.lower())


def parse_categories(v: str, unknown_tokens: list) -> list[str]:
    """Standardise the category field into a list of canonical genre codes.

    Steps: split on separators; split glued codes ('CtxxDram'); case-fold
    against the official vocabulary; apply the documented typo map; treat
    Ctxx (uncategorised) as missing; record anything else as inconsistent.
    """
    out: list[str] = []
    canon = {c.lower(): c for c in CATEGORY_NAMES}
    raw_tokens = [t for t in re.split(r"[,;/\s]+", v or "") if t]
    tokens: list[str] = []
    for t in raw_tokens:
        # split glued CamelCase codes such as 'NoirSusp', 'RomtAdvt'
        parts = re.findall(r"[A-Z][a-z.]*|[a-z.]+", t) if len(t) > 4 else [t]
        if len(parts) > 1 and all(p.lower() in _CAT_CODES_LC for p in parts):
            tokens.extend(parts)
        else:
            tokens.append(t)
    for t in tokens:
        tl = t.lower().strip(".") if t.lower() not in CATEGORY_TYPOS else t.lower()
        if tl in canon:
            c = canon[tl]
        elif t.lower() in CATEGORY_TYPOS:
            c = CATEGORY_TYPOS[t.lower()]
        else:
            unknown_tokens.append(t)
            c = None
        if c and c not in out:
            out.append(c)
    return out


def parse_process(v: str) -> tuple[str | float, bool, bool]:
    """Return (colour_type, is_tv, is_widescreen) from the process field."""
    toks = [t.lower() for t in re.split(r"[,\s]+", (v or "").replace("\\", " ")) if t]
    groups = [PROCESS_GROUP[t] for t in toks if t in PROCESS_GROUP]
    is_tv = "tv" in groups
    wide = any(t in ("cs", "pan", "vst", "trama", "tr") for t in toks)
    groups = [g for g in groups if g != "tv"]
    if not groups:
        return (np.nan, is_tv, wide)
    # priority: colorised > colour > silent b/w > b/w  (a mixed 'bnw,col'
    # film contains colour sequences and is treated as colour)
    for g in ("colorized", "color", "silent_bw", "bw"):
        if g in groups:
            return (g, is_tv, wide)
    return (np.nan, is_tv, wide)


MAJOR_STUDIOS = {
    # classical Hollywood "Big Five" and "Little Three" (+ common spellings)
    "mgm": "MGM", "paramount": "Paramount", "warners": "Warner Bros",
    "warner": "Warner Bros", "warner bros": "Warner Bros", "fox": "Fox",
    "20th century fox": "Fox", "twentieth century fox": "Fox", "rko": "RKO",
    "universal": "Universal", "u-i": "Universal", "columbia": "Columbia",
    "u.a.": "United Artists", "ua": "United Artists", "united artists": "United Artists",
}


def parse_studio(v: str) -> tuple[str | float, str | float]:
    """Return (primary_studio, studio_country_from_SL)."""
    v = v or ""
    country = np.nan
    m = re.search(r"SL:\s*\\?([A-Za-z]{2,3})", v)
    if m and m.group(1) in COUNTRY_CODES:
        country = COUNTRY_CODES[m.group(1)]
    m = re.search(r"(?:^|,\s*)(?:St|S|SN):\s*([^,]+)", v)
    if not m:
        return (np.nan, country)
    name = re.sub(r"\s+", " ", m.group(1)).strip().strip("*").strip()
    if not name or name in ("?", "*"):
        return (np.nan, country)
    return (name, country)


def studio_group(name) -> str | float:
    if not isinstance(name, str):
        return np.nan
    key = name.lower().strip()
    if key in MAJOR_STUDIOS:
        return MAJOR_STUDIOS[key]
    if key.startswith("shamley"):
        return "Shamley (Hitchcock TV)"
    return "Other studio"


def parse_producer(v: str) -> tuple[str, int]:
    """Return (producer_status, n_producers).

    P:name = producer referenced in PEOPLE; PN:name = name given;
    P: alone = no specific producer; PU:/empty PN: = unknown (missing).
    """
    v = (v or "").strip()
    m = re.match(r"^(P[NUZ]?):\s*(.*)$", v)
    if not m:
        return ("unknown", 0)
    code, names = m.group(1), m.group(2).strip()
    if code == "PU" or (code in ("PN", "PZ") and not names):
        return ("unknown", 0)
    if code == "P" and not names:
        return ("none", 0)
    n = len([x for x in re.split(r"[,;]", names) if x.strip()])
    return ("referenced" if code == "P" else "named", n)


def parse_awards(v: str) -> dict:
    """Decode the awards field into numeric indicators.

    Halliwell stars: 'H' = cited (0 stars) ... 'H****' = 4 stars, '+' adds
    half a star; '-H' = explicitly *not* in Halliwell; '(H)' = cited.
    AA = Academy Award (Oscar) won, AAN = nomination.
    """
    v = v or ""
    out = dict(halliwell_stars=np.nan, oscar_wins=0, oscar_noms=0,
               other_awards=0, worst_film_list=0, awards_missing=v.strip().lower() in AWARD_MISSING)
    toks = [t for t in re.split(r"[,;]+", v) if t.strip()]
    for t in toks:
        t = t.strip()
        hm = re.match(r"^\(?(H)(\*{0,4})(\+?)\)?(-?)$", t)
        if hm and not t.startswith("-H"):
            stars = len(hm.group(2)) + (0.5 if hm.group(3) else 0)
            if np.isnan(out["halliwell_stars"]) or stars > out["halliwell_stars"]:
                out["halliwell_stars"] = stars
            continue
        if re.match(r"^AAN\b", t):
            out["oscar_noms"] += 1
        elif re.match(r"^AA\b", t):
            out["oscar_wins"] += 1
            out["oscar_noms"] += 1      # a win implies a nomination
        elif t.startswith("W50"):
            out["worst_film_list"] = 1
        elif t.lower() not in AWARD_MISSING and not t.startswith("-H"):
            out["other_awards"] += 1
    return out


_MPAA = {"G": "G", "PG": "PG", "PG13": "PG-13", "PG-13": "PG-13", "R": "R",
         "X": "X", "NC17": "NC-17", "NC-17": "NC-17"}


def _note_field(notes: str, code: str) -> str | None:
    m = re.search(rf"\b{code}\(([^)]*)\)", notes or "")
    return m.group(1).strip() if m else None


def parse_length_minutes(s: str | None) -> float:
    """Parse Le(...) into minutes.  Accepts '90m', '1:45', '2h', '90min'.

    Lengths given in feet or reels (silent era) are not convertible to
    minutes reliably and are treated as missing; 'short' -> NaN as well
    (captured by the is_short flag instead).
    """
    if not s:
        return np.nan
    s = s.lower()
    if "feet" in s or "ft" in s or "reel" in s and "m" not in s:
        return np.nan
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*h", s)
    if m:
        return float(m.group(1)) * 60
    m = re.match(r"^\s*(\d+)\s*(m|min)", s)
    if m:
        return float(m.group(1))
    return np.nan


_LANG = {**COUNTRY_CODES, "En": "English", "Pg": "Portugal"}


def parse_notes(notes: str) -> dict:
    """Extract structured sub-fields from the free-text notes attribute."""
    notes = notes or ""
    r = _note_field(notes, "R")
    mpaa = np.nan
    if r:
        key = r.replace(" ", "").upper()
        mpaa = _MPAA.get(key, np.nan)
    lang = _note_field(notes, "Lang")
    if lang:
        code = lang.replace("\\", "").strip()[:2]
        lang = _LANG.get(code, lang.replace("\\", "").strip())
    le = _note_field(notes, "Le")
    return dict(
        mpaa_rating=mpaa,
        language_note=lang if lang else np.nan,
        length_raw=le if le else np.nan,
        length_min=parse_length_minutes(le),
        is_short=int(bool(le and "short" in le.lower())),
        has_book_source=int(_note_field(notes, "B") is not None),
        has_writer_note=int(_note_field(notes, "W") is not None),
        has_codirector=int(_note_field(notes, "CoD") is not None),
        is_series=int(_note_field(notes, "Ser") is not None),
        n_alt_titles=len(re.findall(r"\bAlt\(", notes)),
        author_error_flag=int(re.search(r"\bE[Rr]\(", notes) is not None),
    )


_US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH "
                 "NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC US USA".split())
_SETTING_COUNTRIES = {"england": "UK", "scotland": "UK", "wales": "UK", "london": "UK", "uk": "UK",
                      "britain": "UK", "france": "France", "paris": "France", "italy": "Italy",
                      "germany": "Germany", "japan": "Japan", "india": "India", "russia": "Russia/USSR",
                      "ussr": "Russia/USSR", "spain": "Spain", "mexico": "Mexico", "africa": "Africa",
                      "space": "Space", "china": "China", "greece": "Greece", "egypt": "Egypt",
                      "canada": "Canada", "austria": "Austria", "ireland": "Ireland", "west": "USA"}


def parse_setting(v: str) -> str | float:
    """Coarse country/region where the film is set, from the locale field."""
    v = (v or "").strip()
    if v.lower() in ("lc", ""):
        return np.nan
    first = re.split(r";", v)[0]
    first = re.sub(r"T\([^)]*\)", "", first)
    parts = [p.strip() for p in first.split(",") if p.strip()]
    for p in reversed(parts):
        if p in _US_STATES or p.upper() in _US_STATES:
            return "USA"
        if p.lower() in _SETTING_COUNTRIES:
            return _SETTING_COUNTRIES[p.lower()]
    return "Other"


# ==========================================================================
# MAIN – full cleaning procedure
# ==========================================================================
def clean_movies(log: CleaningLog) -> pd.DataFrame:
    raw = read_table("main.html")
    n_raw = len(raw)
    log.add("MAIN", "load", "*", "Data rows parsed from main.html (<tr><td> lines)", n_raw, "loaded")

    # ---- 1. noisy rows: too few fields to identify a film ---------------
    keep, noisy = [], 0
    for r in raw:
        if len(r.fields) < 4 or not r.fields[0] or not r.fields[1].startswith("T"):
            noisy += 1
            continue
        keep.append(r)
    log.add("MAIN", "noise", "*", "Rows with <4 fields, empty film_id or no title cell (corrupted rows)",
            noisy, "removed")

    # ---- 2. realign malformed rows ---------------------------------------
    recs, n_realigned = [], 0
    for r in keep:
        rec, realigned = realign_main_fields(r.fields)
        rec["source_line"] = r.line_no
        rec["n_raw_fields"] = len(r.fields)
        rec["was_realigned"] = int(realigned)
        n_realigned += realigned
        recs.append(rec)
    df = pd.DataFrame(recs)
    log.add("MAIN", "noise", "all fields", "Rows with != 11 cells (merged/split HTML cells) realigned by "
            "monotone slot matching on field syntax", n_realigned, "realigned")

    # ---- 3. parse & standardise each attribute ---------------------------
    df["title"] = df["title_raw"].str.replace(r"^T\w{0,2}:\s*", "", regex=True).str.strip()
    bad_title = df["title"].str.lower().isin(["", "unknown"])
    log.add("MAIN", "missing", "title", "Empty or 'unknown' title (record cannot be identified)",
            bad_title.sum(), "removed")
    df = df[~bad_title].copy()

    yrs = df["year_raw"].apply(parse_year)
    df["year"] = [y for y, _ in yrs]
    df["year_uncertain"] = [int(u) for _, u in yrs]
    log.add("MAIN", "missing", "year", "Year missing or unparseable (e.g. '19xx', blank)",
            df["year"].isna().sum(), "imputed later (group-based)")
    invalid_year = (df["year"] < 1880) | (df["year"] > 1999)
    log.add("MAIN", "invalid", "year", "Year outside plausible range 1880-1999 (dataset frozen 1999)",
            invalid_year.sum(), "set to missing")
    df.loc[invalid_year, "year"] = np.nan

    df["director"] = df["director_raw"].str.replace(r"^D[NU]?:\s*", "", regex=True).str.strip()
    df.loc[df["director"] == "", "director"] = np.nan
    pseudo = df["director"].fillna("").str.match(r"^(Unknown|UnYear|Un\d|Anon)", case=False)
    df["director_known"] = (~pseudo & df["director"].notna()).astype(int)
    log.add("MAIN", "missing", "director", "Pseudo-director placeholders (Unknown*, UnYear) - director unknown",
            pseudo.sum(), "flagged (director_known=0)")

    prod = df["producer_raw"].apply(parse_producer)
    df["producer_status"] = [p for p, _ in prod]
    df["n_producers"] = [n for _, n in prod]
    log.add("MAIN", "missing", "producers", "Producer unknown (PU:, empty PN:)",
            (df["producer_status"] == "unknown").sum(), "constant imputation: 'unknown' category")

    st = df["studio_raw"].apply(parse_studio)
    df["studio"] = [s for s, _ in st]
    df["studio_country_code"] = [c for _, c in st]
    df["studio"] = df["studio"].str.replace(r"\s*,\s*", ", ", regex=True)
    n_spacing = (df["studio_raw"].str.contains(r",\S", regex=True)).sum()
    log.add("MAIN", "inconsistent", "studios", "Inconsistent comma spacing in studio names "
            "(e.g. 'Shamley,CBS' vs 'Shamley, CBS') standardised", n_spacing, "standardised")
    df["studio_group"] = df["studio"].apply(studio_group)
    log.add("MAIN", "missing", "studios", "Studio unknown (SU:, blank)", df["studio"].isna().sum(),
            "kept as missing; 'Unknown' category in categorical encodings")

    pr = df["process_raw"].apply(parse_process)
    df["color_type"] = [p for p, _, _ in pr]
    df["is_tv"] = [int(t) for _, t, _ in pr]
    df["is_widescreen"] = [int(w) for _, _, w in pr]

    unknown_cat: list[str] = []
    df["genres"] = df["category_raw"].apply(lambda v: parse_categories(v, unknown_cat))
    raw_cat_tokens = df["category_raw"].str.split(r"[,;/\s]+", regex=True).explode()
    canon_lc = {c for c in CATEGORY_NAMES}
    n_case = raw_cat_tokens.dropna().apply(lambda t: t not in canon_lc and t.lower() in {c.lower() for c in canon_lc}).sum()
    canon_lower = {c.lower() for c in canon_lc}
    n_typo = raw_cat_tokens.dropna().apply(
        lambda t: t.lower() not in canon_lower and t.lower() in CATEGORY_TYPOS
        and CATEGORY_TYPOS[t.lower()] is not None).sum()
    log.add("MAIN", "inconsistent", "category", "Category codes with wrong letter case (e.g. 'romt', 'DRam')",
            n_case, "case-folded to canonical code")
    log.add("MAIN", "inconsistent", "category", "Misspelt category codes (e.g. 'Ducu', 'Muusc', 'Biop') mapped "
            "with explicit typo table", n_typo, "corrected")
    log.add("MAIN", "inconsistent", "category", "Category tokens not in the documented vocabulary "
            f"(examples: {sorted(set(unknown_cat))[:8]})", len(unknown_cat), "dropped token")
    df["n_genres"] = df["genres"].apply(len)
    df["primary_genre"] = df["genres"].apply(lambda g: g[0] if g else np.nan)
    log.add("MAIN", "missing", "category", "No valid category (Ctxx/CTxx placeholder or blank)",
            (df["n_genres"] == 0).sum(), "kept; genre flags all 0, primary_genre missing")

    aw = pd.DataFrame(list(df["awards_raw"].apply(parse_awards)), index=df.index)
    df = pd.concat([df, aw], axis=1)
    df["any_oscar_nom"] = (df["oscar_noms"] > 0).astype(int)

    nt = pd.DataFrame(list(df["notes_raw"].apply(parse_notes)), index=df.index)
    df = pd.concat([df, nt], axis=1)
    log.add("MAIN", "noise", "notes", "Records flagged by the author as possibly erroneous (Er()/ER())",
            df["author_error_flag"].sum(), "flagged, kept")
    df["setting"] = df["locale_raw"].apply(parse_setting)

    # ---- 4. duplicates -----------------------------------------------------
    core = ["film_id", "title", "year", "director", "producer_raw", "studio_raw",
            "process_raw", "category_raw", "awards_raw", "locale_raw", "notes_raw"]
    exact = df.duplicated(subset=core, keep="first")
    log.add("MAIN", "duplicates", "all fields", "Exact duplicate records", exact.sum(), "removed")
    df = df[~exact].copy()
    tkey = df["title"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    near = df.assign(_t=tkey).duplicated(subset=["_t", "year", "director"], keep="first")
    log.add("MAIN", "duplicates", "title+year+director", "Same film entered twice under different ids "
            "(normalised title, year and director identical)", near.sum(), "removed (first kept)")
    df = df[~near].copy()
    dup_id = df["film_id"].duplicated(keep=False)
    n_coll = df["film_id"].duplicated(keep="first").sum()
    log.add("MAIN", "duplicates", "film_id", "Distinct films sharing the same film_id (key collision)",
            n_coll, "made unique with suffix _b, _c ...")
    if n_coll:
        rank = df.groupby("film_id").cumcount()
        suffix = rank.map(lambda r: "" if r == 0 else "_" + "abcdefghij"[r])
        df["film_id"] = df["film_id"] + suffix

    # ---- 5. outliers / invalid lengths ------------------------------------
    bad_len = (df["length_min"] < 1) | (df["length_min"] > 600)
    log.add("MAIN", "invalid", "length_min", "Length outside 1-600 minutes (unit errors)", bad_len.sum(),
            "set to missing")
    df.loc[bad_len, "length_min"] = np.nan
    return df.reset_index(drop=True)


# ==========================================================================
# CASTS
# ==========================================================================
def clean_casts(log: CleaningLog) -> pd.DataFrame:
    raw = read_table("casts.html")
    log.add("CASTS", "load", "*", "Data rows parsed from casts.html", len(raw), "loaded")
    rows = []
    noisy = 0
    for r in raw:
        f = r.fields
        if len(f) < 3 or not f[0]:
            noisy += 1
            continue
        f = f + [""] * (7 - len(f))
        rows.append(dict(film_id=f[0].strip(), cast_title=re.sub(r"^T\w?:\s*", "", f[1]),
                         actor=f[2].strip(), roletype=f[3].replace("\\", "").strip(),
                         role=f[4], cast_award=f[5], uncertain=int(f[1].startswith("TZ"))))
    log.add("CASTS", "noise", "*", "Rows with <3 fields or empty film_id", noisy, "removed")
    df = pd.DataFrame(rows)
    unk = df["actor"].str.lower().isin(["", "sa", "s.a.", "?"])
    log.add("CASTS", "missing", "actor", "Actor unknown ('sa' = some actor, blank)", unk.sum(), "removed")
    df = df[~unk].copy()
    df["roletype"] = df["roletype"].replace({"Und": np.nan, "": np.nan, "und": np.nan})
    log.add("CASTS", "missing", "roletype", "Role type unassigned (\\Und)", df["roletype"].isna().sum(),
            "kept as missing")
    dup = df.duplicated(subset=["film_id", "actor"], keep="first")
    log.add("CASTS", "duplicates", "film_id+actor", "Same actor listed twice for the same film "
            "(key of the CASTS relation)", dup.sum(), "removed")
    df = df[~dup].copy()
    df["actor_award_nom"] = df["cast_award"].str.contains(r"\bAAN?\b", regex=True).astype(int)
    return df.reset_index(drop=True)


# ==========================================================================
# ACTORS
# ==========================================================================
def clean_actors(log: CleaningLog) -> pd.DataFrame:
    raw = read_table("actors.html")
    log.add("ACTORS", "load", "*", "Data rows parsed from actors.html", len(raw), "loaded")
    rows, noisy = [], 0
    for r in raw:
        f = r.fields + [""] * (11 - len(r.fields))
        if not f[0] or f[0].lower() in ("stagenm", "stage"):
            noisy += 1
            continue
        dob = re.search(r"(18|19)\d{2}", f[5])
        dod = re.search(r"(18|19)\d{2}", f[6]) if "x" not in f[6].lower() else None
        orig = f[8].replace("\\", "").strip()
        rows.append(dict(actor=f[0].strip(), gender=f[4].strip().upper()[:1] if f[4].strip() else "",
                         birth_year=float(dob.group(0)) if dob else np.nan,
                         death_year=float(dod.group(0)) if dod else np.nan,
                         origin=COUNTRY_CODES.get(orig, np.nan) if orig not in COUNTRY_MISSING else np.nan,
                         role_type_text=re.sub(r"^R[UZ]?:", "", f[7]).strip()))
    log.add("ACTORS", "noise", "*", "Template/header rows inside the table (e.g. 'stagenm')", noisy, "removed")
    df = pd.DataFrame(rows)
    bad_g = ~df["gender"].isin(["M", "F"])
    log.add("ACTORS", "missing", "gender", "Gender not M/F (X = unknown, G = group, A = animal, blank)",
            bad_g.sum(), "set to missing")
    df.loc[bad_g, "gender"] = np.nan
    inv = df["death_year"] < df["birth_year"]
    log.add("ACTORS", "invalid", "birth/death", "Death year before birth year", inv.sum(), "death year set missing")
    df.loc[inv, "death_year"] = np.nan
    dup = df.duplicated(subset=["actor"], keep="first")
    log.add("ACTORS", "duplicates", "actor", "Repeated stage names (key of ACTORS)", dup.sum(), "removed")
    return df[~dup].reset_index(drop=True)


# ==========================================================================
# PEOPLE (directors etc.)
# ==========================================================================
def clean_people(log: CleaningLog) -> pd.DataFrame:
    raw = read_table("people.html")
    log.add("PEOPLE", "load", "*", "Data rows parsed from people.html", len(raw), "loaded")
    rows, noisy = [], 0
    for r in raw:
        f = r.fields + [""] * (10 - len(r.fields))
        if not f[0] or f[0].lower() in ("ref_name", "id-name"):
            noisy += 1
            continue
        yrs = re.findall(r"(18|19|20)(\d{2})", f[3])
        years = [int(a + b) for a, b in yrs]
        bk = f[8].replace("\\", "").strip()
        notes = f[9]
        rows.append(dict(person=f[0].strip(), pcodes=f[1].strip(), is_director=int("D" in f[1]),
                         director_code=f[2].strip(),
                         career_start=years[0] if years else np.nan,
                         career_end=years[-1] if years else np.nan,
                         person_birth_year=float(re.search(r"(18|19)\d{2}", f[6]).group(0)) if re.search(r"(18|19)\d{2}", f[6]) else np.nan,
                         background=COUNTRY_CODES.get(bk[:2], np.nan) if bk and bk not in COUNTRY_MISSING else np.nan,
                         is_female=int(bool(re.search(r"G[en]\(?F|GnF", notes)))))
    log.add("PEOPLE", "noise", "*", "Template/header rows", noisy, "removed")
    df = pd.DataFrame(rows)
    log.add("PEOPLE", "missing", "background", "Birth country unknown (\\Un/blank)", df["background"].isna().sum(),
            "kept as missing")
    dup = df.duplicated(subset=["person"], keep="first")
    log.add("PEOPLE", "duplicates", "person", "Repeated id-names", dup.sum(), "removed")
    return df[~dup].reset_index(drop=True)


# ==========================================================================
# REMAKES
# ==========================================================================
def clean_remakes(log: CleaningLog) -> pd.DataFrame:
    raw = read_table("remakes.html")
    log.add("REMAKES", "load", "*", "Data rows parsed from remakes.html", len(raw), "loaded")
    rows, noisy = [], 0
    for r in raw:
        f = r.fields
        if len(f) < 7 or not f[0]:
            noisy += 1
            continue
        fr = re.search(r"\d*\.?\d+", f[3])
        y = re.search(r"(18|19)\d{2}", f[2])
        py = re.search(r"(18|19)\d{2}", f[6])
        rows.append(dict(film_id=f[0].strip(), year=float(y.group(0)) if y else np.nan,
                         fraction=float(fr.group(0)) if fr else np.nan,
                         prior_film=f[4].strip(), prior_year=float(py.group(0)) if py else np.nan))
    log.add("REMAKES", "noise", "*", "Rows with <7 fields", noisy, "removed")
    df = pd.DataFrame(rows)
    inv = df["year"] < df["prior_year"]
    log.add("REMAKES", "invalid", "year", "Remake dated before its original (violates documented constraint)",
            inv.sum(), "flagged")
    df["temporal_violation"] = inv.astype(int)
    dup = df.duplicated(subset=["film_id", "prior_film"], keep="first")
    log.add("REMAKES", "duplicates", "film_id+prior_film", "Repeated remake links", dup.sum(), "removed")
    return df[~dup].reset_index(drop=True)

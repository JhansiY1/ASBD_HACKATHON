"""
Loader for the UCI "Movie" relational dataset (Wiederhold, 1990).

The original files are hand-written HTML pages in which every record is a
single line of the form

    <tr><td> field1<td> field2<td> ... <td>|

(see Appendix B "CONVERSION" of data/raw/movie/doc.html).  Header lines start
with <tr><th> and describe per-director sub-tables; they are not data.

This module only *reads* the files: it splits rows into raw string fields and
keeps the source line number for auditing.  It deliberately performs no
cleaning – all cleaning decisions are made (and logged) in
src/preprocessing.

Input : path of one of the HTML files in data/raw/movie/data
Output: list[RawRecord] with the raw field strings
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import RAW_DIR

_ROW_RE = re.compile(r"^\s*<tr>\s*<td>", re.IGNORECASE)
_HDR_RE = re.compile(r"^\s*<tr>\s*<th>", re.IGNORECASE)
_TD_SPLIT = re.compile(r"<td>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

# TeX-style accent escapes used throughout the files, e.g. "Les Mis\'erables".
_TEX_ACCENTS = {
    "'": "\u0301",   # acute
    "`": "\u0300",   # grave
    '"': "\u0308",   # umlaut / diaeresis
    "^": "\u0302",   # circumflex
    "~": "\u0303",   # tilde
}
_TEX_RE = re.compile(r"\\([`'\"^~])\{?([A-Za-z])\}?")


@dataclass
class RawRecord:
    """One data row of an HTML table, before any cleaning."""
    line_no: int
    fields: list[str]
    section: str = ""            # last <th> header seen (e.g. director table)
    raw_line: str = field(default="", repr=False)


def detect_encoding(path: Path) -> str:
    """Return 'utf-8' if the file decodes as UTF-8, otherwise 'latin-1'.

    Five of the six main files are pure ASCII; casts.html and remakes.html
    contain a handful of single Latin-1 bytes, which are not valid UTF-8.
    """
    data = path.read_bytes()
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def normalise_text(value: str) -> str:
    """String normalisation applied to every field.

    * strips residual HTML tags (e.g. <A HREF=...>),
    * converts TeX accent escapes to Unicode (\\'e -> e-acute),
    * replaces the '~' word-joiner used inside id-names by a space,
    * collapses repeated whitespace.
    """
    import unicodedata
    value = _TAG_RE.sub("", value)
    value = _TEX_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + _TEX_ACCENTS[m.group(1)]), value)
    value = value.replace("~", " ")
    value = value.replace("&amp;", "&")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def read_table(filename: str, normalise: bool = True) -> list[RawRecord]:
    """Parse all data rows of an HTML table file.

    The trailing '|' end-of-record marker is removed.  Fields are stripped
    of surrounding whitespace.  If ``normalise`` is True the string
    normalisation of :func:`normalise_text` is applied.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    enc = detect_encoding(path)
    text = path.read_bytes().decode(enc)
    records: list[RawRecord] = []
    section = ""
    for i, line in enumerate(text.splitlines(), start=1):
        if _HDR_RE.match(line):
            section = _TAG_RE.sub(" ", line).strip()
            continue
        if not _ROW_RE.match(line):
            continue
        parts = _TD_SPLIT.split(line)[1:]
        parts = [p.strip() for p in parts]
        # drop the end-of-record marker '|' (possibly glued to the last field)
        if parts and parts[-1] in ("|", ""):
            parts = parts[:-1]
        if parts and parts[-1].endswith("|"):
            parts[-1] = parts[-1][:-1].strip()
        if normalise:
            parts = [normalise_text(p) for p in parts]
        records.append(RawRecord(line_no=i, fields=parts, section=section, raw_line=line))
    return records


def file_inventory() -> list[dict]:
    """Describe every file shipped with the dataset (name, size, encoding, rows)."""
    out = []
    for p in sorted(RAW_DIR.iterdir()):
        info = {"file": p.name, "bytes": p.stat().st_size, "format": p.suffix.lstrip(".") or "text"}
        if p.suffix == ".html":
            info["encoding"] = detect_encoding(p)
            txt = p.read_bytes().decode(info["encoding"])
            lines = txt.splitlines()
            info["data_rows"] = sum(1 for l in lines if _ROW_RE.match(l))
            info["header_rows"] = sum(1 for l in lines if _HDR_RE.match(l))
        else:
            info["encoding"] = "binary (MS Word)" if p.suffix == ".doc" else "ascii"
            info["data_rows"] = None
            info["header_rows"] = None
        out.append(info)
    return out

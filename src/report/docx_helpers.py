"""
python-docx helpers for the academic report: styles, inline **bold**/*italic*
markup, bullet lists, data tables, figures/tables with Word SEQ caption fields,
TOC / list-of-figures fields, page-number fields and equation placeholders.

Equations are written as linear UnicodeMath in paragraphs with the style
"Equation Linear"; src/report/word_postprocess.py converts them into native
Word (OMML) equations through Word's own BuildUp() and refreshes all fields.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_FONT = "Calibri"
HEAD_FONT = "Calibri"
ACCENT = RGBColor(0x1C, 0x3D, 0x6E)
HEADER_FILL = "1C3D6E"
ZEBRA_FILL = "EEF3FA"


# ------------------------------------------------------------------ document
def new_document() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)          # A4
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.3))
    sec.top_margin, sec.bottom_margin = Cm(2.3), Cm(2.2)
    sec.header_distance, sec.footer_distance = Cm(1.1), Cm(1.0)
    st = doc.styles
    normal = st["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for lvl, size, before, after in ((1, 16, 18, 8), (2, 13, 14, 6), (3, 11.5, 10, 4)):
        h = st[f"Heading {lvl}"]
        h.font.name = HEAD_FONT
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = ACCENT
        h.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if False else None
        rfonts = h.element.rPr.rFonts
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(a), HEAD_FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rfonts.get(qn(a)) is not None:
                del rfonts.attrib[qn(a)]
        h.paragraph_format.space_before = Pt(before)
        h.paragraph_format.space_after = Pt(after)
        h.paragraph_format.keep_with_next = True
        if lvl == 1:
            h.paragraph_format.page_break_before = True      # every chapter starts on a new page
    cap = st["Caption"]
    cap.font.name = BODY_FONT
    cap.font.size = Pt(9.5)
    cap.font.italic = False
    cap.font.bold = False
    cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    cap.paragraph_format.space_after = Pt(10)
    eq = st.add_style("Equation Linear", WD_STYLE_TYPE.PARAGRAPH)
    eq.base_style = normal
    eq.font.name = "Cambria Math"
    eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.space_before = Pt(4)
    eq.paragraph_format.space_after = Pt(6)
    code = st.add_style("Code Block", WD_STYLE_TYPE.PARAGRAPH)
    code.base_style = normal
    code.font.name = "Consolas"
    code.font.size = Pt(8.5)
    code.paragraph_format.space_after = Pt(0)
    code.paragraph_format.line_spacing = 1.0
    small = st.add_style("Table Text", WD_STYLE_TYPE.PARAGRAPH)
    small.base_style = normal
    small.font.size = Pt(8.5)
    small.paragraph_format.space_after = Pt(0)
    small.paragraph_format.line_spacing = 1.0
    return doc


# ------------------------------------------------------------------ fields
def add_field(paragraph, instr: str, placeholder: str = "") -> None:
    """Insert a complex field (TOC, SEQ, PAGE ...) into a paragraph."""
    run = paragraph.add_run()
    fc1 = OxmlElement("w:fldChar"); fc1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "
    fc2 = OxmlElement("w:fldChar"); fc2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = placeholder
    fc3 = OxmlElement("w:fldChar"); fc3.set(qn("w:fldCharType"), "end")
    for el in (fc1, it, fc2, t, fc3):
        run._r.append(el)


def header_footer(doc, header_text: str) -> None:
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    hp.text = header_text
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in hp.runs:
        r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    _bottom_border(hp)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("Page "); r.font.size = Pt(9)
    add_field(fp, "PAGE", "1")
    r = fp.add_run(" of "); r.font.size = Pt(9)
    add_field(fp, "NUMPAGES", "1")


def _bottom_border(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "A0A0A0")):
        b.set(qn(k), v)
    bdr.append(b)
    pPr.append(bdr)


# ------------------------------------------------------------------ text
_MARK = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_rich(paragraph, text: str, size: float | None = None) -> None:
    """Add text with inline **bold**, *italic* and `code` markup."""
    for part in _MARK.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = paragraph.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = paragraph.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        elif part.startswith("*"):
            r = paragraph.add_run(part[1:-1]); r.italic = True
        else:
            r = paragraph.add_run(part)
        if size:
            r.font.size = Pt(size)


def para(doc, text: str, style: str | None = None, align=None, size=None, space_after=None):
    p = doc.add_paragraph(style=style)
    add_rich(p, text, size)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items: list[str], style: str = "List Bullet") -> None:
    for it in items:
        p = doc.add_paragraph(style=style)
        add_rich(p, it)
        p.paragraph_format.space_after = Pt(2)


def equation(doc, unicode_math: str, label: str | None = None) -> None:
    """Linear UnicodeMath; converted to a native Word equation in post-processing."""
    p = doc.add_paragraph(style="Equation Linear")
    p.add_run(unicode_math)
    if label:
        p2 = doc.paragraphs[-1]
    return p


def code_block(doc, text: str) -> None:
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph(style="Code Block")
        p.add_run(line if line else " ")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def page_break(doc) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ captions
class Counter:
    def __init__(self):
        self.fig = 0
        self.tab = 0


def caption(doc, kind: str, text: str, counter: Counter) -> int:
    """Caption with a SEQ field so Word's List of Figures/Tables picks it up."""
    if kind == "Figure":
        counter.fig += 1; n = counter.fig
    else:
        counter.tab += 1; n = counter.tab
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{kind} "); r.bold = True
    run_holder = p
    add_field(run_holder, f"SEQ {kind} \\* ARABIC", str(n))
    for r in p.runs[-1:]:
        r.bold = True
    r = p.add_run(f". {text}")
    return n


def figure(doc, path: Path, title: str, counter: Counter, max_w_cm: float = 16.0, max_h_cm: float = 19.5) -> int:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    width = max_w_cm
    if h / w * width > max_h_cm:
        width = max_h_cm * w / h
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width))
    return caption(doc, "Figure", title, counter)


def _shade(cell, fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true")
    trPr.append(el)


def table(doc, df: pd.DataFrame, title: str, counter: Counter, col_widths_cm: list[float] | None = None,
          font_size: float = 8.5, decimals: int = 3) -> int:
    """Data table with caption above, shaded repeating header, zebra rows."""
    n = caption(doc, "Table", title, counter)
    # caption above the table must stay with it
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    df = df.copy()
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0]
    _repeat_header(hdr)
    for i, c in enumerate(df.columns):
        cell = hdr.cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(font_size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _shade(cell, HEADER_FILL)
    for ri, (_, row) in enumerate(df.iterrows()):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            if isinstance(v, float):
                txt = "" if pd.isna(v) else (f"{v:.{decimals}f}" if abs(v) < 1e6 else f"{v:.3g}")
                if txt.endswith(".000") and decimals == 3 and float(v).is_integer():
                    txt = str(int(v))
            else:
                txt = str(v)
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            add_rich(p, txt, font_size)
            p.paragraph_format.space_after = Pt(0)
            if ri % 2 == 1:
                _shade(cells[i], ZEBRA_FILL)
    if col_widths_cm:
        for row in t.rows:
            for i, w in enumerate(col_widths_cm):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return n


def add_hyperlink(paragraph, url: str, text: str, color: RGBColor | None = ACCENT, underline: bool = True):
    """Add a clickable hyperlink into a python-docx paragraph."""
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    if color is not None:
        c = OxmlElement("w:color")
        c.set(qn("w:val"), f"{color[0]:02X}{color[1]:02X}{color[2]:02X}")
        rPr.append(c)
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rPr.append(u)
    new_run.append(rPr)
    new_run_text = OxmlElement("w:t")
    new_run_text.text = text
    new_run.append(new_run_text)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def toc(doc, title: str, instr: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(title); r.bold = True; r.font.size = Pt(16); r.font.color.rgb = ACCENT
    p2 = doc.add_paragraph()
    add_field(p2, instr, "Right-click and choose 'Update field' to build this list.")

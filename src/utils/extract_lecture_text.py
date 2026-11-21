"""
Extract plain text from the course lecture PDFs (reference_materials/notes)
so that the lecture concepts can be studied and summarised in
reference_materials/lecture_summary.md.

Usage: python src/utils/extract_lecture_text.py
Uses PyMuPDF (fitz) if available, otherwise pypdf.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reference_materials" / "notes" / "notes"
DST = ROOT / "reference_materials" / "lecture_text"


def extract(pdf: Path) -> str:
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf) as doc:
            return "\n".join(f"--- page {i + 1} ---\n{p.get_text()}" for i, p in enumerate(doc))
    except ImportError:
        from pypdf import PdfReader
        r = PdfReader(str(pdf))
        return "\n".join(f"--- page {i + 1} ---\n{p.extract_text() or ''}" for i, p in enumerate(r.pages))


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(SRC.glob("*.pdf")):
        txt = extract(pdf)
        (DST / (pdf.stem + ".txt")).write_text(txt, encoding="utf-8")
        print(f"{pdf.name}: {len(txt)} chars")

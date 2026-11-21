"""
Finalise the report with Microsoft Word (COM automation, Windows):

1. convert every "Equation Linear" paragraph into a native Word equation
   (OMaths.Add + BuildUp: UnicodeMath linear format -> professional format),
2. update all fields (table of contents, lists of figures/tables, SEQ caption
   numbers, PAGE/NUMPAGES),
3. save the DOCX (still fully editable),
4. export the identical document to PDF (wdExportFormatPDF = 17).

If Word is not available, docx2pdf is tried for the PDF and the equations
remain in linear form.
"""
from __future__ import annotations

from pathlib import Path


def finalise(docx_path: Path, pdf_path: Path) -> dict:
    info = {"equations_built": 0, "word_used": False}
    try:
        import pythoncom
        import win32com.client as win32
    except ImportError:
        from docx2pdf import convert
        convert(str(docx_path), str(pdf_path))
        return info
    pythoncom.CoInitialize()
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx_path.resolve()))
        info["word_used"] = True
        # 1. equations: locate paragraphs styled "Equation Linear" with Word's
        #    own Find (fast, and robust to tables), then build them up in
        #    reverse order so earlier ranges stay valid.
        spans = []
        rng = doc.Content
        f = rng.Find
        f.ClearFormatting()
        f.Text = ""
        f.Style = doc.Styles("Equation Linear")
        f.Format = True
        f.Forward = True
        f.Wrap = 0                                     # wdFindStop
        guard = 0
        while f.Execute() and guard < 10000:
            guard += 1
            for k in range(1, rng.Paragraphs.Count + 1):
                pr = rng.Paragraphs(k).Range
                spans.append((pr.Start, pr.End))
            end = rng.End
            rng.SetRange(end, end)
            if end >= doc.Content.End - 1:
                break
        n = 0
        for start, end in sorted(set(spans), reverse=True):
            r = doc.Range(start, end - 1)              # exclude the paragraph mark
            if not r.Text.strip():
                continue
            try:
                om = doc.OMaths.Add(r)
                om.OMaths(1).BuildUp()
                n += 1
            except Exception as exc:                   # leave the linear form in place
                info.setdefault("equation_errors", []).append(f"{r.Text[:40]}: {exc}")
        info["equations_built"] = n
        # 2. fields: two passes so page numbers in the TOC are final
        for _ in range(2):
            doc.Fields.Update()
            for t in doc.TablesOfContents:
                t.Update()
            for t in doc.TablesOfFigures:
                t.Update()
        info["pages"] = doc.ComputeStatistics(2)       # wdStatisticPages
        doc.Save()
        doc.ExportAsFixedFormat(str(pdf_path.resolve()), 17, OpenAfterExport=False, OptimizeFor=0,
                                CreateBookmarks=1, DocStructureTags=True)
        doc.Close(False)
    finally:
        try:
            word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()
    return info

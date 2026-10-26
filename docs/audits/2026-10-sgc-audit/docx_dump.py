import sys, io
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"G:\SimplCore-V 0.1-ALPHA\simplcore\docs\SGC_v2_诚实版报告.docx"
d = Document(p)
idx = 0; ti = 0
for child in d.element.body.iterchildren():
    tag = child.tag.split("}")[-1]
    if tag == "p":
        para = Paragraph(child, d)
        style = para.style.name if para.style is not None else "?"
        print(f"[P{idx:03d}][{style}] {para.text}")
        idx += 1
    elif tag == "tbl":
        t = Table(child, d)
        print(f"[TABLE {ti}] rows={len(t.rows)} cols={len(t.columns)}")
        for ri, row in enumerate(t.rows):
            print(f"  T{ti}R{ri}: " + " | ".join(c.text.replace("\n", " / ") for c in row.cells))
        ti += 1

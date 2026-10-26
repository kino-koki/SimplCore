# -*- coding: utf-8 -*-
import os
try:
    import docx
    print("python-docx OK", docx.__version__)
except Exception as e:
    print("NO docx:", e)
p = os.path.join(r"G:\SimplCore-V 0.1-ALPHA\simplcore\docs", "SGC_v2_诚实版报告.docx")
print("report exists:", os.path.exists(p))
if os.path.exists(p):
    d = docx.Document(p)
    heads = [par.text for par in d.paragraphs if par.style.name.startswith("Heading") or par.style.name=="Title"]
    print("HEADINGS:", len(heads))
    for h in heads[:60]:
        print("  -", h[:80])
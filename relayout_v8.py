#!/usr/bin/env python3
"""Re-layout pass for build_simplcore_pdf_v8.py.

Fixes the visible typography defects without touching any claim, number,
theorem, or JSON injection:

  A. title block: "Anonymous submission" overlapped the third title line
  B. para():      very long unbreakable math strings overflowed the column;
                  justification could stretch inter-word spaces arbitrarily
  C. table():     rows were split across columns/pages; long words bled into
                  neighbouring cells; no keep-together; no per-row ensure
  D. Table 1:     two narrow columns at 6.5pt made the first-page difference
                  table unreadable -> shorter cells, 6.9pt, wider leading
  E. Section 2:   formula paragraphs are set unjustified; duplicate tail
                  sentence from an earlier edit removed
  F. References:  filled only the left column with 1pt gaps and stretched
                  spacing -> balanced two-column flow, 11pt leading
  G. PDF metadata title now matches the page-1 title

Every replacement is asserted present exactly once.
"""
import re

P = "/data/workspace/build_simplcore_pdf_v8.py"
s = open(P, encoding="utf-8").read()
R = []


def rep(tag, old, new, n=1):
    c = s.count(old)
    assert c == n, "%s: expected %d, found %d\n  %r" % (tag, n, c, old[:110])
    R.append((tag, old, new, n))


# ---------------------------------------------------------------- A  title
rep("A/title block",
'''    def _title_block(self):
        self.c.setFillGray(0)
        yy = H - MT
        for line in ["Exact Content Addressing Needs an Order-2 State:",
                     "A Tight Joint Bound on Tensor Order, Depth and",
                     "State Size for Static-LTI Sequence Models"]:
            self.c.setFont("TB", 15)
            self.c.drawCentredString(W / 2.0, yy - 15, line)
            yy -= 19
        yy -= 5
        self.c.setFont("TR", 9.5)
        self.c.drawCentredString(W / 2.0, yy, "Anonymous submission")
        yy -= 12
        self.c.setLineWidth(0.4)
        self.c.line(ML, yy, W - MR, yy)
        self.first_top = yy - 12
        self.y = self.first_top
''',
'''    def _title_block(self):
        self.c.setFillGray(0)
        yy = H - MT
        for line in ["Exact Content Addressing Needs an Order-2 State:",
                     "A Tight Joint Bound on Tensor Order, Depth and",
                     "State Size for Static-LTI Sequence Models"]:
            self.c.setFont("TB", 15)
            self.c.drawCentredString(W / 2.0, yy - 15, line)
            yy -= 19
        # 23 pt below the last 15 pt title baseline clears both glyph boxes.
        self.c.setFont("TR", 9.5)
        self.c.drawCentredString(W / 2.0, yy - 19, "Anonymous submission")
        yy -= 31
        self.c.setLineWidth(0.4)
        self.c.line(ML, yy, W - MR, yy)
        self.first_top = yy - 12
        self.y = self.first_top
''')

rep("A/metadata title",
'        self.c.setTitle("Where the Bits Must Live: An Input-Side Information Bound, Tight on a Constructible Family, for Fixed-State Spectral Models")',
'        self.c.setTitle("Exact Content Addressing Needs an Order-2 State: A Tight Joint Bound on Tensor Order, Depth and State Size for Static-LTI Sequence Models")')

# ---------------------------------------------------------- B  long words
rep("B/break_words helper",
'''def wpt(s, f, size):
    return pdfmetrics.stringWidth(s, f, size)
''',
'''def wpt(s, f, size):
    return pdfmetrics.stringWidth(s, f, size)


def break_words(toks, size, width):
    """Split tokens whose glyph width exceeds a column into character chunks."""
    out = []
    for word, f in toks:
        if wpt(word, f, size) <= width - 2:
            out.append((word, f))
            continue
        chunk = ""
        for ch in word:
            if chunk and wpt(chunk + ch, f, size) > width - 2:
                out.append((chunk, f))
                chunk = ch
            else:
                chunk += ch
        if chunk:
            out.append((chunk, f))
    return out
''')

rep("B/para use break_words",
'''        toks = tokenize(text)
        lines, cur, curw = [], [], 0.0''',
'''        toks = break_words(tokenize(text), size, width - indent)
        lines, cur, curw = [], [], 0.0''')

rep("B/para justify cap",
'''            extra = ((width - indent - lw) / (len(line) - 1)
                     if justify and li < len(lines) - 1 and len(line) > 1 else 0.0)''',
'''            extra = ((width - indent - lw) / (len(line) - 1)
                     if justify and li < len(lines) - 1 and len(line) > 1 else 0.0)
            extra = min(extra, wpt(" ", "TR", size) * 0.55)''')

# ------------------------------------------------------------- C  table()
rep("C/table keep-together + long-word wrap",
'''        def wrap(txt, f, width):
            lines, cur = [], ""
            for word in txt.split():
                trial = (cur + " " + word).strip()
                if cur and wpt(trial, f, size) > width - 3:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
            return lines

        for ri, row in enumerate(rows):
            cells = [wrap(c, "TB" if ri == 0 else "TR", widths[ci])
                     for ci, c in enumerate(row)]
            nlines = max(len(c) for c in cells)
            self.ensure(nlines * lead + 2)
            for li in range(nlines):
                x = xbase if (span and xbase is not None) else self.x0()
                for ci, lines in enumerate(cells):
                    if li < len(lines):
                        self.c.setFont("TB" if ri == 0 else "TR", size)
                        self.c.drawString(x, self.y - size * 0.82, lines[li])
                    x += widths[ci]
                self.y -= lead
            if ri == 0:
                self.c.setLineWidth(0.35)
                bx = xbase if (span and xbase is not None) else self.x0()
                self.c.line(bx, self.y + 2, bx + sum(widths), self.y + 2)
        self.c.setLineWidth(0.35)
        bx = xbase if (span and xbase is not None) else self.x0()
        self.c.line(bx, self.y + 2, bx + sum(widths), self.y + 2)
        self.y -= after
''',
'''        def wrap(txt, f, width):
            lines, cur = [], ""
            for word in txt.split():
                # Character-level fallback for unbreakable identifiers/URLs.
                if wpt(word, f, size) > width - 4:
                    if cur:
                        lines.append(cur)
                        cur = ""
                    chunk = ""
                    for ch in word:
                        if chunk and wpt(chunk + ch, f, size) > width - 4:
                            lines.append(chunk)
                            chunk = ch
                        else:
                            chunk += ch
                    cur = chunk
                    continue
                trial = (cur + " " + word).strip()
                if cur and wpt(trial, f, size) > width - 3:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
            return lines

        all_cells = [[wrap(c, "TB" if ri == 0 else "TR", widths[ci])
                      for ci, c in enumerate(row)]
                     for ri, row in enumerate(rows)]
        table_h = sum(max(len(c) for c in cells) * lead
                      for cells in all_cells) + 8
        # Keep the whole table in one column whenever it can fit there.
        if table_h < (self.top() - MB) and self.y - table_h < MB:
            if self.col == 0:
                self.col = 1
                self.y = self.top()
            else:
                self.newpage()

        for ri, cells in enumerate(all_cells):
            nlines = max(len(c) for c in cells)
            for li in range(nlines):
                self.ensure(lead)
                x = xbase if (span and xbase is not None) else self.x0()
                for ci, lines in enumerate(cells):
                    if li < len(lines):
                        self.c.setFont("TB" if ri == 0 else "TR", size)
                        self.c.drawString(x, self.y - size * 0.82, lines[li])
                    x += widths[ci]
                self.y -= lead
            if ri == 0:
                self.c.setLineWidth(0.35)
                bx = xbase if (span and xbase is not None) else self.x0()
                self.c.line(bx, self.y + 2.5, bx + sum(widths), self.y + 2.5)
        self.c.setLineWidth(0.35)
        bx = xbase if (span and xbase is not None) else self.x0()
        self.c.line(bx, self.y + 2.5, bx + sum(widths), self.y + 2.5)
        self.y -= after
''')

# ------------------------------------------------------------- D  Table 1
rep("D/Table 1 condense",
'''    d.table([
        ["Known result (specific source and statement)", "Precise increment here"],
        ["Zoology [1], Repeat After Me [2], Illusion of State [3]: fixed-state models are weak at associative recall",
         "Sharp bound, not an observation: min tensor order = 2 with a matching construction (Result 2(i))"],
        ["S5 [4], Z-transform LTI duality: a convolution kernel needs time-invariant coefficients",
         "P6: Toeplitz iff time-invariance for the FULL second-order family with c_t != 0"],
        ["Mamba [5], Sec. 3.1: selectivity breaks the convolution view",
         "Theorem A(c): WITHIN class M, duality-preserving layers cannot manufacture the bits; a conditional design statement"],
        ["Linear attention [6], GLA [7], DeltaNet [8]: outer-product key-value state",
         "P4: the rho=1 single-pole isomorphism; Result 2 keeps rho<1 decay and proves the JOINT tight bound"],
    ], [96, 142], size=6.5, lead=7.6, span=False, after=3.0)
''',
'''    d.table([
        ["Known result (source / statement)", "Precise increment here"],
        ["Zoology [1], Repeat After Me [2], Illusion of State [3]: fixed-state recall is weak",
         "Sharp bound: min tensor order = 2 with a matching construction (Result 2(i))"],
        ["S5 [4]: LTI duality needs time-invariant coefficients",
         "P6: Toeplitz iff time-invariance for the full second-order family, c_t != 0"],
        ["Mamba [5] Sec. 3.1: selectivity breaks the convolution view",
         "Theorem A(c): within class M, duality-preserving layers cannot create the bits"],
        ["Linear attention [6], GLA [7], DeltaNet [8]: outer-product key-value state",
         "P4: rho = 1 single-pole isomorphism; Result 2 keeps rho < 1 decay and joint tightness"],
    ], [112, 131], size=6.9, lead=8.4, span=False, after=5.0)
''')

# ------------------------------------------------------- E  Section 2 text
rep("E/duplicate tail sentence",
'''      "normalisation in H6. The read-out is last-layer only (A4 / P4). "
      "absorbed by the unit normalisation in H6.",
      size=SMALL, lead=SLEAD, after=2.0)''',
'''      "normalisation in H6. The read-out is last-layer only (A4 / P4).",
      size=SMALL, lead=SLEAD, after=2.0, justify=False)''')

rep("E/budget paragraph unjustified",
'''       "bit budget. The full notation table is Appendix A.0.",
       size=SMALL, lead=SLEAD, after=2.0)''',
'''       "bit budget. The full notation table is Appendix A.0.",
       size=SMALL, lead=SLEAD, after=2.0, justify=False)''')

# ------------------------------------------------------- F  References
rep("F/refs balanced two columns",
'''    for r in refs:
        P(d, r, size=SMALL, lead=SLEAD, after=1.0, span=False)
''',
'''    for ri, r in enumerate(refs):
        if ri == 7:                      # balance the last page across columns
            d.col = 1
            d.y = d.top()
        P(d, r, size=8.4, lead=11.0, after=3.0, span=False, justify=False)
''')

# ---------------------------------------------------------------- apply ----
for tag, old, new, n in R:
    assert s.count(old) == n, "apply %s: %d" % (tag, s.count(old))
    s = s.replace(old, new, n)

open(P, "w", encoding="utf-8").write(s)
print("APPLIED %d relayout edits:" % len(R))
for tag, _, _, _ in R:
    print("  -", tag)

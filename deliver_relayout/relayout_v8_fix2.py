#!/usr/bin/env python3
"""Second relayout pass: long headings overflowed the right column and the
dependency-map boxes overlapped their neighbours.  Both are pure geometry
fixes; no claim, number or theorem is touched."""
P = "/data/workspace/build_simplcore_pdf_v8.py"
s = open(P, encoding="utf-8").read()
R = []


def rep(tag, old, new, n=1):
    c = s.count(old)
    assert c == n, "%s: expected %d, found %d\n %r" % (tag, n, c, old[:100])
    R.append((tag, old, new, n))


# ------------------------------------------------- headings wrap to column
rep("heading wrap",
'''    def heading(self, text, size=10.4, before=6.0, after=2.5, span=False):
        self.y -= before
        self.ensure(size + 5)
        self.c.setFont("TB", size)
        self.c.drawString(ML if span else self.x0(), self.y - size * 0.82, text)
        self.y -= size + after
''',
'''    def heading(self, text, size=10.4, before=6.0, after=2.5, span=False):
        self.y -= before
        width = (W - ML - MR) if span else CW
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if cur and wpt(trial, "TB", size) > width - 2:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        self.ensure(len(lines) * (size + 1.5) + 5)
        x = ML if span else self.x0()
        for ln in lines:
            self.c.setFont("TB", size)
            self.c.drawString(x, self.y - size * 0.82, ln)
            self.y -= size + 1.5
        self.y -= after
''')

# --------------------------------------------- dependency-map box labels
rep("figure2 label wrap",
'''        c.setFont("TR", 6.2)
        c.setFillGray(0)
        words = label.split()
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if len(t) > 34:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        ty = by + hh * fh - 8
        for ln in lines[:2]:
            c.drawString(bx + 3, ty, ln)
            ty -= 7
''',
'''        lab_size = 6.6
        c.setFont("TR", lab_size)
        c.setFillGray(0)
        boxw = ww * fw - 6
        words = label.split()
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if cur and wpt(t, "TR", lab_size) > boxw:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        ty = by + hh * fh - 8
        for ln in lines[:2]:
            c.drawString(bx + 3, ty, ln)
            ty -= 7.6
''')

for tag, old, new, n in R:
    assert s.count(old) == n, "apply %s" % tag
    s = s.replace(old, new, n)
open(P, "w", encoding="utf-8").write(s)
print("APPLIED", len(R), "edits:")
for tag, _, _, _ in R:
    print("  -", tag)

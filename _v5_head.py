"""
build_simplcore_pdf.py — generate simplcore.pdf in ICML submission format.

Layout: US Letter, two columns, Times-metric serif, 8-page main paper,
unlimited references / impact statement / appendix, fully anonymous,
English throughout.  Every numeric value is read from the raw result files.
"""
import io
import json
import re
import os
import html
import numpy as np
from PIL import Image
from matplotlib.mathtext import math_to_image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

# ------------------------------------------------------------------ data
ROOT = "/data/workspace"
R = ROOT + "/train/superposition_probe/results/"


def J(name):
    return json.load(open(R + name))


T = J("tensor_results.json")
FUS = J("fusion_results.json")
BC = J("breakthrough_check.json")
AL = J("analytic_length.json")
PP = J("physics_precision.json")
PF = J("physics_frontier.json")
P5 = J("physics_retrain_5seed.json")
PC = J("physics_check.json")
PT = J("physics_throughput.json")
FS = J("physics_fixedscale.json")
PB = J("paired_bootstrap.json")


def acc(arm):
    return float(np.mean([r["final_acc"] for r in T[arm]]))


def nseed(arm):
    return len(T[arm])


def par(arm):
    return T[arm][0]["params"]


SCALE = float(np.mean([r["final_acc"] for r in FUS["UP_softmax"]]))
FP = BC["parity"]["fisher_p"]
W4R = BC["efficiency"]["ratio"]
LENS = AL["length_extrapolation"]
CHK = PC["summary"]
G5 = P5["summary"]
BOOT = PB["G2_bipole-vs-G0_osc"]

# ------------------------------------------------------------------ fonts
FD = "/usr/share/fonts/truetype/liberation2/"
pdfmetrics.registerFont(TTFont("TR", FD + "LiberationSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("TB", FD + "LiberationSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("TI", FD + "LiberationSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("TBI", FD + "LiberationSerif-BoldItalic.ttf"))

BODY, LEAD = 9.2, 10.9
SMALL, SLEAD = 8.0, 9.6
W, H = 612, 792
ML = MR = 54
MT, MB = 66, 58
GAP = 18
CW = (W - ML - MR - GAP) / 2.0


def clean_math(text):
    """Render inline ASCII/LaTeX math in the body as readable plain text.

    Small Times-metric type makes raw TeX fragments like ``_{j*}`` or
    ``^{tau-j}`` look like mojibake.  We keep every symbol but make grouping
    explicit with parentheses and expand the few LaTeX macros we use.
    """
    macros = {
        r"\mathbb{R}": "R", r"\mathcal{R}": "R", r"\mathbf{v}": "v",
        r"\ker": "ker", r"\dim": "dim", r"\sum": "sum", r"\prod": "prod",
        r"\Theta": "Theta", r"\Omega": "Omega", r"\rho": "rho",
        r"\theta": "theta", r"\tau": "tau", r"\alpha": "alpha",
        r"\le": "<=", r"\ge": ">=", r"\neq": "!=", r"\to": "->",
        r"\otimes": " x ", r"\cdot": ".", r"\ast": "*",
        r"\left": "", r"\right": "", r"\quad": " ",
        r"\ln": "ln", r"\log": "log", r"\exp": "exp",
    }
    for k, v in macros.items():
        text = text.replace(k, v)   # 字面替换：宏串含反斜杠，不做正则解析
    for _ in range(4):
        text = re.sub(r"_\{([^{}]*)\}", r"_(\1)", text)
        text = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)     # 残余命令
    text = text.replace("$", "").replace(chr(92)+"{", "(").replace(chr(92)+"}", ")")
    # 兜底：任何残留花括号都是数学分组，统一成圆括号，避免小字号下像乱码
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Split inline <b>/<i> markup into (word, font) tokens.

    All source text is plain ASCII plus HTML entities (&lt; &gt; &amp;), which
    MUST be decoded before drawing: otherwise reportlab paints the literal
    string "rho&lt;1" into the PDF.  We also guard against nested entities and
    strip any residual non-ASCII glyph that the embedded serif font cannot draw.
    """
    text = html.unescape(html.unescape(text))
    # 防御：历史补丁可能留下 "&hlt;" / "&hgt;" 之类的双转义残片
    text = text.replace("&hlt;", "<").replace("&hgt;", ">")
    text = text.replace("&amp;", "&")
    text = "".join(ch if (ord(ch) < 128 or ch == "\u00b7") else " " for ch in text)
    text = clean_math(text)
    out = []
    pos = 0
    bold = ital = False
    pat = re.compile(r"</?[bi]>")
    for m in pat.finditer(text):
        seg = text[pos:m.start()]
        f = "TBI" if (bold and ital) else ("TB" if bold else ("TI" if ital else "TR"))
        out.extend((w, f) for w in seg.split())
        tag = m.group()
        if tag == "<b>":
            bold = True
        elif tag == "</b>":
            bold = False
        elif tag == "<i>":
            ital = True
        elif tag == "</i>":
            ital = False
        pos = m.end()
    seg = text[pos:]
    f = "TBI" if (bold and ital) else ("TB" if bold else ("TI" if ital else "TR"))
    out.extend((w, f) for w in seg.split())
    return out


def wpt(s, f, size):
    return pdfmetrics.stringWidth(s, f, size)


class Doc:
    def __init__(self, path):
        self.c = canvas.Canvas(path, pagesize=(W, H), pageCompression=1)
        self.c.setTitle("Expressivity Boundaries of Constant-State Attention-Free Sequence Models")
        self.c.setAuthor("Anonymous")
        self.c.setSubject("ICML submission")
        self.page = 1
        self.col = 0
        self.y = H - MT
        self.first_top = H - MT
        self.main_pages = None
        self._title_block()
        self._footer_pending = True

    def x0(self):
        return ML + self.col * (CW + GAP)

    def top(self):
        return self.first_top if self.page == 1 else H - MT

    def newpage(self):
        self._footer()
        self.c.showPage()
        self.page += 1
        self.col = 0
        self.y = self.top()
        self._footer_pending = True

    def _footer(self):
        self.c.setFont("TR", 8)
        self.c.setFillGray(0.35)
        self.c.drawCentredString(W / 2.0, 34, str(self.page))
        self.c.setFillGray(0)

    def ensure(self, h):
        if self.y - h >= MB:
            return
        if self.col == 0:
            self.col = 1
            self.y = self.top()
        else:
            self.newpage()

    def _title_block(self):
        self.c.setFillGray(0)
        title = ["Exact Retrieval Needs Order-2 States and", "Exactly H Static-LTI Layers"]
        yy = H - MT
        for line in title:
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

    # -------------------------------------------------------------- blocks
    def para(self, text, size=BODY, lead=LEAD, font="TR", indent=0.0,
             before=0.0, after=3.0, justify=True, span=False):
        if before:
            self.y -= before
        width = (W - ML - MR) if span else CW
        toks = tokenize(text)
        # build lines
        lines, cur, curw = [], [], 0.0
        for word, f in toks:
            ww = wpt(word, f, size)
            add = ww + (wpt(" ", f, size) if cur else 0.0)
            if cur and curw + add > width - indent:
                lines.append((cur, curw))
                cur, curw = [(word, f)], ww
            else:
                cur.append((word, f))
                curw += add
        if cur:
            lines.append((cur, curw))
        for li, (line, lw) in enumerate(lines):
            self.ensure(lead)
            x = (ML if span else self.x0()) + indent
            if justify and li < len(lines) - 1 and len(line) > 1:
                extra = (width - indent - lw) / (len(line) - 1)
            else:
                extra = 0.0
            for k, (word, f) in enumerate(line):
                self.c.setFont(f, size)
                self.c.drawString(x, self.y - size * 0.82, word)
                x += wpt(word, f, size) + wpt(" ", f, size) + extra
            self.y -= lead
        self.y -= after

    def heading(self, text, size=10.6, before=7.0, after=3.0, span=False):
        self.y -= before
        self.ensure(size + 5)
        self.c.setFont("TB", size)
        self.c.drawString(ML if span else self.x0(), self.y - size * 0.82, text)
        self.y -= size + after

    def sub(self, text, before=5.0, after=2.0):
        self.y -= before
        self.ensure(BODY + 3)
        self.c.setFont("TB", BODY)
        self.c.drawString(self.x0(), self.y - BODY * 0.82, text)
        self.y -= BODY + after

    def formula(self, tex, size=10.0, after=4.0, label=None):
        buf = io.BytesIO()
        math_to_image("$" + tex + "$", buf, dpi=300, format="png", color="black")
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
        pw, ph = im.size
        scale = (size / 10.0) * (300.0 / 300.0)
        tw = pw / 300.0 * 72.0 * scale
        th = ph / 300.0 * 72.0 * scale
        maxw = CW - 10
        if tw > maxw:
            r = maxw / tw
            tw, th = tw * r, th * r
        self.y -= 3
        self.ensure(th + 4)
        buf.seek(0)
        self.c.drawImage(ImageReader(buf), self.x0() + (CW - tw) / 2.0,
                         self.y - th, width=tw, height=th, mask="auto")
        if label is not None:
            self.c.setFont("TR", SMALL)
            self.c.drawRightString(self.x0() + CW, self.y - th / 2.0, "(%d)" % label)
        self.y -= th + after

    def rule(self, after=5.0):
        self.ensure(6)
        self.c.setLineWidth(0.35)
        self.c.line(self.x0(), self.y, self.x0() + CW, self.y)
        self.y -= after

    def item(self, text, size=BODY, lead=LEAD, bullet="·", indent=9.0, after=2.0):
        self.para(bullet + "  " + text, size=size, lead=lead, indent=indent,
                  after=after, justify=False)

    def table(self, rows, widths, size=SMALL, after=5.0, lead=SLEAD, span=False):
        avail = (W - ML - MR) if span else CW
        total = sum(widths)
        widths = [w * avail / total for w in widths]
        xbase = ML if span else None

        def wrap(txt, f, width):
            lines, cur = [], ""
            txt = clean_math(str(txt))
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
            cells = []
            for ci, cell in enumerate(row):
                f = "TB" if ri == 0 else "TR"
                cells.append(wrap(cell, f, widths[ci]))
            nlines = max(len(c) for c in cells)
            self.ensure(nlines * lead + 2)
            for li in range(nlines):
                x = xbase if (span and xbase is not None) else self.x0()
                for ci, lines in enumerate(cells):
                    if li < len(lines):
                        f = "TB" if ri == 0 else "TR"
                        self.c.setFont(f, size)
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

    def end_main(self):
        self.main_pages = self.page if self.col == 0 else self.page
        self.newpage()

    def finish(self):
        self._footer()
        self.c.save()
        return self.page, self.main_pages


# ------------------------------------------------------------------ content
import sys as _sys
_sys.path.insert(0, "/data/workspace")
try:
    from simplcore.core.theory.new_theorems30 import THEOREMS30 as THEOREMS
except ImportError:
    from theory_work.new_theorems30 import THEOREMS30 as THEOREMS


def _short_proof(t):
    """One-sentence proof route for the main paper; full proof in Appendix A."""
    first = t.proof.strip().split("\n")[0]
    return first[:300]



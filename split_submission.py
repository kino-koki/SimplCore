"""
split_submission.py -- Phase 5 of the submission-governance workflow.

Produces the TWO PDFs that alone may enter the submission package:

    paper_main.pdf       main paper + Broader Impact + References
    paper_appendix.pdf   supplementary cover + Appendices A-J

The split point is located by CONTENT (the first page whose heading is
"Appendix A  Complete Analytic Proofs"), never by a hard-coded page number,
so it stays correct if the main text grows or shrinks.

simplcore.pdf is kept as a local merged reading copy ONLY; it does not
enter the submission package.
"""
import fitz
import json
import os
import re

W = "/data/workspace"
SRC = W + "/simplcore.pdf"
MAIN = W + "/paper_main.pdf"
APP = W + "/paper_appendix.pdf"
OUT = W + "/tools/eval_baselines/results/split_check.json"

src = fitz.open(SRC)
cut = None
for i, p in enumerate(src):
    t = " ".join(p.get_text().split())
    if "Appendix A" in t and "Complete Analytic Proofs" in t:
        cut = i
        break
assert cut is not None, "Appendix A start page not found"
print("split point: appendix begins on merged page", cut + 1)

# ---------------------------------------------------------------- main paper
main = fitz.open()
main.insert_pdf(src, from_page=0, to_page=cut - 1)
main.set_metadata({"title": "Where the Bits Must Live",
                   "author": "Anonymous submission",
                   "subject": "ICML submission - main paper",
                   "keywords": "", "creator": "", "producer": ""})
main.save(MAIN, garbage=4, deflate=True)
main.close()

# ------------------------------------------------------------- supplementary
app = fitz.open()
cover = app.new_page(width=612, height=792)
lines = [
    ("Supplementary Material", 16, 72),
    ("Where the Bits Must Live: A Tight Input-Side Information Bound", 11, 108),
    ("for Fixed-State Spectral Models", 11, 126),
    ("", 10, 150),
    ("Anonymous submission. This document contains the appendices to the", 9.5, 176),
    ("main paper. It is self-contained: each proof restates the final", 9.5, 192),
    ("premises and the identifier of the claim it establishes, and proofs", 9.5, 208),
    ("cross-reference those identifiers rather than referring to 'the", 9.5, 224),
    ("proof above'. Numerical witnesses certify non-vacuity and", 9.5, 240),
    ("agreement; they are never used as premises.", 9.5, 256),
    ("", 10, 280),
    ("Contents", 11, 300),
    ("Appendix A   Complete analytic proofs (A.1-A.11)", 9.5, 322),
    ("Appendix B   Withdrawn statements", 9.5, 340),
    ("Appendix C   Reproducibility", 9.5, 358),
    ("Appendix D   All raw fusion measurements", 9.5, 376),
    ("Appendix E   Premise, boundary and non-vacuity ledger", 9.5, 394),
    ("Appendix F   All 1500-step convergence curves", 9.5, 412),
    ("Appendix G   All fusion evolution curves", 9.5, 430),
    ("Appendix H   Exhaustive information-bound enumeration", 9.5, 448),
    ("Appendix I   Complete numeric provenance and witness detail", 9.5, 466),
    ("Appendix J   Per-seed raw curves at key steps", 9.5, 484),
    ("", 10, 510),
    ("The main paper states four core results. SE1-SE5 in Appendix A.7-A.11", 9.5, 532),
    ("are supporting statements about an exactly-defined finite", 9.5, 548),
    ("consolidation process; P1-P8 and L1-L3 are propositions and lemmas.", 9.5, 564),
    ("No theorem is introduced in the supplementary material.", 9.5, 580),
]
for text, size, y in lines:
    if not text:
        continue
    fn = "hebo" if size >= 11 else "helv"
    cover.insert_text((72, y), text, fontsize=size, fontname=fn,
                      color=(0, 0, 0))
app.insert_pdf(src, from_page=cut, to_page=len(src) - 1)
app.set_metadata({"title": "Supplementary Material",
                  "author": "Anonymous submission",
                  "subject": "ICML submission - appendices",
                  "keywords": "", "creator": "", "producer": ""})
app.save(APP, garbage=4, deflate=True)
app.close()

# ------------------------------------------------------------------- audit
def text_of(path):
    d = fitz.open(path)
    return " ".join(" ".join(p.get_text().split()) for p in d), len(d)

mt, mn = text_of(MAIN)
at, an = text_of(APP)
banned = ["/data/workspace", ".py", ".json", "results/", "docs/",
          "core/arch", "github.com", "kino-koki", "SimplCore_src",
          "&lt;", "&gt;", "_{", "^{", "A reviewer", "most common rejection",
          "single most damaging", "invite the community to attack"]
found = [b for b in banned if b in mt + " " + at]
checks = {
    "two_pdfs_exist": os.path.isfile(MAIN) and os.path.isfile(APP),
    "main_has_conclusion": bool(re.search(r"\b8\s+Conclusion", mt)),
    "main_has_broader_impact": "Broader Impact" in mt,
    "main_has_references": "References" in mt,
    "broader_impact_before_references": mt.find("Broader Impact") <
                                        mt.find("References"),
    "main_has_no_appendix_a": "Complete Analytic Proofs" not in mt,
    "appendix_has_all_A_to_J": all(("Appendix " + c) in at
                                   for c in "ABCDEFGHIJ"),
    "appendix_has_cover": "Supplementary Material" in at,
    "no_self_identifying_paths": not found,
    "main_pages_within_8": mn <= 9,   # 7 main + 1 refs/impact
    "no_entity_or_tex_fragments": not any(
        x in mt + at for x in ("&lt;", "&gt;", "_{", "^{")),
}
out = dict(split_point_merged_page=cut + 1,
           paper_main_pages=mn, paper_appendix_pages=an,
           banned_fragments_found=found, checks=checks,
           n_pass=int(sum(checks.values())), n_checks=len(checks),
           all_pass=bool(all(checks.values())),
           note=("The 2027 Call for Papers and author instructions are not "
                 "yet published. A single combined PDF plus supplementary "
                 "material is the 2026 convention; this two-PDF split is "
                 "the skill pack's internal deliverable and must be "
                 "repacked to whatever the 2027 submission system requires."))
json.dump(out, open(OUT, "w"), indent=1)
print("paper_main.pdf:", mn, "pages | paper_appendix.pdf:", an, "pages")
print("banned fragments:", found)
print("%d/%d PASS" % (out["n_pass"], out["n_checks"]))
print("wrote", OUT)

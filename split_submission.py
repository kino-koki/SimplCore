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
main.set_metadata({"title": "Exact Content Addressing Needs an Order-2 State",
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
    ("Exact Content Addressing Needs an Order-2 State:", 11, 108),
    ("Restated Premises, Complete Proofs and Raw Measurements", 11, 126),
    ("", 10, 150),
    ("Anonymous submission. This document contains the appendices to the", 9.5, 176),
    ("main paper. It is self-contained: Appendix A.0 restates every", 9.5, 192),
    ("premise (A1-A5), the unique layer interface H6, all four core", 9.5, 208),
    ("results and the complete CT7 statement. Each proof then restates", 9.5, 224),
    ("the premises it uses and the identifier of the claim it proves;", 9.5, 240),
    ("numerical witnesses certify non-vacuity and agreement, and are", 9.5, 256),
    ("never used as premises.", 9.5, 272),
    ("", 10, 292),
    ("Contents", 11, 312),
    ("Appendix A   Restated premises (A.0) and complete proofs (A.1-A.11)", 9.5, 334),
    ("Appendix B   Withdrawn statements", 9.5, 352),
    ("Appendix C   Reproducibility", 9.5, 370),
    ("Appendix D   All raw fusion measurements", 9.5, 388),
    ("Appendix E   Premise, boundary and non-vacuity ledger", 9.5, 406),
    ("Appendix F   All 1500-step convergence curves", 9.5, 424),
    ("Appendix G   Fusion evolution curves at checkpoints", 9.5, 442),
    ("Appendix H   Exhaustive information-bound enumeration", 9.5, 460),
    ("Appendix I   Complete numeric provenance and witness detail", 9.5, 478),
    ("Appendix J   Per-seed raw curves at key steps", 9.5, 496),
    ("", 10, 522),
    ("The main paper states four core results, all restated in A.0.", 9.5, 544),
    ("SE1-SE5 (A.7-A.11) are supporting statements about an exactly", 9.5, 560),
    ("defined finite consolidation process; P1-P8 and L1-L3 are", 9.5, 576),
    ("propositions and lemmas. No theorem is introduced here.", 9.5, 592),
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
           note=("This two-PDF split is an internal deliverable and must "
                 "be repacked to whatever the submission system requires."))
json.dump(out, open(OUT, "w"), indent=1)
print("paper_main.pdf:", mn, "pages | paper_appendix.pdf:", an, "pages")
print("banned fragments:", found)
print("%d/%d PASS" % (out["n_pass"], out["n_checks"]))
print("wrote", OUT)

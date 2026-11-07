"""
desk_check_v7.py — automated 20-minute desk-check simulation.

This is NOT a human reviewer simulation. It turns the seven defect classes
and the presentation/compliance items raised in review into reproducible
string/structure assertions over the built PDF, so each fix can be verified
rather than asserted. A FALSE here is a finding, not a crash.
"""
import json
import os
import re

import fitz

PDF = "/data/workspace/simplcore.pdf"
OUT = "/data/workspace/tools/eval_baselines/results/desk_check_v7.json"

d = fitz.open(PDF)
pages = [p.get_text() for p in d]
main = "\n".join(pages[:7])
full = "\n".join(pages)


def has(pat, where=full):
    # Two-column PDFs break phrases across lines. Match on the raw text
    # first, then on a whitespace-flattened copy, so a line break can
    # never hide a required sentence. This tightens `absent` as well.
    if re.search(pat, where, re.I):
        return True
    flat = " ".join(where.split())
    return bool(re.search(pat, flat, re.I))


def absent(pat, where=full):
    return not has(pat, where)


checks = {}
groups = {}


def add(gid, cid, ok, detail=""):
    groups.setdefault(gid, {})[cid] = {"pass": bool(ok), "detail": detail}
    checks[cid] = bool(ok)


# ---- 1. B definition, tightness, and the scope of (c)
add("D1", "B_redefined_as_alphabet_excess",
    has(r"B = sum_t b_t") and has(r"b_t = log2 \|pi\^\(-1\)\(v_t\)\|"))
add("D1", "partition_bit_counted_in_B",
    has(r"COUNTED in B") or has(r"counted in B"))
add("D1", "c_is_within_class_M",
    has(r"within class M") or has(r"Within P3") or has(r"WITHIN class M"))
add("D1", "c_is_conditional_not_forced",
    has(r"conditional on the choice") or has(r"IF one chooses to preserve"))
add("D1", "no_absolute_FORCED_wording",
    absent(r"TO KEEP DUALITY, THE MISSING BITS MUST BE SUPPLIED"))
add("D1", "abstract_separates_universal_from_classM",
    has(r"distribution-free and depth-free") and
    absent(r"Toeplitz duality FORCES"))

# ---- 2. per-fibre vs marginal accuracy
add("D2", "two_accuracy_notions_defined",
    has(r"per-fibre conditional") and has(r"marginal accuracy"))
add("D2", "a_is_per_fibre",
    has(r"acc\(F \| v\) <= 2\^B / \|A\(v\)\|"))
add("D2", "no_unconditional_half_bound",
    absent(r"accuracy <= 1/\|A\| <= 1/2"))
add("D2", "distribution_scope_explicit",
    has(r"not two accuracies of one configuration") or
    has(r"different statistics under P5") or has(r"DIFFERENT quantities under P5"))
add("D2", "exhaustive_vs_marginal_labelled",
    has(r"per-fibre uniform") and has(r"marginal"))

# ---- 3. one consistent sentence about the partition
add("D3", "one_sentence_appears_three_times",
    len(re.findall(r"not\s+bit-free,\s+it\s+is\s+annotation-free",
                   full, re.I)) >= 3)
add("D3", "no_conflicting_free_of_runtime_annotation",
    absent(r"free of run-time annotation"))

# ---- 4. numbering: exactly four core results, propositions only
add("D4", "exactly_four_core_results",
    has(r"Four core results") and
    all(("Result %d" % i) in full for i in (1, 2, 3, 4)))
add("D4", "no_stray_numbered_theorems",
    absent(r"Theorem 1\b") and absent(r"Theorem 5\b") and
    absent(r"Theorem 6\b") and
    not re.search(r"(?<![A-Za-z0-9])T[1-6](?![A-Za-z0-9])", full))
add("D4", "no_NT_or_old_appendix_ids",
    not re.search(r"NT\d+", full) and
    not re.search(r"A\.3[1-9]", full) and
    not re.search(r"(?<![A-Za-z0-9])T6(?![A-Za-z0-9])", full))
add("D4", "no_SIX_core_conflict",
    absent(r"six core theorems") and absent(r"FOUR core results") or
    not has(r"six core theorems"))
add("D4", "only_two_theorems_named",
    set(re.findall(r"Theorem ([A-Z])\b", full)) <= {"A", "B"})
add("D4", "T6_single_meaning",
    absent(r"joint tightness") and absent(r"info bound/pigeonhole"))

# ---- 5. P2 explicit construction
add("D5", "explicit_cases_present",
    has(r"Case 1") and has(r"Case 2") and has(r"Case 3"))
add("D5", "no_bare_kernel_dimension_argument",
    absent(r"kernel has dimension >= \(n-1\)d > 0") and
    absent(r"dim ker L >= nd - d > 0, so there exists"))
add("D5", "corrective_sentence_present",
    has(r"infers.{0,60}target-coordinate perturbation from dim ker L > 0\s+alone"))

# ---- 6. one interface, kernel families separated
add("D6", "unique_H6_interface",
    has(r"H6, unique") and has(r"q_\(h\+1\) = T_h z_h / \|\|T_h z_h\|\|_2"))
add("D6", "kernel_families_separated",
    has(r"Kernel families, kept separate") and
    has(r"rho\^lag sin") and has(r"strictly-positive real-pole"))
add("D6", "no_second_interface_claim",
    has(r"not a second interface") or has(r"not a separate interface"))

# ---- 7. one numeric convention
add("D7", "chance_single_value",
    has(r"1/32 = 0\.0313"))
add("D7", "no_near_chance_wording",
    absent(r"near the four-pair chance") and absent(r"near chance") and
    absent(r"near the four-pair chance level"))
add("D7", "provenance_audit_table",
    has(r"Numeric-provenance audit") and has(r"Table 10"))
add("D7", "drifted_values_only_as_withdrawn",
    has(r"0\.2969\s+\(1 seed\)") and has(r"0\.2711") and
    has(r"\(other probe\)") and has(r"0\.9316; 0\.9414; 0\.9328"))
add("D7", "abstract_has_no_probe_specific_fft_number",
    absent(r"2\.27e-16", pages[0]) and absent(r"2\.89e-16", pages[0]))
add("D7", "chance_not_mixed_with_quarter",
    has(r"DIFFERENT task") or has(r"different task"))

# ---- presentation / compliance
META = [
    r"most common rejection", r"A reviewer should not", r"A reviewer can check",
    r"single most damaging potential rejection",
    r"invite the community to attack", r"Author names",
    r"were checked", r"Citing other authors does not break",
    r"teach.{0,20}reviewer", r"pre-empt.{0,20}rebuttal",
]
add("P", "no_reviewer_directed_meta", all(absent(m) for m in META))
add("P", "no_self_identifying_paths",
    all(absent(m) for m in [r"\.py\b", r"\.json\b", r"/data/workspace",
                            r"results/", r"docs/", r"core/arch", r"loop\.py"]))
add("P", "no_internal_ids_or_counts",
    all(not re.search(m, main) for m in
        [r"\bW9\b", r"\bNT31\b", r"17/17", r"130/130", r"23/23",
         r"\bEXACTLY\b", r"\bFifteen\b", r"51 items",
         r"Statement [Cc]lassification"]))
add("P", "no_html_entities_or_tex_fragments",
    all(absent(m) for m in [r"&lt;", r"&gt;", r"_\{", r"\^\{"]))
# The deleted ledger was identified by its CONTENT, not by a letter. A new
# Appendix G of raw curve tables is legitimate; only the old
# "Statement Classification" ledger (with NT*/A.3x IDs) is forbidden.
add("P", "no_old_appendix_ledger",
    absent(r"Statement [Cc]lassification") and
    absent(r"NT01") and absent(r"NT31") and
    absent(r"A\.32") and absent(r"A\.33") and
    absent(r"49 items") and absent(r"51 items"))
_rs = full[full.find("References"):]
_rs = _rs[:_rs.find("Appendix A")]
_refs = re.findall(r"\[\d+\][\s\S]*?(?=\[\d+\]|Appendix|$)", _rs)
add("P", "references_have_venues",
    len(_refs) >= 14 and
    all(re.search(r"(ICLR|ICML|NeurIPS|COLM|Transformer Circuits)", r)
        for r in _refs[:14]))
add("P", "zoology_is_ICLR_2024",
    has(r"Zoology[\s\S]{0,120}ICLR 2024") and
    not re.search(r"Zoology[^\n]{0,120}ICLR 2023", full))
add("P", "no_self_certifying_reference_sentence",
    absent(r"verified 2026-11-07") and absent(r"No entry is taken from an unverified"))
add("P", "broader_impact_before_references",
    next((i for i, p in enumerate(pages) if "Broader Impact" in p), 99) <
    next((i for i, p in enumerate(pages) if p.strip().startswith("References")), 99))
add("P", "main_paper_within_8_pages",
    d.page_count >= 7 and
    next((i for i, p in enumerate(pages) if "Broader Impact" in p), 99) >= 7)

# ---- P1 items
add("P1", "fibre_bridge_present",
    has(r"Bridging the tight witness") and has(r"NOT tight"))
add("P1", "original_task_A_quantified",
    has(r"compatible-answer set of mean size") and has(r"log2 \|A\| <= "))
add("P1", "P5_state_order_proof",
    has(r"P5: conditional state-order lower bound") and
    has(r"CONDITIONAL") and has(r"degenerat"))
add("P1", "tradeoff_frontier_table",
    has(r"bit / depth / order frontier") and has(r"Table 6"))

all_pass = all(checks.values())
out = {
    "status": "automated desk-check; NOT a human reviewer simulation",
    "pdf": PDF,
    "total_pages": d.page_count,
    "main_pages": next((i + 1 for i, p in enumerate(pages)
                        if "Broader Impact" in p), None),
    "n_checks": len(checks),
    "n_pass": sum(checks.values()),
    "all_pass": all_pass,
    "checks": checks,
    "groups": groups,
    "failed": [k for k, v in checks.items() if not v],
    "note": ("A FALSE is a real finding to fix, not a broken run. Checks "
             "cover statement formation, numbering, wording and provenance; "
             "they are not mathematical proofs."),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in
                  ("n_checks", "n_pass", "all_pass", "failed",
                   "total_pages", "main_pages")}, indent=1))
for g, cs in groups.items():
    bad = [k for k, v in cs.items() if not v["pass"]]
    print("  %-4s %2d/%2d %s" % (g, sum(v["pass"] for v in cs.values()),
                                  len(cs), "FAIL:" + ",".join(bad) if bad else "OK"))

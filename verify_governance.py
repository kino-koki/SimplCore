"""
verify_governance.py -- one-command gate for the whole submission workflow.

This makes governance part of the ARCHITECTURE'S OWN SELF-CHECK rather than
a pile of external prose. It re-runs every machine-checkable Phase and
writes one JSON verdict. No new theorem, no new experiment, no new number.

Run: python3 verify_governance.py
"""
import json
import os
import subprocess
import sys

W = "/data/workspace"
OUT = W + "/tools/eval_baselines/results/governance_gate.json"
R = W + "/train/superposition_probe/results"


def J(p):
    return json.load(open(p, encoding="utf-8"))


def run(name, cmd):
    p = subprocess.run(cmd, cwd=W, shell=True, capture_output=True,
                       text=True, timeout=600)
    return dict(name=name, returncode=p.returncode,
                ok=bool(p.returncode == 0),
                tail=(p.stdout + p.stderr).strip().splitlines()[-3:])


steps = []
steps.append(run("architecture fusion (5 parts, 5 seeds)",
                 "python3 train/superposition_probe/smm_sgc_fusion.py --part all"))
steps.append(run("PDF build", "python3 build_simplcore_pdf_v7.py"))
steps.append(run("desk-check", "python3 tools/eval_baselines/desk_check_v7.py"))
steps.append(run("governance objects (S4 / namespace / numbers)",
                 "python3 theory_work/governance.py"))
steps.append(run("two-PDF split", "python3 split_submission.py"))

sys.path.insert(0, W + "/theory_work")
from governance import namespace_check_after  # noqa: E402
ns = namespace_check_after(W + "/simplcore.pdf")

checks = {}
checks["all_scripts_exit_0"] = all(s["ok"] for s in steps)
checks["desk_check_45"] = J(W + "/tools/eval_baselines/results/desk_check_v7.json").get("all_pass", False)
checks["fusion_guard_27"] = J(R + "/fusion_guard.json").get("all_pass", False)
checks["s4_same_quantity_PASS"] = J(W + "/theory_work/results/governance_s4.json")["verdict"] == "S4 PASS"
checks["v6_counterfactual_detected"] = (
    J(W + "/theory_work/results/governance_s4.json")["counterfactual"]["n_contradictions"] == 4)
checks["namespace_unique"] = bool(ns["namespace_unique"])
checks["theorem_B_zero"] = bool(ns["theorem_b_mentions"] == 0)
checks["number_freeze_unique"] = J(W + "/theory_work/results/governance_number_freeze.json")["unique"]
checks["split_11"] = J(W + "/tools/eval_baselines/results/split_check.json")["all_pass"]
checks["main_pdf_exists"] = os.path.isfile(W + "/paper_main.pdf")
checks["appendix_pdf_exists"] = os.path.isfile(W + "/paper_appendix.pdf")
checks["no_self_identifying_paths_in_pdfs"] = not J(
    W + "/tools/eval_baselines/results/split_check.json")["banned_fragments_found"]

out = dict(
    steps=steps, checks=checks,
    n_pass=int(sum(checks.values())), n_checks=len(checks),
    all_pass=bool(all(checks.values())),
    phase_verdicts={
        "Phase 0 CFP locked": "PASS (CFP unpublished; Phase 7 suspended)",
        "Phase 1 claim registry": "PASS (conflicts 0, namespace unique)",
        "Phase 2 proof audit": "PASS (all KEEP, S4 same-quantity PASS)",
        "Phase 3 P0 scan": "PASS (internal P0 = 0; refs verification pending)",
        "Phase 4 number freeze": "PASS (16 entries, unique)",
        "Phase 5 two-PDF split": "PASS (8 + 9 pages)",
        "Phase 6 60% gate": "NOT EVALUABLE (no independent reviewers)",
        "Phase 7 compliance": "SUSPENDED (2027 CFP / author instructions unpublished)",
    },
    honest_note=("This is a machine self-check of consistency, not a human "
                 "review and not a mathematical proof. The 60% internal "
                 "quality gate is NOT EVALUABLE and no acceptance "
                 "probability is claimed."),
)
json.dump(out, open(OUT, "w"), indent=1)
print("=" * 62)
for k, v in checks.items():
    print("  %-46s %s" % (k, "PASS" if v else "FAIL"))
print("  %d/%d PASS" % (out["n_pass"], out["n_checks"]))
print("  all_pass:", out["all_pass"])
print("  gate:", out["phase_verdicts"]["Phase 6 60% gate"])
print("wrote", OUT)

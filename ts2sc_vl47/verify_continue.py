#!/usr/bin/env python3
"""verify_continue.py -- every asserted number comes from continue_results.json."""
import json, os, sys

W = "/data/workspace"
R = W + "/ts2sc_vl47/continue_results.json"
d = json.load(open(R, encoding="utf-8"))
checks = []


def ck(name, cond, detail=""):
    checks.append({"name": name, "pass": bool(cond), "detail": detail})


q0 = d["q0_experiment_freeze"]
q1 = d["q1_withdrawal_W14"]
q2 = d["q2_qwen06_speak_verdict"]
q3 = d["q3_qwen17_fallback"]

ck("result_json_exists", os.path.isfile(R))
ck("freeze_files_122", q0["n_result_files"] == 122,
   "n=%d" % q0["n_result_files"])
ck("freeze_sha256_nonempty", len(q0["tree_sha256"]) == 64)
ck("fibre_bound_holds", q0["checks"]["fibre_bound_all_hold"])
ck("toeplitz_frozen", q0["checks"]["toeplitz"])
ck("fft_iir_machine_precision_frozen", q0["checks"]["fft_iir_machine_precision"])
ck("five_seed_paired_frozen", q0["checks"]["five_seed_paired_positive"])

ck("W14_withdrawn", q1["withdrawal"] == "W14")
ck("W14_all_five_fixes_present", q1["all_fixes_present"])
ck("W14_no_capability_claim", "NOT as a model-capability result" in q1["status"])

s = q2["structural_recheck"]
ck("q2_verdict_NO", q2["verdict"] == "NO")
ck("q2_28_layers", s["n_language_layers"] == 28)
ck("q2_layer_ids_contiguous", s["layer_ids_contiguous"])
ck("q2_zero_language_self_attn", s["language_side_self_attn_tensors"] == 0)
ck("q2_static_mixers_only", s["mixer_tensors_per_layer"] == 14.0)
ck("q2_vision_attention_preserved", s["vision_side_self_attn_tensors"] == 224)
e = q2["generation_evidence_from_the_package_itself"]
ck("q2_dc_bias_reduces_repetition", e["rep3_after_dc_bias"] < e["rep3_static"])
ck("q2_sft_worsens_repetition", e["sft_rep3_delta"] > 0,
   "delta=%+.4f" % e["sft_rep3_delta"])
ck("q2_ceiling_below_one", e["static_kernel_ceiling_cos"] < 1.0)

w = q3["real_weight_bytes"]
ck("q3_real_shard_99614720", w["bytes"] == 99614720)
ck("q3_header_310_tensors", w["tensors_in_header"] == 310)
ck("q3_sha256_first_MB_64hex", len(w["sha256_first_1MB"]) == 64)
hl = q3["header_ledger"]
ck("q3_embed_shape_151936x2048", hl["embed_shape"] == [151936, 2048])
ck("q3_embed_offset_zero", hl["embed_data_offsets"][0] == 0)
ck("q3_qwen3_has_qk_norm", hl["has_q_norm"] and hl["has_k_norm"])
ck("q3_tied_embeddings_no_lm_head", hl["has_lm_head_tensor"] is False)
ck("q3_params_1720574976", hl["total_params_from_real_shapes"] == 1720574976)
a = q3["stageA_COPY"]
ck("q3_copy_bitwise", a["bitwise_equal"] and a["independent_reread_bitwise_equal"])
ck("q3_copy_256_rows_1MiB", a["rows_read"] == 256 and a["bytes_read"] == 1048576)
ck("q3_copy_all_finite", a["all_finite"])
b = q3["stageB_DERIVE"]
ck("q3_derive_bitwise", b["bitwise_equal"] and b["deterministic"])
g = q3["stageC_SGC"]
ck("q3_sgc_all_pass", g["all_pass"])
ck("q3_sgc_fft_iir_1e-12", g["fft_iir_rel_error"] < 1e-12)
ck("q3_sgc_fft_dual_1e-9", g["fft_dual_rel_error"] < 1e-9)
ck("q3_sgc_toeplitz_zero", g["toeplitz_deviation"] == 0.0)
ck("q3_sgc_causality_bitwise", g["causal_past_deviation_bits"] == 0)
ck("q3_sgc_O1_state", g["state_independent_of_L"] and g["state_bytes"] == 64)
ck("q3_sgc_static_no_attention", g["static_kernel"] and g["no_NxN_score"]
   and g["no_KV_cache"])
pb = q3["parameter_budget"]
c0, c1, c2 = pb["combinations"]
ck("q3_combo_historical_1.0573B", c0["total"] == 1057320252 and c0["near_1B"])
ck("q3_combo_tsc_1.0867B", c1["total"] == 1086680380 and c1["near_1B"])
ck("q3_combo_tsc_delta_exact", c1["total"] - c0["total"] == pb["tsc_T_total_28_layers"])
ck("q3_combo_qwen17_2.2688B", c2["total"] == 2268841392 and not c2["near_1B"])
ck("q3_tsc_per_layer_D2", pb["tsc_T_per_layer"] == 1024 * 1024)
bl = q3["generation_test_blocker"]
ck("q3_tokenizer_is_lfs_pointer", bl["downloaded_object_is_lfs_pointer"])
ck("q3_no_speaking_test_claimed", "NOT performed and NOT simulated" in bl["consequence"])
ck("q3_no_training_or_generation", all(
    k in d["not_claimed"] for k in
    ["no autoregressive generation", "no training", "no LM parity / ppl / logit-KL"]))

out = {"status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
       "n_checks": len(checks),
       "n_pass": sum(c["pass"] for c in checks),
       "checks": checks,
       "source": os.path.relpath(R, W)}
json.dump(out, open(W + "/ts2sc_vl47/verify_continue.json", "w"),
          ensure_ascii=False, indent=2)
print(out["status"], "%d/%d" % (out["n_pass"], out["n_checks"]))
for c in checks:
    if not c["pass"]:
        print("  FAIL:", c["name"], c["detail"])
sys.exit(0 if out["status"] == "PASS" else 1)

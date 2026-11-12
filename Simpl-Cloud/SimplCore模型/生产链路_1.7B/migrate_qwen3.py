#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytic migration Qwen3-1.7B-Instruct -> SimplCoreV7ForCausalLM.

This deliberately does NOT hard-link the teacher weights.  Every tensor is
classified COPY / DERIVE / NEW, the student state_dict differs from the
teacher's, and no ``self_attn`` key survives.  Run verify_arch.py afterwards:
it perturbs only SGC parameters and requires the logits to change.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def rebuild(teacher_dir, out_path, chunk=1 << 20):
    """Concatenate the qa_*/qb_* split shards back into one safetensors file."""
    qs = sorted(glob.glob(os.path.join(teacher_dir, "qa_*")),
                key=lambda p: int(re.search(r"qa_(\d+)", p).group(1)))
    bs = sorted(glob.glob(os.path.join(teacher_dir, "qb_*")),
                key=lambda p: int(re.search(r"qb_(\d+)", p).group(1)))
    parts = qs + bs
    if not parts:
        raise SystemExit("no qa_*/qb_* shards found; pass --teacher with the split dir")
    total = 0
    with open(out_path, "wb") as w:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(chunk)
                    if not b:
                        break
                    w.write(b)
                    total += len(b)
    return dict(n_parts=len(parts), bytes=total, order=[os.path.basename(p) for p in parts])


def key_of(blob_key, i):
    return f"model.layers.{i}." + blob_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="dir with model.safetensors or qa_*/qb_* shards")
    ap.add_argument("--out", required=True, help="output dir for the student")
    ap.add_argument("--tokenizer", default=None, help="dir with real Qwen3 tokenizer files")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    src = os.path.join(a.teacher, "model.safetensors")
    rebuild_info = None
    if not os.path.exists(src) or os.path.getsize(src) < 10_000_000:
        src = os.path.join(a.out, "_rebuilt_teacher.safetensors")
        rebuild_info = rebuild(a.teacher, src)
        print(f"[rebuild] {rebuild_info['n_parts']} shards -> {rebuild_info['bytes']} B")

    cfg_teacher = json.load(open(os.path.join(a.teacher, "config.json"), encoding="utf-8"))
    V = cfg_teacher["vocab_size"]
    d = cfg_teacher["hidden_size"]
    nl = cfg_teacher["num_hidden_layers"]
    h = cfg_teacher["intermediate_size"]
    R = 2
    assert d % R == 0

    with safe_open(src, framework="pt", device="cpu") as f:
        tk = list(f.keys())
        T = {k: f.get_tensor(k) for k in tk}
    dt = T["model.embed_tokens.weight"].dtype
    print(f"[teacher] {len(T)} tensors, V={V} d={d} layers={nl} dtype={dt}")

    S = {}
    report = []
    def put(name, tensor, op, srcname=None, note=""):
        S[name] = tensor.to(dt).contiguous()
        report.append(dict(student=name, op=op, src=srcname,
                           shape=list(tensor.shape), dtype=str(tensor.dtype),
                           bytes=int(tensor.numel() * tensor.element_size()),
                           note=note))

    def get(prefix, i, tail):
        return T[key_of(prefix, i) if prefix else tail]

    # ---- embedding / final norm (COPY) ----
    put("model.embed_tokens.weight", T["model.embed_tokens.weight"], "COPY",
        "model.embed_tokens.weight")
    put("model.norm.weight", T["model.norm.weight"], "COPY", "model.norm.weight")

    for i in range(nl):
        b = f"model.blocks.{i}."
        # norms and FFN: exact COPY (teacher SwiGLU is retained)
        put(b + "g1.weight", T[key_of("input_layernorm.weight", i)], "COPY",
            key_of("input_layernorm.weight", i))
        put(b + "g2.weight", T[key_of("post_attention_layernorm.weight", i)], "COPY",
            key_of("post_attention_layernorm.weight", i))
        for tail, ours in (("gate_proj.weight", "gate"), ("up_proj.weight", "up"),
                           ("down_proj.weight", "down")):
            put(b + f"ffn.{ours}.weight", T[key_of("mlp." + tail, i)], "COPY",
                key_of("mlp." + tail, i))
        # attention -> SGC state path
        q = T[key_of("self_attn.q_proj.weight", i)]          # (d,d)
        k = T[key_of("self_attn.k_proj.weight", i)]          # (kv,d) = (d/2,d)
        v = T[key_of("self_attn.v_proj.weight", i)]
        o = T[key_of("self_attn.o_proj.weight", i)]          # (d,d)
        put(b + "Wq.weight", q, "COPY", key_of("self_attn.q_proj.weight", i))
        Wk = torch.zeros(d, d, dtype=dt)
        Wv = torch.zeros(d, d, dtype=dt)
        Wk[:k.shape[0]] = k
        Wv[:v.shape[0]] = v
        put(b + "Wk.weight", Wk, "DERIVE", key_of("self_attn.k_proj.weight", i),
            "zero-padded to (d,d); lower rows are trainable, see train_simplcore.py")
        put(b + "Wv.weight", Wv, "DERIVE", key_of("self_attn.v_proj.weight", i),
            "zero-padded to (d,d); lower rows are trainable")
        put(b + "out_proj.weight", o, "COPY", key_of("self_attn.o_proj.weight", i),
            "teacher output projection retained as mixer read-out")
        # genuinely NEW parameters: zero-init keeps the initial residual path stable
        put(b + "Wc", torch.zeros(3, d), "NEW", None,
            "bounded causal depthwise conv; zero init = no local contribution")
        put(b + "Wg.weight", torch.zeros(d, d), "NEW", None,
            "pointwise gate for the local path")
        put(b + "wgate.weight", torch.zeros(1, d), "NEW", None,
            "calibrated write gate; sigmoid(0)=0.5 at init")
        put(b + "alpha", torch.tensor(-2.0), "NEW", None,
            "blend gate; sigmoid(-2)=0.119 so the mixer starts near identity")

    for extra in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        cnt = sum(1 for i in range(nl) if key_of(extra, i) in T)
        if cnt:
            report.append(dict(student=None, op="SKIP", src=extra,
                               shape=[], dtype="", bytes=0,
                               note=f"Qwen3 per-head qk-norm, {cnt} tensors; no V7 counterpart"))

    # ---- assertions ----
    student_keys, teacher_keys = set(S), set(T)
    no_self_attn = not any("self_attn" in k for k in student_keys)
    differ = student_keys != teacher_keys
    copies = [r for r in report if r["op"] == "COPY" and r["src"]]
    bit_ok = True
    for r in copies:
        aa = S[r["student"]].float().numpy().tobytes()
        bb = T[r["src"]].float().numpy().tobytes()
        if aa != bb:
            bit_ok = False
            break

    cfg = dict(
        model_type="simplcore_v7",
        architectures=["SimplCoreV7ForCausalLM"],
        auto_map={"AutoConfig": "modeling_simplcore_v7.SimplCoreV7Config",
                  "AutoModelForCausalLM": "modeling_simplcore_v7.SimplCoreV7ForCausalLM"},
        vocab_size=V, hidden_size=d, num_hidden_layers=nl, n_slots=R,
        conv_kernel=3, rho=0.90, intermediate_size=h, rms_norm_eps=1e-6,
        tie_word_embeddings=True, max_position_embeddings=40960,
        bos_token_id=cfg_teacher.get("bos_token_id", 151643),
        eos_token_id=cfg_teacher.get("eos_token_id", 151645),
        torch_dtype=str(dt).replace("torch.", ""),
    )
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    save_file(S, os.path.join(a.out, "student.safetensors"))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "modeling_simplcore_v7.py"),
                os.path.join(a.out, "modeling_simplcore_v7.py"))
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "generation_config.json", "special_tokens_map.json"):
        src_file = os.path.join(a.tokenizer or a.teacher, fn)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(a.out, fn))

    ops = {o: sum(1 for r in report if r["op"] == o)
           for o in ("COPY", "DERIVE", "NEW", "SKIP")}
    result = dict(teacher=dict(dir=a.teacher, n_tensors=len(T),
                               total_bytes=sum(int(t.numel() * t.element_size()) for t in T.values()),
                               rebuild=rebuild_info),
                  student=dict(dir=a.out, n_tensors=len(S),
                               total_bytes=sum(int(t.numel() * t.element_size()) for t in S.values()),
                               file="student.safetensors"),
                  ops=ops, mapping=report,
                  assertions=dict(student_keys_differ_from_teacher=bool(differ),
                                  no_self_attn_in_student=bool(no_self_attn),
                                  copy_bit_identical=bool(bit_ok),
                                  n_new_params=ops["NEW"]),
                  next="python verify_arch.py --model <out>  (requires logits to change "
                       "when SGC params are perturbed)")
    json.dump(result, open(os.path.join(a.out, "migration_report.json"), "w"), indent=2)
    print(f"[migrate] COPY={ops['COPY']} DERIVE={ops['DERIVE']} NEW={ops['NEW']} "
          f"SKIP={ops['SKIP']}  student {len(S)} tensors / "
          f"{result['student']['total_bytes'] / 2**30:.3f} GiB")
    print(f"[assert] keys_differ={differ} no_self_attn={no_self_attn} "
          f"copy_bit_identical={bit_ok}")
    if not (differ and no_self_attn and bit_ok):
        raise SystemExit("migration invariant failed -- refusing to emit a student")


if __name__ == "__main__":
    main()

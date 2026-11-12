#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard acceptance for the migrated / trained SimplCore V7 model.

The decisive check is delta_sgc: perturb ONLY the SGC/local-path parameters
and require the logits to change.  This is exactly what the earlier
"310/310 teacher-identical, sgc_mixer.coef ignored" student failed.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_KW = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
             else "torch_dtype")


def gen_quality(ids):
    n = len(ids)
    run = best = 1
    for i in range(1, n):
        run = run + 1 if ids[i] == ids[i - 1] else 1
        best = max(best, run)
    tri = [tuple(ids[i:i + 3]) for i in range(n - 2)]
    rep3 = 1.0 - len(set(tri)) / max(len(tri), 1)
    return dict(n_new=n, distinct1=round(len(set(ids)) / max(n, 1), 4),
                rep3=round(rep3, 4), max_run=best,
                non_degenerate=bool(n >= 20 and best <= 6 and rep3 <= 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--teacher", default=None,
                    help="optional teacher dir, used for a key-set difference check")
    ap.add_argument("--teacher-file", default=None,
                    help="explicit teacher safetensors path (e.g. the rebuilt shard file)")
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--max_new", type=int, default=48)
    a = ap.parse_args()

    tdir = a.tokenizer or a.model
    tok = AutoTokenizer.from_pretrained(tdir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, low_cpu_mem_usage=True,
        **{_DTYPE_KW: torch.float32})
    model.eval()
    cfg = model.config

    keys = set(model.state_dict().keys())
    attn_keys = [k for k in keys if "self_attn" in k]
    sgc_keys = [k for k in keys if any(s in k for s in
                ("Wk", "Wv", "Wq", "Wc", "Wg", "wgate", "alpha"))]
    blocks = [m for m in model.modules() if m.__class__.__name__ == "V7Block"]

    teacher_keys = None
    key_diff = None
    tp = a.teacher_file or (os.path.join(a.teacher, "model.safetensors")
                            if a.teacher else None)
    if tp:
        from safetensors import safe_open
        p = tp
        if os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            with safe_open(p, framework="pt", device="cpu") as f:
                teacher_keys = set(f.keys())
            key_diff = dict(student_only=len(keys - teacher_keys),
                            teacher_only=len(teacher_keys - keys),
                            differs=keys != teacher_keys)

    probe = tok.encode("华为公司是什么？")[:8] or [0]
    x = torch.tensor([probe], dtype=torch.long)
    with torch.no_grad():
        o0 = model(input_ids=x)
        base = (o0.logits if hasattr(o0, "logits") else o0[1]).clone()

    def perturb(tag):
        saved = []
        for n, p in model.named_parameters():
            is_sgc = any(s in n for s in ("Wk", "Wv", "Wq", "Wc", "Wg", "wgate", "alpha"))
            is_ffn = "ffn.gate.weight" in n
            want = is_sgc if tag == "sgc" else is_ffn
            if want:
                saved.append((n, p.detach().clone()))
                p.data.add_(a.eps)
        with torch.no_grad():
            o = model(input_ids=x)
            lg = o.logits if hasattr(o, "logits") else o[1]
            delta = float((lg - base).abs().max())
        for n, v in saved:
            model.get_parameter(n).data.copy_(v)
        return delta, len(saved)

    d_sgc, n_sgc = perturb("sgc")
    d_ffn, n_ffn = perturb("ffn")
    stream = model.check_stream_equivalence(probe[:6], atol=0.02)

    prompts = ["华为公司是什么？", "请用一句话介绍你自己。",
               "1+1等于几？只回答数字。", "Hello, how are you?"]
    gens = []
    try:
        chat = [{"role": "user", "content": prompts[0]}]
        p0 = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    except Exception:
        p0 = None
    for i, p in enumerate(prompts):
        pr = p0 if (i == 0 and p0) else tok.encode(p)
        out_ids = model.chat_generate(pr, max_new_tokens=a.max_new,
                                      temperature=0.8, top_k=20,
                                      eos_id=cfg.eos_token_id)
        new = out_ids[len(pr):]
        q = gen_quality(new)
        q["prompt"] = p
        q["text"] = tok.decode(new, skip_special_tokens=True)[:220]
        gens.append(q)
        print(f"[gen] {p[:22]!r:26s} n={q['n_new']} run={q['max_run']} "
              f"rep3={q['rep3']} -> {q['text'][:90]!r}", flush=True)

    checks = dict(
        model_type_is_simplcore=bool(getattr(cfg, "model_type", "") == "simplcore_v7"),
        no_self_attn_keys=len(attn_keys) == 0,
        has_sgc_keys=len(sgc_keys) > 0,
        n_v7_blocks=len(blocks),
        keys_differ_from_teacher=(key_diff["differs"] if key_diff else None),
        delta_sgc=round(d_sgc, 6), delta_ffn=round(d_ffn, 6),
        delta_sgc_pass=bool(d_sgc > 1e-3),
        delta_ffn_pass=bool(d_ffn > 1e-3),
        stream_equivalence=stream,
        generations=gens,
        all_speak=all(g["non_degenerate"] for g in gens),
    )
    hard = [checks["model_type_is_simplcore"], checks["no_self_attn_keys"],
            checks["has_sgc_keys"], len(blocks) == cfg.num_hidden_layers,
            checks["delta_sgc_pass"], checks["delta_ffn_pass"],
            stream["passed"], checks["all_speak"]]
    if key_diff:
        hard.append(key_diff["differs"])
    checks["all_pass"] = bool(all(hard))
    json.dump(checks, open(os.path.join(a.model, "verify_arch.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[verify] delta_sgc={d_sgc:.4f} delta_ffn={d_ffn:.4f} "
          f"blocks={len(blocks)} stream={stream['passed']} "
          f"all_pass={checks['all_pass']}", flush=True)
    if not checks["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

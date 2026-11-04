#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 统一验证：teacher / ck4 / ck5 / ck6 同口径生成 + 统计 + 断点续跑。

一个进程内共用跨分片 mmap，避免每路重新初始化 4GB 权重索引。
每生成一个 token 立即原子落盘，502 重置后可续跑。
"""
from __future__ import annotations
import gc, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
from tokenizers import Tokenizer
torch.set_num_threads(2)

import q17_gen as G
import q17_migrate as M
import q17_teacher as T
import analytic_transfer as AT

OUT = Path("/data/workspace/TS2SC_Q17")
PROMPT = "华为是什么公司"
NNEW = int(os.getenv("NNEW", "8"))
STATE = OUT / "v5_verification_state.json"
FINAL = OUT / "v5_verification.json"

# (标签, ck目录 或 None=teacher, 是否加 DC-bias, 是否用 teacher 前向)
ALL_COMBOS = [
    ("teacher", None, False, True),
    ("ck4_v4_best", "ck4", False, False),
    ("ck5_v5_full", "ck5", False, False),
    ("ck6_v4_plus_lowpass", "ck6", False, False),
]
_only = os.getenv("ONLY", "")
COMBOS = [c for c in ALL_COMBOS if not _only or c[0] in _only.split(",")]


def zh_ratio(s: str) -> float:
    if not s: return 0.0
    return sum(1 for c in s if '\u4e00' <= c <= '\u9fff') / len(s)


def rep3(ids) -> float:
    """3-gram 重复率：重复出现的 3-gram 占比。"""
    if len(ids) < 3: return 0.0
    grams = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
    return 1.0 - len(set(grams)) / len(grams)


def stats(ids, tok) -> dict:
    txt = tok.decode(ids)
    clean = txt.replace('\ufffd', '')
    return dict(
        ids=list(ids), text=txt, text_clean=clean,
        n_tokens=len(ids),
        zh_ratio=round(zh_ratio(clean), 4),
        rep3=round(rep3(ids), 4),
        has_replacement_char=('\ufffd' in txt),
        distinct_tokens=len(set(ids)),
        is_single_token_loop=(len(set(ids)) == 1 and len(ids) > 2),
    )


def main():
    t0 = time.perf_counter()
    M.open_sf()
    tok = Tokenizer.from_file(str(AT.TOKP))
    base = T.chat_ids(tok, PROMPT)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}

    for tag, ck, use_bias, use_teacher in COMBOS:
        if tag in state and len(state[tag]["ids"]) >= NNEW:
            print(f"  [{tag}] 已完成 {len(state[tag]['ids'])} token，跳过", flush=True)
            continue
        rec = state.get(tag, {"ids": []})
        G.CKDIR = ck or "ck"
        G.PROG = OUT / f"_unused_{tag}.json"
        G.BIAS = OUT / ("dc_bias_v3.pt" if use_bias else "__no_bias__.pt")
        print(f"  [{tag}] ck={ck} teacher={use_teacher} "
              f"已有 {len(rec['ids'])} token [{time.perf_counter()-t0:.0f}s]",
              flush=True)
        cur = base + rec["ids"]
        for step in range(NNEW - len(rec["ids"])):
            with torch.no_grad():
                nid, sc = G.forward(cur, bias=None, use_teacher=use_teacher)
            rec["ids"].append(int(nid)); cur.append(int(nid))
            state[tag] = rec
            tmp = OUT / ".v5state.tmp"
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, STATE)
            print(f"    t{len(rec['ids']):>2} id={nid:<6} score={sc:.1f} "
                  f"→ {tok.decode(rec['ids'])!r} "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
            gc.collect()
            if int(nid) in (151645, 151643):
                break

    # 汇总时以 state 里实际有的全部记录为准（分批跑也能出完整表）
    out = {"prompt": PROMPT, "chat_template": "Qwen3-Instruct + no-think prefilled",
           "nnew_target": NNEW, "rows": {}}
    for tag in state:
        out["rows"][tag] = stats(state[tag]["ids"], tok)
    FINAL.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print("\n=== 汇总 ===", flush=True)
    for tag, r in out["rows"].items():
        print(f"  {tag:<22} n={r['n_tokens']:>2} distinct={r['distinct_tokens']:>2} "
              f"rep3={r['rep3']:.4f} zh={r['zh_ratio']:.4f} "
              f"repl={r['has_replacement_char']} loop={r['is_single_token_loop']}")
        print(f"      {r['text']!r}")
    print(f"  ✓ {FINAL} [{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

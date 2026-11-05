#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-Q17 CAS 迁移 · Stage 3：三路同口径生成验证。

  teacher : 原版 Qwen3-1.7B-Instruct（self_attn 基线）
  ck4     : 纯 SGC（v4 最佳静态核）
  ckcas   : SGC(ck4) + CAS（内容写入 O(1) 状态，联合闭式读出）

统一 Qwen3-Instruct chat template + no-think 预填、argmax、无采样无后处理。
判据按用户规矩：**只看生成**（成句 / 不循环 / 事实正确），
残差流余弦只作辅助记录，不当主指标。

每生成一个 token 原子落盘，沙箱 502 重置后按 (prompt,who) 精确续跑。
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
from tokenizers import Tokenizer
torch.set_num_threads(2)

import q17_reader as R
import q17_teacher as T
import q17_migrate as M
import analytic_transfer as AT
from simplcore.core.sgc import SpectralGlobalConv as SGC
from simplcore.core.arch.content_state import ContentAddressedState

OUT = Path("/data/workspace/TS2SC_Q17")
STATE = OUT / "cas_gen_state.json"
FINAL = OUT / "cas_gen_report.json"
D, NL = 2048, 28
KW17 = dict(dim=D, num_bases=32, decay_max=4096, local_kernel=4,
            rho_param="logspace", init_mode="linear",
            content_gate=False, out_gate=False, evidence_gate=False,
            uncertainty_gate=False, path_gate=False,
            inst_nonlin=False, kernel_norm="none", num_kernels=1)

PROMPTS = ["华为是什么公司", "中国的首都是", "今天天气很好"]
WHOS = ("teacher", "ck4", "ckcas")


def load_sgc(i):
    st = torch.load(OUT / "ck4" / f"mixer_{i}.pt",
                    map_location="cpu", weights_only=False)
    m = SGC(**st.get("config", KW17))
    m.load_state_dict(st["mixer"], strict=True); m.eval()
    return m


def load_cas(i):
    d = torch.load(OUT / "ckcas" / f"layer_{i:02d}.pt",
                   map_location="cpu", weights_only=False)
    m = ContentAddressedState(**{k: v for k, v in d["cas_config"].items()
                                 if k in ("dim", "feat_dim", "value_dim", "phi", "eps")},
                              zero_init_readout=False)
    m.load_state_dict(d["cas"], strict=True); m.eval()
    return m


def forward(ids, who):
    with torch.no_grad():
        x = T.embed_rows(ids)
        lnf = R.get("model.norm.weight").float()
        cos, sin = T.rope_cos_sin(torch.arange(len(ids)).float())
        for i in range(NL):
            if who == "teacher":
                a = T.attention(None, i, x, cos, sin)
                y = x + a
            else:
                m = load_sgc(i)
                sgc_inj = m(x) - x
                h = T.rms(x, R.get(
                    f"model.layers.{i}.input_layernorm.weight"))
                if who == "ckcas":
                    c = load_cas(i)
                    y = x + sgc_inj + c(h)
                    del c
                else:
                    y = x + sgc_inj
                del m, sgc_inj
            x = y + T.ffn(None, i, y)
            del y
            gc.collect()
        h = x[0, -1]
        z = (h * torch.rsqrt(h.pow(2).mean() + T.EPS)) * lnf
        return T.logits_chunked(z)


def zh_ratio(s):
    s = s.replace("\ufffd", "")
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff") / max(len(s), 1)


def rep3(ids):
    if len(ids) < 3:
        return 0.0
    g = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
    return 1 - len(set(g)) / len(g)


def main():
    t0 = time.perf_counter()
    M.open_sf()
    tok = Tokenizer.from_file(str(AT.TOKP))
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    NNEW = int(os.getenv("NNEW", "10"))
    only = os.getenv("ONLY", "")
    whos = tuple(w for w in WHOS if not os.getenv("WHO") or
                 w in os.getenv("WHO").split(","))
    for prompt in PROMPTS:
        if only and prompt not in only.split("|"):
            continue
        base = T.chat_ids(tok, prompt)
        for who in whos:
            key = f"{prompt}|{who}"
            rec = state.get(key, {"ids": []})
            if len(rec["ids"]) >= NNEW:
                print(f"  [{who}] {prompt!r} 已有 {len(rec['ids'])} token，跳过",
                      flush=True)
                continue
            cur = base + rec["ids"]
            print(f"  [{who}] {prompt!r} 从 {len(rec['ids'])} token 续 "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
            for _ in range(NNEW - len(rec["ids"])):
                nid, sc = forward(cur, who)
                nid = int(nid)
                rec["ids"].append(nid); cur.append(nid)
                state[key] = rec
                tmp = OUT / ".cas_gen_state.tmp"
                tmp.write_text(json.dumps(state, ensure_ascii=False),
                               encoding="utf-8")
                os.replace(tmp, STATE)
                print(f"    t{len(rec['ids']):>2} id={nid:<6} "
                      f"→ {tok.decode(rec['ids'])!r} "
                      f"[{time.perf_counter()-t0:.0f}s]", flush=True)
                gc.collect()
                if nid in (151645, 151643):
                    break
    # 汇总（state 里可能只有部分组合，按实际有的出表）
    rows = {}
    for k, v in state.items():
        ids = v["ids"]
        txt = tok.decode(ids)
        rows[k] = dict(text=txt, ids=ids, n=len(ids),
                       zh=round(zh_ratio(txt), 4), rep3=round(rep3(ids), 4),
                       distinct=len(set(ids)),
                       has_replacement=("\ufffd" in txt))
    FINAL.write_text(json.dumps(
        dict(chat_template="Qwen3-Instruct + no-think prefilled",
             nnew_target=NNEW, argmax=True, rows=rows),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 汇总（原文） ===", flush=True)
    for k, r in rows.items():
        print(f"  {k:<30} n={r['n']:>2} rep3={r['rep3']:.3f} "
              f"zh={r['zh']:.3f} distinct={r['distinct']:>2}")
        print(f"      {r['text']!r}", flush=True)
    print(f"  ✓ {FINAL} [{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

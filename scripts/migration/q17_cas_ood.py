#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-Q17 CAS 迁移 · 分布外诊断：为什么 holdout cos 涨了、生成却崩了。

拟合集是 6 段陈述性中文语料；生成用的是 chat template（含特殊 token +
问句）。本脚本在 **chat prompt 分布上** 逐层量：
    cos(sgc_inj, a_teacher)  vs  cos(sgc_inj + cas_inj, a_teacher)
    以及  ‖cas_inj‖ / ‖sgc_inj‖
若 chat 分布上联合 cos 反而下降 / 注入幅度失衡，即证明是解析拟合的
分布外泛化失败，而不是「CAS 机制无效」也不是「指标算错」。
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
import torch.nn.functional as Fn
from tokenizers import Tokenizer
torch.set_num_threads(2)

import q17_reader as R
import q17_teacher as T
import q17_migrate as M
import analytic_transfer as AT
from simplcore.core.sgc import SpectralGlobalConv as SGC
from simplcore.core.arch.content_state import ContentAddressedState

OUT = Path("/data/workspace/TS2SC_Q17")
KW17 = dict(dim=2048, num_bases=32, decay_max=4096, local_kernel=4,
            rho_param="logspace", init_mode="linear", content_gate=False,
            out_gate=False, evidence_gate=False, uncertainty_gate=False,
            path_gate=False, inst_nonlin=False, kernel_norm="none",
            num_kernels=1)
PROMPTS = ["华为是什么公司", "中国的首都是"]


def cos(p, q):
    return float(Fn.cosine_similarity(p.flatten(), q.flatten(), dim=0))


def run(prompt, tag):
    tok = Tokenizer.from_file(str(AT.TOKP))
    ids = T.chat_ids(tok, prompt)
    x = T.embed_rows(ids)
    L = len(ids)
    cosv, sinv = T.rope_cos_sin(torch.arange(L).float())
    rows = []
    with torch.no_grad():
        for i in range(28):
            p = f"model.layers.{i}."
            h = T.rms(x, R.get(p + "input_layernorm.weight"))
            a = T.attention(None, i, x, cosv, sinv)[0]
            st = torch.load(OUT / "ck4" / f"mixer_{i}.pt",
                            map_location="cpu", weights_only=False)
            m = SGC(**st.get("config", KW17))
            m.load_state_dict(st["mixer"], strict=True); m.eval()
            sgc = (m(x) - x)[0]
            del m, st
            d = torch.load(OUT / "ckcas" / f"layer_{i:02d}.pt",
                           map_location="cpu", weights_only=False)
            cfg = d["cas_config"]
            c = ContentAddressedState(
                dim=cfg["dim"], feat_dim=cfg["feat_dim"],
                value_dim=cfg["value_dim"], phi=cfg["phi"],
                eps=cfg["eps"], zero_init_readout=False)
            c.load_state_dict(d["cas"], strict=True); c.eval()
            cas = c(h)[0]
            del c, d
            cb, cj = cos(sgc, a), cos(sgc + cas, a)
            rows.append(dict(layer=i, base=round(cb, 4),
                             joint=round(cj, 4), delta=round(cj - cb, 4),
                             amp_ratio=round(float(cas.norm() /
                                                   (sgc.norm() + 1e-9)), 4)))
            x = x + a.unsqueeze(0) + T.ffn(None, i, x + a.unsqueeze(0))
            del a, h, sgc, cas
            gc.collect()
    return rows


def main():
    M.open_sf()
    out = {}
    for pi, prompt in enumerate(PROMPTS):
        rows = run(prompt, f"p{pi}")
        out[prompt] = rows
        n_worse = sum(1 for r in rows if r["delta"] < 0)
        print(f"  {prompt!r}: 联合 cos 变差 {n_worse}/28 层；"
              f"平均 Δ={sum(r['delta'] for r in rows)/28:+.4f}；"
              f"平均注入幅度比={sum(r['amp_ratio'] for r in rows)/28:.3f}",
              flush=True)
        for r in rows[::7]:
            print(f"    L{r['layer']:>2} {r['base']:.4f}→{r['joint']:.4f} "
                  f"(Δ{r['delta']:+.4f}) amp={r['amp_ratio']:.3f}", flush=True)
    (OUT / "cas_ood_diagnosis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ✓ cas_ood_diagnosis.json", flush=True)


if __name__ == "__main__":
    main()

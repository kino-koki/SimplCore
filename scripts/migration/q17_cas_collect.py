#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-Q17 CAS 迁移 · Stage 1：逐层收集联合读出拟合所需特征。

两轴互补的解析迁移（零训练）：
  · SGC 分支 = ck4（linear 网格、逐 KV 头 coef）—— 周期 / 连续信号轴
  · CAS 分支 = 内容写入 O(1) 叠加状态 —— key→value 绑定轴

对 6 段语料逐层存：
  h        : teacher 该层 attention 的输入（rms(x, input_layernorm)）
  a        : teacher attention 输出（拟合目标）
  sgc_inj  : ck4 SGC 的注入量 m(x) − x
  Y        : CAS 分支原始读出 (L,V)，φ=1+elu，prefix-cumsum 形式

随后 build 阶段解 read_out 使  sgc_inj + λ·(Y @ Wᵀ) ≈ a ，
5 段拟合 / 1 段 holdout 决定 λ（含 λ=0 即不启用，保证不劣于 ck4）。

全部解析：零梯度、无 N×N、无 KV cache、O(1) 状态。
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
torch.set_num_threads(2)

import q17_reader as R
import q17_teacher as T
import q17_migrate as M
import analytic_transfer as AT
from simplcore.core.sgc import SpectralGlobalConv as SGC

OUT = Path("/data/workspace/TS2SC_Q17")
FEAT = OUT / "casfeat"; FEAT.mkdir(exist_ok=True)
PROJ = OUT / "casproj"; PROJ.mkdir(exist_ok=True)
STATE = OUT / "cas_collect_state.json"
D, NL = 2048, 28
FDIM, VDIM = 128, 256                  # 低秩：状态 (F·V+F)·4 = 131 KiB/层
EPS = 1e-6


def lowrank_proj(i: int) -> dict:
    """从 teacher q/k/v_proj 取行空间主成分作为 CAS 的分离投影。

    A ≈ U diag(S) Vᵀ  ⇒  W = diag(S) Vᵀ  (F, D)，等价于 UᵀA，
    即把输入映射到权重行空间的前 F 个主方向。q 与 k 各自取主成分
    （线性注意力里二者本就是不同投影），value 单独取 V 个主方向。
    行 L2 归一化让 φ 工作在 O(1) 量级，避免内积数值爆炸。
    """
    p = f"model.layers.{i}.self_attn."
    out = {}
    for key, ncomp in (("q_proj", FDIM), ("k_proj", FDIM), ("v_proj", VDIM)):
        A = R.get(p + key + ".weight", torch.bfloat16).float()   # (out,D)
        U, S, Vh = torch.svd_lowrank(A, q=min(ncomp + 8, min(A.shape)),
                                     niter=4)
        W = (S[:ncomp].unsqueeze(1) * Vh[:, :ncomp].T).contiguous()
        W = F.normalize(W, p=2, dim=1)        # 行 L2 = 1
        out[key] = W
        del A, U, S, Vh
        gc.collect()
    return out


def phi(x):
    return F.elu(x) + 1.0


def cas_readout(h, Wq, Wk, Wv):
    """prefix-cumsum 形式，与 content_state.forward 同一数学（单序列）。"""
    q = phi(h @ Wq.T)                      # (L,FDIM)
    k = phi(h @ Wk.T)                      # (L,FDIM)
    v = h @ Wv.T                           # (L,VDIM)
    S = torch.einsum("lf,lv->lfv", k, v).cumsum(0)   # (L,F,VDIM)
    z = k.cumsum(0)                        # (L,FDIM)
    num = torch.einsum("lf,lfv->lv", q, S)
    den = (q * z).sum(-1, keepdim=True) + EPS
    return num / den                       # (L,VDIM)


def main():
    t0 = time.perf_counter()
    M.open_sf()
    tok = Tokenizer.from_file(str(AT.TOKP))
    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": 0, "proj_done": []}
    KW17 = dict(dim=D, num_bases=32, decay_max=4096, local_kernel=4,
                rho_param="logspace", init_mode="linear",
                content_gate=False, out_gate=False, evidence_gate=False,
                uncertainty_gate=False, path_gate=False,
                inst_nonlin=False, kernel_norm="none", num_kernels=1)

    # 投影权重先算好（每层一次 SVD，28 层）
    for i in range(NL):
        f = PROJ / f"proj_{i:02d}.pt"
        if f.exists():
            continue
        w = lowrank_proj(i)
        tmp = PROJ / f".proj_{i:02d}.tmp"
        torch.save(w, tmp); os.replace(tmp, f)
        print(f"  proj L{i:>2} [{time.perf_counter()-t0:.0f}s]", flush=True)

    BATCH = int(os.getenv("BATCH", "2"))
    MAXSEC = float(os.getenv("MAXSEC", "240"))
    n = 0
    for si in range(state["done"], len(AT.CORPUS)):
        if n >= BATCH or time.perf_counter() - t0 > MAXSEC:
            print(f"  ⏱ 本批 {n} 段，退出续跑（{si}/{len(AT.CORPUS)}）", flush=True)
            return
        ids = tok.encode(AT.CORPUS[si]).ids[:64]
        x = T.embed_rows(ids)
        L = len(ids)
        cos, sin = T.rope_cos_sin(torch.arange(L).float())
        with torch.no_grad():
            for i in range(NL):
                dst = FEAT / f"layer_{i:02d}_seg_{si}.pt"
                if dst.exists():
                    continue
                p = f"model.layers.{i}."
                h = T.rms(x, R.get(p + "input_layernorm.weight"))[0]   # (L,D)
                a = T.attention(None, i, x, cos, sin)                    # (1,L,D)
                W = torch.load(PROJ / f"proj_{i:02d}.pt",
                               map_location="cpu", weights_only=False)
                Y = cas_readout(h, W["q_proj"], W["k_proj"], W["v_proj"])
                st = torch.load(OUT / "ck4" / f"mixer_{i}.pt",
                                map_location="cpu", weights_only=False)
                m = SGC(**st.get("config", KW17))
                m.load_state_dict(st["mixer"], strict=True); m.eval()
                sgc_inj = (m(x) - x)[0]                                  # (L,D)
                del m, st
                tmp = FEAT / f".l{i:02d}_s{si}.tmp"
                torch.save(dict(h=h.clone(), a=a[0].clone(),
                                sgc_inj=sgc_inj.clone(), Y=Y.clone(),
                                seg=si, layer=i, L=L), tmp)
                os.replace(tmp, dst)
                x = x + a + T.ffn(None, i, x + a)
                del a, h, W, Y, sgc_inj
                gc.collect()
        state["done"] = si + 1
        tmp = OUT / ".cas_collect_state.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, STATE)
        n += 1
        print(f"  seg {si+1}/{len(AT.CORPUS)} L={L} "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    print(f"  ✓ collect 完成 [{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

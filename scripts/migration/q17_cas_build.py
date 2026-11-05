#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-Q17 CAS 迁移 · Stage 2：逐层闭式联合读出，产出 ckcas/。

对每层解：

    min_W  ‖ (sgc_inj + Y·Wᵀ) − a_teacher ‖²   (5 段拟合)
    W = (YᵀY + αI)⁻¹ Yᵀ (a − sgc_inj)          (D, VDIM)

λ∈{0,.25,.5,.75,1} 与 α 网格由 **holdout 段（si=5）余弦** 决定；
λ=0 始终在候选里 ⇒ 保证不劣于 ck4。λ 直接乘进 read_out.weight，
生成时无需额外缩放，CAS 模块可直接 load_state_dict。

零梯度、无 N×N、无 KV cache；CAS 状态 (F·V+F)·fp32 = 131 KiB/层。
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
import torch.nn.functional as Fn
from simplcore.core.arch.content_state import ContentAddressedState
torch.set_num_threads(2)

OUT = Path("/data/workspace/TS2SC_Q17")
FEAT, PROJ, CK = OUT / "casfeat", OUT / "casproj", OUT / "ckcas"
CK.mkdir(exist_ok=True)
STATE = OUT / "cas_build_state.json"
D, NL, FDIM, VDIM, HO = 2048, 28, 128, 256, 5
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)


def cos(pred, tgt):
    return float(Fn.cosine_similarity(pred.flatten(), tgt.flatten(), dim=0))


def main():
    t0 = time.perf_counter()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": 0, "rows": []}
    rows = state["rows"]
    BATCH = int(os.getenv("BATCH", "10"))
    n = 0
    for i in range(state["done"], NL):
        if n >= BATCH:
            print(f"  ⏱ 本批 {n} 层，退出续跑（{i}/{NL}）", flush=True)
            break
        dst = CK / f"layer_{i:02d}.pt"
        if dst.exists():
            try:
                torch.load(dst, map_location="cpu", weights_only=False)
                state["done"] = i + 1; n += 1
                continue
            except Exception:
                dst.unlink(missing_ok=True)

        fit, hold = [], None
        for si in range(6):
            d = torch.load(FEAT / f"layer_{i:02d}_seg_{si}.pt",
                           map_location="cpu", weights_only=False)
            if si == HO:
                hold = d
            else:
                fit.append(d)
        Y = torch.cat([d["Y"].float() for d in fit])                 # (N,VDIM)
        Rr = torch.cat([d["a"].float() - d["sgc_inj"].float()
                        for d in fit])                               # (N,D)
        Yh, Sh, Ah = (hold["Y"].float(), hold["sgc_inj"].float(),
                      hold["a"].float())
        base = cos(Sh, Ah)

        G = Y.T @ Y
        gmean = float(G.diag().mean())
        best = dict(cos=base, lam=0.0, alpha=None, W=None)
        for ar in ALPHAS:
            W = torch.linalg.solve(
                G + ar * gmean * torch.eye(VDIM), Y.T @ Rr).T          # (D,VDIM)
            for lam in LAMS:
                if lam == 0.0:
                    continue
                c = cos(Sh + lam * (Yh @ W.T), Ah)
                if c > best["cos"] + 1e-6:
                    best = dict(cos=c, lam=lam, alpha=ar, W=W.clone())
            del W
        # λ=0 ⇒ 零读出（不启用 CAS 分支，严格等价于 ck4）
        Wfinal = (best["W"] * best["lam"]) if best["W"] is not None \
            else torch.zeros(D, VDIM)
        pred = Sh + Yh @ Wfinal.T
        final_cos = cos(pred, Ah)

        proj = torch.load(PROJ / f"proj_{i:02d}.pt",
                          map_location="cpu", weights_only=False)
        m = ContentAddressedState(dim=D, feat_dim=FDIM, value_dim=VDIM,
                                  phi="elu1", zero_init_readout=False)
        sd = m.state_dict()
        sd["write_key.weight"] = proj["k_proj"].clone()      # (FDIM,D)
        sd["write_value.weight"] = proj["v_proj"].clone()    # (VDIM,D)
        sd["read_query.weight"] = proj["q_proj"].clone()     # (FDIM,D)
        sd["read_out.weight"] = Wfinal.contiguous()          # (D,VDIM)
        m.load_state_dict(sd, strict=True); m.eval()

        out = dict(
            cas=m.state_dict(),
            cas_config=dict(dim=D, feat_dim=FDIM, value_dim=VDIM,
                            phi="elu1", zero_init_readout=False, eps=1e-6),
            sgc_ck="ck4",            # 周期轴直接复用 ck4，逐字节不动
            lam=best["lam"], alpha=best["alpha"],
            holdout_cos_baseline=round(base, 6),
            holdout_cos_joint=round(final_cos, 6),
            delta=round(final_cos - base, 6),
            state_bytes=m.state_bytes(1),
            method="analytic joint read-out: SGC(ck4) + CAS(ridge on residual)",
            no_training=True, no_NxN=True, no_kv_cache=True)
        tmp = CK / f".layer_{i:02d}.tmp"
        torch.save(out, tmp); os.replace(tmp, dst)
        rows.append(dict(layer=i, baseline=round(base, 4),
                         joint=round(final_cos, 4),
                         delta=round(final_cos - base, 4),
                         lam=best["lam"], alpha=best["alpha"]))
        state["done"] = i + 1; state["rows"] = rows
        tmp = OUT / ".cas_build_state.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, STATE)
        n += 1
        if i % 4 == 0 or i == NL - 1:
            print(f"  L{i:>2} holdout cos {base:.4f}→{final_cos:.4f} "
                  f"(Δ{final_cos-base:+.4f})  λ={best['lam']} "
                  f"α={best['alpha']}  [{time.perf_counter()-t0:.0f}s]",
                  flush=True)
        del m, sd, proj, Y, Rr, G, Wfinal, fit, hold
        gc.collect()
    if state["done"] >= NL:
        (OUT / "cas_build_report.json").write_text(
            json.dumps(dict(rows=rows,
                            mean_baseline=round(sum(r["baseline"] for r in rows)/NL, 4),
                            mean_joint=round(sum(r["joint"] for r in rows)/NL, 4),
                            mean_delta=round(sum(r["delta"] for r in rows)/NL, 4),
                            n_active=sum(1 for r in rows if r["lam"] > 0)),
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✓ ckcas/ 28 层 + 报告 [{time.perf_counter()-t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()

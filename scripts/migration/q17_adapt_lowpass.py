#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融 ck6 = v4 + 仅 A3 低通适配（不填 SVD），定位 v5 退化来自哪一项。

v5 相对 v4 同时改了 A1（out_proj 后 1024 列填 o_proj 左奇异向量 U）与
A3（local_kernel 4→16 + profile 残差抽头）。实测 v5 的 L0 注入 cos
0.9205→0.7670、L27 残差流 cos +0.2484→−0.2445、生成退化为单 token
死循环。本消融只保留 A3，用于判定退化是否由 A1 的未训练正交基引起。
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import sys

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
torch.set_num_threads(2)
import q17_reader as R
import q17_migrate as M
import analytic_transfer as AT
from simplcore.core.sgc import SpectralGlobalConv as SGC

OUT = Path("/data/workspace/TS2SC_Q17")
CK4, CK6 = OUT / "ck4", OUT / "ck6"
FEAT = OUT / "feat"
D, NL, KVD, LOCAL_K, HO = 2048, 28, 1024, 16, 5


def main():
    t0 = time.perf_counter()
    CK6.mkdir(parents=True, exist_ok=True)
    M.open_sf()
    prof = torch.tensor(json.loads((OUT / "profile.json").read_text())["profile"])
    _rr, _rt, rho, theta = AT.analytic_grid()
    rows = []
    for i in range(NL):
        dst = CK6 / f"mixer_{i}.pt"
        st = torch.load(CK4 / f"mixer_{i}.pt", map_location="cpu",
                        weights_only=False)
        kw = dict(AT.SGC_KW); kw["local_kernel"] = LOCAL_K
        m = SGC(**kw)
        sd = m.state_dict()
        for k, v in st["mixer"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v.clone()
        # A3：profile 前 K 抽头 − sin 拟合核前 K 抽头，逐通道复制
        P = AT.psi_bank(rho, theta, LOCAL_K).T
        with torch.no_grad():
            Kfit = P @ st["mixer"]["coef"][:, 0]
            p = prof[i][:LOCAL_K].clone().float()
            taps = (p - Kfit) / LOCAL_K
            sd["local_conv.weight"] = taps.view(1, 1, LOCAL_K).expand(
                D, 1, LOCAL_K).clone()
        # A4：RMS 重标定（out_proj 保持 v4 原值，后 1024 列仍为 0）
        W = sd["out_proj.weight"].float()
        num = den = 0.0
        for si in range(HO):
            d = torch.load(FEAT / f"layer_{i:02d}_seg_{si}.pt",
                           map_location="cpu", weights_only=False)
            z, a = d["z"].float(), d["a"].float()
            num += float(a.pow(2).sum()); den += float((z @ W.T).pow(2).sum())
        s = math.sqrt(num / max(den, 1e-9))
        sd["out_scale"].fill_(s)
        m.load_state_dict(sd, strict=True); m.eval()
        tmp = CK6 / f".mixer_{i}.tmp"
        torch.save(dict(mixer=m.state_dict(), config=kw,
                        method="v4 + A3 lowpass only (SVD fill NOT applied)",
                        rms_scale=s), tmp)
        os.replace(tmp, dst)
        rows.append(dict(layer=i, rms_scale=round(s, 4),
                         residual_tap_l1=round(float(taps.abs().sum()), 5)))
        del m, sd, st, W
        if i % 9 == 0 or i == NL - 1:
            print(f"  L{i:>2} s={s:.4f} [{time.perf_counter()-t0:.0f}s]",
                  flush=True)
    (OUT / "ck6_ablation.json").write_text(json.dumps(
        dict(rows=rows, method="v4 + A3 lowpass only; A1 SVD fill excluded"),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ✓ ck6/ 28 mixer [{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

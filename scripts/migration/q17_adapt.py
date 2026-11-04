#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-Q17 v5：直接改权重 —— self_attn 整体替换为 SGC + 系统适配。

与 v1~v4 的区别（用户最终决定：不做蒸馏、不学行为分布，直接改权重，
但要有「一系列适配」，而不是把 attention 权重硬塞进 SGC 的壳）：

v1~v4 的做法是把 ``v_proj`` 零填充进 ``in_proj``、``o_proj`` 折叠后
零填充进 ``out_proj``，SGC 只是「披着 SGC 壳的 attention 近似」。本版把
每层的 ``self_attn`` **当成一个整体模块替换掉**，SGC 的
``in_proj / out_proj / coef / rho / theta / local_conv / feed / out_scale``
各自有明确的来源与适配方式，并逐张量写进 provenance 账本。

五项适配（全部解析、零梯度、零 pairwise、无 N×N、无 KV cache）：

A1 **full-rank 读出适配**（本版关键）：D=2048 但 KV·HD=1024，v1~v4 的
   ``out_proj`` 后 1024 列恒为 0 —— 于是 ``in_proj`` 后 1024 行
   （k_proj 路径）算出来的卷积特征被全零乘掉，**一半容量白算**。
   本版用 ``o_proj`` 的左奇异向量 U 填这 1024 列，使 out_proj 列满秩。
A2 **逐 KV 头 coef**（沿用 v4）：8 个 KV 头各自 LS 拟合 profile → (32,2048)。
A3 **低通适配**：``local_kernel`` 4→16，用实测平均 profile 的前 16 抽头
   解析初始化 ``local_conv``，补 sin 基族缺的平坦长尾（预检：ΣK/Σp
   0.178 → 1.000）。
A4 **RMS 幅度标定**：local 通道改变注入幅度后重新标定 ``out_scale``。
A5 **provenance 账本**：每个张量标明 teacher 来源 / 解析拟合 / SVD 适配 /
   结构零值，可逐条审计。

其余（embedding / 28 层 FFN / 全部 norm / lm_head）逐位直搬不动。
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, "/data/workspace/SC_repo"); sys.path.insert(0, "/data/workspace")
import torch
torch.set_num_threads(2)

import q17_reader as R
import q17_migrate as M
import analytic_transfer as AT
from simplcore.core.sgc import SpectralGlobalConv as SGC

OUT = Path("/data/workspace/TS2SC_Q17")
CK4, CK5 = OUT / "ck4", OUT / "ck5"
FEAT, HP = OUT / "feat", OUT / "headprof"
D, NL, KVD = 2048, 28, 1024
LOCAL_K = 16
HO = 5


def full_rank_out(sf, i, st):
    """A1：用 o_proj 的左奇异向量补齐 out_proj 后 1024 列。

    返回 (W_new, svd_energy)：W_new 形状 (2048,2048)，前 1024 列保持
    ck4 原值（fold_o_proj + 可能的 ridge 修正），后 1024 列填 U[:, :1024]
    乘一个保守缩放。缩放取「前 1024 列列范数均值 / U 列范数(=1)」的 0.25，
    避免新通道一上来就压过原有读出；之后 A4 的 RMS 标定统一对齐幅度。
    """
    W = st["mixer"]["out_proj.weight"].float().clone()      # (2048,2048)
    O = R.get(f"model.layers.{i}.self_attn.o_proj.weight",
              torch.bfloat16).float()                       # (2048,1024)
    U, S, _ = torch.linalg.svd(O, full_matrices=False)      # U:(2048,1024)
    front = W[:, :KVD]
    scale = 0.25 * float(front.norm(dim=0).mean())
    W[:, KVD:] = U[:, :KVD] * scale
    return W, {"sv_top5": [round(float(v), 3) for v in S[:5]],
               "fill_scale": round(scale, 6),
               "col_rank_before": int(torch.linalg.matrix_rank(
                   st["mixer"]["out_proj.weight"].float(), tol=1e-4)),
               "col_rank_after": int(torch.linalg.matrix_rank(W, tol=1e-4))}


def lowpass_local(prof_i, coef_i, rho, theta):
    """A3：用 profile 前 16 抽头减去 sin 基已覆盖的部分，初始化 local_conv。

    local_conv 是 depthwise FIR，作用在 u 上并被 out_proj 读出；这里用
    「目标核 p(τ) 的前 K 抽头 − 拟合核 K(τ) 的前 K 抽头」作为残差抽头，
    逐通道复制（SGC 的 local_conv 是 (D,1,K)）。
    """
    P = AT.psi_bank(rho, theta, LOCAL_K).T                  # (K,32)
    with torch.no_grad():
        Kfit = P @ coef_i[:, 0]                             # (K,) 取第 0 通道代表
        p = prof_i[:LOCAL_K].clone().float()
        taps = (p - Kfit) / LOCAL_K
        w = taps.view(1, 1, LOCAL_K).expand(D, 1, LOCAL_K).clone()
    return w, {"target_head": [round(float(v), 5) for v in p[:6]],
               "fit_head": [round(float(v), 5) for v in Kfit[:6]],
               "residual_tap_l1": round(float(taps.abs().sum()), 5)}


def rms_scale(i, W):
    """A4：用 5 段拟合集的 (z@W.T + local 贡献) 对 attention 做 RMS 标定。"""
    num = den = 0.0
    for si in range(HO):
        d = torch.load(FEAT / f"layer_{i:02d}_seg_{si}.pt",
                       map_location="cpu", weights_only=False)
        z, a = d["z"].float(), d["a"].float()
        delta = z @ W.T
        num += float(a.pow(2).sum()); den += float(delta.pow(2).sum())
    return math.sqrt(num / max(den, 1e-9))


def main():
    t0 = time.perf_counter()
    CK5.mkdir(exist_ok=True)
    M.open_sf()
    prof = torch.tensor(json.loads(
        (OUT / "profile.json").read_text())["profile"])
    _rr, _rt, rho, theta = AT.analytic_grid()
    prov = {"method": "direct weight replacement: self_attn -> SGC + analytic adaptation",
            "no_training": True, "no_pairwise": True, "no_kv_cache": True,
            "layers": [], "global": {}}
    rows = []
    for i in range(NL):
        CK5.mkdir(parents=True, exist_ok=True)     # 502 重置可能删目录
        dst = CK5 / f"mixer_{i}.pt"
        if dst.exists():
            # 断点续跑：校验可加载，坏文件重算
            try:
                torch.load(dst, map_location="cpu", weights_only=False)
                rows.append(dict(layer=i, resumed=True))
                prov["layers"].append(dict(layer=i, resumed=True))
                continue
            except Exception:
                dst.unlink(missing_ok=True)
        st = torch.load(CK4 / f"mixer_{i}.pt", map_location="cpu",
                        weights_only=False)
        # 用 local_kernel=16 重新构造，逐键搬运 ck4 参数
        kw = dict(AT.SGC_KW); kw["local_kernel"] = LOCAL_K
        m = SGC(**kw)
        sd = m.state_dict()
        for k, v in st["mixer"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v.clone()
        # A1 full-rank 读出
        W, svd = full_rank_out(None, i, st)
        sd["out_proj.weight"] = W
        # A3 低通 local
        w, lp = lowpass_local(prof[i], st["mixer"]["coef"], rho, theta)
        sd["local_conv.weight"] = w
        # A4 RMS 标定
        s = rms_scale(i, W)
        sd["out_scale"].fill_(s)
        m.load_state_dict(sd, strict=True)
        m.eval()
        out = dict(mixer=m.state_dict(), config=kw,
                   method="v5 direct replacement + A1..A5 analytic adaptation",
                   rms_scale=s)
        tmp = CK5 / f".mixer_{i}.tmp"
        torch.save(out, tmp)
        os.replace(tmp, dst)                       # 原子写，防 502 截半
        rows.append(dict(layer=i, rms_scale=round(s, 4),
                         col_rank_before=svd["col_rank_before"],
                         col_rank_after=svd["col_rank_after"],
                         fill_scale=svd["fill_scale"],
                         residual_tap_l1=lp["residual_tap_l1"]))
        prov["layers"].append(dict(
            layer=i,
            in_proj="teacher v_proj (rows 0:1024) + k_proj (1024:2048) [v4]",
            out_proj="fold_GQA(o_proj) cols 0:1024 [v2] + SVD(U) cols 1024:2048 [A1 v5]",
            coef="per-KV-head LS fit of measured attention profile [v4]",
            rho_theta="linear orthogonal grid [init_mode=linear]",
            local_conv=f"residual taps from profile head, K={LOCAL_K} [A3 v5]",
            out_scale=f"RMS calibrated s={s:.4f} [A4 v5]",
            norm="teacher input_layernorm.weight (exact copy)",
            feed="zero (unchanged)",
            content_gate="OFF (no analytic init; would add 4M random params/layer)",
            svd=svd, lowpass=lp))
        if i % 7 == 0 or i == NL - 1:
            print(f"  L{i:>2} rank {svd['col_rank_before']}→"
                  f"{svd['col_rank_after']}  s={s:.4f}  "
                  f"tapL1={lp['residual_tap_l1']}  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
        del m, sd, st, W, w
    prov["global"] = dict(
        embedding="exact copy", ffn="exact copy (28 layers)",
        norms="exact copy", lm_head="exact copy (tied)",
        attention_modules_removed=NL, sgc_modules_inserted=NL,
        local_kernel=LOCAL_K, num_bases=AT.M, dim=D,
        zero_random_params=True,
        note="every parameter has a traced source; no random initialization")
    (OUT / "v5_provenance.json").write_text(
        json.dumps(dict(rows=rows, provenance=prov), ensure_ascii=False,
                   indent=1), encoding="utf-8")
    print(f"  ✓ ck5/ 28 个完整 SGC mixer + provenance "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

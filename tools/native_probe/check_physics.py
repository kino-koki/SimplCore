"""
check_physics.py — 物理边界报告的数字自动核验。

只读 results/physics_*.json，逐条核对 README_physics_frontier.md 的声明。
"""
import json
import os
import numpy as np

R = "results"
C = []


def ck(name, cond, detail=""):
    C.append(dict(name=name, passed=bool(cond), detail=str(detail)))


def j(fn):
    return json.load(open(f"{R}/{fn}"))


def main():
    meta = j("physics_ckpt_meta.json")
    cons = j("physics_consistency.json")
    prec = j("physics_precision.json")
    cap = j("physics_capacity.json")
    st = j("physics_stability.json")
    hd = j("physics_hardening.json")
    fr = j("physics_frontier.json")
    th = j("physics_throughput.json")
    rt = j("physics_retrain.json")
    rf = j("physics_retrain_full.json")

    # 0 checkpoint
    ck("checkpoint acc=1.0 且守恒通过",
       meta["final_acc"] == 1.0 and meta["perm_acc"] == 1.0 and meta["guards"],
       f"acc={meta['final_acc']} perm={meta['perm_acc']}")
    ck("forward ↔ 解析式逐样本一致", cons["identical"])

    # 1 精度相图
    P = prec["phase"]
    ck("fp64/fp32 全长度满分",
       all(v == 1.0 for v in P["fp64"].values()) and
       all(v == 1.0 for v in P["fp32"].values()))
    ck("bf16 在 L=2049 首次掉落", P["bf16"]["L2049"] < 0.7, P["bf16"]["L2049"])
    ck("int8 在 L=2049 首次掉落", P["int8"]["L2049"] < 0.7, P["int8"]["L2049"])
    ck("fp16 掉点晚于 bf16",
       P["fp16"]["L4097"] > P["bf16"]["L4097"],
       f"{P['fp16']['L4097']:.2f} > {P['bf16']['L4097']:.2f}")
    ck("fp16 L=2049 仍满分", P["fp16"]["L2049"] == 1.0)

    # 2 容量
    ck("n_pairs=128 仍满分", cap["pairs"]["n128"]["acc"] == 1.0)
    ck("信息下界 << 状态容量",
       cap["pairs"]["n128"]["info_bit_lowerbound"] < cap["fp16_bits"] / 100,
       f"{cap['pairs']['n128']['info_bit_lowerbound']:.0f} vs {cap['fp16_bits']}")
    ck("噪声 0.20 仍满分", cap["noise"]["e0.2"] == 1.0, cap["noise"]["e0.2"])
    ck("噪声 0.30 开始掉", cap["noise"]["e0.3"] < 1.0, cap["noise"]["e0.3"])

    # 3 ρ 相图
    ck("ρ=0.90 早期崩", st["rho0.9"]["L257"] < 0.7, st["rho0.9"]["L257"])
    ck("ρ=0.999 全长满分（fp16 L≤4097）",
       all(st["rho0.999"][f"L{L}"] == 1.0 for L in (257, 1025, 2049, 4097)))
    ck("补偿单独不足以救 ρ=0.995",
       st["rho0.995"]["L2049_compensated"] < 1.0,
       st["rho0.995"]["L2049_compensated"])

    # 4 加固
    ck("fp16 base 在 L=4097 掉", hd["fp16"]["base"]["L4097"] < 0.7)
    ck("fp16 ρ→1 把 L=4097 救回 1.0", hd["fp16"]["rho1"]["L4097"] == 1.0)
    ck("compensate / renorm 单独无效",
       hd["fp16"]["comp"]["L4097"] == hd["fp16"]["base"]["L4097"] and
       hd["fp16"]["renorm"]["L4097"] == hd["fp16"]["base"]["L4097"])
    ck("int8 硬界：最优加固 L=4097 仍 <0.6",
       max(v["L4097"] for v in hd["int8"].values()) < 0.6,
       max(v["L4097"] for v in hd["int8"].values()))

    # 5 极限前沿
    A = fr["acc"]
    ck("fp32 ρ=0.999 在 L=16385 仍 1.0", A["rho0.999_fp32"]["L16385"] == 1.0)
    ck("fp32 ρ=0.995 在 L=16385 已掉", A["rho0.995_fp32"]["L16385"] < 0.7)
    ck("fp16 ρ=0.9999 推到 L=8193", A["rho0.9999_fp16"]["L8193"] == 1.0)
    ck("fp16 ρ=0.9999 在 L=16385 掉", A["rho0.9999_fp16"]["L16385"] < 0.5)
    ck("ρ=1 明确标为非 BIBO", fr["guards"]["rho1.0"]["BIBO_stable"] is False)
    ck("ρ=0.999 BIBO 稳定", fr["guards"]["rho0.999"]["BIBO_stable"] is True)
    ck("全部守恒断言通过",
       all(g["all_passed"] for g in fr["guards"].values()),
       f"{sum(g['all_passed'] for g in fr['guards'].values())}/{len(fr['guards'])}")

    # 6 吞吐
    L = th["lengths"]
    ck("状态字节与 L 无关", th["independent_of_L"] and
       len({v["us_per_token"] > 0 for v in L.values()}) == 1)
    us = [L[k]["us_per_token"] for k in ("L65", "L257", "L1025", "L4097")]
    ck("每 token 耗时不随 L 增长（±30%）",
       max(us) / min(us) < 1.3, f"{max(us):.2f}/{min(us):.2f}={max(us)/min(us):.2f}")
    ck("状态字节 = 56320", th["state_bytes_fp32"] == 56320,
       th["state_bytes_fp32"])

    # 6.5 从头重训
    ck("重训 ρ_eff=0.999", abs(rt["rho_eff"] - 0.999) < 1e-6, rt["rho_eff"])
    ck("重训同分布四种精度全 1.0",
       all(v == 1.0 for v in rt["heldout"].values()), rt["heldout"])
    ck("重训 fp32 L=16385 仍 1.0", rf["fp32"]["L16385"] == 1.0)
    ck("重训 int8 L=2049 优于后验覆盖",
       rf["int8"]["L2049"] > hd["int8"]["base"]["L2049"],
       f"{rf['int8']['L2049']:.2f} > {hd['int8']['base']['L2049']:.2f}")
    ck("重训 int8 硬界仍在（L=16385 <0.3）", rf["int8"]["L16385"] < 0.3,
       rf["int8"]["L16385"])
    ck("重训 fp16 非单调如实记录（8193<1.0）", rf["fp16"]["L8193"] < 1.0)

    # 7 纯粹性
    import glob, re
    bad = []
    for fn in glob.glob("*.py"):
        if fn.startswith("check_"): continue
        src = open(fn, encoding="utf-8").read()
        code = re.sub(r'"""[\s\S]*?"""', "", src)
        code = re.sub(r"#.*$", "", code, flags=re.M)
        for pat, label in [
            (r'def\s+_?attn', "attention 实现"),
            (r'\bsoftmax\s*\(', "softmax 调用"),
            (r'mode\s*==\s*["\']hybrid', "混合网格分支"),
        ]:
            if re.search(pat, code): bad.append((fn, label))
    ck("活跃 .py 无对照/混合实现", len(bad) == 0, bad)

    n = sum(c["passed"] for c in C)
    print(f"==> {n}/{len(C)} PASS")
    for c in C:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}  {c['detail']}")
    json.dump(dict(checks=C, summary=dict(passed=n, total=len(C),
                   failed=[c["name"] for c in C if not c["passed"]])),
              open(f"{R}/physics_check.json", "w"), indent=1, ensure_ascii=False)
    return 0 if n == len(C) else 1


if __name__ == "__main__":
    raise SystemExit(main())

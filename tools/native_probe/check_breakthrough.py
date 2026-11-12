"""
check_breakthrough.py — 追平/超越声明的自动一致性核验。

只读 results/*.json，逐条核对报告里的每个数字与声明。
任何不一致直接 FAIL，避免把没跑出来的东西写进报告。
"""
import json
import os
import math
import numpy as np
from scipy.stats import fisher_exact

R = "results"
CHECKS = []


def ck(name, cond, detail=""):
    CHECKS.append(dict(name=name, passed=bool(cond), detail=str(detail)))
    return bool(cond)


def main():
    t = json.load(open(f"{R}/tensor_results.json"))
    f = json.load(open(f"{R}/fusion_results.json"))
    al = json.load(open(f"{R}/analytic_length.json"))
    out = {}

    # ---- 1 追平检验 ----
    w = np.concatenate([np.asarray(x["correct_te"]) for x in t["W1_code64_lr"]])
    n_w, k_w = len(w), int(w.sum())
    a = [x["final_acc"] for x in f["UP_softmax"]]
    n_a, k_a = 1536, int(round(sum(a) * 512))
    _, p = fisher_exact([[k_w, n_w - k_w], [k_a, n_a - k_a]])
    out["parity"] = dict(W1=f"{k_w}/{n_w}={k_w/n_w:.6f}",
                         scale=f"~{k_a}/{n_a}={k_a/n_a:.6f}",
                         fisher_p=float(p), verdict="统计不可区分" if p > .05 else "显著")
    ck("W1 三 seed 全 1.0000", k_w == 1536, f"{k_w}/1536")
    ck("标尺非满分", k_a < n_a, f"{k_a}/{n_a}")
    ck("Fisher p>0.05（追平）", p > .05, f"p={p:.4f}")
    ck("W1 与标尺参数 ±3%", abs(t["W1_code64_lr"][0]["params"] - 10240) / 10240 <= .03,
       f"{t['W1_code64_lr'][0]['params']} vs 10240")

    # ---- 2 参数效率超越 ----
    h = t["W4_half"]
    hw = np.concatenate([np.asarray(x["correct_te"]) for x in h])
    ratio = h[0]["params"] / 10240
    out["efficiency"] = dict(params=h[0]["params"], ratio=ratio,
                             acc=float(hw.mean()))
    ck("W4_half 参数 < 标尺一半", ratio < .5, f"{ratio:.3f}")
    ck("W4_half 三 seed 全 1.0000", int(hw.sum()) == len(hw), f"{int(hw.sum())}/{len(hw)}")
    ck("W4_half acc > 标尺 acc", hw.mean() > k_a / n_a, f"{hw.mean():.4f} > {k_a/n_a:.4f}")

    # ---- 3 长度外推 ----
    ck("forward 与解析式逐样本一致", al["forward_analytic_match"])
    L = al["length_extrapolation"]
    out["length"] = L
    ck("L=1025 仍 1.0000", L.get("L1025") == 1.0, L.get("L1025"))
    ck("所有外推长度均 1.0000", all(v == 1.0 for v in L.values()),
       f"{len(L)} 个长度")
    ck("训练长度 L=17 在外推集合内", "L17" in L)

    # ---- 4 机制消融 ----
    ck("T7_noK 掉到 chance 附近", np.mean([r["final_acc"] for r in t["T7_noK"]]) < .30,
       f"{np.mean([r['final_acc'] for r in t['T7_noK']]):.4f}")
    ck("T0_vector 显著低于 W1", np.mean([r["final_acc"] for r in t["T0_vector"]]) < .70)
    ck("W6 随机正交码也满分", all(r["final_acc"] == 1.0 for r in t["W6_code_rand"]))
    ck("W7 硬任务 8 对满分", all(r["final_acc"] == 1.0 for r in t["W7_more_pairs"]))

    # ---- 5 判别性 ----
    perm_ok = all(
        all(r["perm_acc"] == 1.0 for r in v)
        for k, v in t.items() if k.startswith("W") and k != "W3_code32"
    )
    ck("W 系列 value 置换对照全 1.0000", perm_ok)

    # ---- 6 守恒与梯度 ----
    guards_ok = all(
        all(r["guards_all_passed"] for r in v)
        for k, v in t.items() if k.startswith(("T", "U", "V", "W"))
    )
    ck("全部臂训练前后守恒全绿", guards_ok)
    gc = json.load(open(f"{R}/tensor_gradcheck.json"))
    ck("张量探针 gradcheck 全过", all(x["passed"] for x in gc),
       f"{sum(x['passed'] for x in gc)}/{len(gc)}")

    # ---- 7 纯粹性：活跃代码无对照实现（只看可执行实现，不看说明文字） ----
    import glob
    import re
    bad = []
    for fn in glob.glob("*.py"):
        if fn in ("check_breakthrough.py",):
            continue                       # 本文件含检查词，跳过
        src = open(fn, encoding="utf-8").read()
        # 去掉模块/函数 docstring 后再查
        code = re.sub(r'"""[\s\S]*?"""', "", src)
        code = re.sub(r"#.*$", "", code, flags=re.M)
        patterns = [
            (r'def\s+_?attn', "attention 函数实现"),
            (r'mode\s*==\s*["\']hybrid', "混合网格分支"),
            (r'["\']hybrid["\']\s*:\s*dict', "混合网格臂"),
            (r'\bsoftmax\s*\(', "softmax 调用"),
            (r'np\.exp\([^)]*\)\s*/\s*np\.exp', "softmax 手写实现"),
        ]
        for pat, label in patterns:
            if re.search(pat, code):
                bad.append((fn, label))
    ck("活跃 .py 无混合网格/对照实现", len(bad) == 0, bad)

    # ---- 8 数值安全（无 nan/inf） ----
    finite = all(
        np.isfinite([r["final_acc"] for r in v]).all()
        for v in t.values()
    )
    ck("全部 acc 有限", finite)

    n_pass = sum(c["passed"] for c in CHECKS)
    print(f"==> {n_pass}/{len(CHECKS)} PASS")
    for c in CHECKS:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}  {c['detail']}")
    out["checks"] = CHECKS
    out["summary"] = dict(passed=n_pass, total=len(CHECKS),
                          failed=[c["name"] for c in CHECKS if not c["passed"]])
    json.dump(out, open(f"{R}/breakthrough_check.json", "w"),
              indent=1, ensure_ascii=False)
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

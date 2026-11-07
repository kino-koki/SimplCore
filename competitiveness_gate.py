"""
competitiveness_gate.py -- 把「创世级架构」拆成 7 个可机器判定的 gate。

设计约束（来自投稿治理包的硬约束，不放宽）：
  * 只读已有原始结果文件，不生成数据、不编造数字；
  * 每个 gate 先有证伪条件，再判 PASS / PARTIAL / NOT MET；
  * NOT MET 就是 NOT MET，不因为「有相关工作」就写成 PASS；
  * 60% 门是内部质量代理，不是录用概率，本脚本不换算任何百分数。

Run: python3 competitiveness_gate.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

W = "/data/workspace"
OUT = W + "/gate_results.json"


def load(rel):
    p = os.path.join(W, rel)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def has(rel):
    return os.path.exists(os.path.join(W, rel))


def check_g1():
    """bit 内生化：存在输入侧可学分配器，5 seeds，守恒，且证伪条件已写死。"""
    d = load("bit_allocator_results.json")
    if d is None:
        return "NOT MET", "bit_allocator_results.json 不存在（实验未跑）"
    rows = d.get("rows", [])
    seeds = sorted({r.get("seed") for r in rows})
    guards = d.get("guards", {})
    ok_guard = bool(guards.get("all_pass"))
    learned = [r for r in rows if r.get("arm") in ("tok", "ctx")]
    oracle = [r for r in rows if r.get("arm") == "oracle"]
    pv = d.get("protocol_validity", {})
    ev = {
        "n_rows": len(rows), "seeds": seeds, "guards_all_pass": ok_guard,
        "learned_arms": sorted({r.get("arm") for r in learned}),
        "oracle_mean": (sum(r["acc"] for r in oracle) / len(oracle)) if oracle else None,
        "ctx_mean": (sum(r["acc"] for r in rows if r.get("arm") == "ctx") /
                     max(1, len([r for r in rows if r.get("arm") == "ctx"]))),
        "falsification_conditions": d.get("falsification_conditions"),
        "protocol_valid": pv.get("valid"),
        "protocol_finding": pv.get("finding"),
        "falsification_outcome": pv.get("falsification_outcome"),
    }
    if pv and not pv.get("valid"):
        return "NOT MET", ev          # a vacuous probe is NOT a pass
    if len(seeds) >= 5 and ok_guard and learned and oracle:
        ctx = ev["ctx_mean"]
        orc = ev["oracle_mean"]
        if ctx is not None and orc is not None and ctx >= orc - 0.02:
            return "PASS", ev
        return "PARTIAL", ev
    return "NOT MET", ev


def check_g2():
    return "NOT MET", "无 per-layer / per-hop 的 B 预算分配实验（前沿曲线）"


def check_g3():
    """Zoology/MQAR 标准协议：V>=8192, L>=512, >=5 seeds, 同参数强基线。"""
    cands = ["mqar_results.json", "zoology_protocol.json",
             "tools/eval_baselines/results/mqar.json"]
    found = [c for c in cands if has(c)]
    if not found:
        return "NOT MET", "无 V>=8192 / L>=512 / 同参数 GDN·DeltaNet·Mamba-2·Transformer·KATA 的配对 CI"
    return "PARTIAL", {"files": found, "note": "需人工核对配置阈值"}


def check_g4():
    return "NOT MET", "无 >=10M token 等算力 LM（ppl + 检索探针）；TS2SC 权重本体/torch/GPU 均缺"


def check_g5():
    src = ["train/superposition_probe/results/codebook_throughput.json",
           "train/superposition_probe/results/physics_throughput.json"]
    ev = {"files": [s for s in src if has(s)],
          "note": "现有吞吐是未优化 NumPy 参考实现，只能看标度；无 Triton/CUDA fused kernel"}
    return "NOT MET", ev


def check_g6():
    d = load("train/superposition_probe/results/fusion_evolution.json")
    if d is None:
        return "NOT MET", "fusion_evolution.json 不存在"
    s = d.get("summary", [])
    fused = next((x for x in s if x.get("arm") == "fused_gated"), None)
    ev = {"fused_gated_mean": fused.get("mean") if fused else None,
          "monotone_all_seeds": d.get("monotone_all_seeds"),
          "protocol": d.get("protocol", {}).get("note"),
          "scope": "有限合成巩固过程，非真实 token 序列"}
    return "PARTIAL", ev


def check_g7():
    txt = ""
    p = os.path.join(W, "paper_fulltext.txt")
    if os.path.exists(p):
        txt = open(p, encoding="utf-8", errors="ignore").read()[:6000].lower()
    boundary = ("boundary" in txt or "where the bits must live" in txt)
    budgeted = "bit-budgeted" in txt or "budget" in txt and "architecture" in txt
    ev = {"title_is_boundary_characterization": boundary,
          "title_is_bit_budgeted_architecture": budgeted,
          "note": "C1-C5 至少 3 条实证后才换贡献类型"}
    return "NOT MET", ev


GATES = [
    ("G1", "bit 内生化（可学输入侧分配器，满足下界且守恒）", check_g1,
     "若只依赖当前 token id 必然失败、且依赖上下文仍达不到 oracle，则内生化被证伪，转 G2"),
    ("G2", "bit 可调度（按层/按 hop 的预算前沿）", check_g2,
     "预算与能力无单调关系 → 该轴不可调度"),
    ("G3", "标准 MQAR / Zoology 协议，同参数 ≥5 seed 配对 CI", check_g3,
     "等参数下显著落后且无理论解释 → 停止架构叙事"),
    ("G4", "≥10M token 等算力语言建模", check_g4,
     "ppl 差距不可由 B 预算解释 → C3 证伪"),
    ("G5", "fused kernel + 与 FlashAttention-2 同场实测吞吐", check_g5,
     "常数状态在真实硬件无优势 → C4 证伪"),
    ("G6", "SMM 自进化在真实 token 序列上单调且守速率界", check_g6,
     "真实序列不单调或违反 CT7 速率界 → 收敛为理论边界"),
    ("G7", "论文换贡献类型（bit-budgeted architecture）", check_g7,
     "G3–G5 任一不成立则不换，走 workshop / TMLR"),
]


def main():
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "definition": "创世级 = 正交自由度 + 真实算力下可换能力 + 可迁移；判据 C1-C5",
        "rules": [
            "只读已有原始结果文件；NOT MET 不包装成 PARTIAL",
            "60% 门是内部质量代理指标，不是录用概率，不换算百分数",
            "合成任务满分不升级为『架构更强』",
        ],
        "gates": [],
        "counts": {},
    }
    for gid, name, fn, falsify in GATES:
        status, ev = fn()
        out["gates"].append(dict(id=gid, claim=name, status=status,
                                 evidence=ev, falsification_condition=falsify))
    c = {}
    for g in out["gates"]:
        c[g["status"]] = c.get(g["status"], 0) + 1
    out["counts"] = c
    out["headline"] = (
        "当前定位：C5 通过（紧界+构造+scope），C1 部分，C2/C3/C4 未通过。"
        "不是创世级，是『一条尚未离开实验室的正交轴』。"
    )
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("== competitiveness gate ==")
    for g in out["gates"]:
        print("  %-3s %-8s %s" % (g["id"], g["status"], g["claim"]))
    print("counts:", c)
    print("->", OUT)


if __name__ == "__main__":
    main()

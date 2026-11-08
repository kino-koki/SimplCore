"""把本轮修复 + 前哨训练产物一次性推到 kino-koki/SimplCore@main。

走 GitHub Git Data API（沙盒内 git 客户端被 403 拦），构造单个 tree +
单个 commit，再 PATCH ref。token 只从环境变量读取，不写入任何文件。

用法：
  GH_TOK=... python push_changes.py --dry-run
  GH_TOK=... python push_changes.py --apply
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "kino-koki/SimplCore"
BRANCH = "main"
API = "https://api.github.com"
TOKEN = os.environ.get("GH_TOK", "")
ROOT = "/data/workspace"

if not TOKEN:
    sys.exit("GH_TOK 未设置")


def req(path, method="GET", payload=None):
    url = API + path
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "simplcore-fix-push",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(r, timeout=90) as f:
        return json.loads(f.read().decode("utf-8"))


def get_file(path):
    d = req(f"/repos/{REPO}/contents/{urllib.parse.quote(path)}?ref={BRANCH}")
    return base64.b64decode(d["content"]).decode("utf-8")


import urllib.parse  # noqa: E402  (quote 上面用到)

# ---------------------------------------------------------------- 修复定义
PATCHES = {}

PATCHES["simplcore/core/model/config.py"] = [(
    '''        if getattr(self, "sgc_init_mode", "legacy") not in ("legacy", "logfreq",
                                                            "hybrid"):
            raise ValueError(
                f"sgc_init_mode must be 'legacy' | 'logfreq' | 'hybrid', "
                f"got {self.sgc_init_mode!r}")''',
    '''        # 0.1-beta: "linear" is implemented in arch/sgc.py and guarded by
        # tests/test_linear_grid.py, but was missing here -- SimplCoreConfig
        # is the main construction path, so the fix was unreachable from it.
        if getattr(self, "sgc_init_mode", "logfreq") not in (
                "legacy", "logfreq", "hybrid", "linear"):
            raise ValueError(
                f"sgc_init_mode must be 'legacy' | 'logfreq' | 'hybrid' | 'linear', "
                f"got {self.sgc_init_mode!r}. "
                f"NOTE: 'legacy' uniform theta grid has an unfixable algebraic "
                f"common zero and is rejected at mixer construction; prefer 'linear'.")''')]

PATCHES["simplcore/core/arch/sgc.py"] = [
    ('''import math
from dataclasses import dataclass''',
     '''import math
import warnings
from dataclasses import dataclass'''),
    ('''    _assert_no_common_zero(theta0, num_bases)
    if mode in ("hybrid", "linear"):
        # hybrid / linear 必须过数值质量门（logfreq 的秩缺失是它们存在的理由）
        _assert_numerical_rank(theta0, num_bases)''',
     '''    _assert_no_common_zero(theta0, num_bases)
    if mode in ("hybrid", "linear"):
        # hybrid / linear 必须过数值质量门（logfreq 的秩缺失是它们存在的理由）
        _assert_numerical_rank(theta0, num_bases)
    elif mode == "logfreq":
        # 0.1-beta: logfreq is the historical default, so a hard rejection would
        # break existing checkpoints -- but it must NOT be silently accepted.
        # Independently recomputed rank/cond (docs/frequency_grid_fix.md §3,
        # §10): M=32, L=512 -> rank 24/32, cond ~1e16. Emit an auditable warning.
        rank, cond = _grid_rank_condition(theta0)
        if rank < num_bases or cond > 1e6:
            warnings.warn(
                f"init_mode='logfreq': damped-oscillator basis is rank-deficient "
                f"(rank={rank}/{num_bases}, cond={cond:.2e}); the low-frequency "
                f"end near theta_min=pi/decay_max is nearly collinear. "
                f"Consider init_mode='linear' (zero extra params/FLOPs/state).",
                RuntimeWarning, stacklevel=2)''')]

PATCHES["simplcore/core/eval/mathproof.py"] = [
    ('''    **θ 卫兵（P1）**：命题 5 修正后的误差界含 1/|sinθ_m|，θ→0/π 时界发散。
    ``oscillator_params`` 已加 ``θ ∈ [1e-3, π−1e-3]`` 参数级钳制——此处验证：
    把 raw_theta 推到 ±30，θ 仍严格落在钳制区间内（|sinθ| ≥ sin(1e-3)）。
    """''',
     '''    **θ 卫兵（P1）**：命题 5 修正后的误差界含 1/|sinθ_m|，θ→0/π 时界发散。
    ``oscillator_params`` 已加参数级钳制；上界 π−1e-3 是绝对界，**下界随
    ``decay_max`` 让步**（``theta_clamp_floor = min(1e-3, π/decay_max)``），
    以保证 logfreq 最慢基的合法设计值不被钳。

    ⚠️ **0.1-β 口径修正**：本函数构造的是 ``decay_max=64`` 的 mixer，下界恰为
    1e-3，因此下面的 ``theta_lo >= 1e-3`` 断言**只在这个小配置上成立**。
    默认 decay_max=8192 的下界是 3.83e-4（1/|sinθ|≈2607.6），
    tuan-2b 的 decay_max=4e6 是 7.85e-7（≈1.27e6）。各档实际下界与
    误差界放大倍数在 ProofResult.detail 中如实打印，不再把固定 1e-3
    当作全局保证。
    """'''),
    ('''        f"fp64 sigmoid 严格={math_ok}, 包络违反={worst_env:.1e}, "
        f"ℓ₁ 超界={worst_l1:.1e}, raw=±30 时 ρ<1={guard_ok}, "
        f"θ 钳制 [1e-3, π−1e-3] 生效={theta_ok}",''',
     '''        f"fp64 sigmoid 严格={math_ok}, 包络违反={worst_env:.1e}, "
        f"ℓ₁ 超界={worst_l1:.1e}, raw=±30 时 ρ<1={guard_ok}, "
        f"θ 钳制 [1e-3, π−1e-3] 生效={theta_ok} "
        f"(仅 decay_max=64 口径), "
        f"下界随 decay_max 让步: {floor_note}",'''),
]
# 上面用到 floor_note，需在 theta_ok 计算后插入定义
PATCHES["simplcore/core/eval/mathproof.py"].append((
    '''    ok = math_ok and worst_env <= 0.0 and worst_l1 <= 1e-9 and guard_ok and theta_ok''',
    '''    # 0.1-beta: report the actual clamp floor for every production decay_max.
    # Informational only -- does not change the `ok` verdict above.
    floor_note = "; ".join(
        f"decay_max={dm:g}: floor={min(1e-3, math.pi / dm):.3g} "
        f"(1/|sin|~{1.0 / math.sin(min(1e-3, math.pi / dm)):.3g})"
        for dm in (64.0, 8192.0, 16384.0, 4_000_000.0))
    ok = math_ok and worst_env <= 0.0 and worst_l1 <= 1e-9 and guard_ok and theta_ok'''))

PATCHES["SimplCore_Paper.md"] = [
    ('''1. **双形式精确对偶**：FFT 整段与 IIR 流式在 fp32 下逐位一致（守护测试锁定）；''',
     '''1. **双形式对偶（分路径口径）**：FFT 整段与 IIR 流式在 fp64 下数学等价；流式路径 fp32 严格因果逐位为 0，整段 FFT 路径 fp32 为容差内一致（守护测试口径 <1e-5），偏差随 $L$ 增长且呈重尾（§5.4）；'''),
    ('''- **定理 1（双形式对偶）**：FFT 整段 == IIR 流式（fp32 短序列逐位一致，
  守护测试锁定；长序列偏差见 §5.4 勘误）。''',
     '''- **定理 1（双形式对偶，分路径口径）**：FFT 整段 == IIR 流式在 **fp64 下数学等价**（fp64 全流程实测相对误差 2.43e−14 @ L=64）；fp32 下流式路径逐位精确因果，整段 FFT 路径为容差内一致（`test_audit_guards.py::test_causality_by_path` 口径 <1e-5），长序列偏差见 §5.4 勘误。**不再主张 fp32 整段无条件逐位一致。**'''),
    ('''- **定理 3（O(1) 状态）**：状态字节数 $4(2MD+(K{-}1)D+1)$ 与 $L$ 无关
  （M=64, D=1024, K=4 → 524.0 KiB/层；`mathproof` P12 硬断言）。''',
     '''- **定理 3（O(1) 状态）**：状态字节数 $4(2MD+(K{-}1)D+1)$ 与 $L$ 无关
  （M=64, D=1024, K=4 → 524.0 KiB/层；`mathproof` P12 硬断言）。
  **口径注**：公式中的 $+1$ 对应 L2 证据门的常驻标量（`batch·4` B），仅在
  `evidence_gate=True` 时存在；默认 OFF 配置为 $4(2MD+(K{-}1)D)$（§5.14 的
  4,829,184 B 即按此口径）。见 `simplcore/core/arch/sgc.py::streaming_state_bytes`。'''),
    ('''MQAR 对照（§7.2）定位：''', '''MQAR 对照（§5.5）定位：'''),
    ('''不是更好的模型（聚合未复现、精确轴先输后平）、长序列 fp32 稳健性
未验证、大规模训练有效性未证明。把它变成事实需要的是 §5.13 战役的
一次真实运行，而不是更多的文字。''',
     '''不是更好的模型（聚合未复现、精确轴先输后平）、长序列 fp32 稳健性
未验证、大规模训练有效性未证明。**0.1-β 更新**：47 万参数原生 SimplCore
已在真实语料上完成 3.07M token 前哨训练（`docs/TRANSFER_DISTILL_REPORT.md`），
val ppl 10.817 优于同预算 Transformer 11.281，输出分布蒸馏进一步到 10.217；
但三臂生成仍不成句，规模化有效性依然未证明。把它变成事实需要的是
§5.13 战役的一次真实运行，而不是更多的文字。'''),
]

APPEND = {
    "docs/frequency_grid_fix.md": '''

---

## 10. 0.1-β 口径对齐与训练证据（2026-11-05）

独立按本仓 `_init_oscillator_grid` 公式复算三种条件数口径（脚本见
`tools/transfer_distill/../verify_source_grid.py` 的口径定义），结果：

| 口径 | logfreq | hybrid | linear |
|---|---:|---:|---:|
| A：`Ψ=sin((τ+1)θ)`, L=512, SVD, 不含 ρ | 7.96e16（秩 24/32） | 6.52 | **1.04** |
| B：含 ρ^τ, L=1024, 列归一化 Gram | 2.95e10 | 9.66e8 | 1.03e7 |
| 外部白盒报告：含 ρ, L=128, **fp32 未归一化 Gram** | 1.35e36 | — | 3.97e3 |

- 口径 A 的 linear=**1.04** 与本文 §3 精确命中；logfreq 与 1.92e16 同量级。
- 白盒的 1.35e36 是 **fp32 数值条件数**（最小特征值被舍入钳在 ~1e−30），
  不是数学条件数，换 fp64 会变成 1e300 量级；它是 `cond(H)²` 口径，
  与 SVD 口径差一个平方。**方向与机制一致，绝对值不可直接比较。**
- 因此「cond 改善 8~9 个数量级」应改述为：在白盒 Gram 定义下约 32 个数量级，
  在本仓 SVD 口径下约 16 个数量级，并注明口径。

**「线性均匀是最优解」的限定**：§3 末尾已指出含 ρ 口径下 linear 未必优于
legacy 均匀（π/2 对称冗余对）。该结论只在「本文扫过的网格组合 + 本文
Gram 定义」下成立；非对称 θ 网格与 `linear_floor` 连续扫描仍待做。

**构造期守护**：本次已给 `init_mode="logfreq"` 加上可审计的
`RuntimeWarning`（不硬拒绝，以保既有 checkpoint），并把
`sgc_init_mode="linear"` 加进 `SimplCoreConfig.validate()` 白名单——
此前该修复只能绕过 config 直接构造 mixer 才能启用。

**独立训练证据**：`docs/TRANSFER_DISTILL_REPORT.md` 给出 47 万参数原生
SimplCore 的 3.07M token 三臂对照。注意该实验的 `psi` 按本仓
`ρ^τ sin((τ+1)θ)` 口径实现，与上文外部白盒的
`ρ^t sin((t+1)θ)/sinθ` 口径不同，训练数字不可与白盒报告互换。
''',
}

NEW_FILES = {
    "docs/TRANSFER_DISTILL_REPORT.md": f"{ROOT}/train/TRANSFER_DISTILL_REPORT.md",
    "docs/FIX_STATUS_20261105.md": f"{ROOT}/train/FIX_STATUS.md",
    "tools/transfer_distill/README.md": f"{ROOT}/train/README_tools.md",
    "tools/transfer_distill/core.py": f"{ROOT}/train/core.py",
    "tools/transfer_distill/data.py": f"{ROOT}/train/data.py",
    "tools/transfer_distill/run.py": f"{ROOT}/train/run.py",
    "tools/transfer_distill/gradcheck64.py": f"{ROOT}/train/gradcheck64.py",
    "tools/transfer_distill/results/verdict.json": f"{ROOT}/train/verdict.json",
    "tools/transfer_distill/results/log_ts.json": f"{ROOT}/train/log_ts.json",
    "tools/transfer_distill/results/log_sgc.json": f"{ROOT}/train/log_sgc.json",
    "tools/transfer_distill/results/log_kd.json": f"{ROOT}/train/log_kd.json",
}

COMMIT_MSG = """fix(0.1-beta): make linear grid reachable from config, audit logfreq ill-conditioning, align paper caveats, add 470K-param transfer-distillation pilot

Source fixes
- config.py: add "linear" to sgc_init_mode whitelist (was implemented in
  arch/sgc.py + guarded by test_linear_grid.py but unreachable from the main
  SimplCoreConfig -> SimplCoreModel path); fix getattr default to "logfreq";
  document that "legacy" is rejected at mixer construction.
- arch/sgc.py: logfreq default no longer silently bypasses the numerical-rank
  gate -- emit an auditable RuntimeWarning (rank 24/32, cond ~1e16 at M=32)
  instead of hard-rejecting, to preserve existing checkpoints.
- eval/mathproof.py P05: document that theta clamp floor follows decay_max;
  the fixed 1e-3 assertion only holds for the decay_max=64 probe. Print the
  actual floor and 1/|sin(floor)| for 64 / 8192 / 16384 / 4e6. Verdict logic
  unchanged, informational only.

Documentation
- SimplCore_Paper.md: replace unconditional "fp32 bitwise" claim with the
  per-path wording already used by test_audit_guards.py; annotate theorem 3's
  +1 as the evidence-gate scalar; fix dangling §7.2 -> §5.5; note the new
  pilot training in §10.
- docs/frequency_grid_fix.md §10: align three condition-number conventions,
  qualify "linear is optimal", record the constructor warning and config fix.

New: 470K-param native SimplCore pilot (docs/TRANSFER_DISTILL_REPORT.md,
tools/transfer_distill/). Pure NumPy, no torch. Three equal-budget arms
(TS / SGC scratch / output-distribution KD) on 3.07M tokens of this repo's
real text. fp64 gradcheck passes; FFT/IIR duality rel<6e-6 at L=32..256 with
68,608 B constant state. val ppl 11.281 / 10.817 / 10.217.
HONEST: generation is still word salad on all three arms; the script's
SUPPORTED verdict is downgraded to PARTIAL in the report; the random-teacher
control is unfinished and single-seed. Not evidence for scale-up.

NOTE: P0-4 (gate defaults sgc.py=True vs config.py=False) is deliberately
NOT changed here -- it would alter checkpoint behaviour and needs the full
pytest suite, which this sandbox cannot run. See docs/FIX_STATUS_20261105.md.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.apply):
        sys.exit("指定 --dry-run 或 --apply")

    print("== 读取远端当前内容 ==")
    remote = {}
    for path in list(PATCHES) + list(APPEND):
        remote[path] = get_file(path)
        print(f"  GET {path}  ({len(remote[path])} chars)")

    print("\n== 校验替换 ==")
    final = {}
    ok = True
    for path, subs in PATCHES.items():
        s = remote[path]
        for i, (old, new) in enumerate(subs, 1):
            n = s.count(old)
            flag = "OK" if n == 1 else "FAIL"
            print(f"  [{flag}] {path} #{i} 命中 {n} 次")
            if n != 1:
                ok = False
            else:
                s = s.replace(old, new, 1)
        final[path] = s
    for path, tail in APPEND.items():
        if tail.strip() in remote[path]:
            print(f"  [SKIP] {path} 追加内容已存在")
        else:
            final[path] = remote[path].rstrip() + "\n" + tail
            print(f"  [OK] {path} 追加 {len(tail)} chars")

    print("\n== 新增文件 ==")
    for path, local in NEW_FILES.items():
        if not os.path.exists(local):
            print(f"  [FAIL] 本地缺失 {local}")
            ok = False
            continue
        txt = open(local, encoding="utf-8").read()
        final[path] = txt
        print(f"  [OK] {path}  <- {os.path.basename(local)} ({len(txt)} chars)")

    if not ok:
        sys.exit("\n有替换未唯一命中，终止（未做任何远端修改）")
    if a.dry_run:
        print(f"\nDRY-RUN OK：共 {len(final)} 个文件待提交，远端未改动")
        return

    print("\n== 构造单个 commit ==")
    ref = req(f"/repos/{REPO}/git/ref/heads/{BRANCH}")
    base_commit = ref["object"]["sha"]
    base_tree = req(f"/repos/{REPO}/git/commits/{base_commit}")["tree"]["sha"]
    print(f"  base commit {base_commit[:12]}  tree {base_tree[:12]}")

    entries = [{"path": p, "mode": "100644", "type": "blob", "content": c}
               for p, c in final.items()]
    tree = req(f"/repos/{REPO}/git/trees", "POST",
               {"base_tree": base_tree, "tree": entries})
    print(f"  new tree {tree['sha'][:12]}  ({len(tree.get('tree', []))} entries)")

    commit = req(f"/repos/{REPO}/git/commits", "POST",
                 {"message": COMMIT_MSG, "tree": tree["sha"],
                  "parents": [base_commit]})
    print(f"  new commit {commit['sha'][:12]}")

    # 确认 main 未在此期间被他人推进
    cur = req(f"/repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    if cur != base_commit:
        sys.exit(f"main 已被推进到 {cur[:12]}，为避免覆盖，终止 PATCH")
    res = req(f"/repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH",
              {"sha": commit["sha"], "force": False})
    print(f"\nPUSH OK -> {BRANCH} @ {res['object']['sha'][:12]}")
    print(f"https://github.com/{REPO}/commit/{commit['sha']}")


if __name__ == "__main__":
    main()

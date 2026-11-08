"""替换论文主稿并推到 kino-koki/SimplCore@main。

操作（单 commit）：
  MODIFY  SimplCore_Paper.md                    -> 0.2-β 表达力边界刻画主稿
  MODIFY  README.md                             -> 3 处最小一致性修复
  ADD     docs/SimplCore_Paper_v0.1alpha.md     -> 原稿逐字节归档
  ADD     docs/PAPER_REWRITE_NOTE.md            -> 改写清单
  ADD     docs/PAPER_CONSISTENCY_CHECK.json     -> 26/26 自检结果

走 GitHub Git Data API（沙盒内 git 客户端 403）。token 只从 GH_TOK 读取，
不写入任何产物。

  GH_TOK=... python push_paper.py --dry-run
  GH_TOK=... python push_paper.py --apply
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO = "kino-koki/SimplCore"
BRANCH = "main"
API = "https://api.github.com"
TOKEN = os.environ.get("GH_TOK", "")
ROOT = "/data/workspace/train"

if not TOKEN:
    sys.exit("GH_TOK 未设置")


def req(path, method="GET", payload=None):
    r = urllib.request.Request(
        API + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "simplcore-paper-rewrite",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(r, timeout=90) as f:
        return json.loads(f.read().decode("utf-8"))


def get_file(path):
    d = req(f"/repos/{REPO}/contents/{urllib.parse.quote(path)}?ref={BRANCH}")
    return base64.b64decode(d["content"]).decode("utf-8")


def exists(path):
    try:
        req(f"/repos/{REPO}/contents/{urllib.parse.quote(path)}?ref={BRANCH}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


README_PATCHES = [
    ('''> ⚠️ **实验阶段警告**：目前该架构仅仅处于实验阶段，如果不是真的想试试，
> 建议还是去用 **Mamba** 进行开发、设计模型。本仓库没有经过大规模训练验证，
> 所有"优于 Transformer"的主张都是**机制层结构性质**（可运行断言），
> **不是**实测的质量结论。''',
     '''> ⚠️ **实验阶段警告**：目前该架构仅仅处于实验阶段，如果不是真的想试试，
> 建议还是去用 **Mamba** 进行开发、设计模型。本仓库没有经过大规模训练验证，
> 所有"优于 Transformer"的主张都是**机制层结构性质**（可运行断言），
> **不是**实测的质量结论。
>
> 📄 **论文主稿已重构（2026-11-05）**：`SimplCore_Paper.md` 现为 **0.2-β
> 表达力边界刻画稿**，章节号已变更。本文及仓库文档中形如「论文 §7.2」
> 的旧引用请对照归档版 `docs/SimplCore_Paper_v0.1alpha.md`；改写清单见
> `docs/PAPER_REWRITE_NOTE.md`。**当前状态不宜直接投稿**，见主稿 §7。'''),
    ('''> `docs/THEORETICAL_EDGE.md` 实测对照节、`SimplCore_Paper.md` §7.2 轴注记、''',
     '''> `docs/THEORETICAL_EDGE.md` 实测对照节、`docs/SimplCore_Paper_v0.1alpha.md` §7 轴注记（旧稿归档）、'''),
    ('''| `SimplCore_Paper.md` | **架构白皮书（v1.1.2，含 v1.2 扩展）**：定理 1–4 / 命题 5–7（命题 5 误差界 v1.1.2 含 1/\\|sinθ\\| 修正）、§3.5 世界模型理论动机、SGC-R 谱专家路由、负面结果与撤回、13 项数学证明、部署/战役账本、§7 等效参数推测（明确标注为可证伪理论推测） |''',
     '''| `SimplCore_Paper.md` | **论文主稿（0.2-β，表达力边界刻画）**：四条表达力边界（函数类不匹配 / 静态核天花板 / 时变参数破坏 FFT 对偶 / 内容寻址等参数仍输 attention）、三条结构性质的分路径口径、`[R]/[E]/[A]/[C]` 四级证据分级、47 万参数三臂训练与 SFSC-FDCAO 六臂前哨、§7 投稿硬门槛清单。**不是 SOTA 主张；当前不宜直接投稿** |
| `docs/SimplCore_Paper_v0.1alpha.md` | **0.1-α 原稿归档**（逐字节保留）：架构白皮书 v1.1.2 / 定理 1–4 / 命题 5–7 / §7 分轴等效参数推测 / 旧章节号，供口径追溯 |'''),
]

NEW_FILES = {
    "docs/SimplCore_Paper_v0.1alpha.md": f"{ROOT}/PAPER_REMOTE_CURRENT.md",
    "docs/PAPER_REWRITE_NOTE.md": f"{ROOT}/PAPER_REWRITE_NOTE.md",
    "docs/PAPER_CONSISTENCY_CHECK.json": f"{ROOT}/results/paper_consistency.json",
}

COMMIT_MSG = """paper: rewrite main manuscript as an expressivity-boundary characterization (0.2-beta)

Replaces SimplCore_Paper.md. The 0.1-alpha draft framed the work as
"replace attention / cost-viable alternative". Multiple independent experiments
(both in-repo and external white-box) have since established that the
content-addressing and generation-quality weaknesses are STRUCTURAL, not
tuning artifacts. Keeping that framing would be an overclaim and would bury
the most informative results -- the negative ones -- in appendices.

New framing: "What is the capability boundary of constant-state attention-free
sequence models?" Four boundaries:
  B1 function-class mismatch (K(tau) vs K(i,j)=f(q_i,k_j)); exact-kernel
     oracle gives mid-layer injection cosine ~0 despite a perfect kernel.
  B2 layer-wise accumulation (single-layer same-input control 0.5430 vs
     28-layer 0.2484).
  B3 saturated static-kernel ceiling (v4 0.2484 vs oracle 0.2476);
     improved alignment does NOT translate into fluent generation.
  B4 time-varying rho/theta => LTV => non-Toeplitz => FFT duality FAILS
     (is_toeplitz=false, FFT-proxy rel error 0.972); duality and
     context-dependent kernel are mutually exclusive.

Also added:
- Four-level evidence grading [R] in-repo / [E] external white-box /
  [A] analytic ledger / [C] conjecture. [C] excluded from abstract and
  conclusion.
- DOWNGRADED the osc-axis "beats Transformer" claim: in-repo osc_probe.py
  did NOT reproduce it (7 grids 0.1094..0.1354 < copy_last 0.1575, paired
  CI crosses 0). Script present != conclusion present.
- Removed "equivalent 4-7B" / "only cost-viable choice" from abstract and
  conclusion; cost axis and quality axis now accounted separately.
- New experiments into the manuscript: 470K-param three-arm pilot
  (val ppl 11.281 / 10.817 / 10.217, PARTIAL not SUPPORTED -- generation
  still word salad) and SFSC/FD-CAO six-arm probe (equal-param low-rank
  attn 1.0000 vs FD-CAO 0.8672).
- Section 7 rewritten as an ICML-style hard-gate limitation list (>=5 seeds,
  same-token-budget LM main experiment, related-work DOI verification,
  formal proof for B1-B3). Explicitly states: NOT READY TO SUBMIT.
- Appendices: reproducibility commands for the new probes, submission TODO
  table, AI-usage declaration template (placeholder, author fills per CFP).
- No unverifiable arXiv IDs / DOIs anywhere; all citations point to in-repo
  files listed in Appendix A.

NOT changed: no new/fabricated experimental numbers, no source code,
config, checkpoint, or guard test modified, no attention/hybrid module
introduced, B1-B3 explicitly NOT claimed as theorems.

Original draft archived byte-for-byte at docs/SimplCore_Paper_v0.1alpha.md.
README.md: 3 minimal edits to fix dangling section references (old §7.2)
and the now-stale document-index description.

Consistency self-check: 26/26 PASS (docs/PAPER_CONSISTENCY_CHECK.json).
Rewrite rationale: docs/PAPER_REWRITE_NOTE.md.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.apply):
        sys.exit("指定 --dry-run 或 --apply")

    ok = True
    print("== 读取远端现状 ==")
    remote_paper = get_file("SimplCore_Paper.md")
    remote_readme = get_file("README.md")
    print(f"  SimplCore_Paper.md {len(remote_paper)} chars")
    print(f"  README.md         {len(remote_readme)} chars")

    print("\n== 校验 README 三处替换 ==")
    readme = remote_readme
    for i, (old, new) in enumerate(README_PATCHES, 1):
        n = readme.count(old)
        print(f"  [{'OK' if n == 1 else 'FAIL'}] README #{i} 命中 {n} 次")
        if n != 1:
            ok = False
        else:
            readme = readme.replace(old, new, 1)

    print("\n== 校验新稿与归档一致性 ==")
    new_paper = open(f"{ROOT}/PAPER_NEW.md", encoding="utf-8").read()
    archive = open(f"{ROOT}/PAPER_REMOTE_CURRENT.md", encoding="utf-8").read()
    print(f"  归档 == 远端原稿逐字节: {archive == remote_paper}")
    if archive != remote_paper:
        ok = False
    chk = json.load(open(f"{ROOT}/results/paper_consistency.json"))
    print(f"  一致性自检: {chk['passed']}/{chk['total']} PASS, {chk['failed']} FAIL")
    if chk["failed"]:
        ok = False

    print("\n== 检查新增路径占用 ==")
    for path in NEW_FILES:
        e = exists(path)
        print(f"  {'[CONFLICT]' if e else '[free]    '} {path}")
        if e:
            ok = False
    for path, local in NEW_FILES.items():
        if not os.path.exists(local):
            print(f"  [FAIL] 本地缺失 {local}")
            ok = False

    files = {
        "SimplCore_Paper.md": new_paper,
        "README.md": readme,
    }
    for path, local in NEW_FILES.items():
        files[path] = open(local, encoding="utf-8").read()

    if not ok:
        sys.exit("\n校验未过，终止（远端未改动）")
    if a.dry_run:
        print(f"\nDRY-RUN OK：2 修改 + {len(NEW_FILES)} 新增 = {len(files)} 文件待提交，远端未改动")
        return

    print("\n== 构造单个 commit ==")
    base = req(f"/repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    base_tree = req(f"/repos/{REPO}/git/commits/{base}")["tree"]["sha"]
    print(f"  base {base[:12]}  tree {base_tree[:12]}")

    entries = [{"path": p, "mode": "100644", "type": "blob", "content": c}
               for p, c in files.items()]
    tree = req(f"/repos/{REPO}/git/trees", "POST",
               {"base_tree": base_tree, "tree": entries})
    print(f"  new tree {tree['sha'][:12]} ({len(tree.get('tree', []))} entries)")
    cm = req(f"/repos/{REPO}/git/commits", "POST",
             {"message": COMMIT_MSG, "tree": tree["sha"], "parents": [base]})
    print(f"  new commit {cm['sha'][:12]}")

    cur = req(f"/repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    if cur != base:
        sys.exit(f"main 已推进到 {cur[:12]}，终止 PATCH（未覆盖）")
    res = req(f"/repos/{REPO}/git/refs/heads/{BRANCH}", "PATCH",
              {"sha": cm["sha"], "force": False})
    print(f"\nPUSH OK -> {BRANCH} @ {res['object']['sha'][:12]}")
    print(f"https://github.com/{REPO}/commit/{cm['sha']}")


if __name__ == "__main__":
    main()

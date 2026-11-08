"""把 SFSC / FD-CAO 前哨实验推送到 kino-koki/SimplCore@main。

只新增文件、不修改任何现有文件（远端 main 已自行落地 P0/P1 修复，
旧 patch 脚本会失配）。走 GitHub Git Data API 单 tree + 单 commit。
token 只从 GH_TOK 环境变量读取，不写入任何产物。

  GH_TOK=... python push_sfsc.py --dry-run
  GH_TOK=... python push_sfsc.py --apply
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
                 "User-Agent": "simplcore-sfsc-push",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(r, timeout=90) as f:
        return json.loads(f.read().decode("utf-8"))


def exists(path):
    try:
        req(f"/repos/{REPO}/contents/{urllib.parse.quote(path)}?ref={BRANCH}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


NEW = {
    "docs/SFSC_FDCAO_PILOT.md": f"{ROOT}/SFSC_FDCAO_REPORT.md",
    "tools/sfsc_fdcao_probe/README.md": f"{ROOT}/README_sfsc.md",
    "tools/sfsc_fdcao_probe/sfsc_fdcao.py": f"{ROOT}/sfsc_fdcao.py",
    "tools/sfsc_fdcao_probe/results/gradcheck.txt": f"{ROOT}/results/gradcheck.txt",
    "tools/sfsc_fdcao_probe/results/toeplitz.txt": f"{ROOT}/results/toeplitz.txt",
    "tools/sfsc_fdcao_probe/results/toeplitz.json": f"{ROOT}/results/toeplitz.json",
    "tools/sfsc_fdcao_probe/results/results_sfsc_fdcao.json": f"{ROOT}/results_sfsc_fdcao.json",
}

COMMIT = """experiment(SFSC/FD-CAO): falsifiable pilot for two first-principles conjectures (6 arms, fp64 BPTT)

Two conjectures were proposed to beat attention without NxN / KV cache / O(L^2),
staying entirely inside the SGC family:

  SFSC  : rho_t, theta_t modulated by state EMA (linear time-varying IIR)
  FD-CAO: phase-resonance content addressing, cos(theta_m - phi_t)

Six equal-skeleton arms (static / sfsc / fdcao / fdcao_p / attn / attn_lr),
pure NumPy fp64 handwritten BPTT, no torch. Six-arm gradcheck PASS.

KEY NEGATIVE RESULT: SFSC's claim that FFT/IIR duality still holds is FALSE.
A linear TIME-VARYING system has a non-Toeplitz impulse response K(t,s);
check_toeplitz() measures mean abs deviation 17.67 and 0.972 relative error
when approximating it with a single averaged FFT kernel. Consequence: SFSC
cannot use the FFT training path, drops to O(L*M) recurrent scan, and loses
the O(L log L) claim plus theorem 1. Not an implementation bug -- structural.

TASK RESULTS (seed 0 only, held-out 512):
  bind  (2-hop MQAR): static .2422 sfsc .2500 fdcao .2402 fdcao_p .7656
                      attn .2617 attn_lr .2402
    NB: single-layer attention is NOT a valid upper bound here (2 hops needed),
    and fdcao_p injects the adjacent (key,value) prior explicitly. Do NOT read
    .7656 as "beating attention" -- it drops to .6113 on the 1-hop task.
  bind1 (1-hop, valid upper bound):
                      static .4160 sfsc .4141 fdcao .8672 fdcao_p .6113
                      attn 1.0000 attn_lr 1.0000
    -> FD-CAO clearly beats the static kernel, but still LOSES to equal-param
       low-rank attention (5,217 vs 5,152 params). "Beat attention" unproven.
  mod   (context modulation): only 2/6 arms ran (.5000 / .5352); task is
        first-order solvable, not a discriminative test. INCOMPLETE.
  osc   : NOT RUN.

HONEST BOUNDARIES (in docs/SFSC_FDCAO_PILOT.md):
  single seed, no bootstrap, synthetic L=9..25 / V=64 / D=64 / M=16,
  single layer, no 4M long-context / throughput / streaming checks,
  theta_m fixed (key->theta projection not implemented),
  FD-CAO is a frequency-slot variant of content-addressed state (same
  concept as the reverted CAS) and does NOT satisfy a strict "no external
  ideas" reading; fdcao_p's gain is a task prior, not an architecture win.

All files are additive; no existing source, doc, or checkpoint modified.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.apply):
        sys.exit("指定 --dry-run 或 --apply")

    print("== 检查本地文件 ==")
    files, ok = {}, True
    for path, local in NEW.items():
        if not os.path.exists(local):
            print(f"  [FAIL] 本地缺失 {local}"); ok = False; continue
        txt = open(local, encoding="utf-8").read()
        files[path] = txt
        print(f"  [OK] {path} ({len(txt)} chars)")

    print("\n== 检查远端路径占用 ==")
    for path in NEW:
        e = exists(path)
        print(f"  {'[CONFLICT]' if e else '[free]    '} {path}")
        if e:
            ok = False

    if not ok:
        sys.exit("\n存在缺失或冲突，终止（远端未改动）")
    if a.dry_run:
        print(f"\nDRY-RUN OK：{len(files)} 个新文件待提交，远端未改动")
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
             {"message": COMMIT, "tree": tree["sha"], "parents": [base]})
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

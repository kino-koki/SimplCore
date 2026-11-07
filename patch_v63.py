"""
v6.3 -- audit of our OWN v6.2 fairness claim.

v6.2 said "every arm receives exactly the same one-hot visible ids". True for
the ids, FALSE for the computation: only the TSC branch consumed the
predecessor view Xp, and the shared width d=24 < V=64 gave every learned
linear embedding a null space. Removing both asymmetries reverses v6.2's
headline training result (tsc_h2 0.3625 vs attn_x2 0.9543, paired -0.5918).

Rather than hide this, v6.3:
  * WITHDRAWS the v6.2 "trained layer bound" claim (W13);
  * gives explicit ZERO-TRAINING analytic witnesses for H=1 (three
    mechanisms) and H=2 (two mechanisms), all 1.0000, showing the function
    classes COINCIDE on this task and the training gap is optimisation /
    inductive bias, NOT expressivity;
  * keeps W12 (the stored 0.9987 record never reproduced) but sharpens the
    reason: the old low baseline numbers were an unfair implementation,
    not evidence about published models;
  * tightens positioning: NO expressivity advantage over attention is
    claimed anywhere. Theorem B is a constructive existence statement and
    is NOT withdrawn -- only the end-to-end-training corroboration is.

No new theorem. No architecture change. No LM training.
Run: python3 patch_v63.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

W = "/data/workspace"
GEN = W + "/build_simplcore_pdf_v5.py"
R = W + "/tools/eval_baselines/results"


def rw(old, new, count=1):
    s = open(GEN, encoding="utf-8").read()
    if new[:90] in s:
        print("  skip (applied):", old[:50].replace("\n", " "))
        return
    n = s.count(old)
    if n != count:
        raise RuntimeError("expected %d x %r, got %d" % (count, old[:70], n))
    open(GEN, "w", encoding="utf-8").write(s.replace(old, new, count))
    print("  patched:", old[:58].replace("\n", " "))


# ------------------------------------------------- 1. replace Sec 4.13 in full
s = open(GEN, encoding="utf-8").read()
a = s.find("    # --------------------------------------------------------- SEC 4.13 (v6.2)")
b = s.find('    Hh("5  What This Does NOT Establish"', a)
assert a > 0 and b > a, (a, b)

SEC = r'''
    # --------------------------------------------------------- SEC 4.13 (v6.3)
    _CB = _json.load(open(_os.path.join(_os.path.dirname(__file__),
        "tools/eval_baselines/results/corrected_baselines.json")))
    _AE = _json.load(open(_os.path.join(_os.path.dirname(__file__),
        "tools/eval_baselines/results/analytic_equivalence.json")))
    _AH = _json.load(open(_os.path.join(_os.path.dirname(__file__),
        "tools/eval_baselines/results/analytic_h2.json")))

    def _cb(hops, arm):
        _v = [r["acc"] for r in _CB["rows"]
              if r["hops"] == hops and r["arm"] == arm]
        _m = sum(_v) / len(_v)
        _sd = (sum((x - _m) ** 2 for x in _v) / len(_v)) ** 0.5
        return _m, _sd

    Hh("4.13  Fairness audit: two asymmetries removed, and a withdrawn "
       "training claim", size=9.6, before=5.0, after=2.0)
    P("<b>We audited our own comparison.</b> v6.2 stated that every arm "
      "received the same one-hot ids. That was true of the IDS but not of "
      "the COMPUTATION: only the TSC branch consumed the predecessor view "
      "Xp, and the shared width d=24 was below V=64, so every learned "
      "linear embedding had a null space while TSC kept a hard-coded "
      "pairing bias. Both are now removed: all eight arms receive "
      "(U, Up, q, mask) with ONE fixed full-width orthonormal Hadamard "
      "embedding (d=V=64, nothing learned). This is registered as W13.",
      size=SMALL, lead=SLEAD, after=2.0)
    _t = [["hops", "arm", "acc (5 seeds)", "analytic"]]
    for _h, _a, _an in ((1, "tsc_h1", "yes"), (1, "linear_pair", "yes"),
                        (1, "attn_x1", "yes"), (1, "gla_pair", "no"),
                        (2, "tsc_h2", "yes (stack)"),
                        (2, "attn_x2", "yes (stack)")):
        _m, _sd = _cb(_h, _a)
        _t.append([str(_h), _a, "%.4f +/- %.4f" % (_m, _sd), _an])
    d.table(_t, [24, 68, 76, 60], size=6.9, lead=8.2, after=2.0)
    _ci = _CB["paired_ci_h2"]["tsc_h2_minus_attn_x2"]
    P("<b>W13: v6.2's trained-layer claim is WITHDRAWN.</b> Under the "
      "corrected protocol the two-layer static-LTI stack reaches only "
      "%.4f on H=2, while the two-layer cross-attention reference reaches "
      "%.4f; the paired difference is %+.4f (95%% CI [%+.4f, %+.4f]). "
      "v6.2 reported 0.9512 and +0.8457 from the ASYMMETRIC setup. We "
      "therefore withdraw the claim that the exactly-H-layer bound was "
      "corroborated by end-to-end training. <b>Theorem B itself is NOT "
      "withdrawn</b>: it is a constructive existence statement with an "
      "analytic proof under stated premises (Sec 3.6, Appendix A).",
      % (_cb(2, "tsc_h2")[0], _cb(2, "attn_x2")[0], _ci["mean"],
         _ci["ci95"][0], _ci["ci95"][1]),
      size=SMALL, lead=SLEAD, after=2.0)
    P("<b>Why this does NOT become an expressivity result either way.</b> "
      "We give EXPLICIT ZERO-TRAINING weight assignments on the same "
      "corrected data. For H=1, %.4f / %.4f / %.4f for the static-LTI "
      "layer, linear attention and single-layer cross-attention; for H=2, "
      "%.4f / %.4f for the two-layer static-LTI stack and the two-layer "
      "attention stack (5 seeds each). All are 1.0000. The construction is "
      "one algebraic object: Wk=Wv=Wq=I, Wout=C[V_K:,:]^T, "
      "hop(q)=sum_s (q.Up_s)U_s K(tau) or its sharpened softmax, and "
      "M=C[V_K:,:]^T C[:V_K,:] maps the value code to the next key code. "
      "The two function classes therefore COINCIDE on this task; the "
      "observed training gap is OPTIMISATION / INDUCTIVE BIAS, not "
      "expressivity. This is an observation with an explicit witness, "
      "NOT a new theorem and NOT a superiority claim in either direction. "
      "It makes precise our T6 remark on the rho=1 single-pole isomorphism." %
      (next(x["mean"] for x in _AE["summary"] if x["arm"] == "tsc_h1"),
       next(x["mean"] for x in _AE["summary"] if x["arm"] == "linear_pair"),
       next(x["mean"] for x in _AE["summary"] if x["arm"] == "attn_x1"),
       next(x["mean"] for x in _AH["summary"] if x["mode"] == "tsc"),
       next(x["mean"] for x in _AH["summary"] if x["mode"] == "attention")),
      size=SMALL, lead=SLEAD, after=2.0)
    P("<b>W12 retained, reason sharpened.</b> The stored 0.9987 attention "
      "record from the earlier single-hop run still has NO verifiable "
      "same-run reproduction, so the Fisher-parity statement remains "
      "withdrawn. The additional reason is now clearer: v6.2's low "
      "reference numbers came from an UNFAIR implementation, not from the "
      "published mechanisms. We make NO claim of equivalence, superiority "
      "or inferiority with respect to attention anywhere in this paper. "
      "All rows, both versions, are released so the audit can be "
      "falsified.", size=SMALL, lead=SLEAD, after=2.5)

'''
open(GEN, "w", encoding="utf-8").write(s[:a] + SEC + s[b:])
print("  replaced Sec 4.13 (%d -> %d chars)" % (b - a, len(SEC)))

# ------------------------------------------------------------------ 2 abstract
rw('''      "Toeplitz, O(1) construction, and the H-layer bound now also holds after "
      "end-to-end training from random initialisation (five seeds). A same-run "
      "audit failed to reproduce an earlier stored attention baseline, so that "
      "parity claim is withdrawn (W12). We make no language-modeling, "
      "perplexity, throughput, equivalence or superiority claim." %
      (max(q["max_X_dev"] for q in IB_P2),''',
   '''      "Toeplitz, O(1) construction. A fairness audit of our own comparison "
      "removed two input asymmetries, reversed an earlier end-to-end-training "
      "claim (W13), and produced explicit zero-training witnesses showing that "
      "the static-LTI and attention function classes COINCIDE on this task. "
      "An earlier stored attention baseline never reproduced (W12). We make "
      "no language-modeling, perplexity, throughput, equivalence, "
      "expressivity or superiority claim." %
      (max(q["max_X_dev"] for q in IB_P2),''')

# ------------------------------------------------------------------ 3 Sec 5
rw('''    It("<b>Parity claim WITHDRAWN (W12).</b> The earlier 'no significant "
       "difference detected' statement (Fisher p = %.4f) relied on a stored "
       "attention record. The same-run audit of Section 4.13 did not "
       "reproduce it, so the statement is withdrawn. We make NO claim of "
       "equivalence, superiority or inferiority relative to attention." % FP,
       size=SMALL, lead=SLEAD)''',
   '''    It("<b>Parity and trained-layer claims WITHDRAWN (W12 / W13).</b> The "
       "earlier 'no significant difference detected' statement (Fisher "
       "p = %.4f) relied on a stored attention record with no verifiable "
       "same-run reproduction. A later end-to-end-training claim relied on "
       "a comparison that gave the predecessor input to one side only. Both "
       "are withdrawn; Section 4.13 gives explicit zero-training witnesses "
       "showing the two function classes coincide on this task. We make NO "
       "claim of equivalence, superiority, inferiority or expressivity "
       "advantage relative to attention." % FP,
       size=SMALL, lead=SLEAD)''')

rw('''       "Section 4.13 supplies same-condition TOY mechanism references, not "
       "full reproductions of GLA / DeltaNet / Mamba-2 / Transformer; they "
       "are a reproduction audit, not a performance ranking. Earlier O1-O4 "
       "lack >=5-seed paired CIs; the Section 4.13 results do have them.",''',
   '''       "Section 4.13 supplies same-condition TOY mechanism references, not "
       "full reproductions of GLA / DeltaNet / Mamba-2 / Transformer; they "
       "are a FAIRNESS / reproduction audit, not a performance ranking. "
       "Earlier O1-O4 lack >=5-seed paired CIs; the Section 4.13 results "
       "do have them.",''')

# ------------------------------------------------------------------ 4 Sec 8
rw('''"new SOTA sequence model. The H-layer bound also survives end-to-end "
       "training from random initialisation; the earlier attention-parity "
       "claim is withdrawn because it did not reproduce (W12). No claim "
       "of equivalence or superiority is made.", size=SMALL, lead=SLEAD)''',
   '''"new SOTA sequence model. A fairness audit of our own comparison "
       "withdrew the end-to-end-training corroboration (W13) and gave "
       "explicit zero-training witnesses that the static-LTI and attention "
       "function classes coincide on this task; the stored attention parity "
       "record never reproduced (W12). No claim of equivalence, "
       "expressivity advantage or superiority is made.",
       size=SMALL, lead=SLEAD)''')

# --------------------------------------------------------------- 5 appendix F
rw('''        ["W12: single-hop parity with attention (Fisher p=0.4998)", "WITHDRAWN: the 0.9987 baseline came from a stored record and did not reproduce in the same-run audit of Sec 4.13. No equivalence or superiority claim is made."],
    ], [130, 108], size=7.6, lead=9.4)''',
   '''        ["W12: single-hop parity with attention (Fisher p=0.4998)", "WITHDRAWN: the 0.9987 baseline came from a stored record with no verifiable same-run reproduction. No equivalence or superiority claim is made."],
        ["W13: v6.2's 'trained layer bound' (0.9512, paired +0.8457)", "WITHDRAWN: that comparison gave the predecessor view Xp ONLY to the TSC branch and used d=24 < V=64. After removing BOTH asymmetries, tsc_h2 = 0.3625 vs attn_x2 = 0.9543 (paired -0.5918). Theorem B is a constructive existence statement and is NOT withdrawn; only the end-to-end-training corroboration is. Explicit zero-training witnesses for H=1 and H=2 are all 1.0000 for BOTH mechanisms, so the gap is optimisation / inductive bias, not expressivity."],
    ], [130, 108], size=7.6, lead=9.4)''')

# --------------------------------------------------------------------- 6 logs
def prepend(path, title, body):
    s = open(path, encoding="utf-8").read()
    i = s.find("\n## ", 5)
    if i < 0:
        i = s.find("\n", s.find("\n") + 1)
    open(path, "w", encoding="utf-8").write(
        s[:i + 1] + "\n" + title + "\n\n" + body + "\n" + s[i + 1:])
    print("  updated", os.path.basename(path))


CB = json.load(open(R + "/corrected_baselines.json"))
AE = json.load(open(R + "/analytic_equivalence.json"))
AH = json.load(open(R + "/analytic_h2.json"))


def cm(h, arm):
    v = [r["acc"] for r in CB["rows"] if r["hops"] == h and r["arm"] == arm]
    return sum(v) / len(v)


c = CB["paired_ci_h2"]["tsc_h2_minus_attn_x2"]

prepend(W + "/CHANGELOG.md",
        "## v6.3：公平性自审、W13 撤回、解析等价 witness（最新）",
        f"""- **自查发现 v6.2 对照有两处输入不对称**：`Xp`（前驱）只给 TSC 分支；
  d=24 < V=64 使可学线性嵌入有零空间，而 TSC 保留硬编码配对偏置。
- **修正**：八臂统一 (U, Up, q, mask)，固定满宽正交 Hadamard（d=V=64，E 不学）。
- **W13 正式撤回** v6.2 §4.13「训练版层数下界」：修正后
  H=2 `tsc_h2` {cm(2,'tsc_h2'):.4f} vs `attn_x2` {cm(2,'attn_x2'):.4f}，
  配对差 {c['mean']:+.4f}，95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]；
  v6.2 的 0.9512 / +0.8457 来自不公平设置。**Theorem B 本身不撤回**
  （解析构造性存在命题），撤回的是端到端训练的旁证。
- **新证据（正面）**：零训练解析 witness，H=1 三机制
  {[round(x['mean'],4) for x in AE['summary']]}，
  H=2 两机制 {[round(x['mean'],4) for x in AH['summary']]}，全部 1.0000。
  ⇒ 该任务上**函数类等价**，训练差距是优化 / 归纳偏置，**不是表达力**。
  写成观察 + 显式 witness，**不新增定理**。
- **W12 保留但理由收窄**：历史 0.9987 仍无同场复现；v6.2 的低基线数字
  是不公平实现的结果，不是关于已发表模型的结论。
- 定位再收紧：全文**不作任何相对 attention 的表达力优势主张**；
  T6 的 rho=1 同构从「开放态度」升级为显式 witness。
- 论文主文仍 8 页；§4.13 整段重写，摘要 / §5 / §8 / 附录 F / Broader Impact 同步。
- 判定 7/7；无 LM 训练、无 ppl、无生成样本、无私有权重字节。""")

prepend(W + "/REVISION_LOG.md",
        "## 6.11 v6.3：主动证伪自己的公平性声明",
        """| 项 | v6.2 | v6.3 |
|---|---|---|
| `Xp` 前驱输入 | **仅 TSC 分支** | 八臂统一 |
| 宽度 | d=24 < V=64，E 可学 | d=V=64，固定正交 Hadamard |
| H=2 训练 | `tsc_h2` 0.9512，配对 +0.8457 | **`tsc_h2` 0.3625 vs `attn_x2` 0.9543，配对 −0.5918** |
| 训练版层数下界 | 称为主证据 | **撤回 W13** |
| 解析等价 | 无 | H=1 三机制、H=2 两机制零训练全 1.0000 |

**W13（正式撤回）**：v6.2 的「层数下界经端到端训练仍成立」依赖
不对称对照，修正后结论反转，整条撤回。Theorem B 是解析构造性存在命题，
**不撤回**；撤回的只是训练旁证。

**关键区分（本轮最硬的一句话）**：零训练解析 witness 对两类机制都成立
⇒ 函数类在该合成任务上重合 ⇒ 训练差距是优化 / 归纳偏置，不是表达力。
因此本轮既不能读成「TSC 输给 attention」，也不能读成「TSC 赢」。
这把 T6 的 rho=1 单极点同构从开放态度变成显式 witness，并主动堵死
「isn't this linear attention with extra steps」这条必杀质疑。

**W12 理由收窄**：历史 0.9987 记录仍无同场复现，parity 声明保持撤回；
但 v6.2 把低基线数字写成「audit limitation」仍不够精确 —— 真因是
实现不公平，已在 §4.13 与附录 F 写明。

**另修**：v6.2 摘要 / §8 里「H-layer bound also survives end-to-end
training」全部删除；§5 第一条合并 W12/W13；附录 F 增 W13 行。""")

print("\n--- rebuilding ---")
r = subprocess.run([sys.executable, GEN], cwd=W, capture_output=True, text=True)
print((r.stdout or "")[-1000:])
if r.returncode:
    print((r.stderr or "")[-2500:])
    sys.exit(1)
print("PATCH v6.3 OK")

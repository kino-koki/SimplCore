import os, json, base64, urllib.request, hashlib

W = "/data/workspace"

# ------------------------------------------------------------------ logs
p = W + "/CHANGELOG.md"
s = open(p, encoding="utf-8").read()
a = "## 移除最后一个外部 side-channel：码本内生角色分离 + Corollary R2 + 论文 v5.2（最新）"
add = '''## Theorem A：输入侧信息紧下界 + 贡献类型转换 + P0 清理（v6，最新）

### 1) 主交付：把最大软肋做成核心定理
- **Theorem A**：精确 H 跳检索需要 B >= log2|A| bit 输入侧角色信息；
  显式族 |A|=2^m（m<=n）⇒ B>=n；配对也不在 x 里时 B>=log2(n!)
- **诚实划界**：(a)(b) 是普适信息论（任何确定性模型），**不依赖 LTI**，
  不声称首次发现；架构增量是 (c) Toeplitz 对偶**强制** bit 只能在输入端 /
  (d) 三载体守恒且紧 / (e) bit-深度**直角**前沿 / (f) overshoot 非架构定律
- **修正 R3 口径**：旧「恰好 1 bit 必要」是 |A|=2 最坏情况特例；
  定理 A 推广为总 bit 下界并撤回该过度断言，R3 未被推翻

### 2) 判别性 witness（12/12，验证器 18/18）
- P2 前提可证伪：2^m 个角色分配产生的 X 逐位相同，max dev **0.0**
- 穷举 bit 扫描 m=1..4：acc=2^(B-m)，**恰在 B=m 跳到 1.0**
- 真实静态 LTI 前向 H_L=1..8、B<m：同一纤维内输出逐位相同，**深度完全不补偿**
- 三载体：tag 1.0000 / codebook 1.0000（FFT-IIR 2.89e-16）/ format prior 0.2969
- 两种解码器口径（信息论最优 vs 架构实测）**分开标注**，不混用

### 3) 论文 v6：换贡献类型
- 标题/摘要/Table 1/§1.2/§3.7/§4.8/§6/§7/§8/Appendix A.0 全部重写或新增
- 主文 **7 页**（8 页限内，留余量），总 24 页；FOUR core results，不是 30 条
- 定理 A 完整解析证明（6 处 QED + 8 条 SCOPE）放 Appendix A.0 最前
- §3.1/A.1/A.13 **证明漏洞修复**：显式构造 ker 向量（Δv1=β2u, Δv2=-β1u）
- 归一化统一 `CCᵀ=I`；§3.2 (d) 限定 H1-H7 + 末层读出
- 删 §6 与 Table 1 重复的 Table 2；§7 换成定理 A 的可证伪 bit 下界

### 4) P0 逐条清完
- 摘要矛盾 / A.1 证明 / 归一化 / 表格口径 / 引用与匿名 / Broader Impact 顺序 /
  全大写与内部编号 / self-evolution 工程日志整包 —— 全部处理
- 自识别路径 `sgc.py`/`loop.py`/`core/arch`/`.json`/`results/`/`docs/` **全篇 0 次**
- 正文 W9/NT31/EXACTLY/17/17/EXACTLY ONE BIT/boundary-characterization **全 0**
- 修掉中断轮次留下的重复 `%` 格式化 `TypeError`（生成器此前无法编译）
- 修掉两个真渲染 bug：`<b>` 原样画入表格、§1.1 跨栏孤立尾句
- References 15 条联网核验 ID↔标题/venue，统一格式，作者名省略并说明

### 5) 未补（照实）
- P1 的小公平对照（MQAR / GLA / DeltaNet / Mamba-2 / 2 层 TS，>=5 seed CI）**没跑**
- 按建议**不硬塞** >=10M token、val ppl、生成可读性、吞吐对比四类大实验；§7 明写
- 合成内容链任务，无 LM 训练、无 ppl、无生成样本，不是语言能力结论

'''
assert a in s
open(p, "w", encoding="utf-8").write(s.replace(a, add + a))

p = W + "/REVISION_LOG.md"
s = open(p, encoding="utf-8").read()
a = "## 6.7 v5.2：移除外部 side-channel（Corollary R2，前提第三次放宽）"
add = '''## 6.8 v6：Theorem A 输入侧信息下界 + 贡献类型转换 + P0 清理

| 项 | v5.3 | v6 |
|---|---|---|
| 定位 | 边界刻画 / 恰好 1 bit 紧刻画 | **紧的 log2|A| bit 输入侧信息下界 + 设计规则** |
| 主标题 | Exact Retrieval Needs ... Exactly One Role Bit | Where the Bits Must Live: A Tight Input-Side Information Bound |
| 主文核心结果 | 6 定理 | **4 条**（1 主定理 + 2 紧界 + 1 构造） |
| 主文页数 | 8（顶格） | **7**（留 1 页余量） |
| Appendix A 首项 | A.31 层紧性 | **A.0 Theorem A 完整证明** |
| Appendix G | 51 项全量台账 | 一页 Statement Classification |

**新增 Theorem A**：(a) 普适不可区分 ≤1/|A|；(b) B>=log2|A|，显式族 |A|=2^m；
(c) Toeplitz 对偶强制 bit 在输入端（放进核 ⇒ 时变 ⇒ 非 Toeplitz）；(d) 三载体守恒且紧；
(e) bit-深度直角前沿；(f) overshoot 是读出约定产物。验证器 18/18。

**诚实划界（本轮最重要的方法学动作）**：(a)(b) 不做任何架构假设，
明写「universal information theory, we do NOT claim they are new」；
novelty 收窄为位置定理 / 守恒 / 直角前沿。避免「前提过窄的空定理」。

**撤回/修正登记 W10（口径修正，非新撤回）**：旧「恰好 1 bit 必要」
（Lemma R3）降级为 |A|=2 最坏情况特例；「1 bit/token」是充分载体、
既非必要也非信息量最小；显式 role tag 不必要（码本分区已证）。R3 未被推翻。

**P0 修复清单**：摘要矛盾、A.1/A.13 的 ker→Δv 跳步（改为显式构造）、
`CCᵀ=I`/`RI` 混用统一、表格口径、References 规范化与 ID 核验、
Broader Impact 前移至 References 之前、正文全大写与内部编号清零、
self-evolution 工程日志（NT14/15/20/21/22/23/24）从附录过滤、
自识别路径全篇清零、§6 重复 Table 2 删除、§7 脱节的 NT24 预测换主线。
另修生成器重复 `%` 格式化导致的 `TypeError`（此前无法编译）与两个渲染 bug。

**格式校验**：24 页、主文 7 页、实体/TeX 碎片/旧口径残留全 0、
非 ASCII 仅 38 个 `·`、Broader Impact < References < Appendix A 顺序正确。

'''
assert a in s
open(p, "w", encoding="utf-8").write(s.replace(a, add + a))
print("logs updated")

# ------------------------------------------------------------------ push
T = os.environ["GH_TOKEN"]
R = "kino-koki/SimplCore"
H = {"Authorization": "Bearer " + T, "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-v60", "X-GitHub-Api-Version": "2022-11-28"}


def api(path, method="GET", payload=None):
    raw = json.dumps(payload).encode() if payload is not None else None
    q = urllib.request.Request("https://api.github.com" + path, data=raw,
                               method=method,
                               headers={**H, **({"Content-Type": "application/json"}
                                                if raw else {})})
    return json.loads(urllib.request.urlopen(q, timeout=180).read())


head = api("/repos/%s/git/ref/heads/main" % R)["object"]["sha"]
base = api("/repos/%s/git/commits/%s" % (R, head))["tree"]["sha"]
print("base", head[:12])

RES = W + "/train/superposition_probe/results/"
THR = W + "/theory_work/results/"
files = [
    ("simplcore.pdf", W + "/simplcore.pdf"),
    ("build_simplcore_pdf_v5.py", W + "/build_simplcore_pdf_v5.py"),
    ("V60_INFO_BOUND_REPORT.md", W + "/V60_INFO_BOUND_REPORT.md"),
    ("theory_work/info_bound.py", W + "/theory_work/info_bound.py"),
    ("theory_work/results/info_bound_theorem.json", THR + "info_bound_theorem.json"),
    ("theory_work/results/info_bound_check.json", THR + "info_bound_check.json"),
    ("tools/native_probe/info_bound_probe.py",
     W + "/train/superposition_probe/info_bound_probe.py"),
    ("tools/native_probe/results/info_bound_scan.json", RES + "info_bound_scan.json"),
    ("tools/native_probe/results/info_bound_depth.json", RES + "info_bound_depth.json"),
    ("tools/native_probe/results/info_bound_carriers.json", RES + "info_bound_carriers.json"),
    ("tools/native_probe/results/info_bound_adjudication.json",
     RES + "info_bound_adjudication.json"),
    ("CHANGELOG.md", W + "/CHANGELOG.md"),
    ("REVISION_LOG.md", W + "/REVISION_LOG.md"),
]
tree = []
for path, local in files:
    raw = open(local, "rb").read()
    b = api("/repos/%s/git/blobs" % R, "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
    print("  + %-58s %9d B" % (path, len(raw)))

nt = api("/repos/%s/git/trees" % R, "POST", {"base_tree": base, "tree": tree})
msg = """v6: Theorem A - a tight input-side information bound; change the contribution type

"Fixing errors is defence, changing the contribution is offence." This round turns
the single most damaging objection ("the retrieval is solved by the input format")
into the MAIN THEOREM, and clears the P0 desk-rejection items.

THEOREM A (complete analytic proof, Appendix A.0; verifier 18/18)
  (a) Universal impossibility: for ANY deterministic model, if |A| >= 2 distinct
      answers are compatible with the visible input, accuracy <= 1/|A| <= 1/2.
  (b) Bit bound: exact retrieval needs B >= log2|A| input-side bits. Explicit
      family with |A| = 2^m for every m <= n => B >= m, B >= n at m = n; if the
      pairing is also absent from x, B >= log2(n!). Independent of depth, state
      order and kernel parameters.
  (c) LOCATION IS FORCED: within the static-LTI class the bits cannot be created
      internally (the only content channels are the fixed embedding and absolute
      positions, which are independent of the role assignment under arbitrary
      filler spacing). Encoding them in the kernel makes coefficients time-varying;
      by Toeplitz IFF time-invariance (T4) that destroys the single-FFT / IIR path.
      To keep duality, the bits MUST be on the input side.
  (d) TIGHTNESS + THREE-WAY CONSERVATION: explicit tag, alphabet partition V->2V,
      and positional format prior each carry exactly one bit per ambiguous token;
      only the partition is annotation-free AND preserves duality. The prior keeps
      its bit in an INPUT-DISTRIBUTION ASSUMPTION, so arbitrary filler spacing
      drops its effective information to 0 (1.0000 -> 0.2969).
  (e) RIGHT-ANGLE BIT-DEPTH FRONTIER: B < log2|A| => no depth suffices;
      B >= log2|A| => minimum depth is still exactly H. No smooth trade-off.
  (f) The H_L > H overshoot is an artifact of last-layer read-out, not an
      architectural law; selecting the H-th intermediate output makes H_L >= H
      sufficient while the lower bound H remains.

HONESTY ABOUT SCOPE (keeps this from being a vacuous theorem)
  (a),(b) are PURE INFORMATION THEORY and need NO LTI/depth/state/read-out
  assumption. The paper states this explicitly and does NOT claim they are new.
  The architectural novelty is (c),(d),(e),(f). Zoology / Repeat After Me /
  Illusion of State established recall weakness; none quantifies a bit lower
  bound or proves the bits are forced to the input by Toeplitz duality.

RELATION TO OUR OWN EARLIER LEMMA R3 ("exactly one bit is necessary")
  R3 is the |A| = 2 WORST-CASE special case (accuracy 0 on one legal dataset),
  whereas (a) gives the weaker uniform-average bound 1/|A|. R3 is NOT overturned;
  Theorem A generalizes it and REMOVES the over-statement of necessity:
  "one bit per token" is a sufficient carrier, neither necessary nor
  information-theoretically minimal, and an explicit role tag is unnecessary
  (the codebook partition of R2 has none). Recorded as W10 (scope correction).

FALSIFIABLE WITNESSES (12/12 adjudication; certification, NOT the proof)
  - P2 premise tested, not assumed: all 2^m role assignments give BIT-IDENTICAL
    inputs, max deviation 0.0 for m=2,3,4.
  - Exhaustive bit scan m=1..4: accuracy 2^(B-m), step to 1.0000 exactly at B=m.
  - A REAL static-LTI stack, H_L=1..8, B<m: output bit-identical across all
    instances in one side-channel fibre (deviation 0.0) => depth cannot substitute.
  - Three carriers measured; "information-theoretic optimum" and "measured
    static-LTI" decoder regimes are separately labelled, never mixed.

PAPER v6 (simplcore.pdf, 24 pages, MAIN PAPER 7 pages, within 8-page limit)
  - New title, five-sentence abstract, Table 1 gains the Theorem A exact-increment
    row; Sec 1.2 sharpens 30 statements to FOUR core results; Sec 3 renumbered.
  - NEW Sec 3.7: Theorem A with premises P1-P5 stated on the same page.
  - NEW Sec 4.8: bit scan table, depth-invariance check, carrier conservation.
  - Appendix A.0: full analytic proof (six QEDs, eight SCOPE items) placed FIRST.
  - Sec 6 duplicate Table 2 removed; Sec 7 replaces the off-topic zero-order
    self-evolution prediction with Theorem A's falsifiable bit bound; Sec 8
    rewritten around the information bound.

P0 DESK-REJECTION ITEMS, ALL CLEARED
  - Abstract "necessary and sufficient" contradiction -> layout-conditional +
    total-bit bound.
  - A.1/A.13 proof gap (dim ker > 0 does not guarantee a kernel vector perturbs
    the target coordinate) -> explicit construction dV1 = beta2 u, dV2 = -beta1 u.
  - Code normalization unified on C C^T = I (removed R I / delta mixes).
  - Table/chance/overshoot labelling; every cell says what it measures.
  - References: arXiv IDs verified online, uniform format, author names omitted
    with an explicit note (no fabricated authors). Broader Impact moved BEFORE
    References.
  - ALL-CAPS and internal IDs (W9/NT31/EXACTLY/17/17/EXACTLY ONE BIT) removed
    from the main paper: 0 occurrences.
  - Separate self-evolution project's engineering log (NT14/15/20/21/22/23/24)
    filtered out of the appendix; Appendix G shrunk from a 51-item ledger to a
    one-page Statement Classification.
  - Self-identifying paths (sgc.py, loop.py, core/arch, *.json, results/, docs/,
    /data/workspace) are 0 occurrences paper-wide.
  - Two real rendering bugs fixed: <b> drawn literally inside a table cell, and
    a stranded line from a duplicated Sec 1.1 paragraph above the Introduction.
  - Fixed a stale duplicate-% formatting TypeError that made the generator
    uncompilable after an interrupted earlier round.

STILL NOT DONE, NOT CLAIMED: the small, fair, >=5-seed paired-CI comparison
(MQAR / GLA / DeltaNet / Mamba-2 / 2-layer Transformer). Per the review advice
we deliberately did NOT add an unfair >=10M-token equal-compute LM run, val
perplexity, generation readability or throughput comparison; Sec 7 says so.
No LM training, no ppl, no generation sample, no private weight bytes, no token
in this commit.
"""
c = api("/repos/%s/git/commits" % R, "POST",
        {"message": msg, "tree": nt["sha"], "parents": [head]})
api("/repos/%s/git/refs/heads/main" % R, "PATCH", {"sha": c["sha"], "force": False})
print("\nPUSHED:", c["sha"])
print("https://github.com/kino-koki/SimplCore/commit/" + c["sha"])
loc = open(W + "/simplcore.pdf", "rb").read()
print("PDF sha256:", hashlib.sha256(loc).hexdigest()[:20])

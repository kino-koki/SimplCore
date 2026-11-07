"""update_logs_v7.py — register v7 and fix the stale MANUSCRIPT_INDEX path."""
import os

W = "/data/workspace"


def prepend(path, marker, block):
    s = open(path, encoding="utf-8").read()
    i = s.find(marker)
    assert i >= 0, (path, marker)
    j = s.find("\n", i) + 1
    open(path, "w", encoding="utf-8").write(s[:j] + "\n" + block + "\n" + s[j:])
    print("  updated", os.path.basename(path))


# fix stale path references in the two logs
for f in ("CHANGELOG.md", "REVISION_LOG.md"):
    p = W + "/" + f
    s = open(p, encoding="utf-8").read()
    s = s.replace("`docs/MANUSCRIPT_INDEX.md`", "`MANUSCRIPT_INDEX.md`")
    open(p, "w", encoding="utf-8").write(s)

prepend(W + "/CHANGELOG.md", "---", """## v7：一次性重构投稿稿（最新）

- **`simplcore.pdf` 替换为 v7**：10 页，主文 7 页（8 页上限内）；
  新生成器 `build_simplcore_pdf_v7.py` 是**唯一当前稿源**，旧 v5 生成器降为历史。
- **七类硬伤全部修完**：B 重定义为字母表相对 flat 参考的超出位数（分区的 1 bit 正式计入 B）、
  A(c) 降级为类内设计选择、A(a) 改为 per-fibre 条件界并定义两种统计量、
  统一「not bit-free, it is annotation-free」措辞、编号收束为 **4 条核心结果**
  （Theorem A / Theorem B / Construction C / Empirical E，其余 P1–P8 / L1–L3）、
  P2 换成显式三情形构造、H6 层接口唯一化并分离两个核家族、
  一张统一数字表 + Table 10 数字溯源审计。
- **呈现/合规**：删掉首页 meta 话术与全部审稿人喊话；References 14 条统一 venue
  （Zoology 修回 ICLR 2024）、删自证句；Broader Impact 在 References 之前；
  **旧附录 ledger 整包删除**（25 页 → 10 页）。
- **P1 收尾**：新增 `fibre_probe.py` 枚举原任务 |A|（flat/H=4：max 5，
  log₂|A| ≤ 2.3219；分区提供 8 bits，充分但不紧），桥接 tight witness 与主任务；
  P5 状态阶下界升为带完整证明的**条件**定理并写出分支因子 2 时的退化。
- **新增硬核内容**：状态记账推导、P1–P8 完整陈述、线性注意力精确关系、
  bit/depth/order 前沿、完整 5×5 网格、carrier 守恒、深度不变性、
  5-seed 全统计、消融账本、逐条前提—边界—非空洞实例总表、逐条可证伪条件、复现协议。
- **自动化预演 `desk_check_v7.py`：45/45 PASS**（不是真人评审，只验证陈述自洽与口径）。
- W12 撤回不变；六类硬实验（≥10M token LM、同算力强基线、真实 MQAR、
  可学习码本、收敛的公平对照、TS2SC 端到端迁移）仍 `[待补]`。
- 训练/迁移任务继续冻结，零私有权重字节。""")

prepend(W + "/REVISION_LOG.md", "---", """## 7.0 v7：从「补丁」改为「重写」，七类硬伤一次性收口

| 维度 | v6.2 | v7 |
|---|---|---|
| 生成器 | `build_simplcore_pdf_v5.py`（层层打补丁） | `build_simplcore_pdf_v7.py`（自包含重写） |
| 核心结果计数 | FOUR / SIX / 八 混用 | **恰好 4**（A / B / C / E） |
| 定理命名 | Theorem 1/5/6、T1–T6、NT*、A.31+ | 仅 `Theorem A` / `Theorem B`，其余 P1–P8 / L1–L3 |
| B 的定义 | `Σ b_t` 只算显式 side channel | `Σ_t log₂ |π⁻¹(v_t)|`，分区 bit 计入 B |
| A(a) | 无条件 `≤ 1/|A| ≤ 1/2` | per-fibre `acc(F|v) ≤ 2^B/|A(v)|` + 两种统计量定义 |
| A(c) | "duality FORCES" | 类 M 内、IF 保持对偶时的设计选择 |
| P2 证明 | kernel 维数 ⇒ 存在目标扰动 | 显式三情形构造，明写不再用该推断 |
| 层接口 | 四套互斥写法 | H6 唯一 + 核家族分离说明 |
| 数字 | 四处漂移 | Table 2 单表 + Table 10 溯源审计 |
| 页数 | 25 页 | **10 页**（主文 7 页） |

**新增撤回/口径修订登记**：

- **W13（口径修订）**：`B = Σ b_t` 的旧定义不覆盖 alphabet partition 折进 symbol id 的
  1 bit，导致 tightness 形式不成立；本版重定义 B，tightness 恢复成立。
- **W14（口径修订）**：A(a) 的无条件 `accuracy ≤ 1/|A| ≤ 1/2` 被 §4.9 的 marginal
  0.8023 / 1.0000 反证；本版改为 per-fibre 条件界，并定义两种统计量。
- **W15（措辞统一）**：「free of run-time annotation」与「supplies the required
  information」读起来像无 bit；统一为「not bit-free, it is annotation-free」。
- **W16（编号统一）**：Theorem 1 = 最小张量阶 / Theorem 1 = Z 变换对偶、
  T6 三种描述、FOUR / SIX 计数冲突，全部收束。
- **W17（证明自洽）**：`dim ker > 0 ⇒ 存在改变目标坐标的 Δv` 与 §3.1 显式构造并存；
  本版全篇只用显式构造。
- **W18（数字统一）**：0.2969 / 0.2711、0.9316 / 0.9414 / 0.9328、四种 FFT/IIR 误差、
  0.0313 与 0.25 并列 —— 见 Table 10 溯源审计，本稿每个配置只有一个值。

**格式**：ICML 2027 官方 `.sty` 与 CFP 截稿日仍未公布；本版按 ICML 2026 已公布规则排版
（双栏、匿名、主文 8 页上限、Broader Impact 在 References 之前、影响声明不限页）。
待模板发布需重核页数、LLM 使用披露与互审提名 —— 写进 Appendix C，不冒充已合规。

**明确未做**：不承诺过稿率；不把 7 页主文灌成 8 页（再填满只能复制附录证明，
与「无注水」冲突）；不改任何实验数字；不重启训练。""")

# manuscript index: point at the NEW generator, archive the old one
p = W + "/MANUSCRIPT_INDEX.md"
s = open(p, encoding="utf-8").read()
old = """| `simplcore.pdf` | 英文、匿名、ICML 版式。主文 8 页上限内，References / Broader Impact / Appendix 单列。**只有这一份是稿。** |"""
new = """| `simplcore.pdf` | **v7**：英文、匿名、ICML 版式。10 页，主文 7 页（8 页上限内），Broader Impact / References / 精简附录单列。**只有这一份是稿。** |
| `build_simplcore_pdf_v7.py` | v7 的**唯一当前生成器**，自包含；所有数字由原始 JSON 注入。 |
| `build_simplcore_pdf_v5.py` | v6.2 及更早的历史生成器，已归档，**不再产生当前稿**。 |
| `V7_REPORT.md` | v7 修订总报告：七类硬伤逐条对应、P0/P1 落地、45 项预演与诚实边界。 |
| `tools/eval_baselines/desk_check_v7.py` + `results/desk_check_v7.json` | 自动化预演（45/45），**不是真人评审、不是数学证明**。 |
| `tools/eval_baselines/fibre_probe.py` + `results/fibre.json` | P1 桥接：原 content-chain 任务的 \\|A(v)\\| 枚举。 |"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
s += """

## v7 起的结构说明

v7 是一次**重写**而非补丁：核心结果恰好四条（Theorem A 六部分 / Theorem B 两界 /
Construction C / Empirical E），其余一律 P1–P8 或 L1–L3。
旧附录 A.31 之后的 ledger（NT01–NT31、A.32/A.33、130/130、23/23、51 项台账）
已整包删除；附录只保留 A（完整证明）、B（6 行撤回登记）、C（复现说明）。
`CHANGELOG.md` 记录实测进展，`REVISION_LOG.md` 记录口径变更与撤回，两者都不是论文正文。
"""
open(p, "w", encoding="utf-8").write(s)
print("  updated MANUSCRIPT_INDEX.md")
print("LOGS OK")

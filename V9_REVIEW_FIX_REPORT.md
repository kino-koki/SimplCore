# v9 修订报告 · 按评审清单逐条改好

日期：2026-11-08 · 交付：`simplcore.pdf` / `paper_main.pdf` / `paper_appendix.pdf`
验收：**42/42 PASS**；desk-check 46/46；拆分 11/11；主文正文仍在 8 页内。

## 一句话

这轮没加定理、没跑训练、没改任何实验数字、没放宽任何检查。改的全是**陈述口径、内部一致性、编校与附录自包含性**；顺带修掉一个此前没被发现的排版瑕疵（所有表格的横线都压在末行基线上）。

---

## ① 定位：核心不等式是同义反复且自己承认松

| 评审意见 | 改法 |
|---|---|
| 标题信息量大于内容 | 标题改为 `… Joint Bounds for Static-LTI Layers under a Fixed Read-Out Interface`，删掉 `A Tight Joint Bound`；附录封面同步改成「Restated Premises, Complete Proofs and Raw Measurements」 |
| `B ≥ log₂\|A(v)\|` 只是「区分 N 个答案要 log₂N 位」 | 摘要、§1、§8 明写它是 **counting (pigeonhole) necessary condition**；增量是「bit 必须放在哪一侧」的证明 + order/depth 紧刻画，**不是不等式本身** |
| 结论承认 content-chain 上 loose | 摘要直接写 `LOOSE on the content-chain task`，结论写「what transfers is A(b) 与放置位置的证明，不是该任务上的紧数值界」；§6.1 保留「未在 MQAR 上枚举 A(v)」的承认 |

## ② order-2 与 o ≥ H+1 并置像打架

这是全稿最实的内部不一致，做了**集中处理**，不再散落三处：

- **§3.3.2 新增唯一一段 `Reconciliation of the two order bounds`**：Result 2(i) 的 order 是**状态张量阶**（vector=1 / matrix=2，单跳、P2 前提下、紧且可达）；P5 的 `o ≥ H+1` 是 **c 个正交后继前提下线性摘要秩的条件计数界**，二分支退化为 `log₂(H+1)` bit，且不对状态张量阶作断言。Construction C 在每层都用 order-2 张量态，两者不矛盾、互不为下界。
- **设计规则改了**：原文「budget … then choose H layers and an **order-(H+1) state**」是错的，现为「choose H layers … and an **order-2 tensor state per layer**」。
- **摘要 / §1.1 / §3.1(e) / §8** 全部统一到同一口径；§3.1(e) 那句残留的破碎句 `Earlier drafts headline result; …` 也删了。
- **Appendix A.5 重写**：`d/D` 混用统一为 `D`（与 §2.1 状态字节里的 D 同一个量），补上 `D ≥ 1`、H+1 个方向线性独立、联合张成 `(H+1)D` 维、`dim ker Φ ≥ (H+1−o)D ≥ D > 0` 的完整条件，并显式标注这些是前提而非推导结果。

## ③ 「exactly H layers」是接口约定不是架构定律

- §3.2(ii) 标题改成 `Exactly H layers under the stated interface`，正文写明 `H6 + A4 last-layer read-out` 是前提，保留并选取中间输出时 **任意 H_L ≥ H 都充分**（A(f)），明确「not an architecture-independent law」。
- 摘要与 §8 同步；§5 原有 `Overshoot is not an architectural law` 保留。
- order-2 最小的边界也照实：§3.6 明列三种逃逸，其中 outer-product state 正是 Construction C 自己。

## ④ 实验口径偏乐观

- **摘要**：把 `from random initialisation … reaches 1.0000` 改成「层间映射 **FIXED** 为 codebook value-to-key swap，只有外部 query projection 与 read-out 从随机初始化学习」，并补上对照组 `0.2492 ± 0.0325、0/5 seeds converged (Table 14)`。
- **Table 14 说明**补一句：`convergence reported here is NOT evidence that the inter-layer map is learned.`
- **SE2**：在 95% t-CI `[82.72, 177.28]` 之外，新增「CI 跨度约 **73% 的均值**，数值吻合**几乎无区分力**；SE2 的证据价值是前提固定的预测方向，不是 0.10% gap」。保留「均值巧合，不是精度声明」。

## ⑤ 编校

| 项 | 修法 |
|---|---|
| Table 0 挤格 | 4 列窄表 → **2 列宽表**（`symbol / meaning`，17 行），B、b、state bytes 分开定义；目视确认无串行 |
| §4.13 交叉引用 | `Tables 18–19 in Appendix A.7–A.11` → **`in Appendix D`**（per-seed 端点指向 Appendix J）；§8 改为 `Appendices A.9–A.10 and D` |
| Appendix B | `W10/W11` 合并行拆成 **W10 / W11 两条**；补一句说明「ID 是登记时的稳定标识，空缺是已废止且无替代的条目，故意不重编号以保持跨版本引用有效」 |
| Appendix C / Broader Impact / §4.10 / §7 | 删掉全部投稿待办（CFP、截稿日、页数复核、LLM 披露、reciprocal reviewer）与 `we invite the community to falsify` 喊话。全文 `2027` / `call for papers` / `reciprocal` 出现次数 **0** |
| 表格横线 | 发现 `table()` 把横线画在末行**基线上**（`self.y + 2.5` ≈ 文字基线），所有表格都穿字；改为 `self.y - 1.5`。实测 Table 0 底线 491.5 > 末行文字底 489.0，分页不变 |

## ⑥ 附录单发：从「自称 self-contained」到真的自包含

- **Appendix A.0 重写**为 `Restated premises, interface, claim statements and notation`，补齐此前缺的四类：
  1. **A1–A5** 完整前提（逐条）
  2. **H6** 唯一层接口的完整定义（含 `T* = C_val^T C_key`、状态 `2MD+(K−1)D`、因果归纳、B 的定义）
  3. **四个核心结果**的完整陈述（Result 1 的 (a)–(f)、Result 2 的条件、Result 3、Result 4）
  4. **CT7 完整陈述** (a)(b)(c)：零阶必然、`T·w·b` bit 上界、有限过程上改进已证
- 封面声明改成只写真实做到的事，目录同步（A.0 / G checkpoints）。
- **Appendix G** 逐 10 round 的 231 行曲线压成 **40-round checkpoints**，明写完整序列与逐 seed 曲线在匿名 artifact 与 Appendix J；页数不增，单发时不再「翻两页就放下」。
- **Appendix B 修订史保留**（对审稿人是 soundness 加分），但在表头说明它是审计辅助、不是贡献；给一般合作方的分发建议见下。

## 按对象分发（沿用你的判断）

| 发给谁 | 建议 |
|---|---|
| 审稿人 / 帮你看证明的合作者 | **现在可以单发 `paper_appendix.pdf`**（A.0 已自包含，Appendix C 待办已删）；更稳可附 `paper_main.pdf` 的 §2 与 §3.1–3.2 |
| 只想了解结论 | 只给 `paper_main.pdf` 的 §1.1 + §5 + §8 |
| 一次看全 | `simplcore.pdf`（18 页：正文 9 + 附录 9），但它**不含附录封面**；要单发附录请用 `paper_appendix.pdf` |
| 一般合作方、不想暴露修订史 | 可只发 `paper_main.pdf`；Appendix B 的撤回登记按需再给 |

## 校验

| 检查 | 结果 |
|---|---|
| 逐条验收 `verify_v9.py` | **42/42 PASS** |
| desk-check D1–D7 + P + P1 | **46/46 PASS** |
| 拆分 `split_submission.py` | **11/11 PASS**，禁用片段 0 |
| 页数 | 合并 18 页 / `paper_main.pdf` 9 页 / `paper_appendix.pdf` 10 页；主文正文止于第 8 页 |
| 几何（18 页逐 span） | 0 越界、0 小于 6 pt 字、Table 0 横线不穿字 |
| 文本层 | 实体/TeX 碎片 0、自识别路径 0、凭据字面量 0 |

## 没圆的（照实）

- **仍缺**：≥10M token 等算力 LM、同场强基线 + 实测吞吐、真实 MQAR 多层、可学习码本、TS2SC 端到端迁移、ICML 2027 官方 `.sty`（CFP 仍未发布）。
- **防御性写作痕迹**只是精简与去重，没有删掉 §5 的实质限定或撤回登记——它们守住的是 soundness，不是 originality 得分点。
- 核心不等式在 content-chain 上松、任务窄（n=4 / V_K=32 / 单查询 / 相邻绑定）这两点**没有修**，因为修不了；这轮做的是不再让标题和摘要暗示它紧。
- 未跑训练、未改权重、未新增定理、未改任何实验数字；所有数值仍由原始结果 JSON 注入。
- 不作过稿率、SOTA、LM parity、吞吐或「通用自进化」主张。

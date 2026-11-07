# v7.1 · 按投稿技能包治理 + 架构自身完善 — 总报告

本轮按《ICML2027_SimplCore_投稿技能包》的 Phase 0–7 工作流执行，同时把三条治理规则**做成架构里可计算的对象**，而不是停留在文字。
**没有新增定理、没有新增实验、没有新增数字。** 训练仍冻结（纯 CPU 合成探针，秒级）。

---

## 1. 先说最不好听的一句（技能包 §0.2，必须原样复述）

> **60% 门 = 3 名独立模拟评审中 ≥2 人给出 weak accept 及以上，且五维自评无 <3 分项，且 P0 硬错清零。它是内部质量代理指标，不是录用概率。**
> 任何工作流都无法保证 ICML 主会 60% 录用概率。若坚持要真实 60%，唯一诚实的回答是换 venue 或补出足以改变 significance 评分的新结果。

**本轮门实测：`NOT EVALUABLE / 未通过`** —— 因为拿不出 3 名独立真人 blind review。我拒绝把同一执行体生成的三份角色化意见称为「3 名独立评审」。五维自评最低分是 significance = 3，P0 内部清零，但这两项不能替代独立性。

---

## 2. 抓到并修掉的真硬错（不是润色）

技能包基于 v6 基线写成，v7 已修掉其中一部分（B 定义、per-fibre 口径、universal 越界、元话术）。但本轮扫描在**当前 16 页稿**里仍抓到四处真 P0：

| # | 硬错 | 性质 |
|---|---|---|
| 1 | **命名空间双重占用**：Theorem A 前提用 `P1–P5`，支撑命题也用 `P1–P8`；同一节里 `P5: two accuracy notions` 和 `P5 gives a matching state-order bound` 并存 | 20 分钟 desk-check 可发现，直接摧毁 soundness 印象 |
| 2 | **`Theorem B` 同名不同义**：§1.1/§3.2/A.4/Figure 1 用它指 Result 2，§3.1(e) 又说「earlier drafts listed this as a separate Theorem B; it is part (e) here」 | 技能包 §3.1 点名的编号崩塌 |
| 3 | **错引命题**：§3.1(c)「by P1 that destroys the Toeplitz structure」和 Table 1「P1 proves Toeplitz iff time-invariance」实际应为 **P6** | 不是改名，是论证用错了命题 |
| 4 | **完整证明正文/附录各一份**：§3.1.1 / §3.1.2 / §3.2.1 / §3.4 显式构造在正文写了完整证明，Appendix A.1–A.4 又写一份，且 §3.1.1 的「Neither step uses P3」正好撞上命名空间冲突 | 技能包 §4.4 说的「证明越完整，复制进正文的不一致前提越多」 |

修法：Theorem A 前提整体改名 **TAP1–TAP5**（并在 §3.1 写明 namespace 规则）；`Theorem B` 全文归零；两处错引改 P6；四段重复证明移出正文，只留 proof intuition，附录保留唯一完整版。
机器复核：`TAP` 18 次、`Theorem B` 0、残留歧义短语 0、`namespace_unique: true`。

另外修了检查器自身的两处脆弱性（双栏断行导致正则漏判、`TAP5:` 被当成裸 `P5:`），**语义条件一条未放宽**。

---

## 3. 「完善架构本身」落到了哪里

技能包禁止「编号未统一前新增定理/实验/页」，所以架构完善做的是**把治理规则变成架构里的可计算对象**：

`theory_work/governance.py` 实现三件事：

1. **B 的可计算化（S4 同量检查）**。把 §2 的 `B = Σ_t log₂|π⁻¹(v_t)|` 写成 `budget_B()`，在三个载体 + binary-chain witness（m=1..4）上枚举，断言下界 `B ≥ log₂|A|` 与紧性见证 `|A| ≤ 2^B` **使用同一个量**。结果：
   - `S4 PASS`，紧见证 `key_only_tag`（B=m=log₂|A|）
   - partition 确实向 B 加 bit（`partition_adds_bits_to_B=true`）
   - **反事实**：按 v6 旧定义（字母表扩容不计入 B）重跑，partition 得 `B=0` 却声称 `|A|=2^m>1` 精确，产生 **4 条矛盾** —— 这把「为什么必须改公式」从文字变成了可执行证据
2. **命名空间唯一性校验**：扫描 PDF 文本，输出修复前/后 JSON（修复前基线只采一次，永不覆盖）
3. **单一真值源**：16 条冻结数字，`unique: true`

`verify_governance.py` 把 5 个脚本串成一键自检，产出 `governance_gate.json`：**12/12 PASS**。治理从此是架构自检的一部分，不是外部文档。

`split_submission.py` 按**内容**（"Appendix A Complete Analytic Proofs" 首次出现的页）动态切分，不写死页码。

---

## 4. 十项产物（技能包 §0.5 输出契约）

| # | 文件 | 状态 |
|---|---|---|
| 01 | `cfp_lock.md` | ✅ 官网核验：会议日期/地点已确认；**CFP、截止日、Author Instructions 均未公布**；第三方 Jan 22 2027 因与官网已公布地点冲突判为陈旧数据，不采信 |
| 02 | `claim_registry.md` | ✅ 唯一命名空间、四条核心结果、8 项冲突消解 |
| 03 | `proof_audit.md` | ✅ 4 条核心结果 + 支撑命题逐条 S1–S6 卡片，含 S4 可计算表 |
| 04 | `p0_scan.md` | ✅ 7 项修复前后对照；References 逐条核验如实挂起 |
| 05 | `number_freeze.md` | ✅ 21 行冻结表 + v6 八组冲突对逐对消解 |
| 06 | `paper_main.pdf` | ✅ **8 页**（7 正文 + 1 Broader Impact/References） |
| 07 | `paper_appendix.pdf` | ✅ **9 页**（封面 + Appendix A–J，34 张表全量原始数据） |
| 08 | `review_sim.md` | ✅ 三份角色化盲读压力记录 + **门实测 NOT EVALUABLE** |
| 09 | `rebuttal_prep.md` | ✅ 三条必杀质疑的答复骨架 |
| 010 | `submit_checklist.md` | ✅ 12 项挂起/待办，Phase 7 挂起 |

配套模块 B/C/F 未随技能包提供，按技能包正文中可提取的规则自建，已在各文件注明。

---

## 5. 论文与架构状态

- **主文 8 页 → 7 页**（证明移位后），总 16 页；技能包 §4.4 明确说页数不是质量指标，没有为凑第 8 页注水
- 双轨拆分 **11/11 PASS**：自识别路径 0、元数据匿名、Broader Impact 在 References 之前、主文无 Appendix A、附录 A–J 齐全
- `desk_check_v7.py` **45/45**；`fusion_guard.json` **27/27**（架构守恒一字未动）；`verify_governance.py` **12/12**
- 核心结果仍是 **4 条**；SE1–SE5 是支撑命题；`Theorem B` = 0；附录不新增定理

## 6. 没做、也不假装做的

- ❌ **3 名独立真人 blind review** —— 门 `NOT EVALUABLE`，不换算录用概率
- ❌ **References 逐条联网核验** —— v7 已统一 14 条 venue 并修 Zoology→ICLR 2024，但本轮未重新核验 arXiv ID，不写「已核验」自证句
- ❌ ≥10M token LM、同算力 Mamba-2/GLA/DeltaNet/Hyena、真实 MQAR 多层、TS2SC 端到端迁移
- ❌ 2027 官方 `.sty`（CFP 未发布）；双 PDF 是技能包内部交付格式，需按届时提交系统重打包
- ❌ 不作 LM parity / SOTA / 吞吐 / 通用自进化主张；不作任何录用概率承诺

## 7. 一键复现

```bash
python3 verify_governance.py           # 12/12，产出 governance_gate.json
python3 train/superposition_probe/smm_sgc_fusion.py --part all   # 27/27 架构守恒
python3 build_simplcore_pdf_v7.py      # 16 页合并版
python3 split_submission.py            # 两个提交 PDF
python3 tools/eval_baselines/desk_check_v7.py   # 45/45
```

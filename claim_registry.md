# 02 · claim_registry.md — 唯一编号声明注册表 + 冲突消解决议

> 技能包 §3.1：这类稿件的失败模式几乎从不是「定理不够多」，而是「声明体系互相打架」。
> 本文件是 Phase 1 的唯一出口。判定：**编号冲突数 = 0**，机器复核见
> `theory_work/results/governance_namespace_before.json` / `_after.json`。

## 1. 唯一命名空间（修复后）

| 命名空间 | 含义 | 数量 | 位置 |
|---|---|---:|---|
| **R1–R4** | 四条核心结果 | 4 | §1.1 / §3 / §8 |
| **A1–A5** | **仅** Theorem A 的前提（task / visible input / class M / last-layer read-out / two accuracy notions） | 5 | §3.1 |
| **P1–P8** | **仅** 支撑性命题（duality / arbitrary read-out / finite output / linear-attention isomorphism / state order / Toeplitz iff / error bound / grid degeneracy） | 8 | §3.4 / §3.3.3 / Appendix A |
| **L1–L3** | 引理（role-bit / codebook / rank） | 3 | §3.4 / Appendix A |
| **SE1–SE5** | 支撑性命题（SGC×SMM 融合，有限巩固过程） | 5 | Appendix A.7–A.11 |
| **W1–W15** | 撤回登记 | 15 | Appendix B |
| **H1–H7** | 构造 C 的结构前提（其中 H6 = 唯一层接口） | 7 | §3.3 |
| **O / C / F** | 观测值 / 猜想 / 被证伪项 | — | 不进正文头条 |

**铁律**：`TAP*` 与 `P*` 永不相交；`R*` 是结果不是定理编号；Appendix 内不新增定理。

## 2. 四条核心结果（计数唯一）

| ID | 名称 | 类型 | 前提 | 主要证据锚 |
|---|---|---|---|---|
| R1 | Theorem A（输入侧信息下界，a–f） | **唯一的 headline theorem** | A1–A5（a/b 只需 A1,A2,A5） | Table 6 穷举 / Table 7 三载体 / Appendix A.1–A.3 |
| R2 | 结构紧界（order-2 单跳 + exactly-H 层） | 双部分结果，**不再叫 Theorem B** | P2 / H6 / A4 / distinct keys | Table 5 5×5 grid / Table 15 训练网格 / Appendix A.4 |
| R3 | Construction C | **tightness witness + 构造**（不是独立定理） | 静态 LTI / 张量信号 / 缩并读出 | Table 4 守恒 / Figure 1 |
| R4 | Empirical E | **evidence，不是定理** | 1500 步 Adam / 5 seeds | Table 15 / Table 22 / Table 33 |

摘要、§1.1、§3、§8、Broader Impact 对核心结果数的表述统一为 **four core results**。
Theorem B 的计数：**0**（机器复核 `governance_namespace_after.json`: `theorem_b_mentions = 0`）。

## 3. 冲突表与消解决议（修复前 → 修复后）

| # | 冲突 | 修复前（v6.5 合并稿） | 决议 | 修复后 |
|---|---|---|---|---|
| C1 | **命名空间双重占用** | Theorem A 前提用 `P1..P5`，支撑命题也用 `P1..P8`；同一节里 `P5: two accuracy notions` 与 `P5 gives a matching state-order bound` 并存 | Theorem A 前提整体改名 `A1–A5`，并在 §3.1 明写 namespace 规则 | `TAP` 出现 18 次；`P1–P8` 只指支撑命题；残留歧义短语 **0** |
| C2 | **Theorem B 同名不同义** | §1.1/§3.2/A.4/Figure 1 把 R2 叫 `Theorem B`；§3.1(e) 又说「earlier drafts listed this as a separate Theorem B; it is part (e) here」 | R2 只称 Result 2；(e) 的历史说明删除该标签 | `Theorem B` 出现 **0** |
| C3 | **错误引用：Toeplitz** | §3.1(c) 写「by P1 that destroys the Toeplitz structure」；Table 1 写「P1 proves Toeplitz iff time-invariance」。实际应为支撑命题 **P6** | 两处改 P6，并写明 P6 的内容 | 两处均为 P6 |
| C4 | **完整证明在正文与附录各写一份** | §3.1.1 / §3.1.2 / §3.2.1 / §3.4 显式构造在正文写了完整证明，Appendix A.1–A.4 又写一份 | 技能包 §4.4：正文只留「前提 + 结论 + 3–5 行证明直觉 + 证据锚 + scope」，完整证明一律在附录 | 正文重复证明删除 3.7k+1.1k+1.1k 字符，替换为三段 proof intuition；附录保持唯一完整版 |
| C5 | **S4 同量（v6 基线硬错）** | v6 把 `B = Σb_t` 只算显式 side-channel，tightness 却靠 `V→2V` 字母表扩容，未计入 B | v7 已把 B 重定义为 `Σ_t log₂|π⁻¹(v_t)|`；本轮把该定义**做成可计算对象**并做反事实检验 | `governance_s4.json`: S4 PASS；v6 反事实矛盾 **4** 条 |
| C6 | `Theorem 1/3/4/5/6` 与 `T*` 混用 | v6 §3.4 曾并存 | v7 已全部降为 P1–P8 / L1–L3 | 机器检查 `no_stray_numbered_theorems` PASS |
| C7 | 计数口径 | v6 同时声称 FOUR / SIX / 8 个 | 本轮统一为 4 | §1.1 / §3 标题 / §8 / Broader Impact 四处一致 |
| C8 | 数字口径 | `0.2969` 在 Table 28 与 Table 10 两种语境并存 | Table 28 caption 补「SINGLE-SEED / 不同配置」标签，值由原始 JSON 注入 | 见 `number_freeze.md` |

## 4. 交叉引用可解析性

- `Appendix A.1–A.11`：全部存在；无 `A.31 / A.32 / A.34` 悬空引用
- `Table 1–34`：连续且唯一；每张 caption 自带 statistic / seeds / source
- `Figure 1`：legend 改为 `diagonal = exact (Result 2)`，不再引用已删除的 Theorem B
- H6 层接口唯一：`q_(h+1) = T_h z_h / ‖T_h z_h‖₂`（§3.1(c) 唯一化说明 + Appendix A.2）
- 附录证明之间只用注册表 ID 交叉引用，不写「见上文的证明」

## 5. Phase 1 判定

| 闸门 | 结果 |
|---|---|
| 编号冲突数 = 0 | ✅ |
| 核心结果数唯一（4） | ✅ |
| 每条保留结果在摘要/§1.1/§3/§8/Broader Impact 名称一致 | ✅ |
| 每个交叉引用可解析，无悬空 Appendix 编号 | ✅ |
| 未达标禁止进入 Phase 2 | 已达标，放行 |

## v7.2 命名空间终裁

按《v7 详细改稿指令单》N1，Theorem A 的五个前提最终命名为 **A1–A5**（v7.1 曾用 `TAP1–TAP5`，本轮按指令单统一）。P1–P8 保留给支撑命题。机器复核：`A1-A5` 23 处、`TAP` 0、`Theorem B` 0、残留歧义短语 0、`namespace_unique: true`。冲突表 C1–C8 的消解决议全部保持有效，仅命名变更。

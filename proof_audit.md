# 03 · proof_audit.md — 逐条定理六步审计

> 六步 S1–S6 按技能包 §4.1。裁决只有 KEEP / REWRITE→KEEP / DEMOTE / DELETE。
> 完整证明一律在附录，正文只放前提全文 + 结论 + 证明直觉 + 最强证据锚 + scope 句（§4.4）。

## 卡片 1 · R1 = Theorem A（a–f）

- **前提**：TAP1（task）、TAP2（visible input `(v,s)`, `s∈{0,1}^B`）、TAP3（class M：H_L 静态 LTI 层、时不变且内容无关系数、固定维状态、固定线性接口、确定性末层读出）、TAP4（只解码末层）、TAP5（per-fibre / marginal 两种统计量）。**(a)(b) 不需要 TAP3。**
- **结论**：(a) `acc(F|v) ≤ 2^B/|A(v)|`；(b) `B ≥ log₂|A(v)|`；(c) 类 M 内、IF 保持对偶，bit 只能由输入侧提供；(d) 三载体紧性；(e) bit–depth 直角前沿 + 条件状态阶界；(f) 过冲是读出约定。
- **量纲表**：`B` bit，`|A(v)|` 无量纲计数，`acc` ∈ [0,1]，`H_L` / `o` 整数，`2^B` 与 `|A|` 同类（可分辨细化数 vs 答案类数）。
- **S1 前提同页**：✅ §3.1 单页完整列出 TAP1–TAP5 + namespace 规则，无「见 Appendix」。
- **S2 量定义域**：✅ `B = Σ_t log₂|π⁻¹(v_t)|` 在 §2 定义，§3.1(d) 与 Appendix A.3 复述同一式。
- **S3 adversary 覆盖所有实例**：✅ (a)(b) 用鸽巢原理，对**任意确定性解码器**成立；不依赖作者构造的实例。binary-chain witness 只用于证明 |A| 可达 2^m（非空洞），不是下界的前提。
- **S4 见证与下界同一个量**：✅ **本轮做成可计算断言**（`theory_work/governance.py`）。

  | 载体 | \|π⁻¹\| | B (m=2) | \|A\|=2^m | \|A\|≤2^B | 紧 |
  |---|---:|---:|---:|---|---|
  | explicit tag | 2 | 8 | 4 | ✅ | 否（充分） |
  | codebook partition | 2 | 8 | 4 | ✅ | 否（充分） |
  | key-only tag | 2 | 2 | 4 | ✅ | **是** |
  | flat | 1 | 0 | 4 | ❌ | — |
  | position prior | 1 | 0 | 4 | ❌ | 通过 `A(v)` 而非 B |

  机器结果：`governance_s4.json` → `S4 PASS`，tight witness `key_only_tag m=1`，`partition_adds_bits_to_B=true`。
  **反事实**：按 v6 的旧定义（字母表扩容不计入 B），partition 得 `B=0` 却声称 |A|=2^m>1 精确，产生 **4 条矛盾**（m=1..4）。这正是技能包 §4.2 点名的错误，修复后消失。
- **S5 是否用了被撤回/降级结论**：✅ 无。(a)(b) 只用确定性与鸽巢；(c) 用支撑命题 P6（未撤回）；未引用 W1（fp32 逐位一致）、W12（parity）、W13（训练旁证）、W15（SE3 上界措辞）。
- **S6 scope 与摘要一致**：✅ 摘要拆两句——(a)(b) 对任意确定性解码器成立；(c) 明确写 "IF one chooses to preserve duality ... a design choice within M, not a physical force"。§3.6 明写「不主张所有修复路线必然汇合到 pairwise score」。
- **依赖**：P6（Toeplitz iff 时不变）、Result 2(ii)（前沿上界）、Appendix A.1–A.5。
- **裁决**：**KEEP**。

## 卡片 2 · R2（结构紧界，双部分）

- **前提**：P2（固定线性历史摘要下的任意 read-out 不可能性）、H6（唯一层接口）、TAP4（末层读出）、distinct keys、静态系数、n ≥ 2。
- **结论**：(i) 单跳最小张量阶恰为 2；(ii) 末层读出下 H 跳恰需 H 层，`H_L ≥ H` 必要、`H_L = H` 充分。
- **S1**：✅ §3.2 同页给出；完整归纳在 Appendix A.4。
- **S2**：✅ 阶 `o`、层 `H_L`、跳 `H` 均为整数； witness 用 5×5 网格。
- **S3**：✅ 下界对「所有满足 P2 前提的单 mixer + 固定线性摘要」成立；三种逃逸（深层逐位置 FFN / 输入相关门控 / 外积状态）在 §3.6 显式列为例外，并说明它们改变的是前提而非推翻下界。
- **S4**：✅ 达到紧性的 witness 正是同一个 `H_L`（5×5 对角全 1.0000）与同一个 `o`（Construction C 的 order-2 张量状态），不是换了量。
- **S5**：✅ 未用 W13（v6.3 撤回了「层数下界经训练仍成立」这条**旁证**；定理本身是解析归纳，撤回的是训练 corroboration，不是定理）。§5 明写这一点。
- **S6**：✅ (ii) 带 "last-layer read-out" 限定；A(f) 明写 "Exactly H is not an architectural law"；保留中间层输出时 `H_L ≥ H` 充分。
- **裁决**：**KEEP**（本轮去掉 `Theorem B` 标签，名称统一为 Result 2）。

## 卡片 3 · R3 = Construction C

- **身份**：**tightness witness + 构造**，不是独立定理（技能包 §3.3）。
- **前提**：静态 LTI 核、张量阶输入 `u_t[r,n]=a_t[r]b_t[n]`、`ρ` 与 t 无关、缩并读出。
- **S1/S2**：✅ §3.3 同页；`ρ<1` 保证 BIBO，`ρ=1` 明写为临界稳定、架构排除。
- **S3**：✅ 作为 witness 只需存在性；不承担 universal 下界。
- **S4**：✅ 与 Theorem A(d)、Result 2 共用同一码本与同一 H6 接口。
- **S5**：✅ 未引用「内容寻址与 FFT 对偶互斥」（已两次证伪并撤回）。
- **S6**：✅ §3.3.1 精确写明与线性注意力的关系：`ρ=1` 单极点时状态 `Σ k_τ v_τᵀ` 同构；`ρ<1` 多极点保留衰减谱；**不宣称脱离该范式**。
- **裁决**：**KEEP（降级为构造，不计入定理数）**。

## 卡片 4 · R4 = Empirical E

- **身份**：**evidence，不是定理**（技能包 §3.3）。
- **S1/S2**：✅ 协议同页写明（1500 Adam 步、lr 0.02、batch 64、5 seeds、512 held-out/seed、d=64）。
- **S3**：不适用（不是下界）。
- **S4**：✅ 5×5 训练网格对角 1.0000、非对角 ≤0.10；配对 bootstrap CI 全正。
- **S5**：✅ W12 撤回后不作 parity 主张；Table 10 把旧值列为 superseded。
- **S6**：✅ §5 明写「合成任务、零训练解析构造的 lookup-table 风险、不作 LM 主张」。
- **裁决**：**KEEP（作为 evidence）**。

## 卡片 5 · P1–P8 / L1–L3 / SE1–SE5（支撑命题）

| ID | S1 | S2 | S3 | S4 | S5 | S6 | 裁决 |
|---|---|---|---|---|---|---|---|
| P1 Z 变换对偶 | ✅ | ✅ | 对全部稳定二阶 LTI | ✅ 实域等价 | W1 已撤回 fp32 逐位 | ✅ 振荡核与严格正实极族显式分离 | KEEP |
| P2 任意 read-out 不可能性 | ✅ | ✅ | 三情形显式构造覆盖全部 L_j 情况 | ✅ | 未用 `dim ker > 0` 推断（v7 已修） | ✅ 三种逃逸写明 | KEEP |
| P3 有限输出集 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 只驳「O(1) 必输」字面命题 | KEEP |
| P4 线性注意力同构 | ✅ | ✅ | 条件 ρ=1 单极点 | ✅ | ✅ | ✅ 不宣称新范式 | KEEP |
| P5 状态阶界 | ✅ | ✅ | 条件：每跳 d 个正交候选 | ✅ 分支因子 2 时退化为 log₂(H+1) bit | ✅ | ✅ 明写「条件下界」 | KEEP（条件） |
| P6 Toeplitz iff 时不变 | ✅ | ✅ | 逐延迟归纳，c_t≠0 | ✅ | ✅ | ✅ 否定的是单 FFT 路径，不是所有 O(L log L) | KEEP |
| P7 误差界 | ✅ | ✅ | 条件：统一扰动上界 | ✅ | ✅ | ✅ 条件定理 | KEEP（条件） |
| P8 网格渐近秩亏 | ✅ | ✅ | 渐近方向 | ✅ | ✅ | ✅ 只证方向，数值是观测 | KEEP（条件） |
| SE1–SE5 | ✅ Appendix A.7–A.11 各自复述前提 | ✅ | ✅ 非空洞实例 | ✅ | W15 已从「上界」改为精确期望+浓度 | ✅ 有限合成过程边界 | KEEP（支撑） |

## S4 特别判定（技能包 §4.2 / §4.5 的硬闸门）

| 检查 | 结果 |
|---|---|
| 下界变量 == 见证变量 | ✅ 同为 `B = Σ_t log₂\|π⁻¹(v_t)\|` |
| 达到紧性的构造确实向 B 加 bit | ✅ partition 贡献 1 bit/token，机器验证 |
| 未用「information does not disappear; it changes carrier」替代公式更改 | ✅ 公式已在 §2 改写 |
| 所有保留定理卡片为 KEEP 或 REWRITE→KEEP | ✅ |
| 无 DEMOTE/DELETE 遗留引用 | ✅ `Theorem B` = 0，`A.31+` = 0 |

## 一处必须记录的修复

技能包 §4.4 点名「证明越完整，被复制进正文的不一致前提越多」。本轮正文原有 §3.1.1 / §3.1.2 / §3.2.1 / §3.4 四段完整证明与 Appendix A.1–A.4 重复，且其中 §3.1.1 写「Neither step uses P3」而 §3.1 前提 P3 又与支撑命题 P3 同名 —— 这正是「两处都完整写证明却没同步」的实例。**已把四段重复证明从正文移除**，正文只留 proof intuition，附录保留唯一完整版并统一为 TAP 命名空间。

**Phase 2 判定：放行。**

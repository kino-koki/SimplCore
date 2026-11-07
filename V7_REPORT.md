# v7：一次性重构，把 7 类硬伤、呈现合规与 P0/P1 全部落地

> 用户指令：「直接做，一步完成」。
> `simplcore.pdf` 已替换为 v7，**10 页，主文 7 页**（8 页上限内），Broader Impact / References 从 p8 起。
> 训练/迁移任务仍冻结；零 LM 训练、零 ppl、零生成样本、零私有权重字节。

## 0. 一句话

v6.2 是在旧稿上打补丁，编号、口径、B 的定义已经互相打架；**v7 是重写**，不是补丁。
新生成器 `build_simplcore_pdf_v7.py` 自包含，旧 `build_simplcore_pdf_v5.py` 降为历史生成器。
自动化预演 **45/45 PASS**（`desk_check_v7.json`），每一项都可复现。

## 1. 七类致命伤，逐条落地

### ① B 与 tight 构造不是同一个量 —— 已修

**重定义**（§2，全文唯一）：固定 flat 参考字母表 Σ₀ 与抹掉角色的投影 π，
`B = Σ_t b_t, b_t = log₂ |π⁻¹(v_t)|`，即「实际字母表相对 flat 参考的超出位数」。
于是 `V → 2V` 分区的 `log₂(2V_K/V_K) = 1 bit` **正式计入 B**，显式 tag、symbol id 带角色同理；
全 flat 则 B=0。位置格式先验不是逐 token 通道（b_t=0），它是**分布支撑的限制**，通过 A(v) 而不是 B 计入。

- **A(c) 降级**：从「Toeplitz duality FORCES」改成「**IF one chooses to preserve duality**，bits 只能从输入侧进」，明写这是 M 类内的设计选择，并保留「不否定其它 O(L log L) 算法」。
- 摘要把「distribution-free / depth-free」与「within the duality-preserving LTI class」分开陈述，不再连成一句可被拆开的绝对断言。
- A(d) 明写分区的 bit 被 counted in B，tightness 形式成立。

### ② A(a) 被自己的表格反证 —— 已修

**P5 定义两种口径**（§3.1）：
- (i) **per-fibre conditional**：`acc(F|v)`，固定 v、对兼容实例均匀；
- (ii) **marginal**：`E_v[acc(F|v)]`。

A(a) 改写为 `acc(F|v) ≤ 2^B/|A(v)|`，只在 (i) 上成立。
Table 2 / Table 6 / Table 12 每行都带「statistic / seeds / n / source」列，并明写
「per-fibre uniform 0.0625 与 marginal 0.8621 是 P5 下两个不同统计量，不是同一配置的两个准确率」。

### ③ 摘要与 §4.5 措辞打架 —— 已修

全文三处**逐字同一句**（wrap-safe 计数 = 3）：
> "The alphabet partition folds the mandatory bit into the symbol id: it is not bit-free, it is annotation-free."

摘要、A(d)、Table 7 caption 一致；不再出现 "free of run-time annotation" 与「不需要输入侧 bit」的冲突读法。

### ④ 编号体系崩了 —— 已修

**恰好四条核心结果**，全文唯一计数：

| | 内容 |
|---|---|
| Result 1 | **Theorem A**（六部分 a–f，原 Theorem B 的串行深度/状态阶并入 (e)） |
| Result 2 | **Theorem B**（T1+T2 合并：order-2 状态 + exactly H 层） |
| Result 3 | **Construction C**（构造性反例） |
| Result 4 | **Empirical E**（5-seed 训练验证 + 同场审计） |

其余全是 **P1–P8 / L1–L3**，没有第三条 Theorem。
`Theorem 1`（Z 变换对偶）→ P1；`Theorem 5`（任意 read-out）→ P2；
`Theorem 6`（"O(1) must lose is FALSE"）→ P3；T6 统一为 P4（ρ=1 同构），三种描述消失。
**全文无 `NT*`、`A.31+`、`T1–T6`、`Theorem \d`**（desk-check 断言）。

### ⑤ 同一份正文两种论证并存 —— 已修

P2 换成**显式三情形构造**（§3.4 + 附录 A.3）：

- L_j 奇异 → 取 ker L_j 的非零扰动，只改 v_j，ΔA=0 而答案变；
- L_j 可逆且有 L_p≠0（p≠j）→ 取 Δv_p，令 `Δv_j = −L_j⁻¹ L_p Δv_p`，ΔA=0 且目标值变；
- L_j 可逆且其余 L_p=0 → A 只依赖 v_j，把 q 重定向到别的关系对，答案变而 A 不变。

并写死一句：「**never infers a target-coordinate perturbation from dim ker L > 0 alone**」。
旧稿那句「kernel has dimension ≥ (n−1)d > 0, so there exists…」全篇 0 次。

### ⑥ 层接口四套互斥写法 —— 已修

**H6 唯一接口**（§2）：`q_(h+1) = T_h z_h / ‖T_h z_h‖₂`，T_h 可学，解析最优 `T* = C_keyᵀ C_val`。
明写「maps VALUE-half to KEY-half」是 T* 的**作用描述**，`(z_h C_valᵀ) C_key` 是同归一化的特例，不是第二套接口。
**核家族分离**：P1 的振荡核 `ρ^lag sin((lag+1)θ)` 与检索用的严格正实极族
`Σ_j r₁^(lag−j) r₂^j > 0` 明确标成同一 LTI 类的不同子族，并说明后者由 P2 的正性前提要求。

### ⑦ 数字口径四处漂移 —— 已修

- **chance 单一**：`1/32 = 0.0313` 是本任务；`0.25` 只作为**历史四对任务**的 chance，且明写 "DIFFERENT task"，从不与 0.0313 并列。全文 0.25 的其余出现全是 Table 6/11 的 `0.2500` per-fibre accuracy，不是 chance。
- §4.13 旧稿「near the four-pair chance level」整体删除，改成「no arm exceeded 0.3125; none converged」。
- **Table 10 数字溯源审计**：把 0.2969 / 0.2711、0.9316 / 0.9414 / 0.9328、四种 FFT/IIR 误差、chance 混用、near-chance 措辞逐行列成「旧值 / 本稿唯一值 / 统计量 / 来源」。
- Table 7 删掉重复的 acc 列；消融段不再回引被取代的单 seed 值。
- 摘要不放任何探针专属误差值，只说 machine precision；Table 4 单列三个探针，正文报 `≤ 3e−16` 上界。

## 2. 呈现与合规（最容易白丢的分）

- **删掉首页 Table 1 的 meta 话术**：表头改成纯 `Known result / Precise increment here`，表前一句纯陈述。
- **全部审稿人喊话清零**：`most common rejection`、`A reviewer should not`、`A reviewer can check`、`single most damaging potential rejection`、`invite the community to attack`、References 前的「Author names… were checked…」——desk-check 断言 0 命中。
- **References 统一**：14 条全部带 venue + arXiv ID；Zoology 修回 **ICLR 2024**（不再写成 2023）；删掉「verified 2026-11-07 / No entry is taken from an unverified secondary source」自证句。
- **Broader Impact 在 References 之前**（p8 顺序已核验）。
- **瘦身**：25 页 → **10 页**；附录 A.31 之后的旧 ledger（NT01–NT31、A.32/A.33、130/130、23/23、51 项台账、Statement Classification）**整包删除**；附录只剩 A（完整证明）、B（6 行撤回登记）、C（复现说明）。
- 自识别路径、内部编号、HTML 实体、TeX 碎片全 0。
- 2027 CFP / 截稿日未公布、LLM 使用披露与互审义务需按届时模板重核 —— 写进 Appendix C，不冒充已合规。

## 3. P0 / P1 收尾

**P0** 全部完成：4 结果、B 重定义、per-fibre 口径、(c) 降级、H6 唯一、P2 显式构造、核家族标注、一张统一数字表、chance 对齐、meta 清零、旧 ledger 删除。

**P1** 两项硬骨头：

1. **P5 状态阶下界从 witness 升成有证明的条件定理**（附录 A.5）：在「每跳 d 个互相正交候选后继 + H6 秩 ≤ d」前提下，用 `dim ker ≥ (H+1)d − o·d > 0` 配合 P2 构造证明 `o ≥ H+1`；并**主动写出退化**：分支因子为 2 时降为 `log₂(H+1)` bit 要求，不把 `d^(H+1)` 冒充无条件结论。
2. **tight witness 与原任务桥接**（§4.3，新增 `fibre_probe.py` 枚举 5 seeds × 256 样本）：
   - flat / H=4：兼容答案集 `mean |A| = 3.2508`、`max |A| = 5`、`log₂|A| ≤ 2.3219` bits；
   - codebook：所有 H 下 `|A| = 1`；
   - 分区实际提供 `2·n_pairs = 8` bits，显式 tag 提供 `4` bits。
   - **结论写死：原任务上预算充分但不紧**；紧性属于 binary-chain 见证（`|A|=2^m, B=m`）。没有把「为定理量身定做的任务」包装成主任务成就。

另外补了：与线性注意力 ρ<1 多极点的**精确关系**（§3.3.1，只主张条件包含与设计取舍，不宣称新范式）、
bit/depth/order 的可迁移前沿表（§3.3.2 / Table 6）、完整 5×5 网格（Table 5）、
carrier 守恒（Table 7）、深度不变性（Table 11）、5-seed 全统计（Table 12）、
逐条前提—边界—非空洞实例总表（Table 13）、逐条可证伪条件（§4.11）、
完整复现协议（§4.10）。

## 4. 校验

| 项 | 结果 |
|---|---|
| 自动化预演 `desk_check_v7.py` | **45/45 PASS**（D1 6、D2 5、D3 2、D4 6、D5 3、D6 3、D7 6、P 10、P1 4） |
| 章节编号 | 23 个，**无重复** |
| 表格 | Table 2–13，单一数字口径 |
| 定理命名 | 仅 `Theorem A` / `Theorem B`；`Theorem \d` = 0 |
| 禁用短语 | 0 |
| 数字注入 | 抽查 7 个关键值，全部与原始 JSON 逐字一致 |
| 页数 | 总 10 / 主文 7（≤ 8）/ Broader Impact p8 < References p8 |
| 同场协议 | 7/7；Theorem A 18/18；Theorem B witness 130/130 |
| PDF sha256 | `819eca2d21a6727e7e99d767…` |

**预演不是真人评审**：`desk_check_v7.py` 把审稿意见变成可复现的字符串/结构断言，
它验证的是「陈述有没有写对、编号有没有打架、数字有没有漂移」，**不验证数学真伪**，
也不替代 1–2 位未读过的人按 ICML 评分表做 20 分钟真人预演。这一点写进了结果文件的 `status` 字段。

## 5. 诚实边界（不圆）

- **主文是 7 页，不是 8 页**。8 页是 ICML 上限而非填充目标。删掉旧 ledger 与重复证明后，
  再灌满第 8 页只能把附录证明复制进正文，那正是「注水 / thinly sliced」。这一取舍写在这里，不藏在版式里。
- 六种 toy 基线**仍未收敛**；W12 撤回不变，全文不作等价/优越/低劣主张。
- **仍未做**：≥10M token 同算力 LM、val ppl、生成可读性、实测吞吐对标、
  GLA/DeltaNet/Mamba-2/Transformer 完整复现、可学习码本、真实 MQAR 协议、收敛的公平强基线、
  TS2SC 端到端迁移。§7 与 §6.1 逐项明写。
- P5 是**条件**定理；A(c) 是**类内设计选择**；A(a) 是 **per-fibre** 界。三处 scope 都在摘要/定理/结论显眼处。
- 没有承诺过稿率。本轮改变的是**陈述是否自洽、证据是否可核对**，不是「已经能打」。
- 私有权重与凭据字节一律未进仓库。

## 6. 交付物

| 文件 | 内容 |
|---|---|
| `simplcore.pdf` | v7 投稿稿（10 页，主文 7 页） |
| `build_simplcore_pdf_v7.py` | **唯一当前生成器**，自包含 |
| `expand_v7.py` … `expand5_v7.py`、`fix_v7.py` | 五轮扩充 + 修复脚本，全部可重放 |
| `tools/eval_baselines/fibre_probe.py` + `results/fibre.json` | P1 桥接枚举 |
| `tools/eval_baselines/desk_check_v7.py` + `results/desk_check_v7.json` | 45 项预演断言与原始结果 |
| `V7_REPORT.md` | 本报告 |
| `CHANGELOG.md` / `REVISION_LOG.md` / `MANUSCRIPT_INDEX.md` | v7 登记与稿件索引 |

`build_simplcore_pdf_v5.py` 保留为 v6.2 历史生成器，不再是当前稿源。

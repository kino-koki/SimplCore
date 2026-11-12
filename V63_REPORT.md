# v6.3：主动证伪自己的公平性声明

> 用户指令「续」。承接 v6.2 结尾点名的唯一空缺：把对照基线调到收敛。
> 本轮零 LM 训练、零 ppl、零生成样本、零私有权重字节；架构代码一行未改。

## 0. 一句话

上一版说「每个臂拿到同样的一批 one-hot id」——**ids 是对的，计算不是**。
修掉两处不对称后，上一版刚加的「训练版层数下界」主证据**反转**了。
我们不遮，撤回它（W13），并给出更硬的东西：**两类机制的零训练解析解都是 1.0000**。

## 1. 查出来的两处不公平

| | v6.2 | v6.3 |
|---|---|---|
| 前驱视图 `Xp` | **只有 TSC 分支**消费；基线只看 `U = X@E` | 八臂统一 `(U, Up, q, mask)` |
| 宽度 / 嵌入 | d=24 < V=64，E 随机初始化可学 ⇒ 线性嵌入有零空间，而 TSC 保留硬编码配对偏置 | d=V=64，固定满宽正交 Hadamard，**E 不学** |

第二处尤其隐蔽：低维可学 E 会自己收缩成对固定归纳偏置有利的形状，
而要从零学 (前驱→当前) 配对的基线没有这个好处。

## 2. 修正后，v6.2 的主证据反转

5 seeds、400 步、Adam、batch 64、lr 0.02、同一批可见 id：

| hops | arm | acc (5 seeds) | 零训练解析解 |
|---:|---|---:|---|
| 1 | `tsc_h1` | **1.0000 ± 0.0000** | 1.0000 |
| 1 | `linear_pair` | 0.9859 ± 0.0141 | 1.0000 |
| 1 | `attn_x1` | **1.0000 ± 0.0000** | 1.0000 |
| 1 | `gla_pair` | 0.9961 ± 0.0059 | — |
| 1 | `delta_pair` | 0.9840 ± 0.0152 | — |
| 1 | `ssm_pair` | 0.0414 ± 0.0141 | — |
| 2 | `tsc_h1` | 0.1707 ± 0.0176 | — |
| 2 | `tsc_h2` | **0.3625 ± 0.0809** | **1.0000** |
| 2 | `attn_x1` | 0.3727 ± 0.0137 | — |
| 2 | `attn_x2` | **0.9543 ± 0.0315** | **1.0000** |
| 2 | `ssm_pair` | 0.0328 ± 0.0103 | — |

**H=2 配对差 `tsc_h2 - attn_x2` = -0.5918，95% CI [-0.6500, -0.5012]，5/5 全负。**
v6.2 报的是 0.9512 与 +0.8457（CI [+0.8281, +0.8598]）。

### W13（正式撤回）

撤回「层数下界经端到端训练从随机初始化仍成立」这条训练旁证。
**Theorem B 本身不撤回**：它是带解析证明的构造性存在命题（§3.6 / 附录 A），
撤回的只是端到端训练的 corroboration。

## 3. 更硬的东西：函数类等价的显式 witness

在同一批修正后的数据上，**零训练、零可学参数**：

```
Wk = Wv = Wq = I ,  Wout = C[V_K:,:]^T
hop(q) = Σ_s (q · Up_s) U_s K(τ)          [静态 LTI]
       = softmax_τ(q · Up_s / √d) · U_s   [attention]
M = C[V_K:,:]^T C[:V_K,:] :  C[V_K+k] → C[k]
q2 = normalize(hop(q) @ M) ; answer = argmax hop(q2) @ CVAL^T
```

| hops | 机制 | 5-seed 解析 acc |
|---:|---|---:|
| 1 | 静态 LTI 单层 | **1.0000** |
| 1 | linear attention | **1.0000** |
| 1 | cross-attention 单层 | **1.0000** |
| 2 | 静态 LTI 双层 stack | **1.0000** |
| 2 | cross-attention 双层 stack | **1.0000** |

**结论（写成观察 + 显式 witness，不新增定理）**：
在这个合成任务上，两类机制的**函数类重合**；训练里的差距是
**优化路径 / 归纳偏置**，不是表达力。
因此本轮既不能读成「TSC 输给 attention」，也不能读成「TSC 赢」。

这把 T6 的「ρ=1 单极点与线性注意力同构」从**开放态度**升级为**显式 witness**，
并在正文里主动堵死审稿人那句 *isn't this linear attention with extra steps*。

## 4. W12 保留，理由收窄

历史记录文件里的 0.9987 attention 基线**仍然没有可核验的同场复现**，
Fisher-parity 声明保持撤回。但理由要更精确：

> v6.2 把低基线数字写成「audit limitation」还不够准 —— 真因是
> **实现不公平**，不是关于已发表模型的任何结论。

全文现在**不作任何相对 attention 的等价、优越、低劣或表达力优势主张**。
两版所有原始行都发布，审计本身可被证伪。

## 5. 论文 v6.3

`simplcore.pdf` **25 页，主文恰好 8 页**（§4.13 整段重写、缩短，无注水）：

- §4.13 新标题 *Fairness audit and a withdrawn training claim*，
  两处不对称 → 6 行修正表 → W13 → 解析等价 witness → W12 收窄
- 摘要删掉「also survives end-to-end training」，改为公平性审计 / W13 / 函数类重合
- §5 第一条合并 W12/W13；§5 第二条、§7 明确标为 **fairness / reproduction audit**，
  不是性能排名
- §8 结论同步
- 附录 F 新增 **W13 行**，W12 行理由改写
- Broader Impact(p9) < References(p10)；章节编号 3.1–3.10 / 4.1–4.13 / 5–8 **无重复**
- 全大写、标题截断、表头歧义（analytic → witness）、TeX 碎片 / 实体 / 自识别路径残留 **全 0**

## 6. 校验

| 项 | 结果 |
|---|---|
| 协议完整性判定 | **7/7** |
| Theorem A 验证器 | **18/18** |
| Theorem B witness | **130/130** |
| H=1 解析 witness | 3/3 全 1.0000 |
| H=2 解析 witness | 2/2 全 1.0000 |
| PDF | 25 页 / 主文 8 页 |
| 旧口径残留 | 0 |

论文与报告所有数字由结果 JSON 用 f-string 注入，无手抄。

## 7. 诚实边界（不圆）

- **这不等于「基线弱」**：修正后多个基线在 H=1 上达到 1.0000，
  H=2 上 attention 双层达 0.9543。它们仍是**极简机制参考实现**，
  不是 GLA / DeltaNet / Mamba-2 / Transformer 的复现，且放在独立评测目录，
  **不进 SimplCore 架构**（活跃模型代码仍零 N×N / 零 KV cache）。
- `ssm_pair` 全程接近 chance（0.033–0.041）：本轮**没有**为它调参，如实记录；
  它代表递推扫描型归纳偏置，不是对 Mamba 类模型的判断。
- **没有** ≥10M token 同算力 LM、val ppl、生成可读性、实测吞吐对标、
  完整强基线复现、可学习码本。
- 仍是合成内容链、V=64、L≤25、CPU NumPy；不是语言能力结论。
- **不承诺过稿率**。本轮改变的是证据类型与诚实度，不是「已经能打」。
- 三轮撤回序列（W9 格式先验 → W12 parity → W13 训练旁证）是**单调自我收紧**，
  不是反复横跳；每次都用更严的前提替换更宽的声明。

## 8. 交付物

| 文件 | 内容 |
|---|---|
| `simplcore.pdf` / `build_simplcore_pdf_v5.py` | v6.3 投稿稿与唯一生成器 |
| `patch_v63*.py` | 本轮修订脚本 |
| `tools/eval_baselines/fair_probe2.py` | 公平性审计 + 解析 witness 探针 |
| `tools/eval_baselines/results/corrected_baselines.json` | 8 臂 × 2 hops × 5 seeds 原始结果 |
| `tools/eval_baselines/results/analytic_equivalence.json` | H=1 三机制解析 witness |
| `tools/eval_baselines/results/analytic_h2.json` | H=2 两机制解析 witness |
| `tools/eval_baselines/results/corrected_adjudication.json` | 7/7 判定 |
| `V63_REPORT.md` / `CHANGELOG.md` / `REVISION_LOG.md` | 报告与登记 |

v6.2 的探针与结果**全部保留**作证据链，不覆盖、不删除。

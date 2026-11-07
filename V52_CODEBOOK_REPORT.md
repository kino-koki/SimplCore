# 本轮：移除最后一个外部 side-channel（v5.2）

> 「续」这一轮没有新增头条定理、没有新增台账条目、没有刷倍率。
> 做的是把 v5.1 修完之后**剩下的那个软肋**挖出来打掉。
> 全部结果在本沙盒真实运行；无语言建模训练、无 ppl、无生成样本。

---

## 0. v5.1 修完还剩下什么

W9 把 `r[t]=1[(t-1) even]` 换成了「输入符号自带 1-bit KEY/VALUE tag」。
但那个 tag **仍然是运行时外部注入的标注**。审稿人下一句必然是：

> 给每个 token 打了 KEY/VALUE 标签，和格式先验有多大区别？

所以这一轮的目标不是再补一个实验，是**让角色区分变成 SGC 自己码本的内生属性**。

---

## 1. 构造：把角色写进 token id

```
C = Sylvester Hadamard(V),  V = 2 V_K,  C C^T = I_V
key 符号 k  -> id k        -> C[k]          （前 V_K 行）
value 符号 k -> id V_K+k   -> C[V_K+k]      （后 V_K 行）
filler       -> 零 one-hot -> 零码
```

因为 Hadamard 行两两正交，**query `C[k]` 与所有 value 半区码正交** ⇒
链里「同一个符号既是上一跳 value 又是下一跳 key」造成的双峰歧义
**不需要任何 mask 就消失了**：

```
m_h,t = <q_h, A_t>                 只在 k_(h-1) 的 KEY 位置非零
z_h   = Σ_t K_h(L-1-t) m_h,t b_t = c_h C[V_K + k_h]
q_(h+1) = (z_h @ C_val^T) @ C_key = C[k_h]     固定线性换基，无可学权重
logits = z_H @ C_val^T                        解码出 k_H
```

层间接口是**码本内部的固定正交换基**，核外、内容无关、可微；
`K_h(d)=Σ_p w_p ρ_p^d` 仍静态 Toeplitz。

## 2. 七组攻击，全部通过（13/13）

### A. 4 布局 × 4 编码（零训练，H_L=H=2）

| layout | **codebook / 无 gate** | flat + tag (v5.1) | flat + none | codebook + parity |
|---|---:|---:|---:|---:|
| alternating | **1.0000** | 1.0000 | 0.8691 | 1.0000 |
| gapped | **1.0000** | 1.0000 | 0.9609 | 0.2969 |
| dense | **1.0000** | 1.0000 | 1.0000 | 0.2695 |
| padded | **1.0000** | 1.0000 | 0.9922 | 0.3184 |

### B. 层数 × 跳数 5×5（codebook，零训练）
对角全 **1.0000**，非对角 ≤ 0.10（多数 0.0），过冲仍失败。

### C. 5-seed 配对 CI
| 对照 | 差 | CI95 |
|---|---:|---|
| codebook − flat/none（**alternating**，最干净） | **+0.1379** | [0.1324, 0.1473]，5/5 正 |
| codebook − flat/none（gapped） | +0.0266 | [0.0211, 0.0328]，5/5 正 |

gapped 的配对差小，是因为 flat/none 基线已被 filler 隐式修到 0.9609。
**报告里写的是那个小的，没有挑大的那个报。**

### D. SCOPE 量化
重复 key → **0.9414**（不是崩，但确实掉）；fp64/fp32/fp16/bf16/int8 ×
n_pairs=4..64、L≤128 全 1.0000（秩-1 稀疏方向，不推翻此前 L≈2049 掉点）。

### E. 实测吞吐
H_L=1..4、L=9..4097：**~4.8–11.9 µs/token**、状态恒 `1024·H_L` B，
均与 L 无关。未优化 NumPy 参考实现，只有标度有意义。

### F. 守恒（最终构造）
Toeplitz 偏差 0.0、FFT/IIR **2.89e−16**、末时刻 1.45e−16、严格因果、
无 N×N / KV cache / softmax-over-tokens、**无外部 tag、无位置 gate**，全绿。

### G. 真实词表规模（堵「V≫64 不现实」）
| V | 显式矩阵 | FWHT |
|---|---:|---|
| 64..1024 | ≤8.4 MB | 相对误差 2.7e−16 ~ 1.1e−15 |
| **262144**（Qwen3 词表 R=2^18） | **549.8 GB** | **0.010 s**，对合误差 3.27e−16 |

Sylvester Hadamard 有 O(V log V) 快速变换，与 SGC 的 FFT 路径同构；
**不需要显式存码本**。

---

## 3. 最反直觉的一条：把 gate 加回来反而有害

`codebook + parity` 在 gapped 掉到 **0.2969**。

R2(d) 给出解析证明：码本分区后 m_h **只有一个**非零峰；
任何只依赖位置的 gate 只能**删掉**那个峰。一旦删到 key 后继位置，
`z_h = 0`，准确率掉到 chance。所以额外的 role gate 不是「无害」，
是**严格更差**。

这也解释了为什么 §4.4 里 parity 崩掉：它当时是在 flat 编码上做掩码；
换到 codebook 后它连「补上缺失信息」的功能都没有，只剩破坏。

---

## 4. 理论处置：**不新增头条定理**

按「去重归类、不要太多也不要太少、有竞争力就行」：

- **NT31 的 H5 再次放宽**（前提弱化、结论不变 ⇒ 定理变强）：
  「输入编码提供 key/value 角色区分」，三种充分实现按前提强弱排序
  1. **码本正交分区（最弱，无 side-channel）** ← R2 证明充分
  2. 外部 1-bit role tag（v5.1）
  3. 绝对位置奇偶（仅严格交替下充分，W9 已撤回）
- **新增 Corollary R2**（附录 A.33，**NT31 的构造性推论**，
  不是第 32 条头条定理、**不新增台账条目**）：(a) 无需任何 gate 的唯一峰、
  (b) 归纳 z_h = c_h C[V_K+k_h]、(c) 紧性保持、(d) 额外 gate 严格有害、
  (e) 守恒、(f) 前提排序。6 处 QED + 非空洞实例 + 8 条 SCOPE。
- **Lemma R1 不被推翻**：R2 改的是 **P1 输入编码**，不是 P3 层间接口；
  这正是 R1 SCOPE (iii) 明确预留的逃逸。已在 R1 的 boundary 与 source 里标注。
- 台账**保持 51 项**，主文核心定理**仍是 6 条**。附录 G 新增一段
  「Why Corollary R2 does not add a 52nd item」。

验证器重跑：**NT31 17/17、R1 12/12、R2 17/17**（只验陈述与 witness，证明解析）。

---

## 5. 三级前提收敛（这是本轮真正的叙事）

```
v5.0  r[t]=1[(t-1) even]    格式先验          -> W9 撤回
v5.1  external 1-bit tag    运行时外部标注     -> 本轮移除
v5.2  codebook partition    码本内生，无标注   -> 当前
```

**每一步都是假设的单调弱化，不是反复横跳。** 结论始终是
「H 跳恰需 H 层、零训练 1.0、O(1) 状态、静态 Toeplitz」。

---

## 6. 论文 v5.2

`simplcore.pdf`：**20 页，主文 4 页**（8 页上限内；+1 页是 §4.5 的有效证据）。
生成器 `build_simplcore_pdf_v5.py` 仍是唯一来源。

| 位置 | 改动 |
|---|---|
| 摘要 | W9 + 「移除剩余 tag，码本分区使角色内生，无 side-channel」 |
| §3.2 H5 | 换成三级前提弱化叙述 |
| **§4.5（新）** | 码本构造、4×4 表、配对 CI（含较小那个）、gate 有害、grid/scope/守恒、**FWHT 真实词表规模**、诚实残余 |
| §5 | 更新为「role necessity 布局条件 + 无需 side-channel」 |
| Appendix A.33 | Corollary R2 完整前提/证明/边界/用途 |
| Appendix F | W9 行补「CONVERGENCE（非新撤回）」三级链 |
| Appendix G | 保持 51 项 + 明写 R2 不新增条目 |
| 排版 | 缩短 §4.4/4.5 标题、收紧表头，消除贴边与换行 |

校验：实体 / TeX 碎片 / `Fifteen` / `NT01-NT15` / `All 15 proofs` /
`49-Item` / `50-Item` 全 **0**；非 ASCII 仅 55 个 `·`。

---

## 7. 诚实边界（不包装）

- **编码层的归纳偏置仍在**：KEY 符号和 VALUE 符号是不同 id。我们主张的是
  「无运行时标注、无绝对位置规则、无格式先验」，**不是**「没有任何结构约定」。
  R2 SCOPE (iii) 与 §4.5 的 Honest residual 都明写。
- R2 只证明「在 Q1–Q5 下码本分区充分」，不证明它是唯一方案。
- distinct keys、相邻 (key,value) 发射、外部 query、末层读出仍是前提；
  重复 key 掉到 0.9414。
- V 需为 2 的幂；任意 V 取 `R=2^ceil(log2 V)`（NT04/NT05，R<2V）。
- 有限精度只到 L=128；吞吐是未优化参考实现。
- **四类硬实验仍一项没补**：≥10M token 同算力 LM、Mamba-2/GLA/DeltaNet/
  Hyena/Transformer 同场基线 + 实测吞吐对比、真实 MQAR 标准协议多层、
  O1–O4 全配对 CI。TS2SC 端到端迁移仍未跑（无 0.6B 权重本体、无 torch、403）。
- 合成内容链任务，**不是语言能力结论**。

---

## 8. 交付物

| 文件 | 内容 |
|---|---|
| `simplcore.pdf` / `build_simplcore_pdf_v5.py` | v5.2 投稿稿与唯一生成器 |
| `V52_CODEBOOK_REPORT.md` | 本报告 |
| `tools/native_probe/chain_codebook_probe.py` | 七组攻击全部代码 |
| `tools/native_probe/results/codebook_*.json` | 8 份原始数据 + 判定 |
| `theory_work/codebook_corollary.py` | Corollary R2 + 17 项验证器 |
| `theory_work/nt31.py` / `role_lemma.py` | H5 放宽 / SCOPE (iii) 标注 |
| `theory_work/results/*.json` | 定理/引理/推论全文与验证结果 |
| `CHANGELOG.md` / `REVISION_LOG.md` | 进展与撤回/收敛登记 |

私有权重字节、GitHub token 一律未推仓。

# TS2SC-Q17：Qwen3-1.7B-Instruct 解析直接迁移实录（2026-11-04）

**结论：迁移没有成功。** 不能据此推论「可以迁移更大模型」——失败与 0.6B 同源，
模型变大不会自动解决，反而因为 `D=2048 > KV·HD=1024` 让一半读出维度空置的问题更突出。

但这一轮把「失败」从黑箱变成了**三个可定位、已部分修复的解析缺陷**，
并首次拿到了 instruct 基座的正确中文基线。以下全部有落盘证据。

---

## 0. 真正确定的成果：teacher 基线是好的

`Qwen/Qwen3-1.7B-Instruct`（用户上传，4.064 GB / 42 分片 / 311 张量），
用 chat template + no-think 预填（`<think>\n\n</think>\n\n`），argmax 贪心：

```
问：「华为是什么公司」
答：「华为技术有限公司（Huawei Technologies Co., Ltd.）是一家」   (13 token)
```

事实正确、语序正常、无重复。**这证明「会不会说正常汉语」这一侧完全 OK**，
之前十几轮的「中国首都是南京」「阳光明媚，微」是 base model + 裸文本的问题，
不是架构问题。问题 100% 在「用 SGC mixer 替换 self_attn」这一侧。

工程上：42 个分片不重组（沙箱单文件上限 2GB），用 `q17_reader.py`
跨片 mmap + 逻辑 offset 映射流式读取；`q17_teacher.py` 按 token 行读 embedding、
分块算全词表 logits（bf16 matmul），在 2 核 / 2.4GB 沙箱里跑通 1.7B 前向。

---

## 1. 迁移方法（严格直接迁移，非蒸馏）

复用 `analytic_transfer.py` 的解析管线，只换权重源与维度：

| 部件 | 去向 |
|---|---|
| `embed_tokens` / `lm_head` / 28 层 FFN / 全部 layernorm | 逐位直搬 |
| `input_layernorm.weight` | → SGC `norm.weight` 逐位 |
| `v_proj.weight` | → SGC `in_proj` 前 1024 行（零填充到 2048） |
| `o_proj.weight` | → SGC `out_proj` 前 1024 列，`fold_GQA` 解析折叠 |
| `coef` | ← LS 拟合 teacher 平均注意力 profile `p(τ)` |
| `(ρ,θ)` | ← `linear` 正交网格（已验证修复基塌缩） |
| `out_scale` | ← 逐层 LS 标定 |

零梯度、零训练、零 q·k 打分、零 N×N 矩阵、零 KV cache，O(1) 状态保持。
语料 6 段真实中文（含问答格式），profile 增量落盘抗 502。

---

## 2. v1 结果：失败，但定位到根因

**核拟合塌缩**（`Σp` vs `ΣK`，`Ψ` 条件数 3.36e9）：

| 层 | Σp | ΣK | cos(K,p) | rel |
|---:|---:|---:|---:|---:|
| 0 | 1.22 | 0.84 | 0.9932 | 0.117 |
| 3 | 4.56 | **0.31** | **0.1473** | 0.989 |
| 14 | 3.82 | 0.41 | 0.2233 | 0.975 |
| 27 | 4.77 | 0.33 | 0.1526 | 0.988 |

Qwen3 中层注意力是**平坦长尾**分布；SGC 的阻尼 `sin` 振荡基没有直流/低通道，
`linear` 网格又把 θ_min 抬到 0.3（为修正交性），结果中层拟合丢掉约 90% 低频能量。

**逐层残差流漂移**（`q17_diag.py`，teacher-forced 同一段文本）：

```
L 0  cos_M=0.8191  inj=0.8459  ratio=0.837   norm T/M=125/112
L 4  cos_M=0.1626  inj=0.0085  ratio=0.102   norm T/M=15164/94825   ← 饱和
L27  cos_M=0.0518  inj=0.4413  ratio=0.107   norm T/M=14247/11031
```

生成（8 prompt 中的「华为是什么公司」）：
- teacher：`华为技术有限公司（Huawei Technologies Co., Ltd.）是一家`
- v1 migrated：`IPHER` × 6（同一 token 58088 无限重复）
- v1 + DC-bias：`在2\n\n在20`（不动点循环）

`out_scale` 用 LS 投影比 `<Δ,a>/<Δ,Δ>` 标定，当 Δ 与 a 几乎正交时趋近 0，
**完全无法补偿幅度缺失**（实测 `||Δ||/||a||≈0.10`），out_scale 落在 0.5~1.25。
这是可修的标定 bug，不是表达力上限。

---

## 3. v2：解析扩展读出（严谨排除过拟合）

1.7B 的 `D=2048` 但 `KV·HD=1024`，v1 的 `in_proj` 后 1024 行 /
`out_proj` 后 1024 列全为 0 —— **一半读出容量白扔**。
v2 把 `k_proj` 填进 `in_proj` 后 1024 行，并用 teacher-forced 闭式 ridge 解
`W@z ≈ attn_out` 求 `out_proj`（与 `out_proj` 同形，**零额外参数**）。

防过拟合：5 段拟合 / 1 段 holdout，`W = W_fold + λ(W_ridge − W_fold)`，
λ 由 holdout 余弦在 {0, 0.25, 0.5, 0.75, 1.0} 中选。

```
holdout 余弦  fold 均值 0.5088   ridge 均值 0.2530   采用均值 0.5256
28 层中 20 层 λ=0（保持解析折叠），8 层 λ=0.25
```

**ridge 解明显过拟合**（0.253 vs 0.509），与 0.6B 时「读出对齐 W」被排除的结论一致。
改进只有 +0.0168，且没有一层变差 —— 机制有效，收益有限，已如实记录。

---

## 4. v3：RMS 幅度标定（部分修复，实测有效）

把 `out_scale` 从投影比改成 **RMS 比** `s = sqrt(Σ||a||²/Σ||Δ||²)`，
与方向解耦。均值 `1.4557`（v1 投影比均值约 0.72），逐层 1.12~1.81。

| 指标 | v1 | v3 | 变化 |
|---|---:|---:|---|
| L0 注入余弦 | 0.8191 | **0.9041** | +0.085 |
| L27 残差流余弦 | 0.0518 | **0.2221** | +0.170 |
| L4 残差流饱和范数 | 94825 | 78233 | −17% |
| L0 `||Δ||/||a||` | 0.837 | **0.993** | 幅度对齐 |
| +DC-bias 流 L4 余弦 | −0.0001 | **0.5802** | 转好 |

生成（同一 prompt，8 token，argmax）：

| 版本 | 输出 |
|---|---|
| teacher | `华为技术有限公司（Huawei Technologies Co., Ltd.）是一家` |
| v1 | `IPHERIPHERIPHERIPHERIPHERIPHER` |
| **v3** | `的\n\n好像\n\n好像宫宫哪` |
| v3 + DC-bias | `\n` + 替换字符（更差，不再采用） |

**从「单 token 死循环」变成「多个不同中文词」，是实测的实质改善；
但仍然前言不搭后语、后半段退化重复，不是正常汉语，更不是有效回答。**

DC-bias 在 v3 上必须移除：out_scale 放大后 bias 变成大的反向向量，
把注入量抵消回去（B 流范数仅 1166，teacher 是 15164）。

---

## 5. 剩下的边界（已量化，不是猜测）

补一列常数（直流）基到 Ψ 字典后，中层拟合只从 cos 0.15 → 0.39、rel 0.99 → 0.92：

```
L 3 : 0.1473 → 0.3870      L14: 0.2233 → 0.4102
L21 : 0.1159 → 0.3759      L27: 0.1526 → 0.3708
```

说明**缺的不只是直流，而是整个平坦长尾形状**。阻尼 sin 基族本身表达不了这类 profile。
更本质的是：静态核输入无关，做不了 key→value 内容寻址 ——
与 0.6B 时实测的静态平均注意力核上限 cos 0.3695、正交网格在 MQAR 上 0.1133 同源。

**这一条不是调参、不是加步数、不是加核能绕过去的。** 要越过去只有两条：
1. 给 SGC 加一个真正的低通/累积状态通道（一阶 IIR `s_t=ρs_{t-1}+u_t`，
   O(1) 状态、线性、无 pairwise —— 仍符合纯粹性），直接补 sin 基族缺的形状；
2. 接受 hybrid（少量 attention 管内容寻址），放弃纯粹性。

---

## 6. 对用户两个问题的直接回答

**「如果你成功，说明我可以迁移更大模型」** → 没有成功，所以这个推论不成立。
而且 1.7B 暴露了 0.6B 没有的新浪费（D 翻倍但 V 路径仍是 1024），
**模型越大，解析直搬的维度不匹配越严重**，不是更容易，是更需要专门适配。

**「AMD 680M 4GB 能否完整迁移」** → 见 `docs/AMD680M_migration_feasibility.md`。
一句话：解析迁移阶段本身最不吃显存（闭式解 + 逐位直搬，无梯度/优化器状态），
4GB 绰绰有余；卡点在 teacher 前向（1.7B bf16 = 4.06GB 需 Q4 或流式 mmap）、
ROCm 不支持 gfx1035（社区伪装会产生错误结果，应走 Vulkan）、
以及 **llama.cpp 不认 SGC 算子**（需自定义 Vulkan compute shader）。

---

## 7. 本轮未做（如实列出）

- 用户提过的「适当枝剪权重」未做：当前瓶颈不在参数量，剪枝会进一步损伤已很脆弱的注入量方向。
- 「加 MiniCPM-V 4.7 视觉编码器到最大总参数」未做：语言侧尚未跑通，先堆参数没有意义。
- 「每层 500~1000 步训练」未做：上一轮 181 步自监督微调已证伪（holdout −10.4% 但生成 rep3 +36%）。

## 8. 交付物与复现

```
q17_reader.py      42 分片跨片 mmap 流式读取器（311 张量，index 零未映射）
q17_teacher.py     1.7B-Instruct 原版前向（chat template + no-think）
q17_migrate.py     解析直接迁移（profile / fit / gen）
q17_fit2.py        v2 解析扩展读出（5 拟合 / 1 holdout）
q17_calibrate.py   v3 RMS 幅度标定 + DC-bias 重算
q17_diag.py        逐层残差流漂移诊断
q17_gen.py         多版本生成复核（CKDIR / BIASFILE 可切换）

TS2SC_Q17/
  profile.json  fit.json  fit2.json  rms_calibration.json
  drift_ck.json  drift_ck3.json  gen_progress_ck3.json
  ck/ (v1)  ck2/ (v2)  ck3/ (v3，当前最佳)
  Q17/teacher_baseline.json
```

**当前最佳版本 `ck3`**：v2 读出 + v3 RMS 标定，**不加 DC-bias**。
仍未达到「会说正常汉语」，不建议并入任何权重包。

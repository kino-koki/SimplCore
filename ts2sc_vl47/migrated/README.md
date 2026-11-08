# TS2SC-VL-1B — v2

**SimplCore 迁移语言侧 + MiniCPM-V 4.7 视觉侧，权重级合并，可训练参数 1,057,320,252（1.0573 B）。**

| | 参数量 | 来源 |
|---|---:|---|
| 语言侧 | **509,053,836** | Qwen3-0.6B FFN/emb/norm 逐位直搬 + 28 层 SimplCore SGC mixer |
| 视觉侧 | **548,266,416** | MiniCPM-V-4.7-1B `model.vision_tower.*` + `model.merger.*` |
| **可训练合计** | **1,057,320,252** | **1.0573 B** |
| 静态偏置 buffer | 28,672 | `dc_bias.safetensors`，fp32，无梯度，不计入参数量 |
| 含 buffer 总元素 | 1,057,348,924 | 1.05735 B |

111 个 safetensors 分片 / 1017 个索引张量 / 2.136 GiB。**v2 不改动 v1 的任何一个分片字节**，只新增偏置文件、配置、证据与接入脚本。所有分片 ≤ 24 MiB，按 GitHub REST API 的 base64 上限重切。

---

## v2 改了什么：修「前言不搭后语」

Qwen / Llama 用 **RMSNorm——只缩放、不减均值**，残差流的均值向量会原样进入下一层 FFN。把注意力子层换成 SimplCore mixer 后，注入量 `Δ_s = mixer(x) − x` 的逐位置均值与 teacher 的 `attn_out` 均值不同（实测逐层余弦全层均 **0.7456**，深层 `|b_t − b_s|` 达 20~52，残差 std≈48），28 层累积成系统性分布偏移。

修复是每层一个静态向量，**零梯度、零训练、零架构改动**：

```
b_i = mean_pos(attn_out_i) − mean_pos(Δ_s,i)
y_i = mixer_i(x) + β·b_i        (β=1.0，FFN 之前)
```

| 性质 | 状态 |
|---|---|
| q·k 打分 / N×N 矩阵 / KV cache | 一个都没引入 |
| 核输入无关、静态、可 IIR 递推 | 保持 |
| O(1) 流式状态 | **逐字节不变**（偏置不进递推状态） |
| FFT / IIR 对偶 | 保持（逐位置常数，两路径同值） |
| 成本 | 28×1024 fp32 = 114 KB，每层一次向量加 |

生成实测（argmax 贪心，NNEW=16）：

```
「中国的首都是」  旧: 城市，有，北京，北京，北京，北京，北京，北京，        rep3 0.6429
                  v2: 的，是直辖市，人口，是新兴。深圳是新兴。深圳是      rep3 0.2143
「今天天气很好，」旧: 适合户外，微动，天气晴，绿，花儿在树子
                  v2: 晴，心情的植物园圃，绿圃，绿园，公园里
```

推理接入见 `scripts/apply_dc_bias.py`（自带自测），解析推导与七条被排除路径的完整实测见 [SimplCore `docs/TS2SC_coherence_audit.md`](https://github.com/kino-koki/SimplCore/blob/main/docs/TS2SC_coherence_audit.md)。

**被实测排除、因此 v2 没有采用的路径**：λ 逐层校准、读出对齐 W（过拟合，生成退化成「这这这」）、norm 直搬、DPT 投影直搬、静态平均注意力核（表达力上限 cos 0.3695）、v3 解析直接迁移（`otionalotional…` rep3 0.9286）、WSP 循环方向消融。原始证据在 `evidence/coherence_fix.json`。

**训练侧也已实测并排除**：181 步自监督 next-token 微调（严格非蒸馏，冻结 emb/FFN/lm_head，只训 109 万 mixer 参数）让固定 holdout CE 从 4.810 单调降到 4.309（−10.4%），但 8 个 prompt 的生成复核显示 **rep3 0.1719→0.2344（+36% 变差）、中文占比 0.7469→0.7310、事实正确数持平**。holdout 衡量的是「在冻结 FFN 下预测下一个字有多确定」，不是「会不会说正常汉语」。**因此训练权重不并入本包**，v2 保持为最佳版本。完整数据与根因分析见 [SimplCore `docs/TS2SC_sft_negative_result.md`](https://github.com/kino-koki/SimplCore/blob/main/docs/TS2SC_sft_negative_result.md)，原始输出在 `evidence/lm_finetune_eval.json`。

---

> ## ⚠️ 先读这一段：这个包是什么，不是什么
>
> **它是**：一个参数量精确、索引自洽、逐张量可加载、跨模态维度已对齐的**权重级合并产物**，外加一个经生成验证的静态分布修复。
>
> **它不是**：一个能看图说话的 VQA 模型，也不是一个会「回答」的助手。
>
> 没有做跨模态对齐训练；MiniCPM-V 4.7 词表 248094 与 Qwen3 词表 151936 不兼容（图像特殊 token id 248056 等越界）；视觉侧用 mrope/canvas，语言侧用标准 RoPE；语言侧是 base model（teacher 自己对「华为是什么公司」也只会续成追问）；静态平均注意力核的表达力上限只有 cos 0.3695，**pairwise 内容寻址是硬缺口**。
>
> 偏置修复降低重复循环、改善成句性，但**不修事实错误、不教问答行为、不补 pairwise 缺口**。参数量到 1B 与「可用」是两件事。本包的价值是把可对齐的部分精确拼好、把不可对齐的部分明确列出，让后续训练有一个干净起点。

---

## 目录

1. [本轮回答：WSP 对 TS2SC 迁移有没有价值](#1-本轮回答wsp-对-ts2sc-迁移有没有价值)
2. [参数账本](#2-参数账本)
3. [架构与 SimplCore 纯粹性](#3-架构与-simplcore-纯粹性)
4. [跨模态对齐：为什么能拼](#4-跨模态对齐为什么能拼)
5. [文件清单与重组](#5-文件清单与重组)
6. [加载示例](#6-加载示例)
7. [完整性验证结果](#7-完整性验证结果)
8. [已知限制](#8-已知限制)
9. [复现步骤](#9-复现步骤)
10. [与其他仓库的关系](#10-与其他仓库的关系)

---

## 1. 本轮回答：WSP 对 TS2SC 迁移有没有价值

### 1.1 结论

**方法论有价值，具体技术没有落地价值。** WSP（MIT CSAIL《Sparks of Weight-Space Programming》，2026-11-02 博客，未同行评审）修的是**遮蔽**（模型本来会做，但被一个可定位的结构故障挡住写不完），不修**能力缺口**。我们的迁移属于后者。

### 1.2 WSP 到底做了什么

| 项 | 数值 |
|---|---|
| 对象 | Qwen3.5-0.8B |
| 改动 | 131,072 权重 = 0.016% |
| MATH-500 | 7.9% → 48.1% |
| 撞 32768 token 上限 | 93.0% → 33.1% |
| 手段 | 第 15/19 层 MLP 各 64 个输出列，减去「循环方向」`W' = W − α·r(rᵀW)` |

Qwen 官方模型卡早已警示：0.8B 强开思考模式会无限重复。实验把上限撑到 32768 并强制开启，于是 93% 的题不是算错，是**把算力耗在车轱辘话里被截断**。改后多答对的 579 题中，**576 题原本只是超时得零分**；原本能写完的题正确率毫无波动。换成不死循环的 Gemma 3 270M，各种编辑让 MATH-500 变动不超过 1 个百分点。作者原话：权重编辑能移除遮蔽能力的障碍，不能凭空添加模型没有的能力。

这本质是 **abliteration**（秩一投影去掉一个行为方向），不是新技术。

### 1.3 静态审计：权重谱里找不到可减的方向（结论有限）

`scripts/wsp_value_audit.py` 用 power iteration 求每层写入矩阵的最大奇异方向能量占比 σ1²/‖W‖_F²（纯 numpy，零训练）：

| 矩阵 | 最小 | 均值 | 最大 | 随机对照 | 最大/随机 |
|---|---:|---:|---:|---:|---:|
| `o_proj`（attn 写入残差流） | 0.0056 | 0.0076 | 0.0109 | 0.0028 | 3.88× |
| `down_proj`（MLP 写入残差流） | 0.0043 | 0.0083 | 0.0127 | 0.0024 | 5.40× |

- 异常层（|z| > 2.5）：**无**
- 相邻层主导方向余弦均值 **−0.0064**，跨层无共享方向
- 最大第一方向能量占比仅 **0.0127**，谱是平的

→ **从权重谱**找不到可定位的异常主导方向或跨层共享 `r`。

> ⚠️ **这条结论有限，必须配下面的复核一起读。** WSP 的 `r` 是从**正常/循环激活均值差**估计的，不是从权重谱读出来的。静态谱平 ≠ 激活层面没有循环方向。本轮完整激活级消融（`loop_ablate.py`，含安慰剂对照）因沙箱反复重置**未跑完**，所以对本模型的 WSP 适用性应判为**「未证实」，不是「已否定」**。

### 1.4 更根本的判据：我们缺的是能力，不是退出条件

| | Qwen3.5-0.8B（WSP 成功） | TS2SC student（我们） |
|---|---|---|
| 症状 | 会做但重复到超时 | 逐块迁移余弦 ~0.7，输出失真 |
| teacher 生成 | — | `我需要写一篇关于天气的作文` |
| student 生成 | — | `微，孩子们在枝儿在枝` / `在花儿在花儿在花` |
| 性质 | **遮蔽** | **能力缺口** |

按 WSP 作者自己划定的边界，手术刀落不下去。

> **复核结果（NNEW=14，3 prompts，见 `evidence/generation_recheck.json`）**：
>
> | 提示 | student 生成 |
> |---|---|
> | `今天天气很好，` | `适合户外，微动，天气晴，绿，花儿在` |
> | `中国的首都是` | `城市，有，北京，北京，北京，北京，北京，` |
> | `水的化学式是` | `城市，但，也，也。\n这些的，所以，但` |
>
> 三点结论：
> 1. **事实知识真的在**——`北京` 出现了，印证 FFN 逐位迁移余弦 1.0000，知识没丢。
> 2. **重复循环真的存在**——`北京` 连出 6 次，正是 WSP 针对的症状类型。这推翻了「我们这里没有循环」的隐含假设，也说明 1.3 的静态谱审计不能单独定案。
> 3. **但仍不是能力到位**——续写成填空题而非回答，第三句语义完全错乱。按 WSP 作者自定边界，即使消融去掉循环，也只是让答案写完整，**补不上 base model 问答行为与迁移余弦 ~0.7 的能力缺口**。
>
> 所以最终判定：**WSP 类消融值得作为 A 类故障的候选去实测（未做），但不能指望它把模型变成可用助手。**

### 1.5 真正该带走的东西：误差三分

WSP 示范了**把结构性故障从能力缺口里切出来单独处理**。回头看我们十几轮的失败，一直默认「生成崩 = 架构表达力不够」然后硬啃蒸馏，但里面混着一批零训练可修的 bug：

| 类 | 内容 | 手段 |
|---|---|---|
| **A 故障** | `out_scale≈0.02`（mixer 被压 98%）、RoPE 奇偶交错 vs `rotate_half`、`coef` 形状静默跳过、残差流 1000× 尺度爆炸 | 解析修复，**零训练** |
| **B 容量塌缩** | θ_min=π/8192 → 32 基近共线，数值秩 24/32、cond 1.92e16 | `init_mode="linear"`，**零成本**（SimplCore `f4fe67d`） |
| **C 真实缺口** | pairwise key→value 绑定、base model 问答行为 | 蒸馏 / SFT / 换 instruct 基座 |

**A、B 清零之后剩下的余弦误差才是真实的 C。** 之前所有迁移数字都该在 A/B 清零后重测——这是 WSP 给我们最值钱的产出。

---

## 2. 参数账本

精确到个位，三处独立计算已对齐（`param_accounting.json` / `config.json` / 索引实测）：

```
语言侧
  Qwen3-0.6B 原版（独立参数）        751,632,384  ← 含 attention 176,160,768
    其中 embed_tokens 155,582,464（lm_head 与之 tie，不重复计）
    其中 28 层 mlp + layernorm  264,298,496
    其中 final norm                   1,024
  ─ 丢弃 self_attn.*             −176,160,768
  ─ 保留逐位直搬部分              419,881,984
  + 28 层 SimplCore SGC mixer     +89,171,852  （单层 3,184,709）
  ────────────────────────────────────────────
  语言侧小计                        509,053,836  = 0.5091 B

视觉侧
  vision_tower（ViT 27 层 + vit_merger） 522,299,312
  model.merger.mlp.0                       25,967,104
  ────────────────────────────────────────────
  视觉侧小计                        548,266,416  = 0.5483 B

总计                              1,057,320,252  = 1.0573 B
```

**注意口径**：`tie_word_embeddings=True`，`lm_head.weight` 与 `model.embed_tokens.weight` 逐位相同（已验证不等元素数 0 / 155,582,464）。若按「张量总数」而非「独立参数」计，会得到 1.21 B 的虚高数字，那是重复计数。

dtype 分布：BF16 605 张量（原版权重）、F32 392 张量（28×14 个训练得到的 mixer 参数）。mixer 保留 fp32 是因为蒸馏在 fp32 下进行，转 bf16 会引入未量化的精度损失，留待后续统一量化。

---

## 3. 架构与 SimplCore 纯粹性

```
image (980×980, patch14)
  → vision_tower.embeddings (patch_embedding + position_embedding 4900×1152)
  → vision_tower.encoder × 27 (hidden 1152, heads 16, mlp 4304)
  → vision_tower.vit_merger (self_attn + linear_1 17216×4608 + linear_2 1152×17216)
  → model.merger.mlp.0 (4608 → 1024)          ★ 输出 1024
  → 注入语言侧残差流 (D=1024)                  ★ 无需新增投影层
  → [SimplCore SGC mixer + Qwen3 FFN] × 28
  → rms_norm → lm_head (tied)
```

### 纯粹性清单（逐条对照）

| SimplCore 契约 | 本包状态 | 说明 |
|---|---|---|
| 核**输入无关**（静态振荡基） | ✅ 保持 | 未引入 MoK 多核路由，未引入 content-dependent kernel |
| 无 N×N pairwise 打分 | ✅ 保持 | 语言侧 `self_attn` 残留 **0 张量**（已验证） |
| 无 KV cache | ✅ 保持 | 语言侧状态 274,432 B/层恒定，28 层共 7.33 MiB |
| O(1) 状态 vs 序列长度 | ✅ 保持 | 与 L 无关 |
| O(L log L) / IIR 对偶 | ✅ 保持 | FFT/IIR 双路径未改 |
| 频率网格可 opt-in `linear` | ✅ 保留接口 | 本包 mixer checkpoint 是 `logfreq` 训练所得，见第 8 节 |

**视觉编码器自带 ViT self_attn（224 张量）是正常的**：那是图像 patch 之间的注意力，属于视觉侧，不违反语言侧 SimplCore 的纯粹性。验证脚本对此做了范围区分。

### 成本账（与 WSP / MoK 对照）

| 方案 | 参数改动 | 梯度 | 诊断成本 | 状态字节 |
|---|---|---|---|---|
| 频率网格 `linear` | **0** | 无 | 一次性解析 | 不变 |
| WSP 秩一消融 | 0.016% | 无 | 20.3 h / 158 轮 / 50 次前向 | 不变 |
| MoK 多核路由 | +0.1% | 需训练 | — | O(1) 但 readout ×K |

---

## 4. 跨模态对齐：为什么能拼

这不是凑出来的巧合，是三个 1024 撞在一起：

| | 值 |
|---|---|
| MiniCPM-V 4.7 `text_config.hidden_size` | **1024** |
| `model.merger.mlp.0.linear_2` 输出维 | **1024** |
| Qwen3-0.6B / SimplCore 语言侧 `D` | **1024** |

→ 视觉 token 经 merger 投影后**维度逐位对齐**语言侧残差流，**不需要新增任何随机初始化的投影层**（`extra_untrained_params = 0`）。这一点已由 `verify_report.json` 断言验证。

没对齐的部分见第 8 节。

---

## 5. 文件清单与重组

```
TS2SC-VL-1B/
├── config.json                      统一配置（text_config + vision_config）
├── model.safetensors.index.json     997 张量 → 27 分片的映射
├── slices.json                      被 axis0 切分的大张量还原表
├── param_accounting.json            参数账本
├── verify_report.json               完整性验证报告
├── README.md                        本文件
├── shard-00001-of-00111.safetensors … shard-00111-of-00111.safetensors
├── RESTORE.md                       分片说明
└── scripts/
    ├── download_vision.py           从源仓库只拉视觉侧 12 个分片
    ├── extract_vision.py            跨分片按字节区间提取视觉权重
    ├── build_vl1b.py                组装本包（27 大分片）
    ├── repack_vl1b.py               重切为 111 个 ≤24MiB 小分片
    ├── verify_vl1b.py               纯 numpy 完整性验证
    ├── wsp_value_audit.py           WSP 价值审计
    └── load_vl1b.py                 零 torch 加载示例
```

**分片不需要 `cat` 重组。** 每个分片都是独立合法的 safetensors 文件，按 `model.safetensors.index.json` 的 `weight_map` 按 key 取即可。

有 4 个张量因超过单分片上限被按 axis 0 切开，需按 `slices.json` 拼回：

| 原张量 | 原 shape | 块数 |
|---|---|---:|
| `model.embed_tokens.weight` | `[151936, 1024]` | 16 |
| `vision_tower.vit_merger.linear_1.weight` | `[17216, 4608]` | 8 |
| `vision_tower.vit_merger.linear_2.weight` | `[1152, 17216]` | 2 |
| `merger.mlp.0.linear_1.weight` | `[4608, 4608]` | 2 |

块 key 见 `slices.json` 的 `parts` 列表，顺序即拼接顺序。

拼接为**同参数拆分**，不增加参数量（验证脚本已确认拼回 shape 精确一致）。

---

## 6. 加载示例

`scripts/load_vl1b.py` 是零 torch 依赖的参考实现（numpy + struct + mmap）：

```python
from load_vl1b import VL1B

m = VL1B("/path/to/TS2SC-VL-1B")
print(m.n_params)                       # 1057320252
w = m.get("model.layers.0.mixer.coef") # 自动按 index 定位分片
emb = m.get("model.embed_tokens.weight")   # 自动拼回 4 块 → (151936, 1024)
print(emb.shape, emb.dtype)
```

要点：
- BF16 是 safetensors 原生 dtype；用 PyTorch 时 `safe_open(..., framework="pt")` 直接得到 `torch.bfloat16`
- numpy 下 BF16 以 `uint16` 位模式读出，转 fp32 只需 `(u.astype(uint32) << 16).view(float32)`
- `lm_head.weight` 不在 index 里（tie），加载时指向 `model.embed_tokens.weight`

---

## 7. 完整性验证结果

`verify_vl1b.py` 输出（`verify_report.json`）：

```
[1] 分片 111 / 索引张量 1017      header 自洽 ✓ 全部通过
[2] dtype {BF16: 625, F32: 392}  含 NaN/Inf 位模式张量 0 ✓
    索引内参数 1,057,320,252
    4 个切分张量全部按 axis0 无损拼回原 shape ✓
[3] 索引 / 账本 / config 三处参数完全相等，差值 +0（tie 未重复计）✓
[4] 语言侧 self_attn 残留 0 ✓
    mixer 392 张量 = 28 层 × 14 ✓
    视觉侧 468 张量（ViT self_attn 224 属正常）
    merger 输出 1024 == text hidden 1024 ✓
────────────────────────────────────────
★ 模型包完整、可加载、参数口径自洽
```

---

## 8. 已知限制

按严重程度排序，**这些都不是「待优化」，是「尚未做」**：

1. **无跨模态对齐训练。** 视觉特征与语言残差流之间的语义对应关系完全没学过。权重拼上了 ≠ 模态打通了。
2. **词表不兼容。** MiniCPM-V 4.7 vocab 248094，Qwen3 151936。`image_token_id=248056`、`image_start_id=248078`、`slice_start_id=248088` 等全部越界。需扩展 embedding 并重映射 id，本包未做。
3. **位置编码方案不统一。** 视觉侧 mrope / canvas（`mrope_section=[11,11,10]`），语言侧标准 RoPE（θ=1e6）。
4. **语言侧不是对话模型。** base model 只会续写不会回答；「中国的首都是」会续成填空题并重复列举「北京」，而不是给出一个确定的答案。需要 SFT 或换 instruct 基座。
5. **迁移 mixer 保真度有限。** 逐块余弦约 0.7，FFN 逐位等价（1.0000），但注意力子层的 pairwise 内容路由是 SGC 静态卷积核表达不了的。这是结构上限，不是训练步数问题。
6. **频率网格是 `logfreq`。** 本包 checkpoint 在秩 24/32 的塌缩字典上训练所得。SimplCore `f4fe67d` 已提供 `init_mode="linear"`（cond 1.92e16→1.04），但注意：在真实 Qwen3 注意力的闭式 LS 口径下，`logfreq` 反而优于 `linear`（R² 0.391 vs 0.293，L=64~512 未反转）——**正交网格只救周期/连续信号轴，不救语言注意力轴**。换默认前必须在目标任务上重测，不能无条件替换。
7. **「会说话」未成立。** 14-token 复核显示前半可连贯、后半退化为词堆砌与重复循环（`北京`×6），第三句语义错乱。事实知识在，但生成不可用。
8. **mixer 是 fp32 而其余是 bf16。** 混合 dtype，部署前需统一量化并重新验证。

---

## 9. 复现步骤

```bash
# 1) 只拉视觉侧分片（不下载 3.11 GB 全量）
export GH_TOKEN=...
python3 scripts/download_vision.py          # vpart_20..31，1.117 GB

# 2) 跨分片按字节区间提取 vision_tower + merger
python3 scripts/extract_vision.py           # → V47/vision/，548.27 M 参数

# 3) 组装语言侧（去 attention、注入 28 个 mixer ck）+ 合并视觉侧
python3 scripts/build_vl1b.py               # → VL1B/，27 个大分片

# 3b) 重切为 ≤24MiB（GitHub REST API 上限；本沙箱 git push 被 403 拦）
python3 scripts/repack_vl1b.py              # → 111 个小分片，参数不变

# 4) 纯 numpy 完整性验证
python3 scripts/verify_vl1b.py

# 5) WSP 价值审计
python3 scripts/wsp_value_audit.py
```

依赖：Python 3.10、numpy、safetensors（仅 `build_vl1b.py` / torch 加载路径需要）、torch（仅组装阶段的 `torch.load(ck)`）。**验证与加载示例零 torch 依赖**。

源材料：
- `kino-koki/MiniCPM-V-4.7-1B`（`vpart_00..31`，重组后 `model.safetensors` 3,109,054,480 B）
- `Qwen/Qwen3-0.6B`（本地 `mig/model.safetensors`，311 张量，751,632,384 张量参数）
- `QMIG_R2/ck/mixer_0..27.pt`（28 个逐块迁移 checkpoint）

---

## 10. 与其他仓库的关系

| 仓库 | 内容 |
|---|---|
| `kino-koki/SimplCore` | SGC 架构本体；`f4fe67d` 加入 `init_mode="linear"` 与 10 项守护测试 |
| `kino-koki/TS2SC` | A2.0 协议：迁移/蒸馏/验证的完整踩坑与文档体系 |
| **本包** | TS2SC 的**权重级产物**：语言侧迁移结果 + 视觉侧编码器，1.0573 B |

协议版本：TS2SC A2.0。本包**不引入新的训练方法**，只是把两条已验证的资产在同一参数口径下合并并做完整性验证。

---

## 变更记录

- **v1.0**（本轮）：完成 1.0573 B 权重级合并；参数三处对齐；header/有限性/切片拼接/架构断言全通过；附 WSP 价值审计（结论：方法论可用、具体技术不适用）。
- **v1.0-repack**：因 `git push` 403 改用 REST API，重切为 111 个 ≤24 MiB 分片；索引张量 1017（含切分块），独立参数仍为 1,057,320,252。
- **v1.0-recheck**：14-token 生成复核发现「知识在 + 重复循环在 + 能力缺口也在」，据此把 WSP 结论从「已否定」修正为「未证实且不构成能力解」；新增 `evidence/generation_recheck.json`。

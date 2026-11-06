# 本轮交付：论文乱码修复 + TS2SC 真实权重级验证 + 多层架构深化（中间态）

> 全部结果在本沙盒**真实运行**，无编造。训练：仅合成内容链任务的小规模训练；
> **没有语言建模训练、没有 ppl、没有生成样本**。
> 私有权重字节**未上传**到公开仓库，只交付元数据与统计结果。

---

## A. `simplcore.pdf` 乱码修复（已重建，19 页，主文 6 页）

| 问题 | 根因 | 修复 |
|---|---|---|
| 摘要/正文出现 `rho&lt;1`、`0 &lt; rho &lt; 1`、`rho -&gt; 1` 等字面串 | 生成器把 HTML 实体原样画进 PDF | `tokenize()` 内统一 `html.unescape` 两次 + 防御性残片清理 |
| `_{j*}`、`^{tau-j}`、`\mathbb{R}` 在小字号下像乱码 | 内联 TeX 碎片未经处理 | 新增 `clean_math()`：宏展开、`_{...}`/`^{...}` → `_(...)`/`^(...)`、残余花括号转圆括号、非 ASCII 过滤 |
| 摘要写 30 定理，正文却写「Fifteen Theorems」「all 15 theorems」、附录 A 标题「NT01-NT15」 | 上一轮口径漏改 | 全部改为 30 / NT01-NT30，§3 标题改「Thirty Theorems in Six Groups」 |
| 页脚数字混入 pypdf 提取文本 | 提取伪影，非 PDF 缺陷 | 已识别，不影响渲染 |

**校验**：`&lt; / &gt; / &amp; / &hlt; / Fifteen / NT01-NT15 / 15 theorems / _{ / ^{ / \tau / \sum` 全部 **0**；非 ASCII 仅 49 个 `·`（项目符号，字体支持）；A.1–A.30 完整证明与 30/30 正文引用齐全。

---

## B. TS2SC 迁移验证（真实跑过的部分 / 明确没跑的部分）

### B.0 先纠正一个前提：私有仓库里**没有** Qwen3-0.6B

用 token 遍历了账号下全部 10 个仓库（`per_page=100`），实际存在的模型仓库是：

| 仓库 | 内容 |
|---|---|
| `Qwen3-1.7B-Instruct` | **真实 Qwen3 1.7B**：hidden 2048 / 28 层 / 16Q 8KV / head_dim 128 / vocab 151936 / BF16；`model.safetensors.index.json` 有 311 张量、`total_size=4,063,479,808` B；权重切成 `qa_00..qa_34` + `qb_00..qb_06` 共 41 个 ~99.6 MB 分片 |
| `MiniCPM5-1B` | 1B 权重，22 个 `model_part_*` |
| `MiniCPM-V-4.7-1B` / `TS2SC-VL-1B` | VL 模型（后者有 111 个 shard + `evidence/source_safetensors_header.json`，是 248094 词表的 linear-attn 模型，**不是 Qwen3**） |
| `Tuan-Nursery-1B` | Release `v-qwen-migrated` 有 `qm.part_00/01/02` 共 1.06 GB（迁移产物），asset 下载被 403 |
| `TS2SC`（公开） | 迁移协议 + 脚本 + **历史** Qwen3-0.6B 实测数据，但仓库内**无 0.6B 权重本体** |

`TS2SC/scripts/migrate_real.py` 硬编码的 Qwen3-0.6B 配置是真实的：`D=1024, FF=3072, L=28, H=16, KV=8, HD=128, V=151936, BASES=32`，`SRC=/data/workspace/mig/model.safetensors` —— 该路径在本沙盒**不存在**（大文件扫描 0 命中）。

**沙盒环境硬限制**：`pypi.org`、`huggingface.co`、GitHub Release asset 全部 **HTTP 403 / Timeout**；无 `torch` / `transformers` / `safetensors`，且装不上。只有 `api.github.com` 的 JSON / git-blob 可达。

⇒ **本轮无法加载 Qwen3-0.6B 完整权重，也无法跑端到端 logit-KL 蒸馏。这两件事没有伪造。**

### B.1 真实完成 ①：Stage-0 provenance（311 张量，100% 覆盖）

按 `TS2SC/docs/01-架构映射.md` 的真实规则，对每个 teacher 张量判定 `COPY / DERIVE / NEW`：

- **Qwen3-0.6B 规划**：311 张量（28 层 × 11 + 3 全局），`COPY=143 / DERIVE=84 / NEW=84`，`UNCLASSIFIED=0`
- **Qwen3-1.7B 真实 index 交叉验证**：311 张量（真数据），同样 `143/84/84`，**100% 分类覆盖**
- `DERIVE`：`v_proj→mixer.in_proj`、`k_proj→mixer.gate_proj`（GQA 下 KV×HD = D，逐位搬）、`o_proj→mixer.out_proj` 切片 `[:, :D]`
- `NEW`：`q_proj / q_norm / k_norm` 无对应，必须随机初始化（这正是 NT22/NT23 说的「不要逼 SGC 模仿 attention 内部」）

### B.2 真实完成 ②：独立参数对账，与历史实测**精确一致**

从形状独立推导（不读 README 数字反推）：

```
COPY   = 2·V·D + D + L·(2D + 3·FF·D)          = 575,464,448   （lm_head 不 tied）
DERIVE = L · 3 · D²  （o_proj 切片到 (D,D)） =  88,080,384
────────────────────────────────────────────────────────────
逐位保留 = 663,544,832   ← 与 TS2SC README 历史值精确相等 ✅
学生总参数（README）= 664,607,516 → SGC 新增 = 1,062,684（差值，构成未逐项核实）
逐位占比 = 0.998401
```

> 注意：上一版 `ts2sc_verify.json` 里 `COPY+DERIVE fraction = 1.162` 是**错误口径**（DERIVE 按 teacher 原始 shape 计数、未扣 o_proj 切片）。本轮已用 `ts2sc_final.py::fixed_accounting()` 修正，并以 `663,544,832` 的精确吻合交叉验证。

### B.3 真实完成 ③：拿到**真实权重数值**并做逐位复制

从 `Qwen3-1.7B-Instruct/qa_00` 经 git-blob API 下载 **99,614,720 B**（19.4 s），解析出真实 safetensors header（35,648 B，**310 张量**，`format=pt`），第一个张量：

```
model.embed_tokens.weight   BF16   [151936, 2048]   offsets [0, 622329856]
```

读取前 **256 × 2048** 个真实 BF16 值（1,048,576 B），做 TS2SC 通道①的原子操作：

| 检查 | 结果 |
|---|---|
| 逐位复制 `student == teacher` | ✅ True |
| BF16→FP32 视图一致 | ✅ True |
| mean / std | −1.7e−05 / 0.03274 |
| zero fraction / all finite | 0.0 / ✅ |
| 是否真实训练权重（非全零、非随机噪声） | ✅ |

**只取前 2 MB 前缀，完整 4.06 GB 权重没有加载；权重字节不推仓。**

### B.4 真实完成 ④：在**真实 teacher embedding** 上验证 SimplCore 架构

用上面 256 行真实 Qwen3 embedding 的前 64 行 × 8 维，跑静态 LTI 谱卷积（ρ=[0.90,0.99], w=[0.5,0.5]）：

| 判据 | 结果 |
|---|---|
| FFT ↔ IIR 双形式相对误差 | **4.38e−16** ✅ |
| 过去扰动对前 L−1 步输出的偏差 | **0.0**（严格因果）✅ |
| 状态字节 @ L=16/64/256/4096 | **恒 64 B** ✅ |
| 核随输入变化 | 否（ρ、w 与 t、与 U 均无关）✅ |
| N×N 打分 / KV cache | 无 ✅ |

> 这是**架构验证**，不是语言建模结果：没有 logits、没有 loss、没有 ppl。

### B.5 明确**没做**（不包装成成功）

- ❌ Qwen3-0.6B 权重本体加载（仓库里没有，HF 403）
- ❌ 端到端迁移落盘、真实激活缓存、mixer 逐层蒸馏、logit-KL 训练
- ❌ 任何 val ppl / 生成可读性 / 能力测试
- ❌ 「迁移后保住了预训练能力」这类结论 —— README 里的 `FFN 余弦 1.0000` / `mixer 0.1024→0.6659` 是**历史在别的机器跑的**，本轮没有复现

**在有 GPU + 网络的机器上接着跑**（命令来自 TS2SC README，本轮未执行）：

```bash
pip install torch safetensors transformers
python scripts/migrate_real.py --teacher-path ./Qwen3-0.6B   # plan / migrate / verify
python scripts/real_act_distill.py cache
python scripts/real_act_distill.py train                    # RA_LAYER=0..27
python scripts/real_act_distill.py eval && python scripts/test_capability.py
```

---

## C. 架构深化：单层 → 两层内容链（中间态，真实结果）

上一轮 `H4_multihop` 的「二跳」是 `q=k_j ⇒ v_{j+1}`，答案由**对的固定顺序**决定，静态延迟线就能解 —— 用它宣称「验证 NT11 层下界」是错的。本轮换成**真正的内容链**：

```
链 k0→k1→k2→k3，pair 位置随机打乱，q=k0，H 跳答案 k_H
H=2 必须以 k1 为新 query 再做一次内容匹配
```

**两层构造**（`chain_stack_probe.py`，纯 numpy，完全自研）：

```
A_t = Code[x_{t-1}]      固定谱字典；b_t = E[x_t] Wv
Layer1: z1_n = Σ_t K1(L-1-t)·<q1,A_t>·b_t[n]     q1 = Code[k0]
层间:   q2 = z1 Wq2                              ← 输入相关的是 query 信号，不是核
Layer2: z2_n = Σ_t K2(L-1-t)·<q2,A_t>·b_t[n]     K_i(d)=Σ_p w_ip ρ_ip^d > 0
```

保持：每层核静态 Toeplitz、FFT/IIR 对偶、严格因果、状态 `2·P·R·N` 与 L 无关、无 N×N、无 KV cache。守恒 + gradcheck **8/8 全过**。

**训练结果**（合成内容链 bind，held-out 512 × 5 seeds，1000 步，参数全部 4,605）：

| 臂 | acc | 读法 |
|---|---:|---|
| h1_single（1 跳） | **1.0000** | 单层够 |
| single（2 跳） | 0.0312 ≈ chance | **NT11/NT28 层下界的实证** |
| **stack2（2 层 2 跳）** | **0.5918 ± 0.0211** | 配对差 +0.5605，CI [0.5457, 0.5758]，5 seed 全正 |
| stack_h3（2 层 3 跳） | 0.2836 | 高于 chance，但明显掉点 |
| single_h3 | 0.0332 ≈ chance | |
| delay（静态延迟线，无第二跳匹配） | 0.2477 | 位置捷径不够 |
| stack_noq2（掐断内容链） | 0.0266 ≈ chance | 第二次匹配是必要的 |
| stack_rand（随机非正交码） | 0.6016 | 与 NT08 一致：**正交码不是必需** |

**诚实边界**：0.5918 **不是 1.0**；3 跳掉到 0.2836；随机码也行说明 NT04 的零串扰不是这里的瓶颈；3000 步全臂、堆叠到满分、多层 recurrence、语言建模都没做。这是「层下界可达」的**部分构造证据**，不是 NT31 定理。

---

## D. 一句话

**论文乱码已彻底修好并重建；TS2SC 在真实 Qwen3 权重字节上验证了 Stage-0 逐位复制、参数对账与 SGC 架构守恒，但 Qwen3-0.6B 权重本体不在可访问仓库、torch 装不上，所以端到端蒸馏没跑、也没有 ppl；多层内容链从 chance 拉到 0.59，是真实中间态而非完成态。**

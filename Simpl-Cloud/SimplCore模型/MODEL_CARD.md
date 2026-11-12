# SimplCore 模型本体 · 改动后的完整架构

**这一包就是模型**：`model.npz` 是端到端训练出来的权重，`nn.py` + `simplcore_lm.py` 是完整架构，`stream_lm.py` 是 O(1) 状态推理，`train.py` 是真实语料训练入口。纯 NumPy，不需要 torch、网络、云端仓库；scipy 缺失时自动回退到 `math.erf`（实测数值差 0）。

## 一、三条命令

```powershell
python chat.py "Q: Who goes there?`nA:" --n 200     # 全展开推理
python stream_chat.py "HAMLET:`n" --n 200            # O(1) 状态流式推理（推荐）
python verify_final.py                               # 10 项白盒验收
python train.py --eval-only                          # 评估随包权重
python train.py --steps 600 --resume                 # 从 4200 步继续真实训练
python train.py --steps 600 --fresh                  # 从头训一个新模型
```

注：PowerShell 里 `\n` 要写成 `` `n ``；cmd 里用 `\n` 字面量即可。

## 二、权重

| 项 | 值 |
|---|---|
| 文件 | `model.npz`，1,559,564 B，sha256 `ac3838e789569ae93a4a…` |
| 训练步数 | **4200**（Adam，真实 tiny Shakespeare 语料 + 剧本问答模板，仅训练侧） |
| 参数 | **202,036** / 54 组，全部进入前向图 |
| 验证集 bpc | **2.453225 ± 0.067720**（`train.py --eval-only` 现测） |
| 训练轨迹 | 10.0000 → 3.3630(300) → 2.7184(1500) → 2.4622(4200) |
| `self_attn` / `q_proj` 等 Transformer 注意力键 | **0 个** |

先前对外发布的权重是 **1500 步**（val_bpc 2.7184），已归档为 `原始记录/model_step1500_旧发布版.npz`。本包用的是继续训练到 4200 步的最终权重，不是那个旧文件。

## 三、完整架构（ARCH_STEP 8）

3 层 decoder-only，pre-LN 残差，每层并行/融合四路：

| 通道 | 实现 | 状态 |
|---|---|---|
| 局部内容寻址 | 固定窗口 `W=12` 的因果 q·k 打分 + 位置偏置，环形缓冲 | `O(W·d)` |
| 全局静态核（SGC） | 3 极点 Toeplitz `ρ∈{0.6,0.9,0.99}`，可学混合权重，FFT↔IIR 等价 | `O(K·d)` |
| 门控自进化状态 | 输入依赖写入门 `β_t`，`S_t=λS_{t-1}+β_t k_t v_tᵀ`，读出 `q_t S_t/√d` | `O(d²)` |
| 融合与前馈 | 可学 `α/γ` 融合 → `W_o` → GELU FFN | — |

- **无 softmax 全局 attention、无 N×N score、无随 L 增长的 KV cache**。
- 训练用矩阵展开，生成用逐 token 递推；两者在**已训练权重**上最大差 **1.24e−14**。
- 饱和后每层状态 **139,776 B / 3 层**，与序列长度 L 无关。
- ARCH_STEP 8 只改了初始化：`alpha/gamma` 从 0.0 改为 0.02，修掉「训练第一步 Toeplitz 与自进化分支共 9 个参数梯度恒为 0」。已训练权重含学后值，加载后与旧 init 无关，参数契约逐键一致。

## 四、为什么这不是「换个文件名的 Transformer」

`verify_final.py` 的 C6 是决定性判据：**只扰动 SGC/自进化参数，logits 必须变化**；同时放一个未接入图的参数做负对照，必须完全不变。

| 扰动对象 | logits 最大变化 |
|---|---:|
| Toeplitz 混合权重 | 0.0148 |
| 自进化写入门 | 0.1445 |
| 融合门 α/γ | 2.9035 |
| 状态输入 Wk/Wv/Wu | 1.8328 |
| 局部窗口位置偏置 | 0.2613 |
| **未接入参数（负对照）** | **< 1e−12** |

C7 另用全新种子在真实语料上跑 160 步：54/54 参数首步全部拿到非零梯度，loss 下降 1.0664 bpc，4 项解析/数值梯度全过。这正堵死此前「310/310 逐字节相同、`sgc_mixer.coef` 运行时被忽略」那条路。

## 五、验收摘要（`模型自检.json` / `final_arch_verify.json`）

**10/10 ALL_PASS**：Toeplitz 矩阵↔IIR `8.88e−15`、外积↔递推 `0`、窗口↔环形缓冲 `8.88e−16`、全展开↔流式 `1.24e−14`、状态恒定、SGC 扰动、全新种子真训练、9 个提示（含 3 个 held-out）O(1) 生成全非退化（平均真实词率 **0.8793**、rep3 0）、checkpoint 前缀回放 `6.0e−15`。

生成样例（真实输出）：

> *"Why, good marriage, or something do will be lears: / I would be not this the though not for this migh…"*

## 六、包内文件地图

- 根目录 = **模型本体**（权重、架构、两个推理入口、验收、训练入口、语料切片、模型卡）
- `训练/` = 完整 1.1 MB 语料、4200 步原始断点 `step7_ckpt.npz`（含 Adam 状态，可续训）、训练曲线与原始脚本归档
- `生产链路_1.7B/` = 同一架构的 torch/HF 实现 `modeling_simplcore_v7.py`、逐张量迁移 `migrate_qwen3.py`、训练与 `verify_arch.py`、Qwen3 BPE 词表、该链路的 NumPy 参考权重 `student.npz`
- `架构演进/` = STEP0–6 的真实训练脚本与结果（负结果完整保留）
- `原始记录/` = 旧 1500 步权重与各阶段报告
- `MANIFEST.json` = 逐文件 size + sha256

## 七、边界

1. 随包**带权重的模型是字符级 202k 参数**，在 1.1 MB 英文剧本上训练，生成连贯但含伪词的莎剧体英文；不是 1B 级中文助手，不保证事实正确，不与预训练 Transformer 做能力对比。
2. `生产链路_1.7B/` 是**同一架构的 torch 实现 + 迁移脚本**，本环境无 torch，未在此生成 1.7B 权重；需要在你机器上按 `README_V7_PROD.md` 执行 `run_all.ps1`。其能力继承自 teacher 的 embedding/FFN/norm，验证的是架构兼容性与可训练性，不声称更强。
3. 多模态、身份认同、≥10M token LM、标准 MQAR benchmark parity、吞吐、独立真人评审仍缺；本包不含论文，不作录用概率或 SOTA 主张。

# SimplCore 全架构最终版（ARCH_STEP 8）

解压即用，纯 NumPy 链路**不需要 torch、不需要网络、不需要云端仓库**。

## 一句话

这是一个**已经在真实文本上端到端训练过、会说话、并且用 O(1) 状态流式生成**的 SimplCore 语言模型；同时附一条 Qwen3-1.7B-Instruct 的生产迁移链路，以及两层静态/数值白盒验收。

## 包里有什么，两套链路的关系

**A. 已实测的参考实现（沙盒内真跑）**

| 文件 | 作用 |
|---|---|
| `nn.py` | 纯 NumPy fp64 tape 自动微分，每个算子手写解析 adjoint |
| `simplcore_lm.py` | SimplCore LM，ARCH_STEP 8，3 层 decoder-only |
| `model.npz` | 1500 步真实语料训练出的权重，**不是任何 teacher 的副本** |
| `train_record.json` | 训练曲线（val bpc 3.363 → 2.718）、54/54 参数组在图内 |
| `stream_lm.py` | **本轮新增**：同一权重的 O(1) 状态流式推理 |
| `stream_chat.py` | 独立聊天入口，逐 token 不再重算整个前缀 |
| `chat.py` | 旧的全展开聊天入口，保留作对照 |
| `verify_final.py` / `final_arch_verify.json` | 10 项白盒验收与原始结果 |

架构：固定窗口内容寻址 + 静态多极点 Toeplitz（SGC）通道 + 输入门控 O(d²) 外积自进化状态 + 门控融合 + SwiGLU 前馈 + pre-LN 残差。无随 L 增长的 KV cache，无全局 N×N score。

**B. 1.7B 生产链路（在你机器上执行）**

`modeling_simplcore_v7.py` / `migrate_qwen3.py` / `train_simplcore.py` / `verify_arch.py` / `run_all.ps1`。

它**不是** A 的逐算子复制，而是尺度适配：d=64 的单个 O(d²) 状态在 d=2048 会被分槽成 R 个 (d/R)² 块，固定窗口 softmax 换成有界因果卷积以省带宽，Toeplitz 用 IIR 递推（与 A 的矩阵形式严格等价，见 C1）。两条链路共享的**不变量**是：静态核、门控外积状态、有界局部通道、常数大小状态、无 KV cache。此前把两者叫"同一架构"是不准确的，这里更正。

`contract_check.py` 用 AST 推导生产模型的 16 个 state_dict key，与迁移脚本实际写出的 15 个逐一比对（差的一个是 `lm_head.weight`，由 HF 的 tie 机制从 embedding 生成），并核对 TRAIN_KEYS / 扰动选择器确实命中参数名。

## 本轮真修掉的 5 处问题

1. **融合门零初始化**：`alpha=gamma=0` 让 Toeplitz 路径（`Wu`、`wlogits`）和自进化路径（`w_beta`）在训练第一步梯度全为 0 —— 每模型 9 个参数名义可训、实际不在图里。改为 0.02 后，fresh-seed 第一步 54/54 参数全部拿到非零梯度。**这正是此前"权重在 state_dict 里但没进前向图"问题的同构版本**。
2. **O(1) 流式推理**：新增 `stream_lm.py`，三个融合算子分别换成环形缓冲 / K 个 IIR 状态 / 一个 d×d 外积状态；与全展开在训练权重上差 7.1e−15。
3. **验收脚本自身两处假失败**：均匀 `+0.05` 扰动被 softmax 平移不变性吸收（`wlogits`、`pos` 的 delta 假报 0），改为非均匀扰动；gradcheck 曾允许 0/0 假通过，现在要求梯度非零。
4. **契约检查器**：原先只看类体、漏掉 `__init__` 里的 `self.X = ...`，expected 恒为 0；改为 `ast.walk` 遍历方法体。
5. **生产链路兼容**：transformers ≥5 已把 `torch_dtype` 改名 `dtype`；`migrate` 里写死的 `transformers_version=4.51.0` 删除；新增 `--teacher-file`，让分片重建后的权重能参与 key-diff 检查；`run_all.ps1` 自动判断并传递。

## 验收（本轮真跑，`final_arch_verify.json`）

| 检查 | 结果 |
|---|---|
| C1 Toeplitz 矩阵 ↔ IIR | 8.88e−15 |
| C2 外积矩阵 ↔ 递推 | 0.0 |
| C3 固定窗口矩阵 ↔ 环形缓冲 | 8.88e−16 |
| C4 全展开 ↔ 流式（L=1…137） | 最差 7.11e−15 |
| C5 状态字节 | 饱和后恒定，上界 139,776 B，与 L 无关 |
| C6 只扰动 SGC/自进化参数 → logits 变化 | 5 组全过，未链接参数 0 |
| C7 全新 seed 真实语料训练 160 步 | bpc 4.987 → 3.921，54/54 参数首步有梯度，gradcheck 4/4 |
| C8 O(1) 生成 9 个提示（3 个 held-out） | 全非退化，平均真实词率 0.8443，rep3 均值 0 |
| C9 前缀重放 | 2.41e−15 |

**10/10 ALL_PASS**；`contract_check.py` 16/15/tied PASS。

## 怎么跑

```powershell
# 纯 NumPy，秒级验证会说话（不需要 torch）
python stream_chat.py "Q: Who goes there?\nA:" --n 160
python stream_chat.py                     # 6 个内置提示

# 全部白盒验收（约 20 秒）
python verify_final.py --steps 160 --seed 23
python contract_check.py

# 1.7B 生产闭环：真迁移 → 真实语料训练 → 硬验收
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 `
  -Teacher G:\Gpan\model -Tokenizer G:\Gpan\qwen3_tokenizer `
  -Data G:\Gpan\corpus.txt -Out G:\Gpan\student_v8 -Steps 120

# 推仓
powershell -ExecutionPolicy Bypass -File .\push_final_arch.ps1
```

`verify_arch.py` 的硬判据是 `delta_sgc`：只扰动 SGC/局部路径参数，logits **必须**变化；同时要求无 `self_attn` key、学生与教师 key 集不同、流式等价、生成非退化。这条专门堵死"310/310 逐字节相同、`sgc_mixer.coef` 被忽略"那种假迁移。

## 边界（如实）

- A 是 **202k 参数、字符级、1.1 MB 真实英文剧本语料、CPU 训练**的模型；生成的是连贯但含伪词的莎剧体英文，不是 1B 级中文助手，不保证事实正确，不与预训练 Transformer 做能力对比。
- `train_record.json` 记录的是 ARCH_STEP 7 的那次 1500 步训练；当前代码是 ARCH_STEP 8（修了初始化、加了流式）。权重逐参数加载，行为完全一致，C4/C9 已验证。
- B 需要 torch + transformers + safetensors，**沙盒没有这些库，1.7B 链路未在此执行**；它能被交付的前提是 AST 契约检查 + A 的算子级白盒 + 你机器上的 `run_all.ps1`。不声称已在 1.7B 上跑通。
- 能力继承 teacher，不声称"更强"；不承诺任何录用概率、SOTA 或 LM parity。

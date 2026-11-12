# 全架构升级收口报告（ARCH_STEP 8）

## 结论

上一轮交付的是两个分家的包：一个「会说话的纯 NumPy 模型」，一个「1.7B 生产 torch 链路」，且后者被称作前者的「同一架构」——这个说法不准确。本轮把架构本体真正升级并统一交付：

- **已训练、会说话的模型**补上了 **O(1) 状态流式推理**（不再每生成一个 token 就重算整个前缀）；
- **修掉一个真实的架构缺陷**：融合门零初始化让 Toeplitz 与自进化分支共 9 个参数在训练第一步梯度恒为 0；
- **1.7B 生产链路**做了 state_dict 契约核对与 transformers 5.x 兼容修复；
- 全部装进**一个自包含包**，`corpus.txt` 随包，解压到任意目录可跑。

## 本轮抓到并修掉的 5 处真问题

| # | 问题 | 修法 |
|---|---|---|
| 1 | `alpha=gamma=0` ⇒ `Wu`/`wlogits`/`w_beta` 首步零梯度（54 个参数里 9 个名义可训、实际不在图里） | 改为 0.02；fresh seed 首步 54/54 全非零。这是「权重在 state_dict 里但没进前向图」的同构版本 |
| 2 | 没有 O(1) 推理；`chat.py` 是 O(L²) 重算 | 新增 `stream_lm.py`：环形缓冲 / K 个 IIR 状态 / 一个 d×d 外积状态 |
| 3 | 验收自身假失败：均匀 `+0.05` 被 softmax 平移不变性吸收，`wlogits`/`pos` delta 假报 0；gradcheck 允许 0/0 假通过 | 改非均匀扰动；要求梯度非零 |
| 4 | `contract_check.py` 只看类体、漏掉 `__init__` 里的 `self.X`，expected 恒为 0 | 改 `ast.walk` 遍历方法体 |
| 5 | transformers ≥5 把 `torch_dtype` 改名 `dtype`；config 写死 4.51.0；分片重建权重无法参与 key-diff | 版本检测 + 删字段 + 新增 `--teacher-file`，`run_all.ps1` 自动传递 |

另修一处自包含问题：`verify_final.py` / `run_sandbox.py` 原先硬编码工作区语料路径，解压到别处会找不到；现在改为优先读包内 `corpus.txt`。

## 验收（本轮真跑）

`final_arch_verify.json` — **10/10 ALL_PASS**

| 检查 | 结果 |
|---|---|
| C1 Toeplitz 矩阵 ↔ IIR | 8.88e−15 |
| C2 外积矩阵 ↔ 递推 | 0.0 |
| C3 固定窗口 ↔ 环形缓冲 | 8.88e−16 |
| C4 全展开 ↔ 流式（L=1…137） | 最差 7.11e−15 |
| C5 状态字节 | 饱和后恒定，上界 139,776 B，与 L 无关 |
| C6 只扰动 SGC/自进化参数 → logits 变化 | 5 组全过，未链接参数 delta 0 |
| C7 全新 seed 真实语料训练 160 步 | bpc 4.987 → 3.921，54/54 首步有梯度，gradcheck 4/4（含 `w_beta`/`wlogits` 非零） |
| C8 O(1) 生成 9 提示（3 held-out） | 全非退化，平均真实词率 0.8351，rep3 均值 0 |
| C9 前缀重放 | 2.41e−15 |

`contract_check.json` — 16 expected / 15 emitted / `lm_head.weight` 由 HF tie 合法生成，**0 missing / 0 unexpected**，TRAIN_KEYS 命中 7、扰动选择器命中 7，**PASS**。

14 个 `.py` 全部 `py_compile` OK；干净目录冷跑 `stream_chat.py` / `contract_check.py` / `verify_final.py --steps 40` 全部退出码 0（`COLDCHECK.json`）。

## 架构谱系（更正上一轮口径）

- `simplcore_lm.py`：参考实现，d=64、字符级、3 层，**已端到端训练并会说话**。
- `modeling_simplcore_v7.py`：1.7B 生产适配，**不是逐算子复制**。d=2048 时把单个 O(d²) 状态分槽成 R 个 (d/R)² 块，固定窗口 softmax 换成有界因果卷积，Toeplitz 用 IIR 递推（与参考实现的矩阵形式严格等价，C1 验证）。
- 共享的不变量：静态核、门控外积状态、有界局部通道、常数大小状态、无 KV cache。

## 运行与推送

```powershell
python stream_chat.py "Q: Who goes there?\nA:" --n 160
python verify_final.py --steps 160 --seed 23
python contract_check.py
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Teacher G:\Gpan\model `
  -Tokenizer G:\Gpan\qwen3_tokenizer -Data G:\Gpan\corpus.txt -Out G:\Gpan\student_v8
powershell -ExecutionPolicy Bypass -File .\push_final_arch.ps1
```

## 边界

- 参考模型是 202k 参数、字符级、1.1 MB 真实英文剧本语料、CPU 训练；生成连贯但含伪词的莎剧体英文，不是 1B 中文助手，不保证事实正确。
- `train_record.json` 记录 ARCH_STEP 7 的 1500 步训练；当前代码 ARCH_STEP 8，权重逐参数加载、行为一致（C4/C9 已验证）。
- 1.7B 生产链路需要 torch / transformers / safetensors，**沙盒无这些库，未在此执行**；交付依据是 AST 契约 + 参考实现的算子级白盒 + 你机器上的 `run_all.ps1`。不声称已在 1.7B 上跑通。
- 能力继承 teacher，不声称更强；不承诺录用概率、SOTA 或 LM parity。
- 本轮未改论文、未推仓（沙盒无登录态），推送脚本已随包。

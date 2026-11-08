# TS2SC → MiniCPM-V 4.7：前置盘点基线（本轮唯一产出）

状态：**PREFLIGHT ONLY**。用户要求的三步（① 收紧全部实验 ② TS2SC 迁移 MiniCPM-V-4.7-1B ③ 改论文）本轮**一项未完成**，执行被中断。本文只记录已核实的事实，不含推测、不含伪造结果、不含凭据。

日期：2026-11-07 · 沙盒：`/data/workspace`

---

## 1. 设备真实约束（实测，不是估计）

| 项 | 实测 |
|---|---|
| CPU | 2 × x86_64 |
| RAM | 2.3 GiB（可用 1.7 GiB） |
| 磁盘 | 500 MiB 总量，`/data/workspace` 可用 **370 MiB** |
| GPU | 无 `nvidia-smi`，无 `/dev/nvidia*` |
| Python | 3.10.12 |
| numpy | 2.2.6 ✓ |
| scipy | ✓ |
| torch / transformers / safetensors | **全部 MISSING** |
| 网络 | `api.github.com` 200；`huggingface.co` 403；`pypi.org` 403 |

**结论**：3.11 GB 的完整权重**不可能落盘**，也装不上 torch。唯一可行路径是纯 NumPy + GitHub REST 流式分片读取（按需取张量切片，虚拟文件，不物化模型）。这正是「对设备比较友好」的真实含义，不是能全量跑训练。

目录中已存在更早一轮留下的 `speak_test.py` / `speak_staged.py`（纯 NumPy 分阶段前向），以及 `speak_smoke_results.json`：它在 **MiniCPM5-1B**（24 层 / hidden 1536 / vocab 130560）上流式取了 1,463,987,872 B、15 次 API 调用、155.2 s、模型未落盘，跑完一次 full-depth forward。但 top-k 输出为乱码（`;++` / `VILLE` / `nullptr`），**不能当作有效的语言能力验证**，只能当作「流式通路可跑」的证据。该探针针对的是 MiniCPM5-1B，**不是**本轮目标 MiniCPM-V-4.7-1B。

## 2. 私有仓库真实清单（已核实存在）

| 仓库 | private | 大小 | 说明 |
|---|---|---:|---|
| `MiniCPM-V-4.7-1B` | ✓ | 3,129 MB | 目标模型，32 个 `vpart_00..31`，重组后应为 3,109,054,480 B |
| `TS2SC-VL-1B` | ✓ | 2,294 MB | 历史迁移包，111 分片 / 1017 张量 |
| `MiniCPM5-1B` | ✓ | 1,677 MB | 另一模型，非本轮目标 |
| `Qwen3-1.7B-Instruct` | ✓ | 3,146 MB | 历史 teacher |
| `TS2SC` | ✗ | 105 KB | 协议与脚本 |

用户写的「MiniCMP-V 4.7」= 真实存在的 `MiniCPM-V-4.7-1B`。

## 3. 目标模型真实结构（源 `config.json`）

```
architectures : MiniCPMV4_7ForConditionalGeneration
dtype         : bfloat16
text_config   : model_type qwen3_5_text, 24 layers, hidden 1024,
                vocab 248094, head_dim 256,
                linear_attention ×18 / full_attention ×6 (interval 4),
                linear conv kernel 4, 16 key heads / 16 value heads,
                mrope (canvas), max_position_embeddings 262144
vision_config : 27 layers, hidden 1152, image 980, patch 14
special ids   : image 248056, video 248057, slice_start 248088 ...
```

即：**源模型本身已是 3:1 的 linear/full attention 混合体**，不是纯 Transformer。TS2SC 要替换的是那 6 层 full attention 与 18 层 linear attention 的读出方式。

## 4. 历史迁移包 `TS2SC-VL-1B` 的真实口径 —— 最关键的发现

其 `config.json` 明写：

- 语言侧来源：`Qwen/Qwen3-0.6B (FFN/emb/norm verbatim, attention replaced)`，**vocab 151936、28 层、intermediate 3072**
- 视觉侧来源：`MiniCPM-V-4.7-1B model.vision_tower.* + model.merger.*`

> **历史包并不是把 MiniCPM-V 4.7 的语言侧迁到 SimplCore。** 它把 Qwen3-0.6B 的语言侧换核，再拼上 MiniCPM-V 4.7 的视觉塔。所以「MiniCPM-V 4.7 的架构问题一直没成功」这句话，在历史包里**从未被真正测过**——被测的是 Qwen3-0.6B 的语言侧。

自带 `known_limitations` 也承认：`vocab 248094 vs 151936`，图像特殊 token id 越界；mrope/canvas 与标准 RoPE 未统一；无跨模态对齐训练；**是权重级合并，不是可用 VQA 模型**。

## 5. 历史失败的四条硬证据（原始 JSON 已抓取）

| 证据 | 数值 |
|---|---|
| 静态平均注意力核表达力上限 | mean cos **0.3695**（pairwise 内容寻址硬缺口） |
| 读出对齐 W | holdout cos +0.0826，生成退化成「这这这…」rep3 0.8125 → 拒 |
| v3 解析迁移 | 注入 cos 0.5032，`otionalotional…` rep3 0.9286 → 拒 |
| DC-bias 零训练修复 | rep3 0.6429 → 0.2143，但**不修事实错误、不修问答行为** |
| 181 步 next-token 微调 | holdout CE 4.810→4.309（−10.4%），但 rep3 **0.1719→0.2344 变差**、中文占比下降、事实正确数持平 → 训练权重不并入 |

生成复核原文：「中国的首都是」→「城市，有，北京，北京，北京，北京，北京，北京，」；知识仍在（FFN 逐位迁移 cos 1.0000），但循环与语义错乱。

## 6. 新架构与真实权重之间的**尚未接通**的接口

后续理论工作（张量阶 2、码本分区、H 层恰好 H 跳）在**合成内容链**上补上了 pairwise 内容寻址缺口，且保持 Toeplitz / FFT–IIR 对偶 / O(1) 状态。但：

1. Theorem A 要求输入侧提供 `B ≥ log₂|A(v)|` bit 的预算；真实 tokenizer 的 token id **不自带 KEY/VALUE 分区码**，也不自带角色位。把码本分区接到 248094 词表上、且不把答案烘进字母表，这一步**从未在真实权重上做过**。
2. 定理的紧性目前只在显式 binary-chain 构造族上成立；论文 §4.3 已自承原 content-chain 任务上 `|A|≤5`、`log₂|A|≤2.3219`，**界是松的**。
3. 因此「架构基本修好了 → 直接 TS2SC 迁移就能成功」不是已证命题。能先做的是**权重级 Stage-0 provenance + 流式 SGC 对偶验证**；teacher forward / logit-KL / ppl / 生成质量在本设备上仍缺 torch 与算力。

## 7. 本轮未完成清单（不包装）

- [ ] ① 全部实验收到最紧（上轮 v7.2 的 14/14 属更早版本；本轮未重跑）
- [ ] ② TS2SC 迁移 `MiniCPM-V-4.7-1B` 到 SimplCore（**零字节 teacher 权重在本轮被读取**）
- [ ] ③ 论文按本轮结论改稿
- [ ] 推送（本轮未推，远端 main 仍为 `1f6c377`）
- [ ] 凭据 revoke（token 已在对话中明文出现多次，临时文件 `.gh_probe` / `.gh_token` 已删除，环境变量为空）

## 8. 下一步唯一可行的最小动作

复用已抓到的 `SPLIT_RESTORE.md` / 源 config / 历史 header，用纯 NumPy 虚拟文件按 `vpart_*` 分片的 safetensors header 定位张量，**只取 needed slices** 做 Stage-0：逐张量 COPY / DERIVE / NEW 判定 + 在真实 teacher 行上跑 SGC 的 Toeplitz / FFT–IIR / 因果 / O(1) 断言。不落盘、不蒸馏、不声称能力。若这一步通过，再谈语言侧词表 248094 的码本接口设计。

**本文不含任何凭据、不含私有权重字节、不含未实测的性能数字。**

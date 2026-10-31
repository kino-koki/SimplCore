# 青小团 Tuan-1B 训练方案

> **状态注（2026-10-31）**：本文是 1B 先导方案（数据配比 + 流程 + 文献）。
> 当前主力执行计划已升级为 **Tuan-2B × 30 天 × 单 MI250**（`TRAINING_2B_MI250_30D.md`，
> 2026-10-31 启动）；本文的数据配比（`configs/tuan-1b-datamix.json`）与训练流程
> 仍然有效，作为 2B 战役的数据侧输入与先导参考，训练执行口径以 2B 文档为准。
>
> 数据配比 + 全套训练流程 + 文献 + 训练资料下载
> 模型家族：青小团 / Tuan（SGC 非注意力主干，无位置编码、无 KV cache、无 max_len）
> 配套机器可读配比：`configs/tuan-1b-datamix.json`

---

## 0. 一句话定位

**中文为主、英文保底、代码/Computer Use 打底**的 1B 基座，叠加一个 **≤2B 代码+Computer Use 适配器**（冻结基座、只训适配器、在真实电脑环境里自进化），以 **Computer Use 真实环境代码自进化** 与 **SGC 原生超长上下文** 为两大卖点；知识库训练侧接实时联网管线、推理侧靠 Computer Use 直接联网补齐实时性。

---

## 1. 模型家族与卖点

| 组件 | 参数 | 训练方式 | 卖点对接 |
| --- | --- | --- | --- |
| **Tuan-1B 基座** | ~1B（冻结） | 中文为主通用预训练 | 中文主模型、干净底座 |
| **Tuan-Code 适配器** | ≤2B（可学） | envlearn 冻结基座只训 adapter | 全代码 + Computer Use 能力 |
| **超长上下文** | — | SGC 架构原生 | 无 max_len、O(1) 状态、FFT 训练 O(L log L) |
| **知识库实时联网** | — | 训练侧增量爬虫 + 推理侧联网浏览 | 知识不过期 |

与 `weight_budget.py` 一致：**1B 冻结基座 + ≤2B 可学，3B 总上限**。2B 适配器不是另起炉灶，是在 WeightBudget 内的合法可学预算。

**为什么这种架构适合 Computer Use / 自进化：**
- SGC 无 KV cache、状态 O(1)，长程 agent 轨迹（数百步工具调用）不爆显存、不丢历史。
- 六大控制器（accel / nonlinear / reasoning / envlearn / smm / dream）已通过 `train.py --enhance` 接入训练闭环（见 `simplcore/core/training_integration.py`），可直接作为 2B 适配器的**自进化 harness**：dream 巩固（no-regression gate 逐位回滚）、accel 自适应深度、reasoning 迭代复盘。

---

## 2. 1B 基座数据混合配比（核心）

比例与总量无关，可直接套用任意 token 预算。单 MI250 实际先跑 **~300B** 阶段，再决定扩节点到 1T。

| # | 数据域 | 占比 | 语言 | 角色 / 设计理由 |
| --- | --- | --- | --- | --- |
| 1 | 中文通用网页/书籍 | **30%** | zh | 最大单一来源，落实「中文为主」 |
| 2 | 英文通用网页/书籍 | **20%** | en | 「英文保底」——编程/Computer Use 的语言底座，**不能太少** |
| 3 | 代码（多语言，偏 Python） | **15%** | multi | Computer Use 的代码基础，先于 2B 适配器打底 |
| 4 | 数学与推理 | 8% | multi | 提升 agentic 规划/复盘质量 |
| 5 | 知识百科（可联网） | 7% | zh+en | 事实底座 + 实时联网管线（见 §2.3） |
| 6 | 长上下文专项 | 8% | multi | 整本书/长仓库/长文档，喂出超长上下文卖点 |
| 7 | 指令/对话弱监督 | 7% | zh+en | 轻量对齐种子，不喧宾夺主 |
| 8 | Computer Use 轨迹种子 | 5% | multi | GUI/agentic 轨迹文本化，为 2B 适配器预热 |
| | **合计** | **100%** | | |

**设计逻辑：**
- 中文 30% 为最大块但不垄断——英文 20% + 代码 15% + 数学 8% = 43% 的「非中文能力」撑起 Computer Use 所需的英文文档、代码、报错信息、API 文档。英文太少会直接拖垮真实环境编程。
- 代码 15% 放进基座而非全压给 2B：让基座本身就懂语法，2B 只需专精「操作电脑 + 多步 agent 轨迹」。
- 长上下文 8% 与超长上下文卖点对应，配合 §4 的课程学习。

### 2.3 知识库实时联网（双层）

- **训练侧（保底静态 + 增量实时）**：常驻爬虫 + 质量过滤器，把新鲜网页 tokenize 后**增量追加**到 `knowledge_live` 切片，带去重与时间戳，避免概念漂移。静态 Wikipedia/Wikidata 打底。
- **推理侧（实时补齐）**：靠 2B 适配器的 **Computer Use + tool 能力直接联网浏览**（RAG-over-internet）——知识库在推理时就是实时互联网，由 agent harness 拉取并注入上下文。训练侧保静态知识，推理侧补实时性，两者互补。

---

## 3. 2B 代码 / Computer Use 适配器（envlearn + 自进化）

冻结 1B 基座，只训 `LearnedAdapter`（WeightBudget 守 3B 上限）。数据 **全代码 + Computer Use**：

| # | 数据域 | 占比 | 角色 |
| --- | --- | --- | --- |
| 1 | 多语言代码 | **45%** | 代码主体（The Stack v2 / StarCoder） |
| 2 | Computer Use 轨迹 | **25%** | GUI/电脑操作轨迹（UI-TARS / OS-Atlas / Aguvis / InSTA / Explorer） |
| 3 | Agentic 编码轨迹 | **18%** | 真实仓库 agent 轨迹（自进化核心燃料） |
| 4 | 英文底座保底 | 7% | 防遗忘 |
| 5 | 中文底座保底 | 5% | 防遗忘 |
| | **合计** | **100%** | |

### 3.1 真实环境自进化闭环（harness 驱动）

```
沙箱/真实电脑跑任务
   → 收集 (观察, 动作, 结果) 轨迹
   → dream 巩固：更好→提交，退化超容差→逐位回滚（no-regression gate）
   → accel 自适应深度：易样本走浅路径省算力
   → reasoning 迭代复盘：多步思考轨迹
   → 下一轮（模型冻结，程序生长）
```

这与 CMU/JHU 的 **Harness Learning**（arXiv 2609.35738）精神一致：**模型权重冻结，只有「程序/harness」在生长**；区别在于本方案是**手工/规则驱动**的 harness 改造，而非 Proposer 模型 + RL 学出来的。质量是否真因自进化提升，仍待 MI250 战役验证（机制接线 ≠ 质量达标）。

---

## 4. 超长上下文策略（SGC 原生卖点 · 已落地）

SGC 架构无 max_len、状态 O(1)，超长上下文是「免费」的能力，只需在训练里喂够长样本。
**本课程已写入 `simplcore/core/seqlen_curriculum.py` 并由 `train.py` 接入（opt-in、零回归）**：
预置 `context` 的预设（`tuan-1b` / `tuan-870m` / `tuan-1_3b` / `tuan-3b`）自动启用；
也可在任意预设上用 `--seq-len-curriculum 4096:0.1,16384:0.25,...,4000000` 覆盖。
课程切换**不改模型权重**（SGC 长度无关），只拉长数据窗口并按比例调整梯度累积步数。

> ⚠️ 4M 阶段仅受显存约束：需 `grad-checkpoint` + `micro-batch=1`，且 `--cuda-graph`
> 在课程激活时自动禁用（固定形状假设不再成立）。单 MI250 128GB 跑 4M 需实测确认可行性。

| 阶段 | seq_len | 进度 | 配比调整 |
| --- | --- | --- | --- |
| 1 | 4K | 0–10% | 基础配比 |
| 2 | 16K | 10–25% | long_context 权重 ×1.5 |
| 3 | 64K | 25–45% | long_context 权重 ×2.0，加入整本中文书 |
| 4 | 256K | 45–65% | 整仓库/长文档，验证 O(1) 状态外推 |
| 5 | 1M | 65–85% | 百万级上下文，长程依赖专项 |
| 6 | 4M | 85–100% | 4M 总上下文收敛（卖点）；显存受限时此阶段需 grad-checkpoint + micro_batch=1 |

配合 `train.py` 的 `--global-batch-tokens`（自动算梯度累积，课程切换时按比例重算），
各长序列阶段用梯度检查点 + 序列并行（未来）控显存。

---

## 5. 全套训练流程（分阶段 + 命令）

```
阶段 0  数据准备
   ├─ 下载各数据集（见 §7）
   ├─ 按 configs/tuan-1b-datamix.json 的 base_mix 配比 tokenize + 分片
   └─ scripts/build_corpus.py（已有）+ 配比 spec 驱动混比
阶段 1  1B 基座预训练（MI250）
   └─ scripts/train_tuan_mi250.sh --preset tuan-1b
阶段 2  长上下文课程（§4 的 6 阶段 seq_len 爬升到 4M，已落地为 opt-in 课程）
   └─ python -m simplcore.core.train --preset tuan-1b   # 预设 context 自动启用课程
      # 或显式覆盖：--seq-len-curriculum "4096:0.1,16384:0.25,65536:0.45,262144:0.65,1048576:0.85,4000000"
阶段 3  2B 代码适配器（envlearn + 自进化 harness）
   └─ python -m simplcore.core.train --preset tuan-1b \
         --enhance envlearn,accel,reasoning,dream \
         --override dim=...（adapter 预算见 WeightBudget）
阶段 4  轻量对齐（指令 SFT 种子）
   └─ 用 instruction_dialog 切片做薄 SFT
阶段 5  评测 + Computer Use 基准
   └─ OSWorld / WebArena（GUI）、SWE-bench（编码 agent）、长上下文基准
```

**关键 CLI（已接入 `train.py`）：**
```bash
# 默认关闭增强：训练逐位不变（零回归）
python -m simplcore.core.train --preset tuan-1b --dry-run

# 2B 自进化阶段：envlearn 冻结基座只训适配器 + 评估期 harness
python -m simplcore.core.train --preset tuan-1b \
  --enhance envlearn,accel,reasoning,dream
```

---

## 6. 文献

**SGC / 次二次序列模型（架构底座）**
- S4 — *Efficiently Modeling Long Sequences with Structured State Spaces*, Gu et al., arXiv:2111.00396
- Mamba — *Selective State Spaces*, Dao & Gu, arXiv:2312.00752
- Hyena — *Subquadratic Context Scaling*, Poli et al., arXiv:2302.10866
- GLA — *Gated Linear Attention*, Yang et al., arXiv:2312.06635
- RWKV-5/6 — arXiv:2305.13048 ｜ RetNet — arXiv:2307.08621

**Computer Use / GUI Agent（2B 适配器数据源与方法）**
- UI-TARS — *Pioneering Automated GUI Interaction with Native Agents*, ByteDance, arXiv:2501.12326
- Aguvis — *Unified Pure Vision Agents for Autonomous GUI Interaction*, arXiv:2412.04454
- OS-Atlas — *Foundation Action Model for Generalist GUI Agents*, Wu et al., ICLR 2025
- Mind2Web / AndroidControl / GUI-Odyssey / WebArena / OSWorld（基准与轨迹）

**Agentic 编码轨迹（自进化燃料）**
- SWE-bench — arXiv:2308.01867 ｜ SWE-bench-Live — arXiv:2505.23419（自动更新、多语言多 OS）
- SWE-Gym / SWE-smith / Terminal-Bench / τ2-bench
- Tracebench (Contextbench) ｜ CodeTraceBench (NJU-LINK) ｜ OptiQ Code Traces（gold 可验证 agent 轨迹）

**RSI / Harness Learning（自进化参照）**
- Harness Learning — CMU & JHU, arXiv:2609.35738（Proposer 改 Harness + RL 奖励，测试时模型冻结、程序生长）
- 同系列 arXiv:2609.11873
- *The Last AI Built by Humans* — 上海交大 RSI 综述（自主度分级：L2 策略自主）
- *Learning Beyond Gradients* — 翁家翌（把修改策略训进模型，与「把经验写进程序」对照）

**长上下文**
- Ring Attention（块级并行）｜ RoPE/NTK 位置外推 ｜ SGC 原生 O(1) 状态（本项目 `theory.py`）

---

## 7. 训练资料下载（数据集清单 + 脚本骨架）

> 以下为稳定 HuggingFace 数据集 ID；下载前请确认各自许可证（代码类注意 copyleft）。URL 以 `huggingface.co/datasets/<id>` 为准。

| 类别 | 数据集 ID | 用途 |
| --- | --- | --- |
| 中文 | `BAAI/CCI3-HQ` | 中文通用主源 |
| 中文 | `Shanghai_AI_Lab/WanJuan-1.0` | 书籍/网页 |
| 中文 | `Skywork/SkyPile-150B` | 中文网页 |
| 中文 | `map-cc/map-cc` ｜ `ChineseWebText` | 补充 |
| 英文 | `HuggingFaceFW/fineweb-edu` | 英文教育网页（保底） |
| 英文 | `allenai/dolma` ｜ `tiiuae/Falcon-RefinedWeb` | 英文通用 |
| 代码 | `bigcode/the-stack-v2` ｜ `bigcode/starcoderdata` ｜ `codeparrot/github-code` | 代码基础 + 适配器主体 |
| 数学 | `open-web-math/open-web-math` ｜ `DRaiLab/MathPile` ｜ `deepseek-ai/DeepSeek-Math` | 推理 |
| 知识 | `wikipedia` (zh+en) ｜ `wikidata` | 静态知识底座 |
| 长上下文 | `bookcorpus` ｜ `pg19` ｜ `Project Gutenberg` ｜ The Stack v2 长文件 | 超长上下文 |
| 指令 | `teknium/OpenHermes-2.5` ｜ `BAAI/Infinity-Instruct` | 对齐种子 |
| Computer Use | `OS-Copilot/OS-Atlas` ｜ `xlang-ai/aguvis` ｜ `data-for-agents/insta-150k-v3` ｜ UI-TARS 教程 | GUI 轨迹 |
| Agentic 编码 | `Contextbench/Tracebench` ｜ `NJU-LINK/CodeTraceBench` ｜ `mlx-community/optiq-code-traces` ｜ `princeton-nlp/SWE-bench` | 真实环境轨迹（自进化燃料） |

**下载骨架（`scripts/fetch_datamix.py` 雏形）：**
```python
# 伪代码：按 configs/tuan-1b-datamix.json 的 base_mix 拉取并打标
from datasets import load_dataset
import json

spec = json.load(open("configs/tuan-1b-datamix.json"))
for item in spec["base_mix"]:
    for ds in item["datasets"]:
        if ds.startswith("live-crawl"):
            continue  # 实时管线单独跑
        try:
            dst = f"data/raw/{item['id']}/{ds.replace('/', '_')}"
            load_dataset(ds, streaming=True).save_to_disk(dst)
            print(f"[ok] {ds} -> {dst}")
        except Exception as e:
            print(f"[skip] {ds}: {e}")  # 许可证/缺失跳过，不中断
```
配套用 `huggingface-cli download <id> --repo-type dataset --local-dir data/raw/<id>` 拉取大体积数据集。

---

## 8. 诚实边界与风险

- ✅ **机制已设计、配比已落地、harness 已接入**（`--enhance`、WeightBudget、dream/accel/reasoning）。
- ⚠️ **1B 以上规模从未实际训练**；单 MI250 跑 1T 不现实，先 300B 阶段验证混合比例。
- ⚠️ **Computer Use 自进化质量未验证**：轨迹数据多为 7B+/VLM 产生，1B 小模型能否从中自进化出操作电脑能力存疑，需 SWE-bench / OSWorld 实测。
- ⚠️ **多模态缺失**：UI-TARS/OS-Atlas 本质是视觉模型；本方案先把轨迹**文本化**进 1B/2B，纯文本 Computer Use 能力上限待定（视觉版为后续路线）。
- ⚠️ **实时联网推理侧**依赖 Computer Use 落地，而 Computer Use 本身又是最未验证的一环——二者耦合，需优先级最高验证。
- ❌ 不得据此宣称任何 SOTA；比例均为工程经验起点，需用消融实验迭代。

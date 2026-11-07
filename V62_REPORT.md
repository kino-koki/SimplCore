# v6.2：同场训练证据、可复现性审计，以及一条正式撤回

> 用户指令「续」：接着 v6.1 往下补 P1 的两块实验。
> 训练/实验任务全程冻结于「合成小任务」；零 LM 训练、零 ppl、零生成样本、零私有权重字节。

## 0. 本轮最重要的事：一条声明被撤回，不是被修好

上一版 §5 自承写了一处硬伤：

> "the baseline came from a historical record file rather than a same-run re-execution"

评审建议要求补「一组小而公平的对照」。我们补了，结果是**历史数字没复现**：

| 预算（3-seed 均值） | linear attention | 2 层 softmax attention |
|---|---:|---:|
| 400 步 / lr 0.02 | 0.2650 | 0.2428 |
| 400 步 / lr 0.10 | 0.1868 | 0.0762 |
| 1200 步 / lr 0.02 | 0.2819 | 0.2559 |
| 1200 步 / lr 0.10 | 0.1829 | 0.0742 |
| 3000 步 / lr 0.02 | 0.2897 | 0.2663 |
| 3000 步 / lr 0.10 | 0.2227 | 0.0853 |

全部预算区间 **0.0332–0.3125**，围绕「4 对里随机猜一个」的 0.25，没有收敛。
因此历史记录文件里的 **0.9987 attention 基线在独立同场重跑下未被复现**。

 **0.0332–0.3125**，围绕「4 对里随机猜一个」的 0.25，没有收敛。
因此历史记录文件里的 **0.9987 attention 基线在独立同场重跑下未被复现**。

**处理：W12 正式撤回**整条「Fisher p = 0.4998 / 未检测到显著差异 / 1.0000 vs 0.9987」
的 parity 声明。摘要、§5、§7、§8、Broader Impact、附录 F、§4.9 全部同步，
**全文不作任何与 attention 的等价、优越或低劣主张**。

这不是失败掩饰，是把「复现失败」本身变成可审计的方法学记录。所有原始行都发布，
审稿人可以自己证伪这次审计。

## 1. 真正补上的主证据：层数下界经过端到端训练仍成立

七机制共用**一套自包含 autograd、同一个 Adam、400 步、batch 64、lr 0.02、
d = 24、5 seeds、同一批可见 id、同一个外部 query**。参数量逐臂记录，未强制相等。

| hops | arm | params | acc (5 seeds) |
|---:|---|---:|---:|
| 1 | `tsc_h1` | 2304 | **0.9996 ± 0.0008** |
| 1 | `tsc_h2` | 2880 | 0.3805 ± 0.0360 |
| 1 | `linear_attn` | 4032 | 0.2586 ± 0.0178 |
| 1 | `attn2l` | 5184 | 0.2406 ± 0.0079 |
| 2 | `tsc_h1` | 2304 | **0.1055 ± 0.0180** |
| 2 | `tsc_h2` | 2880 | **0.9512 ± 0.0094** |
| 2 | `linear_attn` | 4032 | 0.2625 ± 0.0289 |
| 2 | `attn2l` | 5184 | 0.2223 ± 0.0223 |

H=2 时 `tsc_h2 − tsc_h1` 配对差 **+0.8457**，95% CI **[+0.8281, +0.8598]**，5/5 全正。
即： `tsc_h2 − tsc_h1` 配对差 **+0.8457**，95% CI **[+0.8261, +0.8580]**，5/5 全正。
即：**「H 跳恰需 H 层」不再是零训练解析构造的特解，从随机初始化训练也成立。**
这正是评审要求的「把构造变成可训练层、把 5-seed 从随机初始化学到满分提为主证据」。

## 2. 退化曲线：找到真正的失效机制

5 seeds、H=2、600 步解析梯度训练：

| 条件 | mean acc | 说明 |
|---|---:|---|
| clean | **1.0000** | 基线 |
| tag noise p=.05/.10/.20/.40 | 0.8777 / 0.7797 / 0.6305 / 0.3625 | 单调下降 |
| duplicate keys | 0.9328 | distinctness 前提被量化 |
| dup + 10% noise | 0.7301 | 组合退化 |
| code eps .1→4.0（Gram 非对角 ≤0.476） | **全部 1.0000** | 非正交但满秩，不影响 |
| **rank 16** | **0.5652** | 秩亏损开始掉点 |
| **rank 4** | **0.0781** | 接近 chance 0.0312 |

**结论（写成观察 + Gram 诊断，不新增定理）**：掉点机制是**秩亏损**，不是「失去正交性」。
高维随机单位向量天然近似正交，所以纯旋转扰动无害；一旦码本行张成的空间维度塌缩，
匹配打分向量的可行像空间随之塌缩，内容寻址失效。

「码 C 改成可学习」**没有做** —— 我们只扫描了固定码的非正交度与秩。报告与论文都明写
这是开放项，不冒充 learned-code 结果。

## 3. 纯粹性与公平性口径

- 六个对照机制放在**独立评测目录** `tools/eval_baselines/`，用自带的 `ag.py`。
  SimplCore 架构代码一行未改，活跃模型代码仍**零 N×N、零 KV cache、零对照实现**。
- 它们是**极简机制参考实现**（linear attention / gated linear attention / one-step
  delta-rule surrogate / selective diagonal SSM / 2 层 self+cross softmax attention），
  **不是** GLA / DeltaNet / Mamba-2 / Transformer 的完整复现，论文里逐处标明。
- 因为它们未收敛，**同场表不能被读成性能排名**。§5 明确写「reproduction audit,
  not a performance ranking」，§7 明确写「NOT reproductions / NOT a performance ranking」。
- 参数量 2304–5760，明文写「recorded, NOT forced equal」，不假装严格等参数。

## 4. 论文 v6.2

`simplcore.pdf` **25 页，主文恰好 8 页**（靠压缩 §4.8 的十行审计表腾出空间，无注水）：

- 新增 **§4.13**：统一协议、8 行同场表、训练版层数下界、退化曲线、W12 撤回段
- §5 第一条从「Not equivalence」改为「**Parity claim WITHDRAWN (W12)**」
- §5 第二条、§7「刻意未做」段、§4.9 尾句、摘要、§8 结论、Broader Impact 同步
- 附录 F 补 **W10/W11/W12**；顺带修掉 W9 行里残留的
  「R3 proves the single remaining bit NECESSARY and MINIMAL」——它与 v6.1 的
  Theorem A(b) 总 bit 下界冲突，是上一轮的遗漏
- 章节编号 3.1–3.10 / 4.1–4.13 / 5–8 **无重复**；主文内部编号
  `W9 / NT31 / 17/17 / EXACTLY`、自识别路径、实体与 TeX 碎片残留 **全 0**
- Broader Impact(p9) < References(p10) < Appendix，顺序合规

## 5. 校验

| 项 | 结果 |
|---|---|
| 同场协议完整性 | **7/7** |
| Theorem A 验证器 | **18/18** |
| Theorem B witness | **130/130** |
| 退化方向观察 | tag 噪声单调 ✓、码本 eps 单调 ✓、dup 低于 clean ✓ |
| PDF 页数 | 25 页 / 主文 8 页 |
| 旧口径残留 | `historical record` / `Not equivalence` / `same-run Mamba-2 …` 全 **0** |

数值全部由 f-string 从结果 JSON 注入，无手抄。

## 6. 诚实边界（不圆）

- 基线 toy 实现**没有收敛**，因此本节不能用来支持任何架构优劣结论；
  我们把它定位为复现审计，并主动撤回依赖它的旧声明。
- **没有** ≥10M token 同算力 LM、val ppl、生成可读性、实测吞吐对标；
  **没有** GLA / DeltaNet / Mamba-2 / Transformer 的完整复现；**没有**可学习码本。
- 仍是合成内容链、V=64、L≤25、CPU NumPy；不是语言能力结论。
- 不承诺过稿率。本轮改变的是「证据类型 + 一条正式撤回」，不是「已经能打」。
- 上一轮的 P0 / P1 里，唯一仍完全空缺的是**真正公平且收敛的同条件强基线**；
  这需要把基线调到收敛，而不是继续加我们的结果。

## 7. 交付物

| 文件 | 内容 |
|---|---|
| `simplcore.pdf` / `build_simplcore_pdf_v5.py` | v6.2 投稿稿与唯一生成器 |
| `patch_v62.py` / `patch_v62b.py` | 本轮修订脚本 |
| `tools/eval_baselines/ag.py` / `fair_probe.py` | 独立评测 autodiff 与全部实验 |
| `tools/eval_baselines/results/*.json` | 同场对照、预算校准、退化曲线、判定 |
| `V62_REPORT.md` | 本报告 |
| `CHANGELOG.md` / `REVISION_LOG.md` | v6.2 登记 |

私有权重字节与凭据一律未进仓库。

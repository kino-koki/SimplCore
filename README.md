# SimplCore — 无注意力序列架构（v3.0）

以 PyTorch 为主线的**可训练、彻底非注意力**序列建模仓库。架构叫 **SimplCore**，
主推实现位于 `simplcore/core/`：用**因果谱全局卷积**替代自注意力，
复杂度 `O(L log L)` 训练、`O(1)` 状态流式推理，**无位置嵌入表、无 `max_len` 截断、
无 KV 缓存**。

> **SGC 是 SimplCore 内部的一个 mixer 模块**（`simplcore/core/sgc.py`），
> 不是架构的名字。SimplCore 下面有多个可切换的非注意力 mixer：
> SGC / SRFN / GatedSpectralConv2。架构层面的类名一律是 `SimplCore*`。

---

## 1. 它是什么 / 不是什么

| 是 | 不是 |
|----|------|
| 一套可运行的 PyTorch 序列混合器（SGC / SRFN / GatedSpectralConv2） | 一个已证明 SOTA 的模型 |
| 用 FFT 全局卷积 + 递归替代 softmax 注意力 | Transformer 的变体或混合体（**仓库内已无任何注意力模块**） |
| 严格因果、有 O(1) 流式状态、无位置表 | 依赖 KV 缓存或位置编码的架构 |
| 在 CPU 上验证数学正确 / 因果 / 双形式等价 | 在 1B+ 规模上训练过的系统 |

**架构本体完全不属于 Transformer 框架**：全局交互由 `torch.fft` 谱卷积 +
阻尼振荡基字典完成，**没有** `Q/K/V`、`softmax(QKᵀ)`、`scaled_dot_product_attention`、
RoPE、KV 缓存、位置嵌入表。注意力族（`CausalAttention` / `ReasoningAttention` /
`SlidingWindowAttention` / GQA-YaRN / RoPE）已从产品线**整体删除**，
仅作为历史归档保留在 `.trash/attn_removed/`，早期路线原型归档在 `research/`。
两道防线守护这一点：`simplcore/core/purity.py`（配置 + 实例双重断言）与
`SimplCoreConfig.validate()`（只允许非注意力 mixer）。

---

## 2. 快速开始

```bash
# 环境
python -c "import torch; print(torch.__version__)"

# 核心数学 / 双形式等价 / 因果性 / 梯度可达
python -m pytest simplcore/core/tests/test_mixer_math.py -q

# 主线因果性守护（扰动末位 token，断言前文逐位不变）
python -m pytest simplcore/core/tests/test_architecture_purity.py -q

# 最小示例
python -c "
from simplcore.core import get_preset, SimplCoreModel
import torch
cfg = get_preset('sgc-150m')
m = SimplCoreModel(cfg)
x = torch.randint(0, cfg.vocab_size, (1, 64))
logits, loss = m(x, x); loss.backward()
print('fwd/bwd OK', tuple(logits.shape), float(loss))
st = m.init_stream(1)
_, st = m.stream_step(x[:, 0], st)        # O(1) 状态流式
print('stream bytes:', m.stream_bytes(1))
"

# 理论优势的可执行验证（24 项机制层断言，全过才退出 0）
python -m simplcore.core.theory
```

---

## 3. 架构一图流

```
输入 token id ──► tok_emb ──► [ Block ]×N ──► ln_f ──► head ──► logits
                         │
        ┌────────────────┼───────────────────────────────────────┐
        ▼                ▼                                        ▼
   非注意力 mixer      L4 拒答头                              SwiGLU FFN
   (sgc / srf /        (Linear(dim→1) 标量头)                   + 残差
    gated-spectral)
        │
        ▼
   FFT 全局卷积(训练) ⇄ 二阶 IIR 递推(流式)
   阻尼振荡基字典 + 深度因果卷积 + 内容门控 + L2/L3 闸门
```

数据流的每一层都是 `mixer → FFN`，外加可选 CMT 记忆注入（残差流加性项）
与 L4 拒答头。详见 `docs/ARCHITECTURE.md`。

---

## 4. Mixer 家族（同一套 config 切换，全部非注意力）

| `mixer=` | 混合器 | 注意力 | 流式状态 | 训练 FLOPs 长程项 |
|----------|--------|--------|----------|-------------------|
| `sgc` | 谱全局卷积（默认） | 无 | O(1) IIR | `O(D·log L)` |
| `sgc-nogate` / `sgc-nolocal` | 去内容门控 / 去局部卷积 | 无 | O(1) | `O(D·log L)` |
| `srf` | SGC + selective SSM（输入相关 Δ） | 无 | O(1) chunked scan | `O(D·L/chunk)` |

> 曾经的 `attn` / `hybrid*` / `reason_attn_every` 取值**已删除**：
> `SimplCoreConfig.validate()` 现在只接受上面四类非注意力 mixer。

---

## 5. 文档索引

| 文档 | 内容 |
|------|------|
| `docs/THEORETICAL_EDGE.md` | **逐模块的理论优势论证**：每个模块相对传统 Transformer 的复杂度/状态/机制优势与边界 |
| `docs/ARCHITECTURE.md` | **总体架构设计**：数据流、mixer 分类、流式、记忆、编解码、训练/评测 |
| `docs/SGC_CORE.md` | **SGC mixer 深度解析**：振荡基字典、FFT↔IIR 对偶、内容门控、局部路径、四层闸门、流式状态 |
| `docs/MODULES.md` | 逐模块索引（每个模块一段职责说明） |
| `research/` | 早期路线原型归档（**不发布、不参与 CI**），仅供考古与对照 |

---

## 6. 当前真实能力边界

- ✅ mixer 核心：数学正确、严格因果、FFT↔IIR 双形式等价（fp32 `max|Δ|<4e-5`）、梯度可达
- ✅ 四层抗幻觉闸门（L1 选择门 / L2 证据门 / L3 不确定门 / L4 拒答头）结构完整、可消融、流式==整段
- ✅ O(1) 流式状态 + `checkpoint_stream/restore_stream` 持久化、与 L 无关
- ✅ 无词表模式（字符/字节/BPE 可插拔）、HF 权重迁移桥（只搬 embedding/FFN/norms）
- ✅ 架构纯洁性：产品线内**零注意力模块**，由 `purity.py` + CI 双重守护
- ✅ **理论优势可执行化**：25 项机制层断言（`simplcore/core/theory.py`），
  `docs/THEORETICAL_EDGE.md` 的每条主张都有对应检查，由 `test_theory.py` 守护
- ✅ **支柱 C 闭环**：代码沙箱（`code_exec.py`）+ 独立 verifier + no-regression gate +
  **内置权重巩固器**（`default_consolidator`）→ "真实环境写代码 → 验证 → 写回权重"
  默认就是通的，且退化时逐位回滚
- ⚠️ **1B 以上规模从未实际训练**；配置与链路已打通，曲线待测（目标硬件：1× AMD Instinct MI250）
- ⚠️ 未与 S4 / Mamba / Hyena / GLA 在 LRA 等公认基准做公平对比
- ⚠️ 长上下文外推只有"输出有限"证据，无质量证据
- ❌ 不得据此宣称任何 SOTA

---

## 7. 目录结构

```
simplcore/                       # 发布的产品线（唯一一条）
  core/                          # ★ SimplCore 架构本体
    sgc.py                       #   SGC mixer：谱全局卷积 + FFT/深度卷积 + O(1) 流式状态
    srf.py / ssm.py              #   SRFN mixer / selective 谱状态空间
    selective.py                 #   GatedSpectralConv2（GLA/Mamba2 式通道门，非注意力）
    model.py                     #   SimplCoreModel / SimplCoreBlock / SimplCoreForCausalLM
    config.py                    #   SimplCoreConfig + 规模预设（SIMPLCORE_PRESETS）
    gates.py                     #   L2 证据门 / L3 不确定门（共享数学）
    abstain.py / grounding.py    #   L4 拒答头 + 证据绑定层
    memory.py / dream.py         #   CMT 压缩记忆库 + 自进化 Dreaming 循环
    code_exec.py                 #   代码执行沙箱 + pass@k 验证器
    chat.py / dialogue_*.py / sft.py       # 对话 SFT 与评测
    tokenization.py / tokenizer.py         # 无词表编解码器（char/byte/bpe）
    scaling.py / extrapolation.py / hf_bridge.py / eval.py / train.py / data.py
    purity.py                    #   架构纯洁性守卫（无注意力硬约束）
    tests/                       #   守护测试（623 通过 / 1 跳过）
  safety/                        # 内容安全 / 越狱检测 / RBAC / 加密
  rag/ search/                   # 外围子系统（联网检索 / RAG，可选依赖，未与 core 强耦合）
research/                        # ★ 早期路线原型归档（不发布、不进 CI、不入库）
  subsystems/                    #   自研数值引擎 + 第一代架构 + 认知/AGI/对话/NLU 等子系统
  mix/  pg/                      #   第一代数据流水线 / 第二版 SimplCore 原型
  training/  unity/              #   独立训练基础设施 / 多专家编排原型
  verify_sgc/  legacy_scripts/   #   最早的 SGC 实验脚本 / 遗留脚本
java-high-performance/           # 并行 Java 实现
third_party/                     # 第三方子仓
```

> 打包只发布 `simplcore` / `simplcore.core` / `simplcore.safety`（见 `pyproject.toml`）；
> `research/` 与 `.trash/` 均不入库、不进 CI。

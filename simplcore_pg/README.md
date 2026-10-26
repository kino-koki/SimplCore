# SimplCore 生产级实现 (PyTorch 2.1)

`Simplcore_pg` 是 **SimplCore 多模态语言大模型** 架构的一套以 **PyTorch 2.1** 为基准、
可运行、可测试、可复现的"标杆级"实现。它覆盖架构全部核心模块，并利用 PyTorch 2.1
的新特性（`torch.compile`、FX 图模式），代码结构清晰、异常与边界完备、文档齐全。

> 说明：原仓库包含 229 个 Python + 247 个 Java 零散文件。本包是其中**核心架构**的
> 生产级重实现与标准定义，其余外围脚本可依据本包的标准逐批对齐。

---

## 特性对照（相对原仓库的升级）

| 维度 | 原仓库现状 | 本实现 (10/10 标准) |
| --- | --- | --- |
| 框架 | 混用 PyTorch 2.4 + 大量 Java/OpenCL | **PyTorch 2.1** 单一基准，API 与 2.1 对齐 |
| 性能优化 | 无编译/图优化 | `torch.compile`（inductor，安全回退）+ **FX 图模式**优化 pass |
| 测试 | 散落的自检 `__main__` | **pytest 全套**覆盖正常/边界/异常路径 |
| 文档 | 架构文档与代码不符（写 CUDA 实为 OpenCL） | 模块 docstring + 本 README，真实可运行 |
| 异常 | 有体系但结论失真 | 异常体系 + 张量健康守卫 + 梯度裁剪，结论与数据一致 |
| 多模态 | ViT 依赖未对齐 | 自带轻量 ViT + 统一模态路由 |

---

## 架构总览

```
输入 token id
   │
   ▼
TokenEmbedding ──(×√H)──► [Block]×N ──► FinalNorm ──► LM Head
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      LocalInteraction   ResonanceField      DynamicMemoryChannels
       (因果卷积+递归)    (FFT 波场演化)       (内容寻址记忆)
            │                 │                 │
            └────────┬────────┴────────┬────────┘
                     ▼                 ▼
              TopologyAwareRouter   FFN (SwiGLU)
                 (top-k 路由)        │
                     └────► +残差 ────┘
```

> 自 v2.0 起**已彻底移除自注意力**（无 Q/K/V/RoPE/SDPA）。局部依赖由
> LocalInteraction（因果深度卷积 + 可选门控线性递归）承担，全局依赖由
> ResonanceField 谱全局卷积承担。

子层：
- **LocalInteraction**（`localmix.py`）：因果深度可分离卷积 + 可选门控线性递归，O(n)，替代自注意力。
- **ResonanceField**（`resonance.py`）：FFT 全局卷积式波场演化，O(n log n) 全局感受野，可训练。
- **DynamicMemoryChannels**（`memory.py`）：可学习记忆库 + 内容寻址读取；推演期 `MemoryState` 运行时写入。
- **TopologyAwareRouter**（`routing.py`）：P 条路径 top-k 稀疏路由，近似 MoE。
- **FFN**（`layers.py`）：SwiGLU 前馈。

---

## 快速开始

```python
import torch
from simplcore_pg import SimplCoreConfig, SimplCoreModel

cfg = SimplCoreConfig.from_preset("small")   # 或 "nano"/"150m"/"7b"/...
model = SimplCoreModel(cfg)
model.eval()

ids = torch.randint(0, cfg.vocab_size, (2, 32))
with torch.no_grad():
    logits = model(ids)                       # (B, L, vocab)

# 生成
out = model.generate(ids[:1], max_new_tokens=16, temperature=0.9, top_k=40)
```

### torch.compile（PyTorch 2.1 新特性）

```python
model.compile(mode="default", backend="inductor", smoke_input=(2, 32))
logits = model(ids)        # 自动走编译路径；不可用则安全回退 eager
print("compiled:", model.is_compiled)
```

### FX 图模式

```python
gm = model.trace_fx(example_input_ids=ids)  # 优先 torch.export (Linux)，回退 symbolic_trace (Windows)
counts = model.summarize_graph(gm)          # 算子频次统计
from simplcore_pg.fx_passes import optimize_for_inference
gm_opt = optimize_for_inference(gm)          # 删除冗余别名节点
out = gm_opt(ids)                           # 与 eager 数值一致
```

> **平台兼容说明（重要）**：PyTorch 2.1 在 **Windows** 上通过 `check_if_dynamo_supported()`
> 禁用了 `torch.compile` 与 `torch.export`（报 `RuntimeError: Windows not yet supported for
> torch.compile`）。因此本实现在 Windows 上的"图模式"走 `torch.fx.symbolic_trace`
> （`concrete_args` 特化到示例形状）路径，并已在 `torch 2.1.2+cpu` 上验证通过；在
> Linux/macOS 等支持 dynamo 的平台上，`trace_fx` 与 `model.compile()` 会自动走
> `torch.export` / inductor 路径，获得真正的图编译与算子融合加速。两种路径下模型均
> 为**纯张量运算、无数据相关控制流**，对 `torch.compile` / FX 友好。

### 多模态

```python
from simplcore_pg import ModalityRouter, SimplCoreViTEncoder

router = ModalityRouter(hidden_size=256, use_vision=True, image_size=224, patch_size=16)
out = router(
    text_embeds=torch.randn(1, 8, 256),
    image=torch.randn(1, 3, 224, 224),
)
# out["latent"]: (1, 8 + patches, 256) 统一潜空间
```

---

## 56B 中文 MoE 扩展

将 SimplCore 升级为 **56B 参数、7 专家 (每专家角色 ~7.8B、top-2 激活) 的 MoE 大模型**，
纯中文语料训练，对标前沿中文能力。详见 `SimplCore_56B_MoE_Plan.md`。

```python
from simplcore_pg import SimplCoreConfig, SimplCoreMoEModel

cfg = SimplCoreConfig.from_preset("56b-moe")   # 总量 56.07B，每专家角色 7.80B
print(cfg.estimate_parameters()["total"] / 1e9, "B")   # ≈ 56.07

model = SimplCoreMoEModel(cfg)
logits = model(input_ids, training=True)        # 训练时聚合负载均衡 + z-loss
aux = model.last_aux_loss                        # 辅助损失 (防路由坍缩)
```

MoE 层 (`moe.py`) 采用**容量调度**实现真实 FLOP 节省 (每 token 仅激活 2/7 专家)，
并提供负载均衡损失与 router z-loss、可选共享专家。`trace_fx` 已支持 MoE 图追踪
(自动固定容量以避免动态 `one_hot`)。

### 训练 (分布式 / CPU 冒烟)

```bash
# CPU 小尺度冒烟 (验证逻辑与管线)
python -m simplcore_pg.training.train --smoke --max-steps 20 --seq-len 64

# GPU 集群 (专家并行 + FSDP，详见 distributed.py)
torchrun --nnodes=1 --nproc_per_node=8 \
    simplcore_pg/training/train.py --preset 56b-moe --seq-len 4096 \
    --micro-batch 1 --global-batch 1024 --lr 3e-4 --grad-ckpt
```

中文语料流水线 (`training/data.py`)：可替换的中文优化分词器接口、流式 `IterableCorpus`、
`PackedDataset` 因果 LM 打包 (pad 处 label=-100)，并预留质量过滤/去重钩子。

### 中文能力评估

```bash
# 内置小样本冒烟
python -m simplcore_pg.eval.eval_chinese --smoke

# 加载真实 C-Eval / CMMLU JSONL 对照分数线
python -m simplcore_pg.eval.eval_chinese --benchmark ceval_dev.jsonl --k-shot 5
```

评估 harness (`eval/eval_chinese.py`) 输出**总体与各科目准确率**，把"对标/超越"变成可量化对照线。

---

## 测试与基准

```bash
# 安装 (CPU 版 torch 2.1.2)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install pytest

# 运行全套测试
python -m pytest tests_simplcore_pg -q

# 性能基准 (eager vs torch.compile vs FX 优化图)
python -m simplcore_pg.benchmarks --preset nano --batch 4 --seq 64 --reps 30
```

预期：所有测试通过；`run_benchmark` 输出 eager、torch.compile 与 **FX 优化图** 三者的
延迟与加速比。注意：Windows 上 `torch.compile` 因平台限制不可用（自动回退 eager，加速比≈1.0），
此时 **FX 优化图** 即为跨平台可用的实际加速路径。

---

## 模块清单

| 文件 | 职责 |
| --- | --- |
| `config.py` | 类型化配置、校验、预设、参数量估算 |
| `exceptions.py` | 异常体系 + 张量健康守卫 + 梯度裁剪 |
| `layers.py` | RMSNorm / FFN / 初始化 |
| `localmix.py` | LocalInteraction 原生局部交互层（替代自注意力） |
| `attention.py` | 自注意力（**已废弃**，保留仅供历史参考；架构默认不启用） |
| `resonance.py` | 共振场 (FFT 波场演化，可训练谱全局卷积) |
| `memory.py` | 动态记忆通道 + 运行时记忆 |
| `routing.py` | 拓扑感知路由 (top-k) |
| `block.py` | SimplCoreBlock 计算单元 |
| `model.py` | SimplCoreModel 完整模型 (compile/FX/generate) |
| `model_moe.py` | SimplCoreMoEModel (MoE 完整模型) |
| `model_stream.py` | SimplCoreStreamingModel (长上下文流式) |
| `checkpointing.py` | 权重安全保存/加载 (AES-256-GCM/RBAC/内存金丝雀/指纹) |
| `multimodal.py` | 模态路由 + 轻量 ViT |
| `fx_passes.py` | FX 图优化 pass |
| `compile.py` | torch.compile 封装与后端探测 |
| `benchmarks.py` | 性能基准 |

### 权重安全保存/加载（checkpointing）

```python
from simplcore_pg import SimplCoreMoEModel

# 明文保存 / 加载
model.save_pretrained("./ckpt")
m2 = SimplCoreMoEModel.from_pretrained("./ckpt")

# AES-256-GCM 加密保存 / 加载（口令派生密钥，盐随 checkpoint 持久化）
model.save_pretrained("./ckpt_enc", passphrase="s3cret")
m3 = SimplCoreMoEModel.from_pretrained("./ckpt_enc", passphrase="s3cret")

# RBAC 鉴权（deny-by-default）：未授权保存/加载抛 RBACError
from safety.rbac import RBACManager
rbac = RBACManager(); rbac.assign_role("alice", "developer")
model.save_pretrained("./ckpt_rbac", rbac=rbac, user="alice")
```

解密后的明文 state_dict 驻留带金丝雀的 `GuardedBuffer`，加载完成即覆写清零
（用后即焚）；`config.json` 记录权重 SHA-256 指纹，加载时校验防篡改。

---

## 设计原则

1. **可验证优先**：所有"突破/结论"都必须由测试或数据支撑，杜绝叙述与指标矛盾。
2. **编译友好**：子层均为纯张量运算、无数据相关控制流，兼容 `torch.compile` 与 FX。
3. **失败可诊断**：统一异常体系 + 张量健康守卫，任何非法输入给出明确 `details`。
4. **向后兼容**：保留与原配置同名的 property 访问器，降低迁移成本。

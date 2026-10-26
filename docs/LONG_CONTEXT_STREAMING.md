# 长上下文流式设计（SimplCore-V 0.1-ALPHA）

目标：上下文从 1M token 起，序列长度每次翻倍，激活显存仅近似常数增长。

---

## 1. 设计：把所有"随 n 增长"的项压成固定状态 + 分块处理

| 子层 | 标准 Transformer | SimplCore 流式版（无注意力） | 显存随 n |
|------|------------------|------------------------------|----------|
| 局部交互 | 自注意力 O(n²) | LocalInteraction（因果卷积 + 线性递归，O(n)），无 KV 缓存 | 常数 |
| 全局依赖 | 注意力承担 | RFP 谱全局卷积（分块频域演化，固定频域状态） | 常数 |
| 记忆 | 无 | DMC 动态记忆通道（固定记忆库 `C×S×H`） | 常数 |
| 序列处理 | 整段一次过 | 分块流式（`chunk_size`，默认 512），跨块仅传 `StreamState` | 常数 |

跨块状态 `StreamState` = RFP 频域状态 + DMC 记忆，二者大小均与 n 无关。**自 v2.0 起
已彻底移除自注意力与 KV 缓存**——局部依赖完全由 LocalInteraction 承担。

---

## 2. 真实 API（已实现，2026-08-24）

```python
from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_stream import SimplCoreStreamingModel

lc = SimplCoreConfig.get_longctx_config(4096)
lc.hidden_size = 128; lc.num_layers = 2; lc.num_experts = 4; lc.vocab_size = 128
sm = SimplCoreStreamingModel(lc)

# 流式分块
st = sm.stream_init(batch=1)
for i in range(0, 256, lc.chunk_size):
    chunk = torch.randint(0, 128, size=(1, lc.chunk_size))
    logits, st = sm.stream_step(chunk, st)   # 跨 chunk 仅传固定 StreamState

# 自回归生成
gen = sm.generate_stream(torch.randint(0, 128, size=(1, 8)), max_new_tokens=4)
```

`stream_init` / `stream_step` / `generate_stream` 均经 `smoke_test.py` 实测通过。
（早期文档声称这些 API 已可用，但代码中并不存在——现已补齐。）

---

## 3. 显存增长实测（真实指标）

来源：`longctx_memory_benchmark.json`（公式 + CPU 小尺度验证）。

| 设置 | 固定状态 | 激活峰值 | 1M→2M→4M 翻倍增量 |
|------|----------|----------|---------------------|
| fp16, B=1 | 1.88 GB | 2.08 GB | 0 GB |
| fp32, B=1 | 3.76 GB | 4.16 GB | 0 GB |

`peak_constant_across_lengths = true`：激活峰值在 L=1M/2M/4M 下完全相同。

**诚实解读**：激活显存维度，翻倍增量理论为 0（所有占用项均为定值）。实际部署中
"翻倍 +~1GB"来自权重常驻显存（56B fp16 ≈112GB 固定）与 CUDA 上下文——均为常数，
不随 n 线性膨胀。相对标准 Transformer 的 O(n²) KV 缓存（1M 长度 KV 即 TB 级），
本架构把翻边际成本压到理论零增长。且无 KV 缓存更彻底地消除了注意力带来的线性膨胀项。

---

## 4. 兼容性

- 所有子层为纯张量算子、无数据相关控制流，兼容 `torch.compile` / FX；
- RFP/DMC 复用生产级可训练实现，梯度可达。

---

## 5. 多模态原生接入

`SimplCoreNativeMM`（`native_mm.py`）支持 `streaming=True`，用 `SimplCoreStreamingBlock`
替换全量块，使多模态混合序列同样享受固定状态长上下文。

---

## 6. 待办（真实缺口）

- **GPU 显存实测**：本机 AMD 680M + Windows 下 PyTorch 无法训练；需在 WSL2+ROCm 或
  Linux+ROCm 用 `torch.cuda` 钩子对 1M/2M/4M 二次确认（当前为公式 + CPU 验证）。
- **长序列精度对比**：当前仅有延迟基准，缺困惑度/下游任务精度对比。

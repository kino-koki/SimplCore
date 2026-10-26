# SimplCore 世界模型设计文档

## 概述

在 SimplCore 架构基础上，构建一个通用世界模型（Universal World Model），兼具语言理解、潜空间推演、视觉感知和物理模拟能力。采用渐进式扩展策略，Step1 先实现潜空间推演引擎，Step2 加入视觉编码器。

## Step1: 潜空间推演引擎

### 核心思想

现有 SimplCore 的前向传播是"一次性"的：

加入推演引擎后，模型在输出前会在潜空间进行多轮迭代推演：

```
输入 → RFP → DMC → TAR → FFN → ... → 潜状态 S₀
                                          ↓
         ┌──── 推演循环 (K次, 参数共享) ────┐
         │  Sₜ = RFP(Sₜ₋₁)               │
         │  Sₜ = DMC(Sₜ)                 │
         │  Sₜ = TAR(Sₜ)                 │
         │  Sₜ = FFN(Sₜ)                 │
         └──────────────────────────────┘
                                          ↓
                                        输出
```

### 复用的 SimplCore 组件

| 组件 | 在推演中的作用 |
|------|--------------|
| RFP | 通过波场迭代演化潜状态，物理动力学的天然模拟器 |
| DMC | 跨推演步追踪关键状态，保持长期一致性 |
| TAR | 自适应判断是否停止推演，按需分配计算量 |
| FFN | 每轮推演后做非线性变换，增强表达能力 |

### RolloutEngine

新建文件 `SimplCore/architecture/rollout_engine.py`：

```python
class RolloutEngine(nn.Module):
    """潜空间推演引擎.
    
    在 SimplCoreBlock 的基础上，对潜状态进行多轮迭代推演。
    支持自适应推演深度（TAR 控制停止时机）。
    参数共享：同一套 RFP/DMC/TAR/FFN 每轮推演共用。
    """
    def __init__(self, block: SimplCoreBlock, max_steps: int = 8):
        self.block = block
        self.max_steps = max_steps
    
    def forward(self, latent, memory_state=None, num_steps=None):
        # 对潜状态进行多轮推演
        # 每轮: Sₜ = block(Sₜ₋₁)
        # TAR 判断是否可提前停止
        ...
```

### 延迟解码（Deferred Decoding）

借鉴 LoopWM 的延迟解码策略：推演过程全部在潜空间运行，只在最后一步解码为输出。

```python
# 推演前
latent = model.encode(input)

# 潜空间推演 (K步，不产生中间输出)
for _ in range(K):
    latent = rollout_engine(latent)

# 最后一步才解码
output = model.decode(latent)
```

### 训练策略

```
Step 1.1: 在现有语言任务上加入推演
  - 对简单问题：K=1（不推演，退化为原模型）
  - 对复杂问题：K=3-8（多步推演）
  - 损失函数不变

Step 1.2: 加入推演深度损失
  - 鼓励模型对难题多用推演步数
  - TAR 自适应学习停止时机
```

## Step2: 视觉感知（预留接口）

### 接口设计

```python
class VisionEncoder(nn.Module):
    """视觉编码器 - 将图像/视频编码为潜状态.
    Step2 实现，Step1 只留接口定义。
    """
    def encode_image(self, image) -> torch.Tensor: ...
    def encode_video(self, video) -> torch.Tensor: ...
```

### 模态路由（ModalityRouter）

```python
class ModalityRouter(nn.Module):
    """模态路由 - 将不同模态的输入统一转发到 SimplCore 核心."""
    def forward(self, text=None, image=None, video=None, action=None):
        # 各模态分别编码 → 拼接为统一潜状态序列 → 标记模态类型
        ...
```

## 文件结构

```
SimplCore/
├── architecture/
│   ├── rollout_engine.py      # 新建 - 潜空间推演引擎
│   ├── modality_router.py     # 新建 - 模态路由（Step2 实现）
│   ├── vision_encoder.py      # 新建 - 视觉编码器接口（Step1 占位）
│   └── ... (其余已有文件不变)
├── training/
│   ├── rollout_train.py       # 新建 - 推演训练
│   └── world_model_train.py   # 新建 - 世界模型训练
test_world_model.py            # 新建 - 世界模型测试
```

## 不变量

1. 推演引擎不改变 SimplCoreBlock 的内部实现（只复用现有 forward）
2. 加入推演后，现有语言任务的推理路径不变（当 K=1 时退化为原模型）
3. 视觉编码器接口设计不依赖具体实现（Step2 可以换不同的视觉 backbone）
4. 所有新代码在 2×RTX 4090 上可测试


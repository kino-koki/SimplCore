# MoK：修复 SGC 的输入无关核缺陷（TS2SC 迁移的架构根因）

> 本文记录一个**被实测确认、不是推测**的架构缺陷，以及修复方案、实测证据和使用方式。
> 修复代码：`simplcore/core/arch/sgc.py`；守护测试：`simplcore/core/tests/test_sgc_mok.py`。

---

## 1. 缺陷：SGC 的核是输入无关的

注意力与 SGC 的数学形式对照：

```
注意力   y[i] = Σ_j softmax(q_i · k_j / √d) · v_j
        权重 a[i,j] 是「位置对 (i,j)」的内容函数 —— 二阶 pairwise 交互

SGC     y[i] = Σ_τ K[τ] · u[i-τ]
        核 K 是输入无关的常数；输入相关部分只有逐元素门控
        （content_gate / evidence_gate / uncertainty_gate / path_gate），
        只能逐位置、逐通道地缩放，无法表达
        「位置 i 应当聚合历史的哪一种模式」
```

**后果**：用 SGC 去拟合 Transformer 注意力时，能学到的只是注意力的**平均位置分布**，复现不了逐 token 的内容路由。

### 实测症状（来自 TS2SC 迁移实验）

| 现象 | 数据 |
|---|---|
| 逐块蒸馏余弦长期卡住 | 0.55 ~ 0.97，平均 0.73；深层尤其差 |
| 端到端生成退化 | teacher「阳光明媚，微风」 vs student「理解并进行。」 |
| 累积漂移 | L0–L21 cos ≈ 0.99，L22 0.983，L23 **0.935**（相对误差 0.36） |
| 余弦指标本身会骗人 | 目标设成 `x + attn_out` 时残差主导，cos 恒 1.0000 而 mixer 没训 |

最后一条尤其重要：**前面若干轮"余弦 1.0000"是假指标**。把目标改成注意力的**混合项** `attn_out`、student 取 `SGC(x) - x` 之后，真实余弦立刻掉到 0.55~0.97 —— 这才是缺陷的真实量级。

---

## 2. 修复：Mixture-of-Kernels（逐位置内容路由）

让**核本身成为输入的函数**：

```
y_long[t] = Σ_k g_k(h_t) · conv(u, K_k)[t],    g = softmax(W h_t)
```

- `K` 个核，**共享同一组振荡基 psi**（rho/theta），只有 readout 系数 `coefs` 不同；
- 每个 `conv(u, K_k)` 仍是纯因果卷积 → FFT 仍是 `O(L log L)`；
- `g_k(h_t)` 是当前 token 的因果函数，与序列其余位置无关 → 与 `content_gate` / `path_gate` 同构，**并行(FFT) 与流式(IIR) 逐位一致**。

### 为什么核心契约不破

IIR 递推 `y_new = a·y1 + b·y2 + c·u` 只依赖共享的 psi，与 readout 系数无关。因此：

> **状态 `(y1, y2)` 的形状与字节数与 `num_kernels` 完全无关 —— O(1) 流式状态逐字节不变。**

多核只体现在 readout：

```
nxt_k = einsum("bmd,kmd->bkd", y_new, coefs)
nxt   = Σ_k g_k · nxt_k
```

代价是 readout 的 K 倍常数 FLOPs，**与序列长度无关**。这是"补齐 pairwise 表达力但不牺牲 O(1)"的结构性保证。

---

## 3. 实测证据

### 3.1 表达力判据（冻结多核 teacher → 单核/多核 student 拟合）

固定 seed，teacher = 2 核 SGC（两核系数差异大、路由器强区分），目标能量 1.2953：

| student | 200 步 | 400 步 | 800 步 | 1600 步 |
|---|---:|---:|---:|---:|
| **单核 K=1** | 0.0753 | 0.0697 | 0.0685 | **0.0675（平台期）** |
| 多核 K=2 | 0.0191 | 0.0129 | 0.0106 | 0.0095 |
| **多核 K=4** | 0.00014 | **0.00001** | 0.000002 | 0.000009 |

**两个结论**：

1. 单核存在**不可约的结构误差**（≈5% 相对目标能量），从 200 步加到 1600 步只降约 10% 且卡死 —— 加训练步数补不了表达力缺口。
2. 多核 K=4 在 400 步就精确表示，比单核 1600 步好约 **7000 倍**。

> **直接回答「每层跑 500~1000 步行不行」：不行。** 上面这张表就是答案：单核 1600 步仍在 0.0675 的平台。缺陷在算子的表达力，不在优化步数。

### 3.2 关于第一版判据的诚实说明

最初写了一个「指令决定读最近/最早 token」的任务，结果**单核也解到 mse=0.000000** —— 因为 `in_proj`/`out_proj`/`norm`/`content_gate` 的非线性足以绕过它。那是**假阳性测试，已删除**。现在的 teacher–student 拟合判据是可复现的表达力差异，不是声称。

---

## 4. 踩过的坑（都已修，并有测试守护）

| # | 问题 | 症状 | 修复 |
|---|---|---|---|
| 1 | 余弦目标设成整层输出 | FFN 占大头，SGC 躺平（`out_scale≈0.02`）余弦仍 1.0000 | 迁移侧目标改为注意力**混合项** |
| 2 | 目标设成 `x + attn_out` | 残差范数远大于混合项，`cos(x+A,x+B)≈1` 即使 A≠B | 目标 = `attn_out`，student = `SGC(x)-x` |
| 3 | **MoK 零初始化 router** | 所有核相同 + g 恒均匀 → 每个核梯度完全相同 → **对称性锁定，永远分化不了** | router 用 `normal(0, 0.01)` 打破对称 |
| 4 | 热启动删掉 `coef` | 多核模块仍注册它，`strict=True` 报 missing key | 保留 `coef` 作兼容占位参数（前向不走它） |
| 5 | `forward_chunk` 仍用单核 | 跨块自由响应与 full forward 不一致 | chunk 走多核，且 `_free_response` 按同一组 g 加权 |
| 6 | `RoutedSGC` 静默忽略 MoK | 两套路由叠加却给出不一致结果 | 显式 `ValueError` 拒绝叠加 |

第 3 条是关键：仓库里 `RoutedSpectralGlobalConv.route_proj` 的注释早就警告过「zero init 会让未激活专家成为梯度死区」，MoK 踩的是同一个坑。

---

## 5. 使用方式（TS2SC 迁移）

### 5.1 从头构建多核 mixer

```python
from simplcore.core.arch.sgc import SpectralGlobalConv

mixer = SpectralGlobalConv(
    dim=1024,
    num_bases=32,
    local_kernel=4,
    rho_param="logspace",
    init_mode="logfreq",
    content_gate=True,
    num_kernels=4,          # ← 新增，默认 1
)
```

`num_kernels=1` 是**逐位兼容**的原实现：不创建任何新参数、不走多核分支，既有模型与测试零回归。

### 5.2 从已有单核 checkpoint 热启动（TS2SC 已有权重不作废）

```python
sd = torch.load("mixer_12.pt")["best_mixer"]     # 单核权重
m4 = SpectralGlobalConv(dim=1024, num_bases=32, num_kernels=4, ...)
m4.load_state_dict(sd, strict=True)              # 自动映射，无需手工改 key
```

映射规则：

- `coefs[k] = coef` 对所有 k（所有核初始化为同一个核）；
- `kernel_route` 注入 `normal(0, 0.01)`（**不是零**，理由见坑 #3）；
- `coef` 保留为兼容占位参数，保证单/多核 checkpoint 双向 `strict=True`。

此时输出是**近似等价**于单核（相对误差 ~1% 量级）：起步接近「平均核」，不丢知识、不产生初始化冲击，且训练能真正让各核分化。

### 5.3 需要「精确热启动」时（只换架构、暂不训练）

```python
with torch.no_grad():
    m4.kernel_route.weight.zero_()
    m4.kernel_route.bias.zero_()
```

因为 `Σ_k g_k = 1` 且所有核相同 ⇒ `Σ_k g_k·conv(u,K) = conv(u,K)`，输出与单核**逐位等价**（`atol=1e-6`）。注意：这样会回到对称性锁定，只适合直接推理，不适合继续训练。

### 5.4 迁移脚本的目标函数（必须配套改正）

```python
target  = attn_out                  # 注意力的混合项，不含残差、不含 FFN
student = mixer(x) - x             # 只取 SGC 的混合项
loss    = mse(student, target) + (1 - cos(student, target))
```

用整层输出或 `x + attn_out` 当目标都会得到虚假的高余弦（坑 #1/#2）。

---

## 6. 契约保证与测试清单

`simplcore/core/tests/test_sgc_mok.py` 共 11 项，全部通过；原有 SGC / mixer 测试 63 项无回归。

| 测试 | 守护什么 |
|---|---|
| `test_num_kernels_1_is_backward_compatible` | 单核不创建新参数、输出有限 |
| `test_multi_kernel_does_not_add_state_bytes` | **O(1) 状态字节数不变** |
| `test_mok_parallel_matches_streaming` | FFT/IIR 对偶性 |
| `test_mok_chunk_matches_full_with_state` | 跨块 chunk 一致性（含自由响应加权） |
| `test_mok_route_is_normalized_and_symmetry_is_broken` | 路由归一、小随机打破对称（防坑 #3 回退） |
| `test_mok_is_required_to_represent_content_routed_kernel` | **单核不可约误差 vs 多核精确表示** |
| `test_more_steps_cannot_fix_single_kernel` | **加步数无效**（平台期） |
| `test_single_kernel_checkpoint_hotstarts_to_mok` | 既有 checkpoint `strict=True` 热启动 |
| `test_exact_hotstart_when_router_zeroed` | 置零 router → 逐位等价 |
| `test_mok_can_diverge_after_hotstart` | 热启动后各核确实能分化 |
| `test_routed_sgc_rejects_mok` | 两套路由不静默叠加 |

---

## 7. 边界与未做的事

- **这不等于"迁移后的模型已经会说话"**。MoK 修的是 mixer 的表达力上限；base model 的「续写而非回答」行为仍需 SFT，二者是正交问题。
- 多核 readout 有 K 倍常数 FLOPs 开销（与 L 无关），`K=4` 是实测性价比选择，不是理论最优。
- 本文所有数值来自固定 seed 的可复现实验，规模是诊断用的小模型（D=32 / M=16 / L=24）；**在 Qwen3-0.6B / MiniCPM5-1B 规模上重新跑完整迁移并做生成验证，仍是待办**。

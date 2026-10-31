# SimplUnity（SGC 栈重制版）· 模型粘合 / MoE 效果 设计文档

> 状态：**已实现并接训练闭环（opt-in，默认关闭）**。本文件描述设计、与旧 Archive 的关系、4M 上下文定位、自进化闭环，以及运行方式。
> 配套代码：`simplcore/unity/`（config / router / expert / model）+ `simplcore/core/training_integration.py` 的 `UnityEnhancer`。

---

## 0. 这一段要解决什么

用户原话的两条硬约束：

1. **总上下文 4M**：模型以超长上下文为卖点，序列长度上限设为 4,000,000 token。
2. **真实环境代码自进化 + 后台并行训练 Unity 模块**：模型在真实环境（Computer Use）里学代码操作并把经验写进权重时，**后台应同时长出新的「Unity 小专家」**——即「数个小参数模型合体（训练数据必须不同），组成类似 MoE 的效果」。

> 旧 Archive（`.archive/simplcore_docs/`）里 "Unity" 有两套含混定义：
> - `SIMPLUNITY_AND_UNIT_SYSTEM.md` 的「编排神经系统 / MessageBus 多智能体」——**不是**本任务目标；
> - `UNITY_EXPERT_TRAINING.md` / `UNITY_680M_FEASIBILITY.md` 的「分领域训多个小专家 → `ExpertRouter` + `ExpertEnsemble` 粘合 → MoE 效果」——**正是本任务**。
>
> 旧实现依赖 `simplcore.tensor`（NumPy 后端）+ `LatticeLM`，属迁移清单 G5「未迁移」，当前 SGC/torch 栈上**不存在**。本文件把它**重做到当前栈**。

---

## 1. 架构：Mixture-of-Adapters（MoA）

沿用 SGC 的非注意力、O(1) 状态哲学，Unity 在**冻结的 SGC 基座**之上粘合 N 个「小参数专家」，由路由门做 top-k 选择，加权求和得到 MoE 级别的容量，而显存只吃可学部分（基座冻结）。

```
                   token ids
                      │
              ┌───────▼───────┐
              │  SGC 基座(冻结) │  ← 1B 知识核心，不更新
              └───────┬───────┘
                 hidden h (B,L,D)
                      │
              ┌───────▼───────┐
              │  ExpertRouter  │  ← 线性门 → top-k 软掩码
              │  (gate over h) │
              └───┬───┬───┬───┘
        ┌─────────┘   │   └─────────┐
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │Expert 0 │   │Expert 1 │   │Expert k │   ← 每个 = 小 MLP（域专属，数据不同）
   │(域A代码) │   │(域B数学) │   │(域C长文) │
   └────┬────┘   └────┬────┘   └────┬────┘
        └─────────┬───┴───┬────────┘
             加权求和 (router softmax · expert_out)
                      │
              h' = h + Σ_k w_k · Expert_k(h)
                      │
              ┌───────▼───────┐
              │  base.head     │  ← 共享词表头
              └───────┬───────┘
                  logits + aux(路由负载/ z-loss)
```

### 1.1 为什么是「粘合」而非「堆叠新层」
- 每个专家是**独立的小参数模块**（默认 `expert_hidden = 2·D` 的 2 层 MLP，约数 M 参数），训练数据**必须不同**（域 A/B/C…），天然 specialization；
- 路由门把它们「粘合」成整体，token 级 top-k 路由 → 等效 MoE 的「动态容量」；
- 基座冻结，梯度**只流向路由门 + 专家** → 显存按 `WeightBudget.learned_params` 计，符合「只训 2B 适配器」的预算纪律。

### 1.2 与 SGC 非注意力哲学的兼容性
- 专家是 **per-token MLP**（无时序状态、O(1) 流式），不引入注意力、不破坏 `purity.py` 的无注意力约束；
- 路由是 **线性投影 + top-k**，也是 O(1) 状态，可 streaming；
- 因此 4M 超长上下文下，Unity 不引入 n² 项，状态仍 O(1)。

---

## 2. 自进化闭环（后台并行训练新 Unity 模块）

用户要求「模型在真实环境学代码并把经验写进权重时，后台同时训练新的 Unity 模块」。映射为：

```
真实环境(Computer Use) 跑代码 → 产生 新域轨迹 → 经验写进 adapter / dream 池
        │
        └─ 触发 UnityModel.spawn_expert(domain_name)
              ├─ 在 WeightBudget 守卫下追加一个全新的小专家（零初始化/小随机）
              ├─ 后台线程/进程用该域专属数据训练这个新专家（数据不同于已有专家）
              └─ 训练完成后注册进 ExpertEnsemble，路由门逐步学会在新域激活它
```

- `spawn_expert` 受 `WeightBudget.can_allocate` 守卫：超出 3B 总上限 / 2B 可学上限时拒绝，保证「真实环境生长」不会撑爆预算；
- 每个新专家**只在该域数据上训练**（数据必须不同 = 自进化的硬约束），避免专家同质化；
- 这是「模型冻结、程序/harness 生长」的又一处体现（与 Harness Learning 精神一致，但为规则/手工驱动版，而非 RL Proposer）。

> 当前实现提供 `spawn_expert()` + `train_expert(i, loader, opt)` 接口与预算守卫；后台并行训练调度（多进程/线程）由部署侧决定，本模块给出可安全并发调用的 API。

---

## 3. 配置（UnityConfig）

| 字段 | 默认 | 说明 |
|---|---|---|
| `num_experts` | 4 | 初始专家数（可经 `spawn_expert` 增长） |
| `expert_hidden` | `2·dim` | 每个专家 MLP 中间维 |
| `expert_layers` | 2 | 每个专家 MLP 深度 |
| `top_k` | 2 | 每 token 激活专家数（MoE 稀疏度） |
| `router_hidden` | `dim` | 路由门中间维 |
| `capacity_factor` | 1.25 | 专家容量（令牌溢出时.drop） |
| `router_z_loss` | 1e-3 | 路由 logit 正则（防熵崩） |
| `load_balance_loss` | 1e-2 | 专家利用率均衡正则 |
| `expert_dropout` | 0.0 | 专家内 dropout |

所有可学参数（路由 + 专家）经 `WeightBudget` 计入 2B/3B 上限。

---

## 4. 4M 上下文定位

- 在 `tuan-1b` / `tuan-3b` 预设里将 `max_seq_len`（或等效 seq_len 上限）设为 `4_000_000`；
- Unity 专家为 per-token、O(1) 状态，4M 下不引入 n²，状态代价恒定；
- 训练采用 seq_len 课程（4K→16K→64K→128K→…），推理期靠 SGC 的 O(1) 递推扛 4M；
- 注意：4M 是**上下文窗口上限**卖点，不等于训练时每个 batch 都喂 4M（显存/数据决定实际课程长度）。

---

## 5. 训练接入（opt-in，零回归）

- `simplcore/core/train.py` 的 `--enhance` 新增 `unity`：
  - `--enhance unity`：基座冻结，只训路由 + 专家（MoE 模式，路由在共享流上自学）；
  - 可选 `--unity-experts N`（默认 4）与 `--unity-topk K`（默认 2）：控制初始专家数与每 token 激活数；
  - 空 `--enhance`（默认）：`UnityEnhancer` 不实例化，`train.py` 行为逐位不变。
- `UnityEnhancer` 实现 `train_forward(model, idx, tgt) -> loss` 与 `param_filter()`，与既有 `PonderEnhancer`/`EnvLearnEnhancer` 同契约。
- 路由 aux loss（负载均衡 + z-loss）作为正则并入总 loss。

### 5.1 两种训练模式（满足「训练数据必须不同」）

1. **联合 MoE 路由（默认主循环）**：warm-up 之后，主训练循环在混合数据上训练「路由 + 全部专家」，由路由门自学 token 级分配。有效，但 specialization 弱于显式分域。
2. **按域隔离训练（已实现 CLI）**：`--unity-domains dirA,dirB,...` 把每个域语料目录接进来，第 `i` 个专家**只在第 `i` 个域**上隔离训练（`warmup_unity_domains` → `train_expert(domain=i)`），其余专家与基座冻结；域数多于初始专家时自动 `spawn_expert` 增长。这是「每个专家训练数据必须不同」的**强制落地**。`--unity-warmup-steps` 控制每域步数。

```bash
# 三个域语料（每个 = data/processed/<域> 形态，含 manifest）→ 三个专家，数据互不相同
python -m simplcore.core.train --preset tuan-1b --enhance unity \
  --unity-experts 3 --unity-topk 2 \
  --unity-domains data/processed/code_py,data/processed/bash_cli,data/processed/math
```

> 若未提供 `--unity-domains`，则跳过隔离 warm-up，直接进入联合 MoE 路由（仍有效，但 specialization 弱于显式分域）。

---

## 5.2 后台自进化（模型冻结、程序生长）

`UnitySelfEvolution`（`simplcore/unity/evolution.py`）实现用户要求的「**后台同时训练新的模型模块**」：

```
真实环境(Computer Use) 跑代码 → 产生新域轨迹
        │
        └─ evo.register_domain("python_syntax", loader)   # 入队（该域数据与其他域不同）
              └─ 后台线程 worker：
                  1. 建独立候选专家（不参与活动模型前向 → 无竞态）
                  2. train_standalone_expert(候选, 域数据)   # 在该域专属数据上训
                  3. unity.publish_expert(候选)             # 预算守卫下原子换入活动模型
                  4. 路由门随后逐步学会在新域激活它
```

- 训练在**独立候选专家**上完成，再原子换入活动模型 → 后台训练与主线前向/推理互不竞态；
- 受 `WeightBudget` 约束：总上限（默认 3B）/ 可学上限（2B）耗尽时拒绝生长；
- 线程守护进程，单个域失败不拖垮后台线程；`stop(timeout=)` 等队列排空后退出。

```python
from simplcore.unity import UnityModel, UnitySelfEvolution
evo = UnitySelfEvolution(unity, steps_per_domain=200, lr=1e-2)
evo.start()
evo.register_domain("python_syntax", loader_py)
evo.register_domain("bash_cli",     loader_bash)
evo.stop()                        # 等队列排空
evo.save("runs/tuan-1b/unity.pt")  # 自进化成果落盘
```

> 生产部署建议：本管理器用进程内线程并发；跨进程/多卡可把 worker 改为独立进程，
> 通过队列/共享存储传递候选状态。

---

## 5.3 持久化（自进化跨会话累积）

- `UnityModel.save_learned(path)` / `UnityModel.load_learned(base, path)`：只存可学部分（路由 + 全部专家 + 预算账目），基座按预设恢复；
- `spawn_expert` 长出的新专家随快照恢复（专家数按快照决定）；
- 配合 `UnitySelfEvolution.save`，使「后台生长」在多次训练会话间累积，而非每次从零开始。

---

## 6. 诚实边界

- ✅ 已实现：MoA 结构、top-k 路由、aux loss、预算守卫、`spawn_expert` 自进化接口、
  `--unity-domains` 按域隔离 warm-up（强制数据不同）、`UnitySelfEvolution` 后台线程管理器、
  专家持久化（save/load）、训练闭环接入、CPU 全量验证通过（8 项测试 PASS，持久化权重精确恢复）。
- ⚠️ 未验证：在 4M 上下文 + 真实 Computer Use 轨迹上的**质量收益**——机制接线 ≠ 质量达标，需 MI250/多卡战役实测。
- ⚠️ 后台管理器为进程内线程版；跨进程/多卡部署需改为独立进程（文档 §5.2 已给改造建议）。
- ⚠️ 专家「数据必须不同」依赖 `--unity-domains` 提供真实分域语料；若缺省，则退化为共享流上路由自学的 MoE（仍有效，但 specialization 弱于显式分域）。

---

## 7. 快速验证

```bash
# 单元：基座冻结、域特化、路由梯度、预算守卫、集成、持久化、后台自进化、按域 warm-up
PYTHONPATH=. .venv311/Scripts/python.exe simplcore/core/tests/test_unity.py

# 训练接入（dry-run，unity MoE 模式）
PYTHONPATH=. .venv311/Scripts/python.exe -m simplcore.core.train --dry-run \
  --override dim=64 num_layers=2 num_bases=8 vocab_size=256 --seq-len 32 \
  --micro-batch 2 --max-steps 3 --eval-every 1000 --enhance unity \
  --unity-experts 3 --unity-topk 2 --out-dir tmp/enh_unity

# 按域隔离 warm-up（真实分域语料；此处目录不存在会优雅跳过）
PYTHONPATH=. .venv311/Scripts/python.exe -m simplcore.core.train --dry-run \
  --override dim=64 num_layers=2 num_bases=8 vocab_size=256 --seq-len 32 \
  --micro-batch 2 --max-steps 3 --enhance unity --unity-experts 2 \
  --unity-domains data/processed/code_py,data/processed/bash_cli \
  --out-dir tmp/enh_ud
```

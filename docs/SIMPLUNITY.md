# SimplUnity 技术文档：冻结基座 + 路由门 + 域专属专家的 Mixture-of-Adapters

> **定位**：SimplUnity 是 SimplCore 的「数个小模型合体」子系统——一个冻结的
> SGC 基座，之上粘合 N 个**域专属**小参数专家（gated MLP 栈）与一个 token 级
> top-k 路由门，合成 MoE 的动态容量效果；配合 `UnitySelfEvolution` 后台自进化
> （候选专家并行训练 → 无回归门 → 原子发布），实现「模型冻结、harness 生长」。
>
> **代码**：`simplcore/unity/`（8 模块，~1100 行）
> **守护测试**：`tests/test_unity.py`、`tests/test_simplunity_modular.py`
> **关联**：`SimplCore_Paper.md` §3.5(e)/§7.2 轴 E、§7.4 W1–W3 可证伪判据、
> `docs/SIMPLUNITY_3MONTH_PLAN.md`（三个月路线）
>
> ⚠️ **诚实边界**：域隔离训练、预算守卫、无回归门、双路径一致性均已测试锁定；
> **真实域上的专家特化质量、自进化长期稳定性未验证**（合成域测试已过，
> 真实 Computer Use 场景等 §5.13 战役）。

---

## 1. 设计动机：为什么不是标准 MoE

标准 MoE（Mixtral/DeepSeek 式）把专家做进基座内部，所有参数一起预训练。
SimplUnity 走的是完全不同的四条约束，每条都对应一个明确的工程决策：

| 约束 | 决策 | 落点 |
|------|------|------|
| 1B 主干冻结（预训练成本不重复付） | 基座 `requires_grad_(False)` + `eval()` 常驻 | `UnityModel.__init__` |
| 每个专家的**训练数据必须不同**（域专属，否则 MoE 退化为平均） | `domain=` 强制路由 + 按域 loader | `router.forward(domain=…)`、`train_expert(i, loader)` |
| 可学参数总量守恒（2B 冻结主干 + ≤2B 可学 = 3B 总上限） | `WeightBudget` 三层账（base/learned/total） | `envlearn.WeightBudget`，所有生长入口过守卫 |
| 后台长出新专家不与主线推理竞态 | **独立候选专家**训练，完成后原子 `publish_expert` | `UnitySelfEvolution._train_one` |

一句话：**SimplUnity 是 Mixture-of-Adapters（MoA），不是 Mixture-of-Experts**
——专家不在基座层间，而在基座输出端做残差调制；专家不是预训练产物，
而是域经验的后台沉淀物。

---

## 2. 架构总览

```
                        ┌────────────────────────────────────────────┐
token ids (B,L) ──► 冻结基座 SimplCoreModel（eval 常驻，requires_grad=False）
                        │   tok_emb → [SGC Block]×N → ln_f
                        ▼
                   h = ln_f 隐状态 (B,L,D)
                        │
                        ▼
              ┌── ExpertRouter（线性投影，O(1) 状态）─────────────────┐
              │ logits = proj(h) / temperature                        │
              │ top-k 软选择：w = scatter(topk(logits), softmax(vals))│
              │ aux = lb_loss·负载不均衡 + z_loss·logit 平方均值       │
              │ domain=int/(B,)/(B,L) 时强制路由（隔离训练）           │
              └───────────────────────────────────────────────────────┘
                        │  w (B,L,E) 稀疏（top-k 处非零）
                        ▼
              ┌── 逐专家条件计算（只算被选中的 token，O(active)）──────┐
              │ for e in experts: mask = w[...,e]>0                    │
              │   h_sel = h_flat[active_idx]                           │
              │   he = Expert_e(h_sel)          # gated MLP 栈，可微    │
              │   contrib = index_add(contrib, idx, w_sel·he)          │
              └───────────────────────────────────────────────────────┘
                        │  h2 = h + contrib.reshape(B,L,D)
                        ▼
              复用基座 head（tie_embeddings 时与 tok_emb 绑定）→ logits
```

**前向是一条残差加性路径**：专家输出经路由权重加权后**加回**基座隐状态
（`h2 = h + Σ w·Expert(h)`），不是替换——与 CMT memory_in / sidecar 注入
同一纪律：专家关闭（w=0）时模型退化为纯基座。

---

## 3. 组件精讲（按模块）

### 3.1 `config.py` — UnityConfig / ResolvedUnityConfig

```python
@dataclass
class UnityConfig:
    num_experts: int = 4        # 专家数 E
    expert_hidden: int = 0      # 0 → 2*dim（构造时解析）
    expert_layers: int = 2      # 每专家的 gated MLP 层数
    top_k: int = 2              # 每 token 激活的专家数
    router_hidden: int = 0      # 0 → dim（单层路由）；>dim 则两层
    capacity_factor: float = 1.25
    router_z_loss: float = 1e-3      # 路由 logit 平方均值正则（防熵崩）
    load_balance_loss: float = 1e-2  # 专家利用率均衡正则
    expert_dropout: float = 0.0
```

- `resolve(dim)` 把 `0` 占位字段解析为实际宽度，并做 `top_k ≤ num_experts`
  截断——**所有下游组件只消费 ResolvedUnityConfig**，不重复解析。
- **主干预设**（`UNITY_PRESETS`）：`tuan-2b` / `tuan-1b` 均为
  4 专家 × top-2 × 2 层。dim=2048、hidden=4096 时每专家 ≈50M，
  4 专家 ≈201M 可学——落在 WeightBudget 的「2B 可学 / 3B 总上限」内。
  `for_preset(未知名)` 回落默认**不静默改变预算口径**（文档原话）。

### 3.2 `router.py` — ExpertRouter：token 级 top-k 软选择

路由是**线性投影**（可选一层隐藏），O(1) 状态、可流式，与 SGC 非注意力
哲学一致。前向返回三元组：

```python
w, idxs, aux = router(h, domain=None)
# w    : (B,L,E) 稠密权重（top-k 处为 softmax(vals)，其余 0）
# idxs : (B,L,k) 被选中的专家索引
# aux  : {"lb": 负载不均衡标量, "z": 路由 logit 平方均值}
```

**数学细节**：

1. **top-k 软选择**：`vals, idxs = topk(logits, k)`，然后
   `w.scatter_(-1, idxs, softmax(vals, -1))`——被选中的 k 个专家按
   softmax 归一分配权重，未选中处严格 0。这保证 (a) 每行权重和 = 1、
   (b) 梯度只流向被选中专家（稀疏激活）。
2. **aux loss 两件套**（由 `UnityModel.forward` 按系数合成进总 loss）：
   - 负载均衡 `lb`：惩罚专家利用率方差——防止路由门把所有 token 都送给
     同一个专家（MoE 经典塌缩问题）；
   - z-loss `z`：惩罚路由 logit 平方均值——防止 logits 无界增长导致
     softmax 饱和、梯度消失（ST-MoE 论文的正则）。
   默认系数 `load_balance_loss=1e-2`、`router_z_loss=1e-3`。
3. **domain 强制路由**（域隔离训练的机制核心）：`domain` 为
   `None`（MoE 自由路由）或 `int` / `(B,)` / `(B,L)` 长整型——**强制把
   token 路由到指定专家**（`_domain_weights`：指定专家拿满权重，其余置 0）。
   这是「每个专家训练数据必须不同」的落地：`train_expert(i, loader)` 内部
   调 `forward(domain=i)`，训练期间只有专家 i 参与，其他专家零梯度。
4. **温度**：`router.temperature` 默认 1.0（逐位不变）。被生物计算主义
   耦合器（`bio_computation.BioEnhancer`）动态调节——连续场能量 / 课程
   进度（meso/macro）→ 离散路由（micro）的跨尺度约束接口。
5. `router_hidden > dim` 时路由变为两层 MLP（`Linear(D,H) → Linear(H,E)`，
   无偏置、无激活——保持线性可流式）。

### 3.3 `expert.py` — ExpertMLP / Expert：域专属小模块

```python
class ExpertMLP(nn.Module):     # 单层：norm → silu(gate(x))⊙up(x) → down + 残差
    out_scale = nn.Parameter(torch.ones(1))   # 残差缩放（可学）
class Expert(nn.Module):        # ExpertMLP × expert_layers
```

- **gated MLP 与基座 FFN 同构**（SwiGLU 族），但输出走**残差加性**——
  专家初始化随机时会给基座注入噪声，因此 `out_scale` 可学、配合路由
  权重共同决定扰动强度；
- **显式不引入注意力**：O(1) 状态、4M 上下文下状态仍 O(1)，
  `purity.py` 纯洁性守卫不受影响；
- `num_params()` 供预算账本调用。

### 3.4 `model.py` — UnityModel：粘合模型（核心，421 行）

#### 3.4.1 构造与预算

```python
UnityModel(base: SimplCoreModel, cfg: UnityConfig,
           base_params=None, budget: WeightBudget = None)
```

- 基座冻结三件套：`requires_grad_(False)` + `eval()` + DDP 时 `param_filter()`
  只含路由+专家（显存只随可学权重增长）；
- `dim` 自动推断（`base.config.dim` 优先，回退 `tok_emb.weight.shape[1]`）；
- `_add_expert` 是**唯一**的专家入口：先查
  `WeightBudget.can_allocate(num_params)`（2B 可学上限 **且** 3B 总上限
  双重检查），再 `allocate`——**所有生长路径（构造/spawn/import）都经过
  这里，预算守卫不可能被绕过**；
- `self.head = base.head`：复用基座词表头（tie_embeddings 时与 tok_emb
  绑定——专家调制的隐状态最终仍由基座词表解码）。

#### 3.4.2 前向的梯度工程细节（`forward`）

```python
for e in range(self.router.num_experts):
    we = w[..., e]                          # (B,L)
    mask = we > 0
    if not mask.any(): continue
    active_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    h_sel = h_flat[active_idx]              # (N,D) 只取被选中 token
    he = self.experts[e](h_sel)             # 可微
    contrib = torch.index_add(contrib, 0, active_idx, we_sel * he)
h2 = h + contrib.reshape(h.shape)
```

**为什么用 `index_add` 散射累加而不是就地索引赋值**（`h_flat[mask] = …`）：
对无梯度张量做就地索引赋值会**静默切断梯度**（这是复现实验中 mem_Wq
梯度恒 0 的同族问题）——`index_add` 是可微散射算子，`we_sel` 作为乘数
让路由权重也有梯度。**这条注释是 UnityModel 里最重要的工程决策**。

返回模式矩阵：

| 调用 | 返回 |
|------|------|
| `forward(x)` | `logits` |
| `forward(x, targets)` | `(logits, loss)`——loss = CE + lb·lb_loss + z·z_loss |
| `forward(x, targets, return_aux=True)` | `(logits, loss, aux)` |
| `forward(x, return_hidden=True)` | `(logits, h_base)`（基座 ln_f 隐状态） |
| `forward(x, return_expert_hidden=True)` | `(logits, h2, aux)`（**专家调制后**隐状态，供 BioEnhancer 连续场调制） |

#### 3.4.3 流式：O(1) 递推（生成期也带 MoE）

- `init_stream(batch)`：直接透传基座流式状态（路由/专家是逐 token 的
  无状态计算，**不需要**额外流式状态——这是 MoA 相对标准 MoE 的又一
  简化：没有专家容量/路由缓存需要维护）；
- `stream_step_hidden(token, state)`：基座 O(1) 递推到 ln_f → 路由 →
  专家调制 → 返回 h2（与 `forward(return_expert_hidden=True)` 的逐 token
  等价形式，供 BioEnhancer 施加连续场调制）；
- `stream_step(token, state)`：+ head → 下一 token logits；
- `generate(idx, …)`（0.1-α 补齐）：prefill（`stream_step` 逐步）+ 采样
  循环，复用基座 `_sample`（温度/top-k/top-p 口径与 `SimplCoreModel.generate`
  一致），确定性可测（同 seed 贪心两次逐位一致，`test_generate_o1_streaming`）。

#### 3.4.4 域隔离训练（`train_expert`）

```python
avg_ce = unity.train_expert(i=2, loader=domain_c_loader,
                            optimizer=AdamW(unity.experts[2].parameters()),
                            steps=200, device=dev)
```

内部流程：全部参数 `requires_grad_(False)` → **仅专家 i** 置 True →
`forward(domain=i)`（强制路由，数据与专家一一对应）→ 训练 → 恢复
「基座冻结 + 全专家/路由可学」的主循环状态。**域隔离是硬保证**：
其他专家在这个 loader 上梯度恒 0（路由被 domain 强制 + 未选中专家零权重）。

#### 3.4.5 自进化生长接口

- `spawn_expert(name, hidden, layers, dropout) -> int`：预算守卫下追加
  零/小随机初始化的新专家（「模型学到新域 → harness 长出新模块」的入口）；
- `publish_expert(candidate, name) -> int`：把**独立训练完成的候选专家**
  原子换入活动模型（`UnitySelfEvolution` 的最终一步；预算守卫同样在
  `spawn_expert` 内部执行）。

#### 3.4.6 持久化：只存可学部分

- `save_learned(path)`：路由 + 全部专家 + expert_meta + base_fingerprint +
  预算账目。**基座不重复存**（按预设恢复）——1B 基座的 checkpoint 由
  基线管理，Unity 快照只有几百 MB；
- `load_learned(base, path, cfg=None)`：从快照恢复；**专家数按快照决定**
  （`spawn_expert` 长出的专家也能恢复），预算账目逐字段还原。

#### 3.4.7 无竞态后台训练的三个原语

```python
train_standalone_expert(base, head, candidate, loader, optimizer, steps)  # 候选训练
evaluate_standalone(base, head, expert_or_None, batches)                  # token 加权 CE
publish_expert(candidate)                                                 # 原子换入
```

`train_standalone_expert` 在**独立候选对象**上训练——不触碰活动模型参数，
后台训练与主线前向/推理互不竞态（不需要锁推理路径）。

### 3.5 `bundle.py` — ModuleBundle：模块级最小可搬运单元

`save_learned` 是整机快照；`ModuleBundle` 解决三件它做不到的事：

1. **单专家搬运**：把一个域专家从 2B 主干搬到另一个同构主干；
2. **增量交换**：后台训练完成后只交换某一个模块；
3. **来源审计**：记录「这个模块在哪个基座、多少 token、什么 val loss 上
   训出来」。

```python
bundle = unity.export_expert(2, train_tokens=1_200_000, val_loss=0.87)
bundle.save("experts/finance.pt")
# ……另一个进程 / 另一个同构主干：
bundle = ModuleBundle.load("experts/finance.pt")
idx = unity.import_bundle(bundle, name="finance")
```

- **`base_fingerprint(cfg)`**：结构字段指纹（dim/num_layers/num_bases/
  local_kernel/mixer/norm 的 SHA1[:12]）——**故意不含词表**：专家只依赖
  dim 与残差流语义，不同词表的主干仍可拼接；
- **`compatible_with(unity)`**：dim 一致 + 指纹一致才放行；`import_bundle`
  在不兼容时**抛错拒绝**，而不是让异构主干的专家静默污染路由；
- `audit()`：不含权重的可审计摘要（参数量 + 全部元数据）。

### 3.6 `experience.py` — ExperienceLoader：经验 → 数据面

自进化 = 经验采集 → **本 loader** → 候选训练 → 无回归门 → 发布/回滚。
这里只负责第一步的数据面：

- 输入 = 某域的经验文本（`SMM.experience` 落下的 prompt/response、代码
  沙箱成功轨迹、环境交互日志）；
- `codec.encode` → `seq_len+1` 滑窗（stride 可重叠）→ `(x, y)` next-token 对；
- `repeat` 过采样 / `max_windows` 防“一个域无限膨胀”；
- **域隔离从数据源头保证**：docstring 原话「不要与其他域混用；域隔离是
  专家分工的前提」；
- `stats()` 给出 texts/windows/batches 账本。

### 3.7 `evolution.py` — UnitySelfEvolution：后台自进化管理器

```python
evo = UnitySelfEvolution(unity, device=dev, steps_per_domain=200, num_workers=2)
evo.start()
evo.register_domain("python_syntax", train_loader, eval_loader, max_regression=0.02)
evo.register_domain("bash_cli",     train_loader, eval_loader, max_regression=0.02)
evo.stop()                     # 排空队列后退出
evo.save("runs/tuan-1b/unity.pt")
```

**单域生命周期**（`_train_one`，worker 线程内）：

1. **独立候选专家**：`type(experts[0])(dim, hidden, layers, …)` 全新构造
   （不参与活动模型前向 → 无竞态）；
2. `train_standalone_expert`：`steps_per_domain` 步域专属训练；
3. **GATE-3 无回归门**：若提供 `eval_loader`（必须是**可重复**的 batch
   序列）与 `max_regression`（默认 0.02）——在同一集合上比较
   「无专家基线 CE」vs「候选专家 CE」，**候选 CE 上升超过阈值即拒绝
   发布**（活动模型不变，域名进 `_rejected`）。`max_regression=None`
   关闭门控（旧行为，不推荐）；
4. `publish_expert` 原子换入（锁保护专家列表变更）→ 路由门随后在
   MoE 自由路由下逐步学会在新域激活它。

**并发与容错**：`num_workers` 线程池消费任务队列；单域训练异常**不拖垮
worker**（域名进 `_errors`，打印后继续）；`status()` 给出
pending/done/rejected/errors/num_experts/budget 全账本。
**生产建议**（docstring 原话）：线程版是进程内并发；跨进程/多卡需把
worker 改为独立进程 + 队列/共享存储传候选状态。

### 3.8 `envlearn.WeightBudget` — 三层预算账

```
base_params   = 1e9    （冻结基座，如 1B）
learned_params ≤ 2e9   （路由+专家累计分配）
total_cap     = 3e9    （base + learned 总上限）
```

`can_allocate(n)` 双重检查（可学上限 **且** 总上限）；`remaining_learned()`
/`remaining_total()` 供 UI/日志展示；账目随 `save_learned`/`load_learned`
持久化——**自进化的生长预算跨会话累积，重启不会重置**。

---

## 4. 端到端使用指南

### 场景 1：快速起步（MoE 前向 + 域隔离训练）

```python
import torch
from simplcore.core.model.config import SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel
from simplcore.unity.config import UnityConfig
from simplcore.unity.model import UnityModel

base = SimplCoreModel(SimplCoreConfig(vocab_size=32000, dim=2048,
                                      num_layers=31, num_bases=128)).eval()
unity = UnityModel(base, UnityConfig(num_experts=4, top_k=2))

# 域隔离训练：专家 2 只见"财务报告"域
opt = torch.optim.AdamW(unity.experts[2].parameters(), lr=1e-3)
avg_ce = unity.train_expert(2, finance_loader, opt, steps=200, device=dev)

# MoE 自由路由前向（推理）
logits, loss, aux = unity(x, y, domain=None, return_aux=True)
print("负载不均衡:", aux["lb"], "| z:", aux["z"])

# 生成（O(1) 流式 MoE）
out = unity.generate(prompt_ids, max_new_tokens=64, top_k=50)
```

### 场景 2：后台自进化（经验 → 新专家 → 门控发布）

```python
from simplcore.unity.evolution import UnitySelfEvolution
from simplcore.unity.experience import ExperienceLoader

evo = UnitySelfEvolution(unity, device=dev, steps_per_domain=200, num_workers=2)
evo.start()
for name, texts in domains.items():          # 每域经验文本互不混用
    loader = ExperienceLoader(codec, texts, seq_len=512, batch_size=4,
                              max_windows=2000)
    eval_batches = list(ExperienceLoader(codec, eval_texts[name],
                                         seq_len=512, batch_size=4))
    evo.register_domain(name, iter(loader), eval_batches, max_regression=0.02)
evo.stop()
evo.save("runs/tuan-1b/unity.pt")
print(evo.status())     # pending/done/rejected/errors 全账本
```

### 场景 3：跨主干搬运（专家导出/拼接）

```python
bundle = unity.export_expert(2, train_tokens=1_200_000, val_loss=0.87)
bundle.save("experts/finance.pt")

# 另一个同构主干（dim 相同；词表可以不同）
unity_b.import_bundle(ModuleBundle.load("experts/finance.pt"), name="finance")
```

### 场景 4：手动生长 + 持久化

```python
idx = unity.spawn_expert(name="rust_cli")          # 预算守卫内追加
unity.train_expert(idx, rust_loader, opt_rust, steps=300, device=dev)
unity.save_learned("runs/tuan-1b/unity.pt")        # 只存可学部分
# ……重启后（基座由调用方重建）：
unity = UnityModel.load_learned(base, "runs/tuan-1b/unity.pt")
```

### 场景 5：流式生成（MoE + O(1)）

```python
state = unity.init_stream(1, device=dev)
for t in range(prompt_len):
    logits, state = unity.stream_step(prompt_ids[:, t], state)
# 逐 token 解码 out——状态不随上下文增长（4M 下仍 O(1)）
```

---

## 5. 配置参考

### UnityConfig 字段

| 字段 | 默认 | 说明 |
|------|------|------|
| `num_experts` | 4 | 专家数 E（spawn 可超过初始值，受预算约束） |
| `expert_hidden` | 0→2·dim | 专家 MLP 隐宽 |
| `expert_layers` | 2 | 专家层数 |
| `top_k` | 2 | 每 token 激活专家数（≤E 自动截断） |
| `router_hidden` | 0→dim | 路由隐宽（>dim 时两层路由） |
| `capacity_factor` | 1.25 | 容量因子（记录用；当前实现按 mask 全激活） |
| `router_z_loss` | 1e-3 | 路由 logit 平方正则系数 |
| `load_balance_loss` | 1e-2 | 负载均衡正则系数 |
| `expert_dropout` | 0.0 | 专家内 dropout |

### 预算三层账（WeightBudget 默认）

| 字段 | 默认 | 含义 |
|------|------|------|
| `base_params` | 1e9 | 冻结基座 |
| `max_learned` | 2e9 | 可学参数上限（路由+专家累计） |
| `total_cap` | 3e9 | base+learned 总上限 |
| 守卫 | — | `can_allocate(n)`：n 同时不超 `max_learned-learned` 与 `total_cap-base-learned` |

### UnitySelfEvolution 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `steps_per_domain` | 200 | 每域候选训练步数 |
| `lr` | 1e-2 | 候选专家 AdamW 学习率 |
| `num_workers` | 1 | 后台线程数 |
| `register_domain(max_regression)` | 0.02 | GATE-3 阈值：候选 CE 超基线 + 阈值即拒绝；`None` 关门（不推荐） |

---

## 6. 测试矩阵（守护什么）

| 测试 | 锁定语义 |
|------|----------|
| `test_base_frozen_excluded` | 基座参数全部 requires_grad=False、`param_filter()` 不含基座 |
| `test_experts_different_data_specialize` | 域 A/B 各自训练后，专家 0 在域 A 的 CE 严格低于域 B（域专属特化成立） |
| `test_router_gradient_flows` | MoE 自由路由下梯度能流向路由门 |
| `test_budget_guard_blocks_overspawn` | 超 3B 上限时 `spawn_expert` 抛错（harness 生长受约束） |
| `test_expert_persistence` | `save_learned`/`load_learned` 往返一致（含 spawn 长出的专家） |
| `test_self_evolution_manager` | 队列/worker/status 账本/发布路径 |
| `test_warmup_domains_specialize` | 域隔离 warm-up 后 `warmup_unity_domains` 的特化断言 |
| `test_generate_o1_streaming` | `generate` 输出形状/前缀保留/确定性（O(1) 流式 MoE） |
| `test_simplunity_modular.py` | bundle 导出/拼接/指纹校验/预算集成 |

---

## 7. 诚实边界

1. **域专属特化已在合成域上验证**（`test_warmup_domains_specialize`：
   warm-up 后 expert0 在域 A 的 CE < 域 B）；**真实域（代码/财务/工具调用）
   的特化质量未验证**；
2. **`spawn_expert` 的随机初始化**：新专家初始会给基座隐状态注入噪声
   （out_scale=1 + 随机 MLP），发布前必须经 GATE-3 或显式训练；
3. **`capacity_factor` 当前是记录字段**：实现按 top-k mask 全激活
   （token 级路由没有 token 丢弃），与标准 MoE 的 token dropping 不同——
   极端负载不均衡时显存峰值按最坏专家计；
4. **线程级并发**：`UnitySelfEvolution` 是进程内线程池；跨进程/多卡
   的候选训练需自行改造 worker（docstring 有生产建议）；
5. **路由冷启动**：新发布的专家初始不会被 MoE 路由偏好——需要 MoE
   自由路由的继续训练（或域内推理流量）让路由门学会激活它；
   `register_domain` 只保证「候选不劣于基线」，不保证「路由会选它」；
6. **与 §7.4 W1–W3 的关系**：W1（显式底架增量）/W2（verifier 无冲突）/
   W3（巩固回滚必要性）三个可证伪判据均未运行——SimplUnity 当前是
   **机制完整、判据待跑**的状态。

---

## 8. 与 Paper / 三个月路线的关联

- `SimplCore_Paper.md` §3.5(e)：SimplUnity 是「1B 冻结主干 + ≤2B 自进化
  适配器」完整形态的工程载体；
- §7.2 轴 E：域内代码能力「+1–2B / 域外 ≈0」的正确定义域——SimplUnity
  的专家就是那个「域内增量」；
- §7.4 W1–W3：SimplUnity 的 GATE-3/预算守卫/经验回流是三个判据的
  被测机制；
- `docs/SIMPLUNITY_3MONTH_PLAN.md`：2B 主干（1 月）+ 自进化/模块
  （2 月）的 GCD 分工与硬停日期。

## 9. API 速查

```python
# 配置
UnityConfig(num_experts=4, top_k=2, expert_layers=2)
for_preset("tuan-2b")                    # 主干预设
# 模型
UnityModel(base, cfg)                    # 冻结基座 + 路由 + E 专家
unity.forward(x, y, domain=None|int)     # MoE / 域隔离前向
unity.train_expert(i, loader, opt, steps)
unity.spawn_expert(name)                 # 预算守卫生长
unity.publish_expert(candidate)          # 原子换入
unity.export_expert(i, **meta) / import_bundle(bundle)
unity.save_learned(p) / UnityModel.load_learned(base, p)
unity.generate(idx, max_new_tokens=…)    # O(1) 流式 MoE 采样
unity.stream_step(token, state) / stream_step_hidden(token, state)
unity.as_dict()                          # 预算账本
# 自进化
UnitySelfEvolution(unity, num_workers=…).start/register_domain/stop/save
# 数据面
ExperienceLoader(codec, texts, seq_len=512, max_windows=…).stats()
# 模块包
ModuleBundle.load(p).audit() / .compatible_with(unity) / .save(p)
base_fingerprint(cfg)                    # 结构指纹
# 预算
WeightBudget(base_params=1e9, max_learned=2e9, total_cap=3e9)
```
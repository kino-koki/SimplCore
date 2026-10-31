# 架构补强方案（Track 3）

日期：2026-10-29 · 针对评估中提到的三类结构性短板给出**可证伪**的架构补强与对照方案。
不写训练代码，只出设计与 harness 草图。所有主张都必须能被 `campaign.py` 的
验收闸门拦截或放行。

> **状态注（2026-10-31）**：§1.3 的"分层状态（按时间尺度分桶的 IIR 组）"思想
> 已由 **SGC-R 谱专家路由**（`mixer="sgc-r"`，M 基按频率分 G 组 + 逐 token top-k
> 注入，见 `SGC_CORE.md` §10）以可学习路由的形式部分落地——比本文设想的静态
> 分桶更进一步；§1.2 的检索双塔仍未做。本文其余部分维持设计提案状态。

> 原则：SimplCore 的核心卖点（彻底非注意力、O(1) 流式、可证伪诚实纪律）**必须保留**。
> 补强只在「core 线保持纯谱卷积」的前提下做加法，不在 core 线里塞注意力。

---

## 1. O(1) 状态信息瓶颈（轴 D：精确召回上限）

### 问题本质
O(1) 常驻状态是信息论**上界**（论文定理 3 / P10）：长程路径的容量被冻结在
`2·M·D` 个递推量里，与序列长度无关。这意味着「needle-in-haystack」式的精确
逐位召回**与参数量无关地必输**任何能精确注意的 TS——这是数学事实，不是 bug，
不能「修」只能「绕」。

### 补强方案（不污染 core 纯度）
1. **CMT 记忆注入已存在，需补齐评测**：`memory.py` / `dream.py` 的压缩记忆库是
   正确的补偿方向（把「长程精确信息」外置到可检索记忆，残差流加性注入）。
   **缺口**：没有针对「长上下文精确召回」的对照测试。补 `longctx_eval.py` 的
   S7 诚实账，把「聚合型任务（4M）强、精确 needle 弱」量化出来，作为已知边界公示。
2. **谱卷积 + 检索双塔**：在 core mixer **之外**加一个轻量「检索路由器」
   （cheap kNN / BM25 over chunked memory），命中位直接拼进残差流。
   谱全局卷积负责「平滑长程趋势」，检索负责「精确指针」——两者正交，
   **检索路由不引入注意力、不进 mixer**，core 纯度不变。
3. **分层状态（可选）**：把 O(1) 状态按时间尺度分桶（快/中/慢三组 IIR，
   不同 ρ 窗口），在不破坏 O(1) 的前提下扩大有效记忆跨度。这是
   `decay_max` / `num_bases` 的工程延伸，已有基础（`_init_oscillator_grid`）。

### 可证伪判据
- 若加上检索双塔后，`longctx_eval` 的精确 needle 召回相对纯谱卷积 **提升 < 5pt**，
  说明瓶颈在检索而非容量 → 该补强无效，撤回。
- 若「聚合型 4M 任务」指标**不退化**，则双塔是纯增益。

---

## 2. 路由缺陷（轴 D：带限谱卷积无法精确「指位」）

### 问题本质
带限谱卷积的频域支撑有限，对「定位第 k 个 token / 精确跳到某位置」类任务
（路由型）表现弱（论文数据 0.24–0.27 vs 递归 0.73–0.76）。这直接催生了
**SGC-Hybrid（滑窗注意力）**——但 Hybrid 又把注意力请了回来。

### 补强方案：split the claim，不要在 core 线上妥协
1. **core 线（sgc / srf / gated-spectral）保持彻底非注意力**，对外一律宣称
   「无注意力、无 KV cache」。这是产品线纯洁性的唯一红线（`purity.py` 守护）。
2. **Hybrid 线独立成产品线**，明确标注「含滑窗注意力」，不共享 core 线的
   「彻底非注意力」卖点。两线不得混同宣传（README §6 已部分做到，需强化措辞）。
3. **Hybrid 的注意力必须是「局部滑窗」而非全局**：全局注意力会重新引入
   O(L²) 与 KV cache，违背流式 O(1) 优势。滑窗宽度作为唯一注意力超参，
   纳入 `deploy_budget.py` 的 HBM 账本（滑窗 KV 必须计入，不能只算 O(1) 状态）。
4. **检索路由优先于注意力**：先试 §1.3 的检索双塔解决指位需求；
   只在检索无解的路由型任务才落到 Hybrid 注意力。把「注意力」降级为
   最后手段，而非默认补丁。

### 可证伪判据
- Hybrid 线在路由型任务上若仍 < 递归基线 0.7，则滑窗注意力窗口/实现有误，
  需回炉；core 线对此类任务**本就不承诺**，不计入 core 验收。

---

## 3. §7 等效参数推测 → 改成可测量

### 问题本质
论文 §7 的「4–7B / 中位 5B 等效参数」是**外推 + 公开证据猜测**，每条附可证伪
判据但**尚未跑**。把它当事实传播是危险的（评估已指出）。

### 补强方案：用 `equal_budget.py` 把猜测变成测量
1. **缩放律消融（新增 `scaling_law.py`）**：在 tiny(30M)→small(150M)→medium(350M)
   三档跑同 token 预算，拟合 `loss = a·N^(-α) + b` 的参数-损失曲线；
   用外推区间**预测** 1B/3B 损失，再与实际 1B 训练结果比对，给出外推误差带。
   这把「等效参数」从拍脑袋变成有误差棒的外推。
2. **复用既有 `X-equal-budget` 闸门**：`equal_budget.py` 已完整（TS 对照臂、
   逐窗配对 bootstrap CI、`unqualified_claim_permitted` 机器可读开关）。
   **缺口**：只在 `smoke`/`small` 验证过，没在 1B 跑。补 `1b` preset 的
   真实语料对照，输出 `runs/eval_equal_token.json` 进战役账本。
3. **诚实账分级**：
   - 已测（tiny/small）→ 可带预算口径宣称；
   - 外推（1B/3B，有误差带）→ 标注「外推，误差带 X」；
   - 未测（>3B）→ 一律不得宣称。

### 可证伪判据
- 若 1B 实测损失落在外推带外（>2σ），则缩放律假设失效，§7 结论撤回并修正。
- `X-equal-budget` 闸门未 PASS 时，`unqualified_claim_permitted=False` 强制
  所有对外结论带预算口径与 CI。

---

## 4. 等预算对照 harness 扩展草图

在 `equal_budget.py` 之上补两类实验（不破坏现有契约）：

```
# 扩展 A：路由型任务对照（补强 §2）
python -m simplcore.core.equal_budget \
    --preset 1b --mode equal_token --corpus data/processed/routing \
    --out runs/eval_routing.json
# 实验臂：core(tuan-1b) vs core+retrieval-router vs sgc-hybrid(滑窗)

# 扩展 B：缩放律消融（补强 §3）
python -m simplcore.core.scaling_law \
    --presets tiny small medium --tokens 2_000_000 \
    --out runs/scaling_law.json
# 拟合 N^-α，外推 1B/3B，与实际 1B 对照
```

两路输出都汇入 `campaign.py` 的验收闸门（已有的 `X-equal-budget` /
`X-longctx-retrieval` / `X-heldout-ppl`）。**任何结论在被这些闸门放行前
不得对外宣称**——这正是 README 诚实纪律的落地。

---

## 5. 与 Track 4（闸门可控化）的关系
Track 4 已把 L2/L3 闸门默认关、改为监督可选（`config.py` `evidence_gate` /
`uncertainty_gate` 默认 `False` + `GateConfig` 收敛超参）。本方案的「双塔检索路由」
若采用，其**门控同样默认关、需监督**——与 Track 4 同一条纪律，避免新增一个
「untrained 即空转」的模块。

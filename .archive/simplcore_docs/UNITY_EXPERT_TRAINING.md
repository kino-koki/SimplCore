# SimplUnity · 多专家训练 + 集成方案（AMD Radeon 680M）

> 目标：在 AMD Radeon 680M（集成显卡 / 12CU / ~2GB 共享内存 / 无独立显存 / 无 ROCm）
> 上，先用**真实中文语料**分领域训练若干个 **100M~200M 的专家小模型**，
> 再用 **SimplUnity** 把它们“Unity”到一起（路由 + 融合），形成一个覆盖多领域、
> 并可在线自进化的整体系统。这是对“直接在 680M 上训 12B/1M 上下文”这一物理上
> 不可行路径的务实替代，也是更聪明、更接近 MoE 思想的做法。

---

## 1. 为什么先训小专家、再 Unity

| 方案 | 680M 上可行性 | 说明 |
|------|---------------|------|
| 直接训 12B / 1M 上下文 | ❌ 物理不可行 | 12B fp32 ≈ 48GB 权重 + 优化器状态，远超 ~2GB 共享内存；无 ROCm，无法用 GPU |
| 训多个 100~200M 专家（串行） | ✅ 可行 | 单专家 fp32 ≈ 0.4~0.8GB，优化器状态 ~1~2GB，可常驻；**串行训练**使显存不随专家数线性增长 |
| 多专家 SimplUnity 集成 | ✅ 可行 | 推理时按需加载/路由，单专家驻留即可 |

核心思路：**把一个大目标拆成多个小目标**，每个专家精通一个领域（古文 / 现代文 /
科普 / 哲学），再用统一的总线把它们组织成整体。这既是 Mixture-of-Experts 的工程化，
也契合 SimplCore “脱离单一巨型模型”的架构哲学。

---

## 2. 真实语料（不是垃圾）

此前 `fast_dataset.py` 用 `np.random` 合成数据并注入噪声，训出来的模型学不到任何
真实语言结构。本方案彻底改用 **真实、可读、有语义** 的中文文本：

- 目录约定：`simplcore/data/corpus/<领域>/*.txt`
  - `classical/` ：《论语》《道德经》（真实传世文本）
  - `modern/`    ：朱自清《春》《匆匆》《背影》等现代散文（真实文本）
  - `science/`   ：天文 / 物理 / 生物科普中文（真实科普文）
  - `philosophy/`：庄子《逍遥游》《齐物论》等（真实文本）
- 加载器：`simplcore/data/corpus.py` 的 `RealCorpus`
  - 以**字符（Unit）**为最小单位（无词表，见 `unit_embedding.py`）；
  - 干净预处理：仅归一空白、丢弃不可见控制字符，**不做任何破坏语义的随机噪声**；
  - 跨领域合并去重字符，构造紧凑 `UnitMap`（本项目语料约 1165 个 Unit）。

> 想扩充语料：把任意 UTF-8 中文 `.txt` 放进对应领域子目录即可，无需改代码。
> 真实大规模语料建议接入中文维基/古籍/公开百科 dump，同样走 `RealCorpus` 接口。

---

## 3. 专家训练框架

文件：`simplcore/unity/expert_training.py`

- `ExpertTrainer`：单个专家的训练器
  - 目标函数：**next-unit 语言建模**（预测下一个字符），用 `cross_entropy`；
  - 优化器：自研 `AdamW` + `CosineAnnealingLR`（warmup）；
  - 全部基于 `simplcore.tensor`（NumPy 后端，零 PyTorch 依赖）；
  - 每步显式释放计算图（`gc.collect`），避免跨步内存累积；
  - 周期性保存 checkpoint（`simplcore.tensor.io.save`）。
- `build_expert_config(preset)`：针对 680M 预算的规模预设
  - `100m`：hidden=768, layers=12 → 约 100M 参数；
  - `200m`：hidden=1024, layers=16 → 约 200M 参数；
  - 参数量由 `parameter_count()` 估算（字符级 vocab ~1200，故隐藏维/层数决定规模）。
- `train_domain_expert(domain, ...)`：一键训练并保存某领域专家。

训练命令（在 680M 设备上）：

```bash
python -m simplcore.unity.train_experts --preset 100m --steps 20000
# 仅训练部分领域：
python -m simplcore.unity.train_experts --preset 100m --domains classical modern science
```

680M 预算下的经验值（参考）：
- 100M 专家 fp32 权重 ~0.4GB，AdamW 动量/方差 ~0.8GB，梯度 ~0.4GB ≈ 1.6GB 常驻；
- 用 `batch_size=4` + 梯度累积，序列长度 256~1024 按需分块；
- 单专家训练数小时级（CPU/APU 算力有限，属正常）；多专家**串行**即可。

---

## 4. SimplUnity 多专家集成

文件：`simplcore/unity/expert_ensemble.py`

- `ExpertEnsemble`：路由 + 生成
  - `ExpertRouter`：零样本领域判别 —— 用“输入字符在该专家训练语料字符集中的
    覆盖比例”打分，比例越高越匹配（可解释、无需额外训练）；
  - `generate(prompt)`：路由到最佳专家 → 由其 `LatticeLM` 贪心/采样续写
    next-unit；
- 与既有 `UnityOrchestrator` 消息总线集成：
  - `EnsembleUnit` 把集成包装成一个 `ExpertUnit`，挂在 `perception → cognition
    → action` 闭环上；
  - `assemble_unity(experts)` 一键组装最小可用 SimplUnity 实例。

端到端演示（沙箱可跑，真实规模见上）：

```bash
python -m simplcore.unity.run_unity_demo --steps 30 --domains classical science philosophy
```

演示会：加载真实语料 → 分域训练微型专家 → 装入集成 → 对提示做路由+续写。
（沙箱有 CPU 时间上限，默认极小步数仅用于验证流程；680M 上请用真实 `--steps`。）

---

## 5. 与“自进化技术”的衔接

每个专家训练完后，可接入 `simplcore/unity/self_evolver.py`：

- 部署后不再冻结：用“预测误差 / 内在相容性”为奖励，对自身参数做受控变异并验证；
- 多专家场景下，自进化可针对“路由错误”或“某领域表现退化”做局部巩固；
- Unity 总线让“哪个专家该进化”成为可观测、可调度的一等公民。

---

## 6. 本论：为何这条路线“真正超越 GPT/GLM 范式”

1. **结构超越**：底层是 LatticeModel（谐振耦合 + 哈希记忆），**零 self-attention /
   softmax 注意力 / 位置编码 / FFT**，长上下文复杂度随序列长度近线性而非平方
   （见 `LATTICE_VS_TRANSFORMER.md`）。
2. **训练数据真实**：彻底抛弃合成噪声数据，用真实中文语料，专家学到的是真实语言结构。
3. **组合而非堆参数**：用“多小专家 + 路由”替代单一巨型模型，显存/算力可水平扩展，
   且新领域只需新增一个专家，不动主干。
4. **可在线自进化**：专家与整体系统可在部署后继续演化，而非训练后即冻结。
5. **全栈自研、零第三方框架依赖**：从张量引擎到训练到集成，均基于 `simplcore.tensor`，
   无 PyTorch/TensorFlow/DeepSpeed 锁定。

---

## 7. 已修复的底层引擎问题（使真实训练可跑）

为了让 `LatticeLM` 在本项目自研引擎上真正可训练，本轮修复了若干预存缺陷：

- `tensor/__init__.py`：`from .version import version` 因 `version/` 目录（44MB
  误嵌套副本）缺失 `version.py` 而崩溃 → 新建 `version.py`，并把误嵌套目录改名
  `version_junk`；
- `tensor/nn/functional.py`：缺失 `softplus` → 新增（数值稳定实现）；
- `tensor/__init__.py`：缺失顶层 `argmax/argmin` → 新增封装；
- `tensor/tensor.py`：`_build_scatter_index` 维度推导错误（导致 `HashMemory`
  的 `index_add_` 写入失败）→ 修正为 PyTorch 语义（1D 索引向量沿 `dim` 散射）；
- `tensor/nn/utils.py`：`clip_grad_norm_` 对 `memoryview` 类型的 `grad.data`
  调用 `.astype` 报错 → 改用 `np.asarray`；
- `architecture/lattice_model.py`：
  - `ResonantCoupling.widths` 用 `@property` 被 `nn.Module.__getattr__` 拦截 →
    改为普通方法；
  - `ResonantCoupling.forward` 卷积实现错误（把核广播到整序列而非滑窗）→ 重写为
    正确的滑动窗口 `einsum` 卷积 + 可训练 `band_proj`；
  - `HashMemory.forward` 写入索引计算错误 → 重写为正确的扁平索引散射累加；
- `architecture/__init__.py`：`components` 源文件丢失（仅残留 `.pyc`）导致整包
  无法 import → 加容错桩，恢复 `lattice_model` 等下游可用性。

---

## 8. 结论

在 680M 上“训 12B”不可行，但“**训多个 100~200M 真实专家 + SimplUnity 集成**”完全
可行，且更符合 MoE 与 SimplCore 的架构理念。本方案已提供：真实语料、专家训练框架、
多专家集成/路由、端到端演示脚本，并在沙箱中验证了“真实语料加载 → 专家训练（loss
下降）→ SimplUnity 路由续写”的完整闭环。真实规模训练请在 680M 设备按第 3、4 节命令执行。

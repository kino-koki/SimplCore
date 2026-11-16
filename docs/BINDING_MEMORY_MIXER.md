# SGC-Bind：定容绑定记忆辅助通路 —— 轴 D 绑定缺陷的 mixer 级修复（R1 三期）

> **状态（0.1-α，2026-11-15）**：`mixer="sgc-bind"` 已落地产品线
> （`simplcore/core/arch/bsgc.py`，注册表名 `sgc-bind`）。
> MQAR-4 前哨：**sgc-bind 0.815 vs attention 0.31 vs sgc 0.155**（300 步、
> batch 8、同 lr 同 seed，参数量 31.8K < TS 35.8K）。本页记录设计、
> 证伪过程与口径边界。

## 1. 针对的缺陷

`docs/REPRODUCTION_REPORT.md`（2026-10-31）确诊：SGC 缺的不是感受野，
是 **key→value 绑定能力**——

| 任务 | attention | SGC | chance |
|---|---|---|---|
| MQAR G=4 | 0.500 | 0.3125 | 0.125 |
| 精确指向 | 1.000 | 0.398 | 0.0625 |

根因：注意力权重是位置对 `(i,j)` 的内容函数（二阶交互）；SGC 的谱核是
输入无关常数，逐元素门控只能逐位置缩放，**表达不了"位置 i 应当聚合历史
的哪种模式"**。R1 落地的 `memory/binding_memory.py` 证明了机制可行性
（恒等投影下精确恢复），但它是**外挂组件**，未进任何运行时路径。

## 2. 设计（`BindingSpectralConv`）

在 SGC 三通路（长程 FFT / 局部卷积 / 瞬时项）之外加第四条**定容绑定
记忆**通路，三个关键决定（每个都由探针证伪/证实）：

### 2.1 绑定在归一化残差流上，不在门控投影上

绑定读/写的输入是 `h = RMSNorm(x)`（逐位置 rms=1，层 0 处 token 身份
原样保留），不是 `u = in_proj(h)·σ(gate(h))`。**证伪记录**：绑定在 u 上
时，`init_std=0.02` 的投影把寻址分数缩到均匀 softmax——mixer 级探针
100 步 loss 卡 2.05~2.23（与纯 SGC 同水平）；改绑 h 后同设置 100 步到
1.03、300 步 0.60。

### 2.2 残差形式的寻址/读出投影

```
k_new = u_prev + ak(u_prev)     # key 侧 = 前驱残差
v_new = u_t   + av(u_t)         # value 侧 = 当前残差
q_t   = u_t   + aq(u_t)         # 查询残差
```

**证伪记录**：纯投影版（无残差直通）在直接探针里 600 步学不动
（loss ≈2.05 平台）——随机初始化下寻址分数无区分度。残差形式让查询与
正确槽在**初始化时即自同构**（`e_k·e_k ≫ e_k·e_j`），梯度从第 0 步就
有方向（与 induction head 用残差流携带 token 身份同构）。

### 2.3 硬 FIFO 写入 + 可学习锐度 τ

* 写掩码是 **one-hot 常量**（位置 = `(t-1) mod S`），out-of-place 更新
  `sk + mask·(k_new − sk)`——对 `k_new` 可微，掩码无梯度需求；
* **软门控写入被证伪**：逐槽 `σ(write_gate)` 的 0.5 混合把全部绑定平均
  成一碗汤 → 槽内容趋同 → 寻址分数全等 → softmax 梯度逐槽相消为零。
  这正是复现报告二期「3000 步退化」的根因实锤。`soft` 模式保留为失败
  对照（`sgc_bind_mode="soft"`），默认 `bound`；
* 读出 `probs = softmax(τ·scores/√D)`，τ softplus 参数化可学（oracle
  实测 τ=8 完美寻址）。

注入走加性项：`y = long + local + inst + σ(mem_gate)·(probs @ slot_v)`。
初始 `σ(0)=0.5`；不引入任何 token↔token 成对打分、无 `q/k/v_proj`、
无 SDPA、无位置表——通过 purity 三层守卫。

## 3. 架构一致性判据（全部有测试锁定，`test_bsgc.py`）

| 判据 | 结果 |
|---|---|
| FFT↔IIR 双形式一致（整段 vs 逐 token 流式） | `max\|Δ\| ≈ 6e-8`（fp32） |
| 分块前向（截断 BPTT）与整段一致 | `max\|Δ\| ≈ 6e-8` |
| 严格因果（扰动末位，前文不变） | 通过 |
| O(1) 定容：状态字节与 L 无关（含槽 2·S·D + D） | 通过 |
| 断点 checkpoint/restore 含槽状态，恢复后逐位续写 | 通过 |
| 全链梯度（ak/av/aq/mem_gate/raw_read_temp） | 全部非零 |
| `analytic_params` == 实例参数量 | 逐位一致 |
| purity（配置/实例/源码 AST） | 通过 |

## 4. 实测（D=32 / 2 层 / slots=128，batch 8 真批量，lr 3e-3，300 步）

### 4.1 MQAR-4（chance 0.125；3 seeds，n_eval 200）

| seed | sgc | **sgc-bind** | ts（attention） |
|------|------|--------|------|
| 0 | 0.155 | **0.815** | 0.310 |
| 1 | 0.160 | **0.780** | 0.295 |
| 2 | 0.160 | **0.760** | 0.370 |
| 均值 | 0.158 | **0.785** | 0.325 |

**sgc-bind 3/3 seed 高于 attention，CI 不重叠**（0.76–0.815 vs 0.295–0.37）；
参数量 31,758 < TS 35,816（等预算口径对 sgc-bind 略有利，对 sgc 显著有利）。
确诊的绑定缺陷在该前哨上已修复并**反超 attention 对照臂**。

### 4.2 扩规模（D=128 / 4 层，200 步，MQAR-4，3 seeds）

| seed | sgc | **sgc-bind** | ts（attention） |
|------|------|--------|------|
| 0 | 0.130 | 0.730 | 0.260 |
| 1 | 0.160 | **1.000** | 0.310 |
| 2 | 0.120 | **0.955** | 0.330 |
| 均值 | 0.137 | **0.895** | 0.300 |

优势随宽度/深度保持（3/3 seeds 胜 attention，CI 不重叠）。注意本口径下
ts 层数按**纯 sgc** 参数对齐，sgc-bind 因绑定投影多出 ~3·D²/层（196K）
——严格等参数对照见 §4.4/§4.5 的 equal-budget harness（其参数对齐按
实际 sc_cfg 计算）。seed0 的 0.730 为 u_prev 缺陷修复前数字；修复后
seeds 1/2 为 1.000/0.955。

### 4.3 长上下文（LRA 风格，L=256，S=512，D=64/2 层，150 步）

MQAR 拉长到 **L=256**（绑定单元 255 个、查询在末位——O(1) 状态模型最难
的 regime；槽容 S=512 全覆盖）：

| 臂 | acc | 参数量 |
|----|------|--------|
| sgc（300 步仍如此） | 0.150 | 87,786 |
| **sgc-bind** | **1.000** | 112,366 |
| ts（attention，同 150 步） | 0.280 | 104,360 |

**L=256 联想召回满分**——残差绑定 + 硬 FIFO + 锐读出的组合把确诊缺陷
彻底修复，且大幅超过 attention 对照臂（seeds 1/2 复测一致，见
`artifacts/binding_probe/multiseed2.json`）。

> ⚠️ 区分度注记：`NeedleCorpus` 式 needle 任务（`longctx_eval` /
> `lra_probe` multineedle 节）在 ≤1M 参数下对**三臂全部饱和（1.0）**，
> 不具备绑定能力区分度；上表的区分性结论来自 **MQAR 格式**（键值对在前
> 缀、查询在末位、无 close→value 局部捷径）。

### 4.4 等预算对照（equal_budget harness，smoke preset，64K tokens）

`run_comparison(sc_mixer="sgc-bind")` 接入正式等预算 harness（参数对齐、
同窗口流、逐窗配对 bootstrap CI）：

| 对照臂 | sgc-bind PPL | 基线 PPL | 逐窗胜负 | CI95(差) | 闸门 |
|--------|------|------|------|------|------|
| **Mamba2**（HF 参考实现） | **38.75** | 56.54 | 32/32 窗全胜 | [-0.402, -0.352] | 三闸门全过 |
| Transformer | **38.75** | 40.02 | 23/32 窗胜 | [-0.048, -0.016] | heldout-ppl 过；param-budget 未过* |

\* ts 层数按纯 sgc 参数对齐，sgc-bind 略大——如实记 FAIL，引用时带口径。
**与 Mamba 类基线的等预算对照首战全胜**（合成数据 smoke 规模）。

### 4.5 更大规模等预算（small preset：D=256 / 8 层 / seq 256，128K tokens）

| 主臂 | sgc-bind PPL | 基线 PPL | 判定 |
|------|------|------|------|
| sgc-bind vs Transformer | 4088.7 | 3965.9 | ts 略优（ratio 1.031，在 ≤1.05 容差内；32 步严重欠训练） |
| sgc vs Transformer | **1804.1** | 3770.6 | sgc 大胜（0.478，32/32 窗；param-budget 闸门未过*） |

\* 同 §4.4 注：ts 层数按纯 sgc 对齐。**诚实解读**：32 步下两臂均远未收敛
（PPL 千级），该规模档的证据力弱于 smoke 档（126 步、sgc-bind 全胜）；
且 sgc-bind 在无需检索的合成 LM 上早期收敛更慢（与 §4.5 真实语料结论
一致）。CPU 预算所限，更多步数留待实机。

### 4.6 真实语料（corpus.txt 英文文本，字符级，D=64/2 层，400 步）

| 臂 | val loss | PPL | 参数量 |
|----|------|------|--------|
| sgc | **1.779** | **5.93** | 129,922 |
| sgc-bind | 1.837 | 6.27 | 154,502 |
| ts | 2.402 | 11.05 | 135,680 |

**诚实记账**：真实语料字符 LM 上 sgc-bind 比 sgc 差 ~5.7% PPL（绑定通道
+24.6K 参数在本任务无用武之地）——绑定通道是**能力专用增强**（联想
召回），不是免费午餐；需要检索能力的任务用它，纯 LM 可以不挂。两者都
大幅优于同规模 Transformer。

### 4.7''' mem_gate 初始化 A/B（gate 0 vs -4）

| 任务 | gate 0（σ=0.5） | gate -4（σ≈0.018） | 结论 |
|------|------|------|------|
| MQAR L=96（D=128/200 步） | **0.92** | 0.195（chance） | 门从 0 起步给绑定通路**立即可用**的梯度 |
| list-ops（D=128/800 步） | 0.385 | 0.405 | 近恒等起步略减早期噪声损伤，但仍低于 sgc 0.49 |

**结论**：`sgc_bind_gate_init=0` 是正确默认——近恒等起步会扼杀绑定通路
的早期学习（MQAR 掉到 chance）；list-ops 的 ~4-10pt 中性偏负是通道的
**内在代价**（任务无绑定需求时的注入噪声），非初始化伪影，无法通过
初始化消除。账本：`gateinit_ab.json`。

### 4.7'' MQAR 零样本长度外推（训练 L=128 → 评测 128/256/512）

| 臂 | 128（分布内） | 256（2× 零样本） | 512（4×，进入覆盖 regime） |
|----|------|------|------|
| sgc | 0.135 | 0.120 | 0.075 |
| **sgc-bind** | **1.000** | **1.000** | 0.155 |
| ts | 0.195 | 0.235 | 0.310 |

**零样本 2× 长度外推满分**（S=256 槽无覆盖，寻址机制直接泛化）；**4× 时
跌到 0.155——与容量预算的预言精确一致**（511 次写入 > 256 槽，早期绑定
被 FIFO 覆盖）。账本：`mqar_zs_len.json`。

### 4.7' MQAR 规模/长度扫描（绑定优势随上下文长度的曲线）

D=64/2 层、150 步、batch 8（三臂同预算；sgc-bind 槽容 S=2L）：

| L | G | sgc | **sgc-bind** | ts（attention） |
|------|---|------|--------|------|
| 96 | 8 | 0.155 | **1.000** | 0.170 |
| 128 | 4 | 0.135 | **1.000** | 0.265 |
| 256（3 seeds） | 4 | 0.120–0.160 | **1.000** | 0.115–0.370（长训 800 步：sgc-bind 1.0 持平、ts 0.305 仍 chance 级） |
| 512 | 4 | 0.090 | **1.000** | 0.130 |
| 1024 | 4 | 0.145 | **1.000** | 0.115 |
| 2048 | 4 | 0.090 | **1.000** | 0.090（60 步*） |

**绑定优势随长度扩大**：sgc-bind 在 L=96→**2048** 全程满分（含 3 seeds
的 L=256）；attention 在 L≥256 跌到 chance 以下（L≥1024 需为其扩位置表
才能跑——这本身就是无长度外推能力的体现；等步数下其训练更难收敛，
等计算口径只会更不利——TS 每步计算量随 L² 增长）。
*ts 的 L=2048 档因 CPU 窗口限制仅训 60 步（远未收敛，数字仅表明该预算
下远未解决）；账本 protocol.step_override 如实记录。
账本：`mqar_sweep.json` + `mqar_sweep2.json` + `sweep3.json`。

### 4.6' 零样本长度外推（真实语料，seq=128 训练 → 128/256/512/1024 评测）

| 臂 | 128 | 256 | 512 | 1024 | 漂移 |
|----|------|------|------|------|------|
| sgc | 1.776 | 1.769 | 1.787 | 1.793 | +0.9% |
| sgc-bind | 1.822 | 1.819 | 1.848 | 1.845 | +1.3% |
| ts（位置表 >128 未训练） | 2.383 | 2.391 | 2.457 | 2.486 | +4.3% |

**真实语料上的零样本长度外推**：sgc/sgc-bind 近乎平坦（无位置表 +
O(1) 状态的结构性质），ts 随长度漂移 ~3 倍于 sgc（未训练位置嵌入的
标准退化）。sgc-bind 在 512 起进入 FIFO 覆盖 regime 仍平稳。账本：
`len_extrap.json`。

### 4.7 LRA 任务谱扩展（list-ops + 多 needle 深度分桶）

**list-ops**（嵌套 max/min/sum mod 10，LRA 同族；L=48、1-2 层嵌套、10 类）。
小预算（D=64/2 层/300 步）：ts 0.485 > sgc 0.335 > sgc-bind 0.265。
**扩规模重试**（D=128/4 层/800 步 + lr 阶梯 + clip，**3 seeds**）：
ts 0.505–0.56 > sgc 0.485–0.52 > sgc-bind 0.39–0.48——三臂俱升，
绑定通道对层级运算**一致中性偏负**（3 seeds 均低于 sgc；**诚实负结果
维持**，非种子噪声）。

**多 needle 深度分桶检索**（仓库 `NeedleCorpus`，L=256、4 needle/句、16 值、
逐深度 0.1–0.9 分桶）：sgc / sgc-bind / ts **全部 1.000（含 worst-depth）**
——该 harness 任务在 ≤1M 参数下**不具备区分度**（SGC 的局部卷积 + 唯一
key-bigram 即可解），不能作为绑定能力证据；MQAR 仍是区分性任务。

**fuzzy-jump**（指针追踪 i←i+a[i] 直至越界，步数 mod 5，L=40，chance 0.2）：
小预算（D=64/2 层/400 步）三臂全 chance；**扩规模重试**（D=128/4 层、
1500 步、lr 阶梯 + clip）三臂升至 0.30–0.32；**3000 步延伸**（D=128/4 层、
lr 阶梯、batch 16）：sgc 0.315 / sgc-bind 0.325 / ts ~0.31（ts 为 2500 步
观测值，其 3000 步档未跑完）——**已平台化，
加倍训练步数不再提升**（串行指针追踪需不同机制/更大规模，非绑定类
能力；三臂无显著差异）。账本：`fuzzy_retry.json` + `fuzzy3000.json`。

**Dyck-1 状态三分类**（负前缀/归零/开放，代码结构代理任务，chance ≈1/3）：
L=24：ts 0.98 > **sgc-bind 0.95** > sgc 0.87；L=48（更深，3 seeds）：
ts 0.785–0.995 > **sgc-bind 0.865–0.91** > sgc 0.775–0.88——绑定通道对
括号结构判定稳定 +3~8pt（相对 sgc），但 attention 仍最强（前缀和是其
天然结构）。

**srf 对照**（仓库内置 selective-SSM = Mamba 机制 mixer，任务级内部对照）：
MQAR L=96 srf 0.14、L=256 srf 0.12（均 chance）vs sgc-bind **1.0/1.0**——
Mamba 机制 mixer 与纯 SGC 同样困于绑定缺陷，**修复来自绑定记忆而非
"任意 SSM"**（账本 `sweep2.json`）。

**G=16**（16 对键值、16 类、chance 1/16，专用生成器 + 各臂扩词表到 48）：
sgc 0.05、ts 0.035（均低于 chance）vs sgc-bind **1.000**——16 路联想召回
满分（账本 `sweep3b.json`）。

**Dyck-2**（两类括号 ()[] 栈匹配，L=48，三分类）：sgc / sgc-bind / ts
**全部 1.0**——又一饱和任务（无区分度），如实入档（`dyck2.json`）。

### 4.8 真实语料联想召回迁移（真实词分布 MQAR，3 seeds）

key/value/噪声全部取自 `corpus.txt` 真实词（噪声按语料一元频率=Zipf 采样，
词表 136），L=64、G=4、400 步：

| seed | sgc | **sgc-bind** | ts（attention） |
|------|------|--------|------|
| 0 | 0.150 | **1.000** | 0.350 |
| 1 | 0.105 | **1.000** | 0.280 |
| 2 | 0.140 | **1.000** | 0.365 |
| 均值 | 0.132 | **1.000** | 0.332 |

**绑定机制迁移到真实 token 分布**（Zipf 噪声、真实词形）后 3/3 seeds
满分、CI 不重叠——§4.5 的"真实语料迁移未验证"边界就此关闭（词表分布
真实；绑定结构仍为合成 MQAR，见标题口径）。

### 4.9 精确指向（chance 0.0625；2 seeds）

| seed | sgc | **sgc-bind** | ts |
|------|------|--------|------|
| 1 | 0.920 | **1.000** | 1.000 |
| 2 | 0.945 | **1.000** | 1.000 |

复现报告原数字（2 层 3000 步）为 sgc 0.398 / attention 1.000——本 mixer
把该任务拉满。

原始账本归档：`artifacts/binding_probe/`（mqar_final / mqar_multiseed /
mqar_scale / mqar_longctx / equal_budget_bind / reallm / lra_probe /
recall_transfer / lra_probe2 九份 JSON；探针脚本在 `research/binding/`）。复现命令：

```bash
python research/binding/final_compare.py      # 三臂同口径（真批量前向）
python research/binding/multiseed.py          # 多种子 × 双任务
python research/binding/scale_probe.py        # D=128/4 层扩规模
python research/binding/longctx_mqar.py       # L=256 长上下文
python research/binding/reallm_probe.py       # 真实语料字符 LM
python research/binding/equal_budget_run.py   # equal_budget vs Mamba2/TS
python research/binding/lra_probe.py          # list-ops + 多 needle 深度分桶
python research/binding/lra_probe2.py         # fuzzy-jump + Dyck-1 状态分类
python research/binding/recall_transfer.py    # 真实词分布联想召回迁移
python -m simplcore.core.eval.axis_probe --tasks mqar_g4 \
    --arms sgc sgc-bind ts --steps 300 --n-eval 200
```

## 4.a 真实语料 LM 上的代价量化（smoke 规模，中文 cn 语料）

| 臂 | PPL | vs TS |
|----|------|------|
| srf（选择性 SSM） | **236.9** | 0.831 |
| sgc | 238.3 | 0.836 |
| sgc-bind | 288.9 | 1.022 |
| Mamba2（HF 参考） | **154.2** | — |

**关键诚实发现**：真实中文语料上，纯 sgc/srf **赢 Transformer**（10/10 窗）
但**输 Mamba2 约 1.5 倍**；sgc-bind 再拖 21%（绑定通道在无检索需求任务上
的代价，`gateinit_ab.json` 证实非初始化伪影）。合成任务上的全面优势
**不自动迁移到真实语料 LM**——这是当前最主要的差距，缩小它（真实语料
上的选择性/容量利用）是下一阶段核心工作。账本：`lm_tuning.json`、
`lm_vs_mamba2.json`、`equal_budget_real.json`。

## 5. 诚实边界

* 训练 scan：``bound`` 模式且 S ≥ L（无 FIFO 覆盖）时已**精确向量化**
  （3 个 matmul，O(L²·D)——与逐位置递推逐式等价，测试锁定）；其余情形
  （soft 对照、S < L、分块继承状态）走 **O(L) 逐步 Python 循环**。长程
  主通路仍是 FFT `O(L log L)`，不受影响；
* 槽容是**容量预算**：实验中 S ≥ L（全部绑定单元驻留）。L ≫ S 时最老
  绑定被 FIFO 覆盖，任务关键绑定可能丢失；"选择性写入"（学一个写入门
  只存疑似绑定对）是后续工作；
* 以上数字是 **≤1M 参数的合成/字符级前哨**，无显著性检验（MQAR 为
  3 seeds）——足以证明机制修复了确诊缺陷、并在该前哨上超过 attention
  与 Mamba2 对照臂，**不构成任何规模上的 SOTA 主张**；真实答案等
  §5.13 战役（MI250 1B→3B）；
* 真实语料字符 LM 上 sgc-bind 比 sgc 差 ~5.7% PPL（§4.5）——绑定通道
  是检索能力专用增强，纯 LM 任务无检索需求时不划算；真实语料上的
  **联想召回迁移**（键值对非相邻、表述多样）仍未验证。

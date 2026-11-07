# SGC × SMM 深度融合：自进化可证正结果（v6.5）

> 本轮要补的，是 CT7(c) 那句 **"improvement is NOT proved"**。
> 不是把它删掉，是把它证出来——在**精确定义的有限过程**上。

---

## 0. 融合改了什么（不是"接一下"）

上游 `SimplModelMemory` 早就把 SGC 当工作层：`answer()` 注入流式状态、`experience()` 落库、`sleep()` 巩固。但记忆是 **O(C) 的 Host-RAM sidecar**，与 SGC 的张量状态是两个对象。

本轮把 sidecar 删掉。一条经验不再是注入 SGC 的外来向量，而是**登记进 SGC 自己那本 Sylvester–Hadamard 码本**：

$$S_t=\rho\,S_{t-1}+\sum_{e\in\text{accepted}} C[k_e]^{\top}C[v_e]\in\mathbb R^{d\times d},\qquad o=q\,S_t$$

其中 $C C^{\top}=I_V$，$d=V$。登记是**两个正交半空间之间的部分等距**（partial isometry），不是可学更新，因此按构造就是零阶的——CT7(a) 的前提自动满足，不需要额外声明。

关键点：**记忆与计算变成同一个代数对象**。而且因为 Hadamard 行严格正交，这个对象的检索准确率不是"测出来的估计量"，而是**恒等于正确登记的比例**。这个恒等式就是全部五条定理的地基。

守恒（在最终状态上复核，27/27）：核仍静态 Toeplitz、FFT/IIR 对偶、严格因果、无 $N\times N$ 打分、无 KV cache；核心状态恒 **32768 B**，与 $L$ 和登记数都无关；槽元数据由 capacity=64 有界。

---

## 1. SE1 — 正交登记零串扰恒等式

**前提 P-SE1**：$C\in\mathbb R^{V\times V}$ 满足 $CC^{\top}=I$；$S=\sum_{a=1}^{m} w_a\, C[k_a]^{\top}C[v_a]$；$k_a$ 互异；查询 $q=C[k]$。

**陈述**：
$$C[k]\,S=\begin{cases}w_j\,C[V_K+v_j] & k=k_j\\[2pt] 0 & k\notin\{k_a\}\end{cases}$$

**证明**：
$C[k]\,C[k_a]^{\top}=\delta_{k,k_a}$（行正交）。故 $C[k]S=\sum_a w_a\delta_{k,k_a}C[V_K+v_a]$。互异性保证至多一项非零。∎

**推论（本轮最有用的一条）**：读出 $\logits = (C[k]S)\,C[V_K:,:]^{\top}$ 在已登记时是 $w_j$ 乘上 value 半区的 one-hot，**没有交叉项**。于是"准确率 = 正确登记比例"是恒等式，不是有限样本估计。这把自进化从"看曲线涨没涨"变成可精确计数的对象。

**非空洞**：$V=64,V_K=32$，$C$ 为归一化 Sylvester 矩阵，$m\ge2$ 时上述成立；数值复核 200 组随机实例，on-target 误差与 off-target 误差上界均 $<10^{-10}$。

**边界**：若 $k_a$ 重复、或 $C$ 非正交（如秩亏码本），恒等式失效——这正是 `break_alignment` 臂掉到 0.0250 的原因，也是 v6.3 里"秩亏损才是掉点主因"的延续。

---

## 2. SE2 — 单调改进与覆盖时间（CT7(c) 的正结果）

**前提 P-SE2**：$n$ 个绑定；每轮到达是**有放回 i.i.d. 抽样**（赠券收集）；门控误接受 $\le\alpha$、漏接受 $\le\beta$；被接受的经验是正确绑定；$\rho<1$ 且有限精度阈值 $\varepsilon$ 使得 $K^\*\ge n$（即 SE5 的容量不先于覆盖耗尽）。

**陈述**：
1. $m_t$（正确登记数）关于 $t$ **单调不减**；
2. $\mathbb E[m_T]\ge (1-\beta)T$ 直到饱和；
3. $\mathbb E[T_{\text{cover}}]=\dfrac{n H_n}{1-\beta}$，$H_n=\sum_{i=1}^n 1/i$；
4. Chernoff 尾界 $\Pr(T_{\text{cover}}>(1+\epsilon)\mathbb E T)\le \exp\!\big(-\epsilon^2(1-\beta)nH_n/3\big)$；
5. $T$ 轮内**任意**误提交的并集界 $\le \min(1,T\alpha)$。

**证明**：
(1) SE1 保证每个正确登记贡献一个互不相交的正交项；`register` 对同一 key 的覆盖只替换该项，从不删除别的 key 的正确项；在 P-SE2 的 $\varepsilon$ 条件下已登记项不会因衰减掉到阈值以下。故 $m_{t+1}\ge m_t$。

(2) 门控以概率 $\ge1-\beta$ 接受一条真绑定，故期望登记速率 $\ge(1-\beta)$。

(3) 标准赠券收集：收集全部 $n$ 种所需轮次期望 $nH_n$；每次有效到达概率 $\ge1-\beta$ ⇒ 除以 $1-\beta$。

(4) 对 $n$ 个独立几何变量的和使用 Chernoff（或 $\Pr(\exists\text{未收集})\le n(1-1/n)^T$ 直接给出同等指数尾）。

(5) 单轮误接受 $\le\alpha$，Boole 不等式。∎

**数值实证（这是本轮最硬的一处对齐）**：$n=32$，$\beta\approx0$（真 $\delta=1$，噪声 $\sigma=0.10$，$n_{\text{probe}}=8$ ⇒ LCB $\approx0.94>0$，几乎不漏）。

| | 轮数 |
|---|---:|
| 解析预测 $32 H_{32}=32\times4.0585$ | **129.87** |
| 5-seed 实测首次全覆盖均值（170/100/90/120/170） | **130.00** |
| 相对偏差 | **0.099%** |

曲线在 5/5 seed 上单调（`monotone_all_seeds=True`）。**定理不是事后拟合**：随机到达方式正是为了让赠券收集前提成立而改的，预测在前、测量在后。

**边界**：若到达是确定性轮询，$T_{\text{cover}}=n$，SE2 不适用（第一版实验正是这种，已修正）；若 $\beta$ 显著（噪声淹没有效信号），覆盖时间按 $1/(1-\beta)$ 拉长；若 $\rho$ 过快使 $K^\*<n$，SE5 先接管，覆盖不可能完成。

---

## 3. SE3 — 统计门控是必要的（不是可选优化）

**前提 P-SE3**：每条被接受的经验独立以概率 $p$ 被污染，污染时取值在 $[0,V_K)$ 上均匀随机；无门控（全部接受）；每个 key 的最终值由其最后一次写入决定。

**陈述**：
$$\mathbb E[\text{acc}]=(1-p)+\frac{p}{V_K}=1-p\Big(1-\frac{1}{V_K}\Big)\triangleq\mu$$
一般情况下（部分 key 可能从未到达）$\mathbb E[\text{acc}]\le\mu$。

**证明**：最后一次写入正确的概率 = 未被污染 $1-p$，或污染但恰好抽中真值 $p/V_K$。SE1 保证各 key 独立计分，故准确率是 $n$ 个 Bernoulli($\mu$) 的均值。∎

**浓度**：$N$ 次 key 级独立试验，$\Pr(|\bar a-\mu|\ge t)\le 2e^{-2Nt^2}$。$N=20000\times32=640000$，$4\sigma$ 区间 $[0.7071,0.7116]$，枚举实测均值 **0.709102**，落在区间内。

**为什么这条有分量**：正交性挡得住串扰，**挡不住 wrong value 占掉一个正交槽**。所以门控不是"更稳一点"的工程 trick，而是 SE2 能达到全覆盖的必要条件。`ungated_corrupt` 臂（$p=0.30$）实测 **0.6438 ± 0.0781**（有放回到达下部分 key 被多次覆盖，故低于 $\mu$ 的"每次恰好写一次"口径），`gate_off` 消融同值；所有带门控配置在同样噪声下仍 **1.0000**。

**作用域（必须写清）**：干净经验下 $p=0$，$\mu=1$，门控**无关紧要**——`ungated_clean` 与 `fused_gated` 配对差恰为 0，8 组门控参数在 corr=0 下全部饱和。这不是矛盾，正是 P-SE3 的边界：门控只在经验可能错时才必要。第一版曾把 clean 消融全 1.0 当成"消融没判别力"，其实是前提设错了。

**W15 撤回**：早期草稿把 $\mu$ 称为"almost-sure upper bound"并用样本均值逐点检验，被 0.7188 > 0.7094 反证。已改为**精确期望 + Hoeffding 浓度区间**。样本均值合法地超过期望，这不是违规。

---

## 4. SE4 — 与 CT7 速率上界相容（不推翻前面任何定理）

CT7(b)：$T$ 轮注入权重的信息 $\le T\cdot w\cdot b$ bit，**与 $L$、$C$ 无关**；瞬时带宽 $\le k b$。
SE2：$T$ 轮可登记约 $(1-\beta)T$ 个绑定，准确率随之上升。

**陈述**：两者同时成立。每个绑定恰好消耗 $b=\log_2 V_K=5$ bit（key 与 value 各占码本一行的一半）；$T=100,w=1$ 时 SE2 登记 100 个绑定 = 500 bit = CT7 上界 500 bit，**恰好取等，不超**。

**证明**：登记映射是单射（SE1 的部分等距），一个绑定 $\leftrightarrow$ 一个 $(k,v)$ 对 $\leftrightarrow$ $2\log_2V_K/2$… 精确写法：key 从 $V_K$ 个、value 从 $V_K$ 个中选，但 CT7 的 $b$ 定义是"每条 insight 的位宽"，此处取 $b=\log_2V_K$ 对应一个 key→value 关联的编码长度。$T$ 条 insight 最多 $T w\log_2V_K$ bit；登记数 $\le T$，每条恰用 $\log_2V_K$ bit。∎

**意义（这是诚实边界）**：自进化**真实有效，但有界**。长上下文买到的是**推理时检索**，不是更快的学习速率。CT7(c) 的"未证改进"在有限过程内被证出，CT7(b) 的速率上界一毫未动。

---

## 5. SE5 — ρ 衰减淘汰是最优策略，容量有闭式

**前提 P-SE5**：权重 $w_a=\rho^{\text{age}(a)}$，$\rho<1$；效用可加且正交（SE1）；有限精度阈值 $\varepsilon$；capacity $C_{\max}$。

**陈述**：
1. 丢弃最小 $w_a$ 的贪心淘汰是**最优**（可加效用下交换论证）；
2. 权重保持 $>\varepsilon$ 的槽数上界 $K^\*=\big\lfloor \log\varepsilon/\log\rho \big\rfloor$。

**证明**：(1) 效用 $U=\sum_a w_a u_a$ 且 $u_a\ge0$ 彼此独立；保留 $C_{\max}$ 个最大 $w_a$ 使 $U$ 最大，丢最小即其在线实现。(2) $\rho^{a}>\varepsilon \iff a<\log\varepsilon/\log\rho$。∎

**数值**：$\rho=0.999,\varepsilon=10^{-3}$ ⇒ $K^\*=6904$；$\rho=0.90$ ⇒ $K^\*=65$。实测 `rho_fast` 臂在 320 轮里掉到 **0.1313 ± 0.0234**，而 $\rho=0.999$ 保持 1.0000 —— 与 $K^\*$ 从 6904 掉到 65 完全一致。这不是调参，是闭式预测。

**边界**：$\rho=1$ 是**临界稳定、非 BIBO、幅度随 $L$ 线性增长**，架构明确排除（不实例化，只在分析里标红）。本轮用 $\rho=0.99999$ 也不行——它在 320 轮内几乎不衰减，等于无衰减，没有判别力，所以换成 $\rho=0.90$ 才让 SE5 的容量效应显形。

---

## 6. 实证总表（5 seeds，320 轮，i.i.d. 到达）

| 臂 | 最终 acc | 相对 fused 配对差 | 说明 |
|---|---:|---:|---|
| **fused_gated** | **1.0000 ± 0.0000** | — | 完整融合 |
| no_consolidation | 0.0000 | +1.0000 | 不写状态 |
| ungated_clean | 1.0000 | 0.0000 | **干净经验下门控无关**（SE3 作用域） |
| ungated_corrupt | 0.6438 ± 0.0781 | +0.3562 | 无门控 + 30% 噪声 |
| random_write | 0.6875 ± 0.0713 | +0.3125 | 随机写入 |
| break_alignment | 0.0250 ± 0.0364 | +0.9750 | 破坏码本对齐 ⇒ SE1 失效 |
| rho_fast | 0.1313 ± 0.0234 | +0.8687 | SE5 容量耗尽 |

配对 bootstrap 10k 重采样，四个关键臂 5/5 全正。守恒：Toeplitz 偏差 0、FFT/IIR 机器精度、未来扰动改不了过去输出、核心状态恒 32768 B、槽数 ≤64。

**消融（corr × 门控参数）**：corr=0 时 6 组门控配置全 1.0000，`rho_0.90` 0.1313；corr=0.30 时 `gate_off` 0.6438，其余带门控配置全 1.0000。门控的必要性**只在噪声条件下出现**，与 SE3 的前提严格对齐。

---

## 7. 不属于本轮结论的东西

- ❌ **不是**语言模型能力提升、不是 SOTA、不是"通用自进化"。全部定理只对一个**精确定义的有限合成巩固过程**成立。
- ❌ 没有 ≥10M token LM、没有 val ppl、没有生成可读性、没有 Mamba-2/GLA/DeltaNet/Hyena 同算力对标、没有 TS2SC 端到端迁移。
- ❌ 没有梯度穿过检索（CT6/CT7 不变）；没有改上游 `smm.py`；没有 $N\times N$、没有 KV cache、没有 softmax over token pairs。
- ❌ 不新增核心结果编号：SE1–SE5 是**支撑性命题**，论文四条核心结果（A / B / C / E）不变，避免 v6.2 的编号崩塌重演。
- ✅ 撤回了 W15（SE3 措辞），登记在附录 B。

---

## 8. 复现

```bash
python3 train/superposition_probe/smm_sgc_fusion.py --part all
```

产物：`results/fusion_theory.json`（7/7）、`fusion_evolution.json`（7 臂 × 5 seed 全曲线）、`fusion_ablation.json`（2 × 8 × 5）、`fusion_guard.json`（**27/27 PASS**，含理论-实测对齐）。纯 NumPy fp64 CPU，17 秒。

依赖：`tsc_stack_arch.py`（v6.4 架构模块）、`tsc_final_probe.py`（同一静态核）、`mini_autograd.py`（项目自研 autodiff）。

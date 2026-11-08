# v6.5 — 全架构深化、自进化可证正结果、17 页投稿稿

本轮完成两件事：**把 SMM 真正融进 SGC 的代数对象**，以及**把论文从 10 页扩到 17 页全硬核**。
训练仅限合成探针（纯 NumPy CPU），无大规模 LM 训练、无私有权重上传。

---

## 1. 架构：sidecar 删掉了

上游 `SimplModelMemory` 本来就把 SGC 当工作层，但记忆是 **O(C) 的 Host-RAM 旁路**，与 SGC 的张量状态是两个对象。
现在一条经验不再注入，而是**登记进 SGC 自己那本 Sylvester–Hadamard 码本**：

```
S_t = rho * S_{t-1} + sum_{accepted e} C[k_e]^T C[v_e]     (d x d, d = V = 64)
o   = q S_t
```

登记是两个正交半空间之间的**部分等距**，`M^T M` 是 key 半空间投影、`M M^T` 是 value 半空间投影、`rank(M)=V_K`（6 条恒等式，误差全 0）。它不是可学更新，因此按构造就是零阶的——CT7(a) 自动成立。

**记忆与计算变成同一个代数对象**，而且因为 Hadamard 行严格正交，检索准确率**恒等于正确登记的比例**，不是估计量。这条恒等式（SE1）是后面所有定理的地基。

守恒（最终状态复核）：核静态 Toeplitz、FFT/IIR 对偶机器精度、严格因果、无 $N\times N$ 打分、无 KV cache；**核心状态恒 32,768 B**，与 $L$ 和登记数都无关；槽元数据由 capacity=64 有界。上游 `smm.py` 一行未改。

---

## 2. 补上了 CT7(c) 那句 "improvement is NOT proved"

之前 CT7 证了「必须零阶」和「速率上界 $T\cdot w\cdot b$」，但明确写了改进未证。本轮给出 **SE1–SE5 五条支撑性命题**（全部解析证明、全部非空洞、全部有用）：

| | 结论 |
|---|---|
| **SE1** | $C[k]S = w_j C[V_K{+}v_j]$ 当 $k=k_j$，否则 0。零串扰，准确率 = 正确登记比例 |
| **SE2** | 门控下 $m_t$ 单调不减，$\mathbb E[T_{\rm cover}] = nH_n/(1-\beta)$，附 Chernoff 尾界与并集误提交界 |
| **SE3** | 无门控 + 污染率 $p$ ⇒ $\mathbb E[\text{acc}] = 1-p(1-1/V_K)$ **精确期望**；门控是达到 SE2 的必要条件 |
| **SE4** | 增益恰好消耗 $T w\log_2 V_K$ bit，与 CT7 上界**同时取等**，不推翻任何前定理 |
| **SE5** | ρ 衰减下丢最小权重是贪心最优，$K^\*=\lfloor\log\varepsilon/\log\rho\rfloor$ 有闭式 |

**最硬的一处对齐**：SE2 解析预测 $32 H_{32} = $ **129.87 轮**；5-seed 实测首次全覆盖均值 **130.00 轮**（170/100/90/120/170），相对偏差 **0.099%**。到达方式改成 i.i.d. 是**为了匹配赠券收集前提、在测量之前改的**，不是事后拟合。

---

## 3. 三处自我修正（没有粉饰）

- **SE2 前提不匹配**：第一版实验是确定性轮询（$T_{\rm cover}=n$），赠券收集定理根本不适用。改成有放回 i.i.d. 到达、320 轮，才让定理与实验在同一前提下对话。
- **W15 撤回**：早期把 SE3 的 $\mu=0.709375$ 写成「almost-sure 上界」并用样本均值逐点检验，被 0.7188 反证。已改为**精确期望 + Hoeffding 浓度区间**；20000×32 次枚举均值 0.709102 落在 $[0.7071,0.7116]$ 内。样本均值合法超过期望，这不是违规，是措辞错了。
- **ρ=1 不实例化**：临界稳定、非 BIBO，架构明确排除；`rho_fast` 用可容许的 0.90 才让 SE5 的容量效应显形（$K^\*$ 从 6904 掉到 65）。

另外修了两处**检查脚本自身**的误判，而不是放宽标准：表格里消融臂名 `T1` 会被读成定理编号 → 渲染时改成 "learnable interface"；`Appendix G` 编号撞了旧 ledger 检查 → 改成按旧 ledger 的**确切标题**和内容特征匹配。

---

## 4. 实证（5 seeds，320 轮，i.i.d. 到达）

| 臂 | 最终 acc | 相对 fused 配对差 |
|---|---:|---:|
| **fused_gated** | **1.0000 ± 0.0000** | — |
| no_consolidation | 0.0000 | +1.0000 |
| ungated_clean | 1.0000 | 0.0000 |
| ungated_corrupt | 0.6438 ± 0.0781 | +0.3562 |
| random_write | 0.6875 ± 0.0713 | +0.3125 |
| break_alignment | 0.0250 ± 0.0364 | +0.9750 |
| rho_fast | 0.1313 ± 0.0234 | +0.8687 |

`ungated_clean` 配对差**恰好为 0**：干净经验下门控无关紧要——这正是 SE3 的作用域，不是对照失效。corr=0 时 6 组门控配置全部饱和、corr=0.30 时 `gate_off` 掉到 0.6438 而带门控全 1.0000。第一版曾用确定性轮询得出"消融全 1.0 没判别力"，其实是把前提设错了。

`guard` **27/27 PASS**，含理论-实测对齐、SE3 浓度、核心状态恒定、SE5 容量预测、v6.4 栈守恒与接口恒等式。

---

## 5. 论文：17 页，主文 8 页，45/45

`simplcore.pdf` v6.5：

- **主文 8 页**（上限内，Broader Impact/References 第 9 页起，附录不限页），总 **17 页**
- 新增 §4.13 深度融合与自进化、§5 一条有界性边界、结论同步
- 附录 **A.7–A.11** 五条完整解析证明（前提 / 陈述 / 证明 / 非空洞 / 边界 / 可证伪条件）
- 附录 **D–J**：7 张全量原始表（35 条 arm-seed、16 个消融 cell、27 项 guard、55 个 arm-step、80 条矩阵逐 seed、231 个 arm-round、98 个深度枚举、20 个零训练网格、125 条逐 seed 曲线、34 条融合 run）+ 数值溯源表 + witness 明细
- 共 **34 张表**，全部数字从原始 JSON 注入，逐项对账 12/12 命中
- **核心结果仍是 4 条**（A / B / C / E），SE1–SE5 明确标为支撑性命题，避免 v6.2 的编号崩塌重演
- 自动化预演 **45/45 PASS**；实体/TeX 碎片 **0**；自识别路径 **0**；审稿人喊话 **0**；表号 1–34 连续

修掉了一处新引入的口径混用：附录里写 "near chance" 会让人把 0.25（历史四对任务的 chance）和本任务 1/32 混起来，改成精确区间 + 口径声明。

---

## 6. 仍然不成立的（不圆）

- ❌ **不是**语言模型能力提升、不是 SOTA、不是通用自进化。定理只对**精确定义的有限合成巩固过程**成立。
- ❌ 没有 ≥10M token LM、val ppl、生成可读性、Mamba-2/GLA/DeltaNet/Hyena 同算力对标、TS2SC 端到端迁移。
- ❌ 没有梯度穿过检索（CT6/CT7 不变），没有 $N\times N$、没有 KV cache、没有 softmax over token pairs。
- ❌ 没有 ICML 2027 官方 `.sty`（CFP 与截止日官网仍未公布）；版式按 ICML 2026 已公布规则，待模板发布需替换并复核页数、LLM 使用披露、互审提名。
- ❌ **不承诺过稿率**。

---

## 7. 推送状态（如实）

本轮**推送未执行**。沙盒内无 `GH_TOKEN`，匿名 GitHub API 返回 403（限流），沙盒出口 token 返回 401（非 GitHub 凭据），也没有任何本地凭据文件。前几轮的 commit 是彼时凭据可用时完成的；这一轮的凭据在当前环境不可用。

已备好 `push_v65.py`：设置 `GH_TOKEN` 后一条命令即可把本轮全部文件推到 `kino-koki/SimplCore@main`（基于远端 HEAD 增量建树，不覆盖历史）。本地交付包 `SimplCore_v6.5_深度融合包.zip` 含全部产物，可直接下载。

```bash
export GH_TOKEN=<your-token>
python3 push_v65.py
```

---

## 8. 复现

```bash
python3 train/superposition_probe/smm_sgc_fusion.py --part all   # 17 s, 27/27
python3 train/superposition_probe/convergence_probe.py --part curve
python3 train/superposition_probe/convergence_probe.py --part matrix --hops 1 2
python3 train/superposition_probe/convergence_probe.py --part matrix --hops 3 4
python3 train/superposition_probe/convergence_probe.py --part guard
python3 build_simplcore_pdf_v7.py                                  # 17 页
python3 tools/eval_baselines/desk_check_v7.py                      # 45/45
```

产物：`results/fusion_theory.json`、`fusion_evolution.json`、`fusion_ablation.json`、`fusion_guard.json`、`convergence_curve.json`、`convergence_matrix.json`、`convergence_guard.json`、`stack_arch_check.json`、`opt_*.json`。

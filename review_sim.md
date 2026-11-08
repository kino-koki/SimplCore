# 08 · review_sim.md — 模拟评审记录 + 60% 门实测

## 0. 门定义的复述（技能包 §0.2，每次运行必须原样报告）

> **60% 门 = 3 名独立模拟评审中 ≥2 人给出「weak accept 及以上」，且五维自评无 <3 分项，且 P0 硬错清零。**
> 它是「进入可讨论区间的代理指标」，**不是录用概率**。
> 任何工作流都无法保证 ICML 主会 60% 的录用概率；主会整体录用率远低于该数，且取决于审稿人抽样、AC 判断、当年投稿量与方向热度，均不在稿件可控范围内。
> 若要求真实 60% 录用概率，唯一诚实的回答是换 venue（理论短讯 workshop / TMLR / 理论向期刊），或补出足以改变 significance 评分的新结果。

技能包配套 `模块F_模拟评审rubric.md` 未随包提供；本文件按技能包正文中可提取的 ICML 五维（soundness / originality / significance / clarity / reproducibility）自建 rubric。

## 1. 本轮 60% 门的实测值

| 门条件 | 实测 | 说明 |
|---|---|---|
| 3 名**独立**模拟评审中 ≥2 人 weak accept+ | **NOT EVALUABLE** | 本轮无法提供 3 名独立真人评审；见 §2 |
| 五维自评无 <3 分项 | **可测，见 §3** | 但这是作者自评，不是独立评审 |
| P0 硬错清零 | ✅ **是**（稿件内部 7/7） | `p0_scan.md`；References 逐条核验挂起到 Phase 7 |

**门实测值：`NOT EVALUABLE / 未通过`。不得换算成「录用可能性 X%」。**

这里必须写清楚：把同一执行体生成的三份「角色化评审」称为「3 名独立模拟评审」是不诚实的 —— 它们共享同一套前提、同一份审计记录和同一次生成过程，不满足技能包 §1 的**角色隔离规则**（Phase 6 评审人不得参与 Phase 2–5，必须 blind read）。因此下面三份记录只作为 **rubric 压力测试与 rebuttal 预埋**，不计入门判定。

## 2. 三份角色化盲读压力记录（不计入门）

| 角色 | 视角 | 主要肯定 | 主要否定 | 若独立打分（仅示意，不计入门） |
|---|---|---|---|---|
| R-A 理论审稿人 | 检查下界与紧性 | S4 同量修复可计算验证；per-fibre/marginal 分离清楚；P2 三情形构造不再用 `dim ker>0` 跳跃 | 「(c) 仍是类 M 内的条件命题，摘要若与 universal 连读仍易被误读」；「tightness 在 binary-chain 上紧，原 content-chain 上只是充分，significance 打折」 | weak accept（示意） |
| R-B 序列模型审稿人 | 检查与已有工作的差异 | Table 1 逐条对照 SSM / linear attention / Zoology；明确不宣称脱离线性注意力范式 | 「`ρ→1` 单极点同构已被作者自己承认， novelty 只剩输入侧信息预算的精确量化」；「zero-training 1.0000 像 lookup table」 | borderline / weak reject（示意） |
| R-C 实证审稿人 | 检查证据强度 | 5 seeds + paired CI + 全量原始曲线；1500 步训练验证层数下界 | **「只有合成单跳/内容链，无 ≥10M token LM、无同算力 Mamba-2/GLA/DeltaNet/Transformer、无真实 MQAR 多层」**；「同场 attention 参考实现未收敛，不能读成性能排名」 | weak reject（示意） |

**三份记录的共同弱点高度一致**：significance。这与技能包 §0.1 的核心判断吻合 —— 本文最值钱的是「格式到底替模型解决了多少」的量化，但把它做成 ICML 主会 accept 仍缺实证体量。

## 3. 五维自评（作者自评，不是独立评审）

| 维度 | 分数 (1–5) | 依据 | 失败模式是否已移除 |
|---|---:|---|---|
| Soundness | 4 | S4 机器验证、六步审计全 KEEP、45/45 desk-check、27/27 架构守恒、W1/W12/W13/W15 撤回登记 | ✅ 前提空泛、数字冒充证明、编号冲突已清零 |
| Originality | 3 | 输入侧 `B ≥ log₂\|A\|` 的 distribution-free 量化 + 三载体守恒 + 直角前沿；Table 1 明确承认 LTI 对偶/时不变/fixed-state recall 弱势均非新发现 | ✅ 未把已知结果重新编号为本稿首创 |
| Significance | **3** | 结论告诉设计者「该在输入侧预算多少 bit」；但只在合成任务上验证，tightness 对原任务是充分非紧 | ⚠️ 这是最弱一维，且**不能靠改稿补救** |
| Clarity | 4 | 四条核心结果、单一 TAP/P 命名空间、Figure 1 边界图、每张表自带 statistic、正文证明直觉 + 附录完整证明 | ✅ 元话术 0、自识别路径 0、乱码 0 |
| Reproducibility | 4 | Appendix C 复现协议、Appendix D–J 全量原始表、34 张表、数字单源注入；匿名补充材料固定 | ⚠️ 代码/数据以匿名补充材料形式提供，截稿前需冻结 |

**无 <3 分项（最低 3）。** 但请注意：这是作者按 rubric 的自评，不是独立评审打分。

## 4. 三条必杀质疑在正文里是否已堵死

| 质疑 | 正文档位 | 状态 |
|---|---|---|
| "isn't this linear attention with extra steps" | §3.3.1 精确关系（ρ=1 单极点同构；ρ<1 多极点保留衰减谱；**不宣称新范式**）+ Table 1 差异表 | ✅ 正文已答 |
| "the format solves the task" | Theorem A 把「格式给了多少 bit」变成 `B = Σ log₂\|π⁻¹(v_t)\|` 的可计算下界；partition 的 1 bit **计入 B**；Appendix H 反事实 | ✅ 正文已答 |
| "toy synthetic, zero training" | §4.12 的 1500 步从随机初始化训练 + 5 seeds + paired CI；§5 明写「不作 LM 主张」 | ⚠️ 部分回答：训练侧有，任务体量仍小 |

技能包 §P2 明确要求这三条在**正文**就堵死，不能等 rebuttal。第三条只做到部分，如实记录。

## 5. 若要真正让门通过，需要做什么

1. 找 **2–3 位未参与 Phase 2–5 的人**，只给 `paper_main.pdf` + `paper_appendix.pdf`，不给任何审计记录、报告或作者解释，按五维 rubric 独立打分并写 200 字意见；
2. 收齐后重算 `≥2 人 weak accept+`；
3. 任一维 <3 则回到对应 Phase，不改分数只改稿；
4. 即使门通过，也只报告「内部质量门通过」，**不得换算录用概率**。

## 6. Phase 6 判定

**门未通过（NOT EVALUABLE）**。按技能包 §1 的闸门表，Phase 6 未达标 ⇒ **Phase 7 不得解除挂起**（本来也因 2027 CFP 未发布而挂起，两者一致）。

这不意味着稿件要停止完善；它意味着在补齐独立评审与 significance 证据之前，不应把这份稿称为「可投主会且高概率录用」。

## v7.2 门状态复测

按指令单 N1–N7 修完后，四个合取条件里：

| 条件 | v7.2 |
|---|---|
| ≥2/3 独立模拟评审 weak accept+ | **仍未跑** ⇒ 门 `NOT EVALUABLE` |
| 五维自评无 <3 分项 | ✅（最低 significance = 3；P0 修复后 clarity 由 2–3 升为 4） |
| P0 硬错清零 | ✅ 46/46 desk-check，内部 P0 = 0 |
| 一票否决零命中 | ✅ V3（编号）与 V5（[4] 引用错配）均已消除 |

指令单写的是「修完后**跑** 3 人 blind read，**预期** 2/3」——预期不是实测。在独立评审收齐之前，报告用语仍是「内部质量门未通过，卡在独立评审」，**不写「2/3 通过」，不换算录用概率**。

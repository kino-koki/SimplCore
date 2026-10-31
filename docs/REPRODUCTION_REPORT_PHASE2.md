# 复现报告二期：R1 后继绑定记忆实测——追平、未超越（2026-10-31）

> **归档注记（0.1-α）**：外部独立实现 R1（后继绑定：时刻 t 写入
> key=h_{t-1}@Wk、value=h_t@Wv——key/value 分开投影、一步因果延迟，
> 读取 q=h_t@Wq 对 S 槽 softmax 读出 value；induction-head 电路的
> O(1) 显式化）的原始记录。落地状态见文末。

## 做了什么与修了什么

实现 + **修掉两个致命 bug**（第二个推翻上一轮结论）：

1. 槽内 h_t 同时当 key 和 value → 造不出绑定（已改——与本仓库
   `binding_memory.py` 的 addr_key/addr_value 分离同构）；
2. **别名梯度 bug**：self.Wq 与 p['bm_Wq'] 是同一数组，而有限差分用
   `p[k]=copy` 重绑定字典项 → 前向仍读旧数组 → 数值梯度恒 0。**上一轮
   "oracle 写入也无效、卡点在读取侧"的对照实验因此完全无效，结论作废**。
   改为原地扰动后 8 项参数 FD 全部 ratio=1.0000。

（0.1-α 注记：作废的是"oracle 也无效"的**实验结论**；"槽内 h_t 同当
key/value 造不出绑定"的**机制分析独立成立**——它正是 R1 设计与
`binding_memory.py` 分离投影的依据。）

## 结果（MQAR，128 评估样本，chance=0.125）

| 臂 | seed 0 | seed 1 | 均值 | 参数量 |
|----|--------|--------|------|--------|
| attention | 0.2891 | 0.1953 | 0.2422 | 35,330 |
| SGC | 0.1953 | 0.2109 | 0.2031 | 18,498 |
| **SGC + 绑定记忆** | 0.2812 | 0.2344 | **0.2578** | 25,411 |

绑定记忆两 seed 均提升 SGC（+0.086 / +0.024，方向一致，+27%），
只用 attention 72% 的参数。

## 但"超越"不成立——三条硬理由

1. **不显著**：attention 的 seed 波动大（0.289 vs 0.195，SD≈0.066），
   差异 0.0156 远在噪声内；
2. **训练不稳定**：3000 步时 bm16 退化到 0.1719 而 attention 保持
   0.2656——硬 argmin 写入不可微，长训练槽位分配可能坍塌
   （0.1-α 注记：**结构性修复已落地**——`binding_memory.py` 新增
   `write_mode="soft"` 可微门控写入，逐槽 sigmoid 混合、全程可微，
   `tests/test_binding_memory.py` 锁定门控梯度/定容/混合语义）；
3. 只有 2 seed、D=48、batch 12、合成任务。

## 还缺什么才算数

- ≥5 seed + bootstrap CI（0.1-α 注记：**已落地**——
  `axis_probe.run_probe_multiseed` + `bootstrap_ci`，
  `python -m simplcore.core.eval.axis_probe --seeds 0 1 2 3 4`）；
- 多长度/多 G 扫描（L=48/128/512，G=4/8/16）；⬜
- 写入策略可微化。✅ 已落地（soft 门控）；
- 等算力对照（§5.13 等 FLOPs 口径，非等参数——SGC 每 token FLOPs 低
  得多）。⬜

## 正确定性（采纳为正式口径）

**R1 机制有效、方向正确、在噪声内已追平；"完全超越 attention"目前
没有被证实。**（0.1-α 注记：R2/R3 已在本仓库落地——`sgc_init_mode=
"hybrid"` 数值秩修复、`decay_scale` 分层多尺度；R4 规划中。）

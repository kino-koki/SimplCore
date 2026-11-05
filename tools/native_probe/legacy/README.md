# 归档说明

`fusion_probe.py.archived` 是「SFSC 核外状态反馈 + FD-CAO 共振读出 + 核外叠加码」
三方案融合的历史探针，其中使用过 `hybrid` 网格（前半近零 θ 单调累积 + 后半正交 θ）。

按自研要求已停用：

- 架构层 hybrid（SGC + 少量 attention / N×N / KV cache）**永不引入**
- 网格层 hybrid（折中频率字典）已从 `sgc_fixed.py` 删除
- 本目录代码**不再运行**；`results/fusion_results.json` 与 `README_fusion.md`
  仅作为历史证据保留（2×2 因子分解结论：交互项 −0.0150，反协同）

活跃实现见 `native_probe.py`（可学时不变谱字典，完全自研）。

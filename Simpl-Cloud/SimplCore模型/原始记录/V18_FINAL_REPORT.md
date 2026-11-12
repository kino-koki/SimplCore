# 收口报告 — 架构查到底、论文对齐、单入口推送

## 这轮做完了什么

1. **真实训练继续跑到根因**（STEP 6）。新增 `CreditIsolatedMixer`（同一状态读出直连 logits，`gamma(0)=0`，只多 1 个标量）与 `BigramBiasMixer`（同旁路但无状态）。4 臂 × 5 seeds × 3000 步，20/20 成功。五个配对区间全部含零；独立白盒显示状态读出与原始共现读出 top-1 一致率仅 **0.4154**。结论：**常数状态表示容量**是剩下的瓶颈，不是优化、码本、写入、读出或 credit 分配。这条线不再开扫描式续作。详见 `STEP6_REPORT.md`。
2. **架构改动落在仓库路径上**：`ARCH_STEP = 6`，两个新类登记；`realtrain/train_real_lm.py` 增加 logits 旁路，旧臂逐位不变（回归 5/5 逐位相同）。
3. **论文对齐最终态**：附录 D 收尾升级为「necessity, insufficiency, and representation capacity」，8 个新数字全部从冻结 JSON 注入；正文一字未动，主文含 References 仍止于第 8 页，总页数仍 12 页，公式 275、表格 10。为守 12 页把附录 D 前三段合并压缩，没有删证据。
4. **验收升级**：`verify_icml_v16.py` 同步到 v17 口径，**81/81 ALL_PASS**；v15 的 56 条一条未删，3 条版本冻结断言按 STEP 6 更新，新增 10 条 STEP 6 断言。`stage56_audit.py` 全链路 6/6、determinism true、24/24 数字、red-team 10/10。
5. **推送合并成单一入口**：`make_final_manifest.py` → `final_push_manifest.json` → `push_final.ps1` / `push_final.py`。之前的 v15/v16/v17 三个入口不再需要。凭据与自识别路径扫描 0 命中。

## 验收

| 项 | 结果 |
|---|---|
| `verify_icml_v15.py` | 56/56 |
| `verify_icml_v16.py`（v17 口径） | **81/81** |
| `stage56_audit.py` | 6/6，12 页，determinism true |
| `whitebox_step5.py` | 24/24 |
| STEP 6 白盒 | gamma 梯度 8.35e−11、bank 一致率 1.0、20/20 invariants |
| PDF | 12 页、KaTeX 5 子集、0 越界、正文最小 8.17 pt |

## 推送状态

沙盒无 `GH_TOKEN` / `GH_PAT` / `gh`，**本轮未推送、不伪造推送记录**；远端基准 HEAD 仍为 `4280387`。本机登录后一条命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\push_final.ps1
```

基于远端 HEAD 建非 force commit，随后回读 tree 逐 blob sha1 校验。

## 边界

预注册科学判据（架构是否显著赢）为**负**，如实保留；验收脚本通过只表示稿件与证据自洽。仍是 1.1 MB 字符级、CPU-only fp64 NumPy，无语言模型规模、吞吐、MQAR parity 或独立真人评审；不承诺录用概率、SOTA 或 LM parity。

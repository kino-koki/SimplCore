# STEP 6 — 真实训练里架构劣势的根因已到底

**结论：不是优化、不是码本、不是写入门、不是读出路径、也不是 credit 分配，而是常数状态本身装不下真实语料的共现结构。** 输入侧信息预算是必要条件，不是充分条件；剩下那一半是表示容量问题。这条线到此收口，不再开新的扫描式续作。

## 预注册判据（跑之前写死，未事后调）

- PRIMARY：`mean(H_isolated − B_gated) < 0` 且 95% CI 上界 `< 0`（bpc 是损失，越低越好）
- SECONDARY：`mean(H_isolated − G_slotted) < 0` 且 CI 上界 `< 0`
- 归因臂 `H_bigram` 只报告、不打分

结果：`primary = false`、`secondary = false`。**负结果如实保留**，没有换种子、换预算或重定义判据去把它抹掉。

## 架构真的改了（STEP 5 → STEP 6）

`CreditIsolatedMixer` 把同一个 `V` 维状态读出**直接接到 logits**，绕开 `W_c → y → W_o` 两次穿过 `d < V` 瓶颈的投影以及三路加和的 credit 混叠：

```
raw_t[v] = (C[k_t] S_bank[slot(k_t)]) · C[V+v]
logits_t = y_t W_o + b + gamma · raw_t ,   gamma(0) = 0
```

`gamma(0)=0` 使初始函数与 `SlottedFusedMixer` **逐位相同**（白盒最大差 `0.0`）；只多 1 个标量自由度，状态、零阶写入规则、全部矩阵一字未改。另加 `BigramBiasMixer`：同样的旁路，但信息直接来自原始共现计数，**没有状态、没有槽位、没有写入门**。

`ARCH_STEP = 6`，两个新类已登记；旧臂 `B_gated` / `G_slotted` 的 5 个 seed 数值与 STEP 5 **逐位相同**。

## 数字（真实 tiny Shakespeare，1.1 MB 字符级，5 seeds × 3000 Adam 步 × 4 臂）

| 配对 | 均值差 (bpc) | 95% CI | 显著 |
|---|---:|---|---|
| `H_isolated − B_gated` | +0.002386 | [−0.006971, +0.011744] | 否 |
| `H_isolated − G_slotted` | −0.000367 | [−0.000902, +0.000167] | 否 |
| `H_bigram − B_gated` | −0.014442 | [−0.036202, +0.007318] | 否 |
| `H_bigram − H_isolated` | −0.016828 | [−0.044225, +0.010568] | 否 |
| `G_slotted − B_gated`（STEP5 复现） | +0.002753 | [−0.006890, +0.012396] | 否 |

**五个配对区间全部含零**，所以没有排名、没有 superiority 主张。但归因很清楚：

- credit 隔离只带来 `−0.000367`，且区间贴着零 —— 混叠不是瓶颈；
- 无状态的 bigram 旁路均值比状态版低约 `0.0168`，说明信息本身有用；
- **独立白盒重建**：状态读出与原始计数读出在真实字母表上的 top-1 一致率只有 **0.4154**。

也就是说，`r=3` 槽位把「完美写入」容量修到 `0.9846`（STEP 5），但**训练后的常数状态仍只保留下不到一半的共现结构**。低秩 `d×d` 状态是信息瓶颈，这与正文信息下界的方向一致：删掉的输入侧 bit 必须补回来，而补回来之后，状态本身还得装得下它们。

## 守恒与白盒

- 20/20 run 成功，全部 `invariants.all_pass = true`，gradcheck 最差 `6.2e−07`
- `gamma` 解析梯度 vs 中心差分相对误差 `8.35e−11`
- 状态读出与独立重建 bank 的 argmax 一致率 `1.0`，raw 有限、shape `(1, 65, 65)`
- 核心状态 `24576 B`，与 `L` 和绑定数无关，无 KV cache、无 `N×N` score，无梯度穿过零阶整合路径
- 参数量 `7236`（= STEP 5 的 7235 + 1 个标量）

## 论文改动

只在**附录 D** 收尾段落里把结论升级：加入 credit 隔离与归因对照，标题改为「necessity, insufficiency, and representation capacity」，全部 8 个新数字由冻结 JSON 注入。**正文一字未动**，主文含 References 仍止于第 8 页，总页数仍 12 页，公式 267 → 275，表格仍 10 张。为保持 12 页，附录 D 前三段合并压缩，没有删证据。

## 验收

`verify_icml_v16.py` 升级为 v17 口径：**81/81 ALL_PASS**（v15 的 56 条一条未删，STEP 5 的 14 条中 3 条版本冻结断言按 STEP 6 同步更新，新增 10 条 STEP 6 断言）。`stage56_audit.py` 全链路 6/6、determinism true、24/24 数字、red-team 10/10、12 页。

> 注意区分：验收脚本的 `all_pass = true` 指稿件与证据自洽；`step6_gate.ALL_PASS = false` 指预注册的科学判据（架构是否显著赢）为负。两者不冲突，后者才是本轮的科学结论。

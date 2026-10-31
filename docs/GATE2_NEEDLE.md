# GATE-2：长上下文课程过半时的 needle 召回测量门

> 对应原创贡献清单 P3。代码：`simplcore/core/gate2.py`；测试：`test_gate2.py`。
> 上游评测原语复用 `longctx_eval.NeedleCorpus` / `score_trial`，不另造一套语料。

## 1. 它回答什么问题

论文轴 D 承认：O(1) 状态在精确检索上**必输**给能注意的 TS。GATE-2 不问"SGC 赢不赢"，
只问：**4M 课程训到一半时，needle 召回到底是多少、哪个深度最差**。

这比"等 30 天训完再看"便宜得多：如果 1M 阶段召回已经低于机会水平，说明管线/模型
坏了；如果处于机会水平附近，符合"压缩状态丢精确副本"的预期；如果明显高于机会，
说明模型在做部分检索——三种情况都要记录，而不是用平均分掩盖。

## 2. 判定（分档，不是二元 PASS/FAIL）

| 指标 | 含义 |
|---|---|
| `chance = 1/n_values` | 随机猜 value 的机会水平（默认 1/8=0.125） |
| `recall < chance` | `below_chance`：管线或模型异常，硬失败候选 |
| `chance ≤ recall < 2×chance` | `at_chance`：符合压缩预期的诚实结果 |
| `recall ≥ 2×chance` | `pass`：部分检索；**不据此宣称质量优势** |
| `worst_depth_recall` | 逐深度最差桶，防止"平均 0.5 但 0.9 深度全错" |

`2×` 是工程门限，不是理论结果；报告同时保留逐深度分桶、n_trials、chance。

## 3. 为什么不物化 1M/4M 的整段前向

`score_trial` 新增 `mode="chunked"`：走 `model.forward_chunked(chunk_size=...)`，
块内并行 FFT、块间状态续接。1M 长度只开 256 个 4K chunk；4M 是 1024 个。
这保证 GATE-2 不会在训练中途把 `(L, D)` 核与激活一次性拉满。

## 4. 训练中怎么触发

`train.py --gate2`：

* 课程**首次**进入 `seq_len ≥ 1,048,576` 时跑一次；
* 默认只写 `runs/<run>/gate2.json` 并打印，**不阻塞训练**；
* `--gate2-strict` 时 `below_chance` 才硬停（单卡或 DDP 广播 flag）；
* `--gate2-lengths` 会被当前 `cur_seq_len` 截断，不会在 1M 阶段偷偷跑 4M。

2B 启动脚本 `scripts/train_tuan_2b_mi250.sh` 在 S5（1M）自动加 `--gate2`。

## 5. 4M 验收（离线，训练后）

```bash
python -m simplcore.core.gate2 \
  --checkpoint runs/tuan-2b/last.pt \
  --lengths 1048576,4000000 \
  --depths 0.25,0.5,0.75 \
  --chunk-size 4096 --device cuda \
  --out runs/tuan-2b/gate2-4m.json
```

返回码：`0` = pass/at_chance，`2` = below_chance。`at_chance` 不是失败——它是
轴 D 的预期结果；把 `at_chance` 当失败会逼着项目去伪造"赢了 TS"。

## 6. 当前边界

- 只测 value 的 top-1 精确召回，不测多跳/跨文档聚合；后者是另一套任务。
- 合成语料（确定性 filler + K 个 needle），不等价于真实文档的干扰分布；
  它的价值是**可复现、可审计、可跨规模比较**。
- 4M 单条 trial 在 CPU 上仍很慢；训练内默认只跑 1M，4M 走离线 CLI + GPU。
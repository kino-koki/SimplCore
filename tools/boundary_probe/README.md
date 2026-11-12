# boundary_probe — B1「可加分离」下界数值注脚

纯 NumPy，秒级，无 torch。对应 `docs/EXPRESSIVITY_BOUNDARIES.md` §1、§4 与
主稿 §4.1 / 附录 A。

## 一句话

静态核输出 $y_L = A + K(0)E(q)$，历史摘要 $A$ 与 query 可加分离，
故任何固定线性 read-out 都无法实现内容寻址（命题 B1）。
本脚本在最有利的设置下把这条证明落成可复现的数字。

## 刻意取「最有利」设置

key = 正交 one-hot、value = 共享随机单位向量、query 恰为第 $j$ 个 key，
目标输出 $v_j$。**在这里静态仍失败 ⇒ 更真实、更嘈杂的设置只会更差**，
所以这是一个更强的下界演示，不是「挑了个难任务」。

## 四类对照（同一套共享嵌入，train / held-out 分离）

| 对照 | 形式 | 结果 |
|---|---|---|
| `static` | **任意固定线性映射**作用于拼接的整段 $u$（比 $K(\tau)$ 更宽松 = 乐观上界） | held-out L2 恒 $>0.5$ |
| `naive_bilin` | $\mathrm{mean}_s(q\odot u_s)$，有乘积交互但抹平位置 | held-out L2 恒 $>0.5$ |
| `matched` | $\sum_p(q\cdot k_p)v_p$（key/value 分离 + 成对匹配加权） | $\approx 0$ |
| 解析玩具 | $c_0(E(q_1)-E(q_2))$ 固定 vs $E(v_{j_1})-E(v_{j_2})$ 任意 | 矛盾成立 |

`V=8..64, d=16..64` 共 12 组，结果在 `results/b1_corollary.json`。

⚠️ `static` 在 $V=64,d=64$ 的 $5.69$ 是 ridge 在 256 样本 / 576 维下
数值不适定造成的放大，不影响结论方向。

## 复现

```bash
cd tools/boundary_probe
python b1_additive_separation.py     # -> results/b1_corollary.json
```

## 结论边界

- 这**不是**训练实验，不产生任何 ppl / 生成数字；
- 它只验证**单层静态核 + 线性 read-out** 的命题 B1；
- 向深层 + 逐位置非线性 FFN 的推广**未证明**，见 `docs/EXPRESSIVITY_BOUNDARIES.md` §1.5、§6；
- 「仅加乘积交互但抹平位置也失败」这一项说明：
  打破可加分离需要 $(q,k_p)$ 的**成对匹配**，不是任意非线性或任意门控。

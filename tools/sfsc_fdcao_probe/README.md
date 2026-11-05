# sfsc_fdcao_probe — 两个第一性原理猜想的判别性原型

纯 NumPy / fp64 / 手写 BPTT，**无 torch 依赖**。完整报告见
`../../docs/SFSC_FDCAO_PILOT.md`。

## 一句话

两个猜想都实现了、六臂 fp64 梯度检验全过，但**没有任何一臂超过等参数
attention**；SFSC 的「FFT/IIR 双形式对偶仍成立」被数值检验**直接证伪**
（时变核非 Toeplitz，单次 FFT 近似相对误差 0.972）。

## 六臂（共享 embedding / 残差 / 分类头，只换 mixer）

| 臂 | 机制 | mixer 参数 |
|---|---|---:|
| `static` | LTI 二阶 IIR，静态 $\rho,\theta$ | 1,056 |
| `sfsc` | $\rho_t,\theta_t$ 由状态 EMA 调制（LTV） | 3,136 |
| `fdcao` | 相位共振 $\cos(\theta_m-\phi_t)$，写入用当前相位 | 5,217 |
| `fdcao_p` | 同上，写入用**前一 token** 相位 | 5,217 |
| `attn` | 全秩 causal self-attention | 17,440 |
| `attn_lr` | 秩 8 低秩 attention（**与 FD-CAO 等参数**） | 5,152 |

## 任务

| 任务 | 判什么 | chance |
|---|---|---|
| `bind` | 两跳 MQAR（4 对 key/value + filler + query，L=13） | 1/16 |
| `bind1` | **单跳**内容寻址（组合 token 同编 key+value，L=9） | 1/8 |
| `mod` | 上下文调制（A/B 占比决定 regime，L=25） | 1/16 |
| `osc` | 周期信号频率分类（SGC 主场，L=32） | 1/8 |

## 复现

```bash
cd tools/sfsc_fdcao_probe
python sfsc_fdcao.py --mode gradcheck     # 6/6 PASS
python sfsc_fdcao.py --mode toeplitz      # is_toeplitz=false, fft_rel_error=0.972
python sfsc_fdcao.py --mode train --task bind1 --arm fdcao   --seed 0
python sfsc_fdcao.py --mode train --task bind1 --arm attn_lr --seed 0
python sfsc_fdcao.py --mode all           # 全任务 × 全臂 × seed 0/1/2
```

结果增量落 `results/results_sfsc_fdcao.json`，已完成的
`(task, arm, seed)` 自动跳过，可断点续跑。

## 已跑出的数字（**仅 seed 0**）

```
bind : static .2422 | sfsc .2500 | fdcao .2402 | fdcao_p .7656 | attn .2617 | attn_lr .2402
bind1: static .4160 | sfsc .4141 | fdcao .8672 | fdcao_p .6113 | attn 1.0000 | attn_lr 1.0000
mod  : static .5000 | sfsc .5352   （其余 4 臂未跑）
osc  : 未跑
```

⚠️ 三处别误读：

1. `bind` 上 `attn` 只有 .2617 是**单层 attention 一跳不够**（任务要两跳），
   不是 FD-CAO 赢了；`fdcao_p` 的 .7656 来自把「相邻 (key,value)」先验
   显式写进写入相位，换成 `bind1` 立刻掉到 .6113。
2. `bind1` 才是有效上界：等参数 `attn_lr` 5,152 参数 **1.0000**，
   FD-CAO 5,217 参数 **0.8672** → 超过静态核，**没超过 attention**。
3. `mod` 只跑 2/6 臂、单 seed、且任务一阶可解（`static` 的全局平均就能算），
   **不构成对 SFSC 的判别性检验**。

## 文件

| 文件 | 用途 |
|---|---|
| `sfsc_fdcao.py` | 六臂前向/反向、三+一任务、fp64 gradcheck、Toeplitz 检验、断点续跑 |
| `results/gradcheck.txt` | 六臂梯度检验原始输出 |
| `results/toeplitz.json` / `.txt` | 时变核非 Toeplitz 的数值证据 |
| `results/results_sfsc_fdcao.json` | 三臂×若干 seed 的逐 250 步曲线与最终 acc |

## 未完成（照实）

`mod` 剩 4 臂、`osc` 全六臂、≥3 seed、配对 bootstrap、4M 长上下文、
吞吐、流式 IIR 一致性、key→θ 动态映射、多层混合堆叠。
**本目录不是成功验证，是判别性排除 + 可续跑的前哨。**

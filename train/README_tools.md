# transfer_distill — 47 万参数原生 SimplCore 前哨训练

纯 NumPy 实现（**无 torch 依赖**），用于填补「无规模化训练空白」的第一个
可复现对照。完整报告见 `../../docs/TRANSFER_DISTILL_REPORT.md`。

## 为什么单独放一份白盒实现

本仓主线是 PyTorch + ROCm/CUDA 后端，需要 GPU 与完整依赖。这套代码是
**独立重实现**：核公式按 `simplcore/core/arch/sgc.py::build_kernel` 对齐
（`ψ[τ,m]=ρ_m^τ sin((τ+1)θ_m)`），但不 import 本仓任何模块，
因此在 2 核 CPU / 2 GB 内存里也能跑完三臂等预算对照。

它验证的是「**架构族能不能在真实语料上学**」，
不是「本仓那份 PyTorch 实现逐位正确」。两者结论不可互换。

## 文件

| 文件 | 用途 |
|---|---|
| `core.py` | SGC mixer（Toeplitz 整段 / IIR 流式）、causal Transformer、FFN、手写反向、AdamW |
| `data.py` | 字符级词表 / 语料编码 / batcher |
| `run.py` | 三臂训练、分块续跑、评估、生成、判决 |
| `gradcheck64.py` | fp64 中心差分梯度检验（训练前置） |
| `results/` | 三臂原始 log 与 `verdict.json` |

`corpus.txt`（6.3 MB）由本仓真实包内 700 个文本文件生成，未纳入版本控制；
首次运行会自动生成 `arr.npy` / `chars.json`。

## 复现

```bash
cd tools/transfer_distill
python gradcheck64.py          # 期望：GRADCHECK PASS (fp64) | ts True sgc True
for i in $(seq 1 6); do python run.py --arm ts  --total 3000 --chunk 500 --tag ts  --lr 6e-4; done
for i in $(seq 1 6); do python run.py --arm sgc --total 3000 --chunk 500 --tag sgc --lr 6e-4; done
for i in $(seq 1 6); do python run.py --arm kd  --total 3000 --chunk 500 --tag kd  --teacher ts --lr 6e-4; done
python run.py --final
```

单 chunk 500 步在 2 核 CPU 上约 25~130 s（KD 臂需额外跑冻结 teacher 前向）。

## 主结果（3.07M token/臂，单 seed）

| 臂 | 参数量 | val ppl | KL(T=1) vs teacher |
|---|---:|---:|---:|
| TS from scratch | 524,928 | 11.281 | 0.0000 |
| SGC from scratch | 436,096 | 10.817 | 0.7395 |
| SGC + 输出分布 KD | 436,096 | **10.217** | **0.4739** |

⚠️ **不要只看表**：三臂生成都仍是词堆砌、不成句；脚本自动给出的
`SUPPORTED` 判据过松，报告里已人工降级为 **PARTIAL**；
随机 teacher 对照（`kdrand`）未跑完；单 seed、字符级 V=512、L=64。
详见报告 §3 / §7。

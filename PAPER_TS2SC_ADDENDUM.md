# 论文增量：TS2SC 真实权重级观察（**独立补充材料，不进投稿包**）

适用稿件：`simplcore.pdf` / `paper_main.pdf`（v7.x，边界刻画 + 构造性反例定位）

**结论先行：本轮不修改投稿 PDF 正文，也不新增定理、观察编号或实验节。**

理由：本轮产出是工程验证（Stage-0 provenance、守恒断言、参数预算）与一次撤回登记，不是理论增量。按既定口径，工程日志整包进附录会让审稿人怀疑是几篇稿的剩料拼盘；而稿件定位是 boundary-characterization，混入权重迁移账本会稀释单一叙事。

## 若下一版决定吸收，只允许以下三处、且必须带 scope

1. **Limitations 增补一句**（不改变主张）：
   > Weight-level TS2SC provenance was verified on real BF16 teacher bytes
   > (310 tensors, 1.7206 B parameters from actual shapes); a static SGC
   > kernel evaluated on these values preserves Toeplitz structure,
   > FFT–IIR duality (7.0e−16), strict causality and O(1) state. This is a
   > structural check, not a language-modelling result.

2. **Appendix 可选一张预算表**（若且仅若 TSC 接口真的落进权重并跑通生成）：
   historical 1.0573 B / +learned T_h 1.0867 B / full Qwen3-1.7B 2.2688 B。
   现在**不放**，因为没有实现、没有验证。

3. **撤回登记 W14**：上一轮 `speak_smoke_results.json` 的乱码 top-k 来自修复前前向
   （RoPE 绝对位置、mask 广播、embedding/lm_head 基址四处缺陷），**不构成模型能力结论**。
   修正版完整 logits 因沙箱 502 中断未得到，故两个方向都不主张。

## 明确禁止写进论文的东西

- 「Qwen3-0.6B 不会说话」作为**架构定理**或**通用结论**（它只是对某个已有迁移包的判定）
- 「Qwen3-1.7B 基座能说话」（本轮**没测**，LFS tokenizer 403）
- 任何 ppl / 生成样本 / logit-KL / 吞吐 / LM parity / SOTA / 过稿率数字
- 「1.0867 B 的 TSC 接口已实现」——它目前只是一个参数预算对象

## 待解锁条件（解锁后再重写这一节）

| 条件 | 现状 |
|---|---|
| 真实 `tokenizer.json`（非 LFS blob） | 缺失，签名 URL 403 |
| torch / transformers 运行时 | 缺失，PyPI 403 |
| 4.06 GB 模型的算力预算（GPU 或足量 CPU） | 2 CPU / 2.3 GiB RAM / 370 MiB 可用磁盘 |
| 中文 prompt 的正确 id 编码 + 自回归生成 | 未做 |
| 与 teacher 的同条件生成对照 | 未做 |

**本文件不是论文正文，不随稿提交。**

# TS2SC 权重与迁移包 · 先看这个

打包日期：2026-11-08 · 目的：绕开 GitHub 直链，直接拿到本地已有的**真实权重字节**与完整 TS2SC 工具链。

---

## 一句最重要的话

**这个包里没有完整模型权重。** 沙盒磁盘只剩 372 MiB，装不下 3.11 GB 的 `MiniCPM-V-4.7-1B / TS2SC-VL-1B`，也装不下 4.06 GB 的 `Qwen3-1.7B-Instruct`；私有仓需要 `Contents: read` 的 PAT，匿名 404。所以包里给的是：

| 你拿到的 | 性质 |
|---|---|
| **2,000,000 B 真实 Qwen3-1.7B 分片前缀** | 真 BF16 字节，含完整 safetensors header + `embed_tokens` 前 256 行，可离线跑核守恒验证 |
| **9.89 MB 真实 tokenizer.json** | MiniCPM5-1B 侧真实文件（Qwen3 侧那个是 133 B LFS 指针，已如实保留） |
| **完整 index / header / 视觉元数据 / slices / 参数对账** | 权重的**结构描述**，虚拟读取靠它定位 offset |
| **全部 TS2SC 脚本** | 含 `ts2sc_680m.ps1`：在你机器上按 offset 只拉需要的分片，不用把整个仓库拖下来 |
| **全部验证结果 JSON** | 参数对账逐位相等、FFT/IIR 4.38e−16、真实 embedding 上的 Toeplitz/因果/O(1) |

完整权重仍在你私有仓里，由 `ts2sc_680m.ps1` 在你机器上**虚拟流式**按需读取——这正是「直链太慢」的解法：不拖全仓，只取要用的分片，且写本地缓存、断点可续。

---

## 目录

```
00_真实权重字节/      2 MB qa00 前缀 + 真实 header + 探针状态 npz
01_权重结构与索引/    model.safetensors.index.json、source header、vision_meta、
                      slices、param_accounting、各 config / generation_config
02_TS2SC_脚本/        ts2sc_680m.ps1（Windows/680M 驱动）
                      ts2sc_numpy_migrate.py、ts2sc_final.py、ts2sc_verify.py
                      migrated/scripts__*.py（build/repack/verify/load/extract/wsp）
                      speak_test.py、speak_staged.py、continue_round.py、verify_continue.py
03_验证结果/          ts2sc_*.json、speak_*.json、verify_*.json、evidence__*.json
04_文档/              PREFLIGHT / CONTINUE / QWEN3_VERIFY / PAPER_ADDENDUM /
                      MIGRATION_TIMING_ACCEPTANCE / migrated README / RESTORE /
                      上游 4 份 TS2SC 审计文档
05_tokenizer/         m5_tokenizer.json（真实 9.89 MB）、q17 tokcfg 与 133 B 指针
```

## 在你机器上怎么用（AMD 680M / Windows）

```powershell
$env:GH_TOKEN = '<你的 PAT>'   # 只在当前会话，脚本不打印、不写文件
powershell -ExecutionPolicy Bypass -File .\02_TS2SC_脚本\ts2sc_680m.ps1 -Mode Preflight
powershell -ExecutionPolicy Bypass -File .\02_TS2SC_脚本\ts2sc_680m.ps1 -Mode Stage0
powershell -ExecutionPolicy Bypass -File .\02_TS2SC_脚本\ts2sc_680m.ps1 -Mode Verify
```

`-Mode Full` 是**显式空操作**：teacher forward / logit-KL / ppl / 生成需要 torch-directml、248094 词表的码本分区、mrope/canvas 统一与跨模态对齐训练，目前没接通，不用一个无法解释的数字冒充成功。

## 成功 / 失败怎么判

- 退出码 `0` 正常、`4` 缺 token、`5` Verify 硬失败
- `out/stage0_provenance.json`：`UNCLASSIFIED == 0`、`virtual_data_bytes ≈ 3.109e9`、`model_materialised_on_disk == False`
- `out/verify_sgc.json`：`all_pass` + FFT/IIR `< 1e-8`、Toeplitz ✓、因果改变位 `0`、状态与 L 无关、零 N×N / KV cache

Stage0+Verify 只证明**权重可解析、SGC 核守恒在真实数值上成立**，不等于模型会说话。详细口径、耗时估算、常见失败对照表见 `04_文档/MIGRATION_TIMING_ACCEPTANCE.md`。

## 隐私与安全

- 包内**不含任何 token / PAT**（打包前已对全部文本文件做常见 PAT 前缀扫描，命中 0）
- 你之前在对话里贴过的 PAT 请尽快 revoke；脚本只从环境变量读，不落盘
- `qa00_prefix.bin` 是公开 Qwen3-1.7B 分片的只读前缀，不是私有权重本体

## 仍未接通（照实）

完整 3.11 GB 权重离线副本、真实 `q17_tokenizer.json`（现在是 LFS 指针）、torch/transformers 运行时、248094 词表码本分区、跨模态对齐训练、同条件 teacher 对照。因此本包**不是**「迁移完能说话的模型」，是「让你自己跑迁移、且能立刻验证核守恒」的全套证据与工具。

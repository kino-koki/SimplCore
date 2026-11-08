# TS2SC 迁移：耗时估算 / 成功失败验收 / 仍需修的地方

日期：2026-11-08 · 适用脚本：`ts2sc_680m.ps1` v1.0 · 目标机：AMD Radeon 680M（Windows）

---

## 一、大概要多久

下面的估算**全部来自这台沙盒的真实计时**，不是拍脑袋；目标机网络（国内访问 `api.github.com`）可能慢 2–5 倍或触发重试，所以给出区间。

### 实测基准

| 实测项 | 数值 |
|---|---|
| 单个 99.6 MB 分片下载 | **20.8 s**（4.80 MB/s） |
| 一次流式 full forward | 155.2 s，抓 1.464 GB，模型未落盘 |
| 分阶段 24 层前向 | stage0 157.2 s / stage1 84.7 s / stage2 71.1 s / stage3 104.7 s = **417.7 s ≈ 7.0 min**，共 2.330 GB（含跨阶段重复分片） |
| 3.109 GB 纯下载 | 3.109 GB ÷ 4.80 MB/s = **648 s ≈ 10.8 min** |
| 32 片 × 150 ms 强制暂停 | **4.8 s** |

### 按模式估时

| 模式 | 实际做什么 | 沙盒估计 | 你机器上的现实区间 |
|---|---|---|---|
| `Preflight` | 只读本机信息，不联网取权重 | < 5 s | 5–15 s |
| `Stage0` | **只取 part_00**（含完整 header），按真实张量名分类，不读数据区 | 20–40 s | **30–90 s** |
| `Verify` | 随机 offset 直读真实 embedding 前 256 行 + SGC 断言 | 30–90 s | **1–4 min** |
| 两者连跑（首次） | 分片写磁盘缓存 | 1–3 min | **2–6 min** |
| 缓存命中后重跑 | 不再联网，只做解析与数值断言 | 秒级 | **< 1 min** |
| 全量 32 片首次下载 | 只在显式扫描全部张量时才发生，默认路径**不会触发** | ≈ 11–15 min | **12–35 min**（视重试） |

**关键设计**：虚拟 safetensors 是**随机访问**，不是顺序流。`read(off, len)` 直接定位到目标分片并只下载那一片，不会为了读第 20 片而把前 19 片全拉下来。所以 Stage0 / Verify 绝大多数情况下只触碰 1–3 片。

### 「完整端到端迁移」是另一件事

`-Mode Full` 在这版脚本里是**显式 NO-OP**，且必须加 `-AllowFullCompute` 才放行。真正的 teacher forward / logit-KL / ppl / 生成需要：torch-directml 运行时、4.06 GB 模型算力、248094 词表的码本分区、mrope/canvas 统一、跨模态对齐训练。那是**小时到天级**的工程，当前没有实现，任何「几分钟迁移完就能说话」的说法都是假的。

---

## 二、怎么判定成功 / 失败

### 2.1 退出码（脚本 `exit $rc`）

| rc | 含义 | 处置 |
|---:|---|---|
| 0 | 正常完成 | 看下面 JSON 断言 |
| 2 | `-Mode Full` 未加 `-AllowFullCompute` | 加开关，或停在 Verify |
| 3 | Python / numpy 缺失 | `pip install numpy` |
| 4 | `GH_TOKEN` 未设置 | 只在当前会话设，不写文件 |
| 5 | **Verify 断言失败**（硬失败） | 看 `out/verify_sgc.json` 哪一项 False |
| 1 | 其它异常 | 看 `logs/ts2sc_<Mode>_<时间>.log` |

### 2.2 Stage0 成功判据（`out/stage0_provenance.json`）

```
status == "PLAN_ONLY_NO_WEIGHTS_WRITTEN"
n_tensors > 0                       # 真实 header 解析成功
class_counts["UNCLASSIFIED"] == 0   # 全部张量被归类，没有漏网
sum(class_counts.values()) == n_tensors
virtual_data_bytes ≈ 3.109e9        # 与仓库 32 个 vpart 的总字节一致
network.model_materialised_on_disk == False
open_interfaces 列出 4 条已知未接通项
```

`class_counts` 里 `COPY / DERIVE / NEW` 各自非零即可；**不要**把它读成「迁移已完成」——这是 provenance 计划，一个权重都没写。

### 2.3 Verify 成功判据（`out/verify_sgc.json`）

```
all_pass == true，且：
fft_vs_iir_rel_error         < 1e-8      # 实测同型断言 8.5e-16
conv_vs_iir_rel_error        < 1e-8
toeplitz_pass                == true
causal_future_perturbation_changed_past_bits == 0
state_independent_of_L       == true
state_bytes_L9 == state_bytes_L256
kernel_static == true
nn_score_or_kv_cache == false
embedding_tensor 是真实张量名；rows_read == 256；dtype == "BF16"
```

这五项分别钉住：FFT/IIR 对偶、Toeplitz 静态核、严格因果、O(1) 状态、无 N×N 打分与 KV cache。**它们是在真实 BF16 teacher 数值上跑的，不是合成信号**。

### 2.4 常见失败模式

| 现象 | 原因 | 处置 |
|---|---|---|
| 401 / 403 | token 失效、无 `Contents: read`、或 LFS 签名 URL 被拒 | 换新 token；确认仓库权限 |
| `IncompleteRead` ×4 后 `RuntimeError: part N failed` | 大 blob 传输中断 | 脚本已内置 4 次退避重试；再失败就分次跑（缓存保留已成功分片） |
| 下载到 133 B 的 `version https://git-lfs...` | 该对象是 LFS 指针不是权重 | 换普通 blob 分片；`vpart_*` 本身是真字节 |
| `size mismatch` | 磁盘缓存分片损坏 | 删除 `cache/part_NN.bin` 让它重下 |
| `offset outside virtual file` | header 与分片布局不一致 | 删除 `cache/layout.json` 重建 |
| Verify rc=5 | 数值断言失败 | **这是发现，不是崩溃**；保留 JSON，别改阈值 |

### 2.5 分层成功标准（最重要的一条）

```
Stage0 PASS + Verify PASS
  ⇒ 权重级 provenance 可解析、SGC 核守恒在真实数值上成立
  ⇏ 模型会说话 / 有语言能力 / ppl 合理 / 多模态可用
```

真正的能力验证需要：`tokenizer.json` 真实文件（现在私有仓里是 133 B LFS 指针，签名 URL 403）、torch/transformers、跨模态对齐训练、同条件 teacher 对照。**这四项目前一项都没有**，所以脚本刻意把 Full 留空，不用一个无法解释的数字冒充成功。

### 2.6 一条命令验收（PowerShell）

```powershell
$env:GH_TOKEN='<your-PAT>'
powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Stage0
powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Verify
$j = Get-Content .\ts2sc_680m_run\out\verify_sgc.json | ConvertFrom-Json
"all_pass = $($j.all_pass) | fft-iir = $($j.fft_vs_iir_rel_error) | state = $($j.state_bytes_L256) B"
if (-not $j.all_pass) { throw "Verify failed" }
```

---

## 三、仍需修的地方：本轮已全部落盘

| # | 问题 | 修法 | 落在哪 |
|---|---|---|---|
| 1 | Theorem A(a) 的 `≤1/2` 不总成立 | 主文改为 `acc(F\|v) ≤ 2^B/\|A(v)\|`；`≤1/2` 只在 `B ≤ log₂\|A(v)\|−1` 时成立；附录 A.1 补推论条件；写进 `\|A\|=3, B=1 ⇒ 2/3` 反例 | 主文 §3.1、附录 A.1 |
| 2 | Result 2(ii)「exactly H layers」需条件化 | 摘要、贡献、引言、A(e)、§3.3.2、§4.12、结论、附录 A.3 全部加 H6 接口 / A4 末层读出 / 不同键 / 每层至多一个内容匹配 / 分支因子与接口秩 c 前提；明写「不是架构无关定律」，并指出二元分支时 P5 的状态阶陈述相应退化 | 全文 10 处 |
| 3 | SE2 的 0.10% 是均值巧合 | 5 个 cover time `170,100,90,120,170`，mean 130.00，sample SD **38.08**，SEM **17.03**，t(4) 临界 2.776 ⇒ 95% CI **[82.72, 177.28]**（跨度约为均值的 73%）。三处（Table 16 说明、附录 A.8、Table 28）全部改口：均值与预测一致，但**不是精度验证**，需更多种子 | `ts2sc_vl47/se2_ci.json` 注入，正文/附录/表 28 |
| 4 | `B` / `b` / state bytes 记号混用 | 新增 **附录 A.0 记号与单位表**：`B` 总输入侧 bit、`b_t` 每位 alphabet excess、`b` 每轮每 insight bit（此处 5 = log₂V_K）、`w,T`、`state bytes`（字节，不是 bit）、`32768`（d=64 fp64 融合配置）、`c,o`；Table 8 表头 `state B` → `state bytes`；明确 Table 8 与融合表是**不同配置，不可数值比较** | §2 指引 + 附录 A.0 + Table 8 |
| 5 | 表格清理与转置一致性 | 附录 D 指向机器可读长表 `appendix_D_fusion_long.csv`（35 arm-seed 行 → 105 长表行，无 OCR 转录）；Table 10 第二列改名 `superseded (NOT USED)` 并加粗警示「只有五种子列进入任何主张，单种子值仅作溯源、绝不作证据」；`T*` 统一为 **`C_val^T C_key`**（value→key 映射，1/R 缩放由 H6 单位归一化吸收），消除 §2 与 §4.12 的转置歧义；顺手删掉 §2 重复句 | 全文 + 两个 CSV |

### 校验结果

| 检查 | 结果 |
|---|---|
| 生成器 `py_compile` | PASS |
| PDF 重建 | **17 页，主文 8 页，在 8 页限内** |
| 五项修正逐条 PDF 文本核验 | **16/16 PASS** |
| 既有 desk-check 回归 | **46/46 PASS**（D1–D7 + P + P1） |
| 拆分投稿包 | `paper_main.pdf` 9 页 / `paper_appendix.pdf` 9 页，禁用片段 0，**11/11 PASS** |

> 两个页数口径不冲突：生成器报「主文正文止于第 8 页」，desk-check / 拆分把 Broader Impact 与 References 计入第 9 页；正文仍在 8 页预算内。

### 没有圆的地方

Theorem A 的紧性仍只在显式 binary-chain 构造族上成立，原 content-chain 任务上界是松的；≥10M token LM、同算力强基线、真实 MQAR、fused kernel、TS2SC 端到端生成、ICML 2027 官方 `.sty` 与截稿日，一项未补。不作 LM parity / SOTA / 过稿率承诺。

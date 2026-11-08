# TS2SC 完整代码包 · 清单与边界

打包日期：2026-11-08 · 与上一轮 `TS2SC_权重与迁移包.zip` 互补：那个给权重字节 + 脚本 + 结果，这个给**全部代码文件**。

---

## 一句话

按「文件名 + 文件内容」双重扫描了整个工作区，把 TS2SC 迁移链路的代码收齐；上一轮包里只有 16 个脚本，本包补上了**上游迁移库、HF 权重桥接、导出、相关测试、SGC 架构本体**，并附一份完整上游源码快照兜底。

**总计 259 个文件**（含 README 与 MANIFEST，去重后），压缩后 1.07 MiB。

---

## 分组

| 目录 | 内容 | 数量 | 性质 |
|---|---|---:|---|
| `00_核心迁移脚本/` | 本地实际跑过的 TS2SC 脚本 | 16 | 上一轮已给，本包原样保留（去掉打包时的 `migrated__` 前缀，恢复真实文件名） |
| `01_上游迁移库/` | `simplcore/core/migration/`、`model/hf_bridge.py`、`model/export.py`、`core/common.py` 等 | 8 | **本轮新补**，迁移链路的库代码 |
| `02_相关测试/` | `test_dc_bias.py`、`test_hf_bridge.py`、`test_export.py`、`test_sgc_mok.py` | 4 | **本轮新补** |
| `03_架构依赖/` | `arch/sgc.py`（81 KB）、`arch/__init__.py`、远程副本 `remote_simplcore_core_arch_sgc.py` | 3 | **本轮新补**，TS2SC 迁移的目标架构，不是迁移脚本本身 |
| `04_完整上游源码快照/simplcore/` | 上游 `simplcore/` 包全部 `.py`，保留原目录层级 | 226 | 兜底：防止任何相关文件被漏掉；**大部分不是 TS2SC 专属** |

### 00 组逐文件

```
ts2sc_680m.ps1                       AMD 680M / Windows 低占用驱动（内嵌纯 numpy worker）
ts2sc_numpy_migrate.py               无 torch 的真实 safetensors 虚拟读取 + COPY/DERIVE
ts2sc_final.py                       真实权重字节上的参数对账与 SGC 对偶验证
ts2sc_verify.py                      Stage0 provenance / 守恒断言复核
ts2sc_vl47/continue_round.py         VL47 续跑
ts2sc_vl47/speak_test.py             MiniCPM5-1B 流式 full forward 冒烟
ts2sc_vl47/speak_staged.py           分阶段前向
ts2sc_vl47/verify_continue.py        续跑结果复核
ts2sc_vl47/migrated/scripts__apply_dc_bias.py
ts2sc_vl47/migrated/scripts__build_vl1b.py
ts2sc_vl47/migrated/scripts__download_vision.py
ts2sc_vl47/migrated/scripts__extract_vision.py
ts2sc_vl47/migrated/scripts__load_vl1b.py
ts2sc_vl47/migrated/scripts__repack_vl1b.py
ts2sc_vl47/migrated/scripts__verify_vl1b.py
ts2sc_vl47/migrated/scripts__wsp_value_audit.py
```

> 注意：`scripts__*.py` 是本地归档时的扁平化命名，原仓库里它们位于 `scripts/` 目录下。本包保留该文件名以免与历史证据链脱节。

---

## 文档引用、但沙盒里**不存在**的脚本（不伪造、不编造）

上游 4 份 TS2SC 审计报告里提到过下面这些脚本，它们属于**私有 TS2SC 仓库**，本工作区没有副本，因此**不在包内**。逐名列出以便你到私有仓里核对：

```
q17_migrate.py  q17_teacher.py  q17_reader.py  q17_calibrate.py  q17_diag.py
q17_fit2.py     q17_gen.py      q17_oracle.py  q17_oracle_head.py q17_v4.py
analytic_transfer.py  bias_fix.py  optimize_fix.py  verify_profile_ceiling.py
eval_ckpt.py    train_lm.py     test_capability.py
TS2SC/scripts/migrate_real.py   TS2SC/scripts/real_act_distill.py
```

另：`chain_stack_probe.py` 在 `train/superposition_probe/` 里存在，但它是合成内容链探针，**不是 TS2SC 迁移代码**，故未收入。

---

## 明确排除的类别（避免注水）

- 论文 PDF 生成器 `build_simplcore_pdf_*.py`、补丁脚本 `patch_*.py`
- 推送工具 `push_*.py`、日志更新器 `update_logs_v7.py`
- 全部 `.md` 报告、`.json` 结果、`.pdf`（这些在上一轮的权重包里）
- 上游 `.archive/java-high-performance/` 的 Java 视觉模块：它是 SimplCore 早期视觉管线，与 TS2SC 权重迁移无调用关系

如果你要的是「连论文生成器一起的全部源码」，说一声，我另出一个全仓源码包。

---

## 校验与安全

- `MANIFEST.json`：每个文件的字节数 + sha256，解压后可逐个核对
- 打包前对全部文本文件做 PAT 前缀扫描，**命中 0**；包内不含任何凭据或私有权重字节
- zip 完整性 `testzip()` PASS

## 仍未接通

`migrate_real.py` / `real_act_distill.py` 等缺失脚本、真实 `q17_tokenizer.json`（133 B LFS 指针）、torch/transformers 运行时、248094 词表码本分区、跨模态对齐训练。因此本包是**代码全集**，不是「跑完就能说话的模型」。用法与验收口径见上一轮包的 `04_文档/MIGRATION_TIMING_ACCEPTANCE.md`。

# MI250 战役就绪报告（CPU 侧预演验收）

> **日期**：2026-11-16　**环境**：torch 2.14.1+cpu / Windows（无 ROCm）
> **结论**：1B→3B 战役的**数据管线、训练入口、闸门清单、启动预检**已在
> CPU 上全链路预演通过；战役实跑（1B→3B 训练、4M 长上下文、LRA 榜）仍需
> MI250 硬件。预演过程中发现并修复 **2 个会在上卡首日爆发的真实缺陷**。

## 1. 预演结果（`research/binding/mi250_rehearsal.py`）

| 步骤 | 内容 | 结果 |
|------|------|------|
| 1 数据管线 | `corpus/cn_*.jsonl`（1500 docs）→ `pack_documents` → `data/processed/tuan_zh_smoke/`（train/val/test `.bin` + `index.json` 含逐分片 SHA-256；87/10/4 窗，vocab 423） | ✅ |
| 2 闸门清单 | `campaign --json`：P0（后端一致性 9 硬闸）/ P1（1B 训练 5 闸）/ 复盘（3 闸）/ P2（2B 适配器 4 闸）/ X（6 评测门）全量落盘 | ✅ |
| 3 启动预检 | `train.py --dry-run --preset tuan-1b --mi250 --train-chunk 8192 --max-steps 0`：1B 配置构建 + ROCm tuning + 4M 课程阶梯（4K→1M→4M）+ 分块预算断言 | ✅ rc=0 |
| 4 训练环路 | toy preset + 冒烟语料实跑 3 步（数据→前向→反向→eval→checkpoint 落盘） | ✅ rc=0 |

账本：`artifacts/binding_probe/mi250_rehearsal.json`；
闸门全文：`runs/rehearsal/dryrun_1b.log` 与本报告 §3。

## 2. 预演发现并修复的缺陷（上卡首日必爆）

1. **`python -m simplcore.core.train` 在 compat-alias 层下必炸**
   （`ModuleNotFoundError: No module named 'simplcore.model'`）。
   根因：`_compat_aliases` 只把**顶层**旧名映射进 `sys.modules`
   （`simplcore.core.train` → 真实包 `simplcore.core.training.train`），
   而 `-m` 入口会以**别名包名**重新加载 `__main__`/`main`/`cli`——
   别名名比真实名少一段（`train` vs `training.train`），包内的三级相对
   导入 `from ...model.config import ...` 因此解析到 `simplcore.model`。
   **修复**：`_alias_package_submodules` 把别名包的全部真实子模块
   （跳过 `__main__`）按真实名预先 import 并 setdefault 进 `sys.modules`——
   别名查找直接命中已初始化模块，不再在别名包名下重跑。
2. **启动器缺 `--train-chunk`**：4M 上下文课程的既定路径是分块 BPTT
   （campaign `P1-chunked-4m-nooom`），整段 FFT 的 4M 核需 ~30 GiB >
   默认 `--max-kernel-gib 4`，启动预检直接拒绝。**修复**：两个 MI250
   启动器默认 `TRAIN_CHUNK=8192`（可置 0 关闭），并把 `-m` 模块路径改为
   规范名 `simplcore.core.training.train`。

## 3. 闸门清单与验收归属

| 闸门 | 验收场所 | 状态 |
|------|----------|------|
| P0-backend-conformance（16 内核） | MI250（ROCm/HIP 编译） | ⬜ 待硬件 |
| P0-stream-bitexact / chunked-carry / frozen-mask / bf16-accum / shard-2gcd / no-attention / budget-frozen | MI250 实机 | ⬜ 待硬件 |
| P1-chunked-4m-nooom / doubling-claim / stream-bitexact-4m / loss-stable / token-budget | MI250 训练期 | ⬜ 待硬件 |
| R-*（复盘）/ P2-*（2B 适配器） | MI250 训练期 | ⬜ 待硬件 |
| X-equal-budget / X-heldout-ppl | **CPU 已预演**（smoke 全过；small 档 32 步欠训练已入账） | 🟡 smoke ✅ / small 待补 |
| X-longctx-retrieval / X-stream-consistency | 依赖 1B 训练产物 | ⬜ 待硬件 |
| X-code-passk / X-throughput | 依赖 1B 训练产物 | ⬜ 待硬件 |
| 数据管线（pack + SHA-256 manifest + hash 校验消费） | **CPU 已验** | ✅ |
| 训练入口（args 兼容 MI250 启动器 / dry-run 预检 / checkpoint 落盘） | **CPU 已验** | ✅ |

## 4. 硬件到位后的执行清单（runbook）

```bash
# 0) 上卡前自检（本报告 §1 的 1-4 步应在目标机重跑一遍）
python scripts/probe_batch.py --preset tuan-1b --seq-len 4096   # 定 micro-batch

# 1) Phase 0：后端一致性 + 数据冻结（30 天硬停）
bash scripts/build_ref_backend.sh
python -m simplcore.core.backend_contract
bash scripts/train_tuan_mi250.sh --preset tuan-1b --micro-batch <probe 输出> \
    --global-batch-tokens 2000000 --max-steps 100000 \
    --data-dir data/processed/tuan_zh           # 正式语料（生产清洗）

# 2) Phase 1：1B 基座（4M 课程，chunked BPTT 已默认开启）
# 3) 复盘 → Phase 2：2B 适配器（基座冻结）
# 4) X 闸门评测：equal_budget / longctx_eval / code_passk / throughput
# 5) 回填 README「当前真实能力边界」与本文档状态列
```

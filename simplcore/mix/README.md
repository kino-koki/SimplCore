# SimplCore Mix Pure Text Training Base

`simplcore/mix` is the only active model base for the current training route.

The architecture name remains `SimplCore Mix`. Python 3.11 is the required
Python development target, while Java26 is the high-performance hot-path target.
The current local validation environment may still run an earlier Python
interpreter.

## Causality Status (2026-11)

三处非因果算子已于 2026-11 修复（见 `docs/HONEST_STATUS.md` 第 0.2 节）：

- `cin.py` 上下文自注意力：加入有限大负数因果掩码（`-1e4` triu，与 DCGD 同策略），
  位置 t 只聚合 s ≤ t 的上下文。
- `memory.py` 工作记忆更新：全序列均值 `x.mean(dim=1)` 改为**因果前缀均值**
  （cumsum/cumcount 前缀和形式），第 t 位置的 gate/decay/新记忆只由 s ≤ t 的输入决定。
- `blocks.py` Hebbian 在线适应：全局均值改为因果前缀统计，逐位置外积 + 时间维
  前缀累积，末位值只依赖全序列过去。

守护测试：`tests_simplcore_mix/test_mix_causality.py`（5 项，CPU 全绿）。

> 注意：本路线仍为自研 NumPy 自动微分引擎（`import simplcore.tensor as torch`），
> 非 PyTorch，AMP 关闭。因果性修复使其可用于自回归 LM 训练，但性能与可训练性
> 边界以本机实测为准。

## Scope

- Pure text causal language model only.
- No vision, speech, camera, multimodal fusion, or world-model training in this route.
- Simplified Chinese pretraining data only.
- The local AMD Radeon Graphics 680M machine is used for code checks, data cleaning, smoke training, and stacking dry-runs.
- Full 7B and 14B training requires a larger GPU environment.

## Route

1. Repair and validate `simplcore/mix` imports.
2. Prepare legal public Simplified Chinese text data.
3. Run tiny and 150M smoke training.
4. Train multiple approximately 150M text models.
5. Stack their parameters into a 7B target model.
6. Continue-train the 7B model for calibration.
7. Extend from calibrated 7B to 14B.

## Active Modules

- `core/`: SimplCore-owned public module, parameter, layer, and backend aliases.
- `config.py`: model configurations.
- `model.py`: `SimplCoreMix` model.
- `fusion.py`: deep SimplCore fusion components from the Java SimplCore line.
- `blocks.py`: Prelude, recurrent, and coda blocks.
- `attention.py`: text attention modules.
- `moe.py`: text MoE modules.
- `memory.py`: dynamic memory channel.
- `tokenizer.py`: text tokenizer.
- `data_pipeline/`: Simplified Chinese data preparation.
- `training/`: smoke and full training entrypoints.
- `stacking/`: parameter stacking utilities.
- `tests/`: local validation tests.

## Self-Owned Surface

Core model files use SimplCore-owned public names such as `SCModule`,
`SCLinear`, `SCEmbedding`, `SCParameter`, `SCDropout`, and `SCRMSNorm`.
The default backend is `simplcore_torch`, which is a transitional hidden
backend rather than the architecture identity. Future work can replace it with
`simplcore_native` or the Java high-performance backend without changing the
public SimplCore Mix module names.

The recurrent block now integrates the Java SimplCore component semantics:

- `CIN/DeepCIN`: causal intervention network (DeepCIN: dynamic graph + do-operator + back-door adjustment).
- `IB`: information bottleneck.
- `MRD`: meta representation distiller.
- `PUM`: plasticity update module.

## Frontier Target

`SimplCore Mix` currently reports `current_tier=T3+` and `target_tier=T0`.
The T0 target is an engineering and research contract, not a claim that the
model has already reached T0. Required gates include complete 150M training,
7B stacking calibration, Chinese benchmark evaluation, ablation studies,
Java26 hot-path validation, Python3.11 ecosystem orchestration, long-context
stability checks, and weight export/runtime consistency.

## Prepared Local Data

- Public corpus download target: `Fineweb-Edu-Chinese-V2.2` under the E-drive text-data directory.
- User-authorized `.docx` corpus: 13 Simplified Chinese documents from the user-provided E-drive corpus folder.
- Converted user corpus output: `processed/user_docx/user_docx_corpus.jsonl`, with `user_docx_corpus.manifest.json` recording source files, sample count, chunk size, and SHA-256.
- Use `simplcore.mix.data_pipeline.prepare_docx_corpus(...)` to regenerate the `.docx` JSONL shard.

## Deprecated For This Route

The old `qingxiaotuan` package route is not used by this training path. Multimodal, vision, speech, camera, and world-model modules remain in the repository but are outside this route.

## Local Validation

Run the pure-text route tests:

```powershell
python -m pytest simplcore/mix/tests simplcore/mix/test_fullstack.py -q
```

### Handoff Notes (merged from TRAINING_HANDOFF.md)

Local machine (AMD Radeon 680M) is validated for: import tests, tiny forward/backward,
Chinese data cleaning, JSONL shard writing, `.docx` corpus extraction, parameter stacking
dry-run, tiny smoke training. **Not validated for full 7B/14B pretraining.**

Active corpus paths:
- Public shards: `E:\qingxiaotuan_text_data\raw\Fineweb-Edu-Chinese-V2.2\2_3`
- User `.docx` JSONL: `E:\qingxiaotuan_text_data\processed\user_docx\user_docx_corpus.jsonl`

7B preparation gates:
- Python 3.11 required before 150M/7B prep.
- 680M local target ≈ 150M params; 7B built by stacking eligible 150M checkpoints + guarded calibration.
- Training loss target < 1.5; absolute max 3.9 (stop & rollback above 3.9).

Verified (2026-07-27..29): `py_compile` OK; `pytest simplcore/mix/tests` 16 tests passed;
`run_smoke_training(steps=1, batch_size=1, seq_len=8)` completed one optimizer step,
backward OK, wrote `artifacts/mix_smoke/train_smoke.pt`. Data-pipeline tests (8) passed;
610 records from 13 `.docx` sources converted, manifest SHA-256
`f6055df3eb91505bc21a2e1f13f28a0faaeab9ef0b87f7bc98dd4e3ee809065f`.

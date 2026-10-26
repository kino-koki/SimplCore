# SimplCore Mix Training Handoff

## Local Machine

The current AMD Radeon Graphics 680M machine is validated for:

- Import tests.
- Tiny forward/backward.
- Simplified Chinese data cleaning.
- JSONL shard writing.
- User-authorized `.docx` corpus extraction.
- Parameter stacking dry-run.
- Tiny smoke training.

It is not validated for full 7B or 14B pretraining.

## Active Corpus Paths

- Public corpus shards: `E:\qingxiaotuan_text_data\raw\Fineweb-Edu-Chinese-V2.2\2_3`
- User-authorized `.docx` JSONL: `E:\qingxiaotuan_text_data\processed\user_docx\user_docx_corpus.jsonl`
- User-authorized `.docx` manifest: `E:\qingxiaotuan_text_data\processed\user_docx\user_docx_corpus.manifest.json`

Latest local shard check in this handoff:

- Parquet count: `6840`
- Continuous range: `000000.parquet` through `006839.parquet`
- Missing shard IDs in that range: none observed
- Total public shard bytes: `12825972317`

## 7B Preparation Gates

- Python runtime must be Python `3.11` before starting real 150M or 7B preparation runs.
- AMD 680M local training target is approximately `150M` parameters.
- `7B` is prepared by stacking eligible `150M` checkpoints, then running guarded calibration.
- Gradient checkpointing is enabled for `mix_150m_text` and `mix_7b_text` training configurations.
- Deep-thinking and web-search capability flags are enabled in the 7B text target config for tool-augmented runtime integration.
- Training loss target is `< 1.5`; absolute maximum is `3.9`.
- If loss exceeds `3.9`, training must stop and roll back to the last stable checkpoint.
- If loss is between `1.5` and `3.9`, training may continue only with mitigation such as learning-rate decay and data-batch inspection.

## Local Commands

```powershell
python -m py_compile 'simplcore/mix/model.py' 'simplcore/mix/config.py'
python -m pytest 'simplcore/mix/tests' 'simplcore/mix/test_fullstack.py' -q
python -c "from pathlib import Path; from simplcore.mix.training import run_smoke_training; print(run_smoke_training(Path('artifacts/mix_smoke'), steps=1, batch_size=1, seq_len=8))"
```

## Verified Result

Final local verification completed on 2026-07-27:

- `python -m py_compile simplcore/mix/model.py simplcore/mix/config.py` completed successfully.
- `python -m pytest simplcore/mix/tests simplcore/mix/test_fullstack.py -q` passed with 16 tests.
- `run_smoke_training(Path("artifacts/mix_smoke"), steps=1, batch_size=1, seq_len=8)` completed one optimizer step, ran backward propagation, and wrote `artifacts/mix_smoke/train_smoke.pt`.

Additional local data-pipeline verification completed on 2026-07-29:

- `python -m pytest simplcore/mix/tests/test_data_cleaning.py simplcore/mix/tests/test_shard_writer.py simplcore/mix/tests/test_docx_corpus.py -q` passed with 8 tests.
- The user-authorized `.docx` corpus was converted into `processed/user_docx/user_docx_corpus.jsonl`.
- The converted `.docx` shard contains 610 training records from 13 source documents, with manifest SHA-256 `f6055df3eb91505bc21a2e1f13f28a0faaeab9ef0b87f7bc98dd4e3ee809065f`.

Smoke training output:

```text
{'steps': 1, 'batch_size': 1, 'seq_len': 8, 'loss': 4.162770748138428, 'optimizer_steps': 1, 'backward_ran': True, 'checkpoint_path': 'artifacts\\mix_smoke\\train_smoke.pt'}
```

## Next Training Stages

1. Finish downloading and validating Simplified Chinese public corpus shards.
2. Merge public shards with the user-authorized `.docx` JSONL shard.
3. Run 150M smoke training.
4. Train multiple 150M checkpoints.
5. Build 7B stacking manifest.
6. Stack 7B target checkpoint.
7. Continue-train 7B for calibration on a larger GPU environment.
8. Expand calibrated 7B to 14B.

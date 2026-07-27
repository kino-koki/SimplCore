# SimplCore Mix Training Handoff

## Local Machine

The current AMD Radeon Graphics 680M machine is validated for:

- Import tests.
- Tiny forward/backward.
- Simplified Chinese data cleaning.
- JSONL shard writing.
- Parameter stacking dry-run.
- Tiny smoke training.

It is not validated for full 7B or 14B pretraining.

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

Smoke training output:

```text
{'steps': 1, 'batch_size': 1, 'seq_len': 8, 'loss': 4.162770748138428, 'optimizer_steps': 1, 'backward_ran': True, 'checkpoint_path': 'artifacts\\mix_smoke\\train_smoke.pt'}
```

## Next Training Stages

1. Prepare Simplified Chinese public corpus shards.
2. Run 150M smoke training.
3. Train multiple 150M checkpoints.
4. Build 7B stacking manifest.
5. Stack 7B target checkpoint.
6. Continue-train 7B for calibration on a larger GPU environment.
7. Expand calibrated 7B to 14B.

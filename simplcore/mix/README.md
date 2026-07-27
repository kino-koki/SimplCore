# SimplCore Mix Pure Text Training Base

`simplcore/mix` is the only active model base for the current training route.

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

- `config.py`: model configurations.
- `model.py`: `SimplCoreMix` model.
- `blocks.py`: Prelude, recurrent, and coda blocks.
- `attention.py`: text attention modules.
- `moe.py`: text MoE modules.
- `memory.py`: dynamic memory channel.
- `tokenizer.py`: text tokenizer.
- `data_pipeline/`: Simplified Chinese data preparation.
- `training/`: smoke and full training entrypoints.
- `stacking/`: parameter stacking utilities.
- `tests/`: local validation tests.

## Deprecated For This Route

The old `qingxiaotuan` package route is not used by this training path. Multimodal, vision, speech, camera, and world-model modules remain in the repository but are outside this route.

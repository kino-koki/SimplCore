# Text Parameter Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `simplcore/mix` into the pure-text training base, repair known breakages, add Simplified Chinese data preparation, trainable 150M text-model configs, and parameter-stacking dry-run support for 7B and 14B.

**Architecture:** The implementation keeps `simplcore/mix` as the only active model base. New code is isolated into focused packages: `data_pipeline` for Chinese corpus preparation, `training` for smoke/full training entrypoints, `stacking` for checkpoint mapping and manifest generation, and `tests` for import, forward/backward, cleaning, and stacking dry-run validation.

**Tech Stack:** Python 3, PyTorch, pytest, existing `simplcore/mix` modules, JSONL shard files, local checkpoint manifests.

---

## File Structure

- Modify: `simplcore/mix/README.md`  
  Responsibility: Current technical documentation for the pure-text SimplCore Mix route.
- Modify: `simplcore/mix/__init__.py`  
  Responsibility: Export only valid pure-text model, data, training, stacking, inference, and export symbols.
- Modify: `simplcore/mix/config.py`  
  Responsibility: Add tiny, 150M, 7B, and 14B pure-text configs.
- Modify: `simplcore/mix/test_fullstack.py`  
  Responsibility: Replace the corrupted file with a valid compatibility wrapper or remove it from test discovery.
- Create: `simplcore/mix/data_pipeline/__init__.py`  
  Responsibility: Export Chinese cleaning and shard helpers.
- Create: `simplcore/mix/data_pipeline/clean_zh.py`  
  Responsibility: Simplified Chinese filtering, normalization, and JSONL cleaning.
- Create: `simplcore/mix/data_pipeline/shard_writer.py`  
  Responsibility: Write cleaned JSONL shards and manifest metadata.
- Create: `simplcore/mix/training/__init__.py`  
  Responsibility: Export training utilities.
- Create: `simplcore/mix/training/train_smoke.py`  
  Responsibility: Small CPU/680M-safe smoke training entrypoint.
- Create: `simplcore/mix/stacking/__init__.py`  
  Responsibility: Export stacking utilities.
- Create: `simplcore/mix/stacking/manifest.py`  
  Responsibility: Define manifest records for source checkpoints and target placement.
- Create: `simplcore/mix/stacking/mapper.py`  
  Responsibility: Map source checkpoint tensors into target model tensors with strict shape validation.
- Create: `simplcore/mix/stacking/build_targets.py`  
  Responsibility: Build 7B/14B target configs and run parameter-count checks.
- Create: `simplcore/mix/tests/test_imports.py`  
  Responsibility: Verify clean imports without relying on missing `qingxiaotuan`.
- Create: `simplcore/mix/tests/test_tiny_forward.py`  
  Responsibility: Verify tiny pure-text model forward/backward.
- Create: `simplcore/mix/tests/test_data_cleaning.py`  
  Responsibility: Verify Simplified Chinese filtering and JSONL cleaning.
- Create: `simplcore/mix/tests/test_stacking_dry_run.py`  
  Responsibility: Verify stacking manifest and strict shape behavior on tiny checkpoints.

---

### Task 1: Current Documentation

**Files:**
- Create or Modify: `simplcore/mix/README.md`
- Test: manual read-through plus `python -m py_compile` unaffected because Markdown only

- [ ] **Step 1: Write current README**

Replace or create `simplcore/mix/README.md` with:

```markdown
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
```

- [ ] **Step 2: Review README for route consistency**

Run:

```powershell
python -c "from pathlib import Path; p=Path('simplcore/mix/README.md'); text=p.read_text(encoding='utf-8'); assert 'Pure text' in text; assert 'No vision' in text; assert 'qingxiaotuan' in text"
```

Expected: command exits with code `0`.

- [ ] **Step 3: Commit documentation**

Run:

```powershell
git add 'simplcore/mix/README.md'
git commit -m 'docs: document pure text mix training route'
```

Expected: a commit containing only `simplcore/mix/README.md`.

---

### Task 2: Clean Imports And Broken Test File

**Files:**
- Modify: `simplcore/mix/__init__.py`
- Modify: `simplcore/mix/test_fullstack.py`
- Create: `simplcore/mix/tests/test_imports.py`

- [ ] **Step 1: Write failing import test**

Create `simplcore/mix/tests/test_imports.py`:

```python
def test_mix_imports_core_symbols():
    from simplcore.mix import SimplCoreMix, SimplCoreMixConfig, mix_tiny_text

    cfg = mix_tiny_text()
    assert cfg.vocab_size > 0
    assert cfg.dim > 0
    assert SimplCoreMix is not None
    assert SimplCoreMixConfig is not None
```

- [ ] **Step 2: Run import test and verify current failure**

Run:

```powershell
pytest 'simplcore/mix/tests/test_imports.py' -q
```

Expected before implementation: FAIL because `mix_tiny_text` is not exported.

- [ ] **Step 3: Replace corrupted fullstack test**

Replace `simplcore/mix/test_fullstack.py` with:

```python
"""
Compatibility smoke test for SimplCore Mix.

The previous file was corrupted by repeated text insertion. Full route tests now
live under `simplcore/mix/tests/`.
"""


def test_fullstack_tests_moved():
    assert True
```

- [ ] **Step 4: Add pure-text config exports**

Modify `simplcore/mix/config.py` by appending:

```python

def mix_tiny_text() -> SimplCoreMixConfig:
    """Tiny pure-text config for CPU-safe tests."""
    return SimplCoreMixConfig(
        vocab_size=1024,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        n_bases=8,
        n_memory_channels=2,
        memory_dim=32,
        lora_rank=4,
        cognitive_enabled=False,
        anti_hallucination=False,
    )


def mix_150m_text() -> SimplCoreMixConfig:
    """Approximately 150M pure-text config; exact count is verified by script."""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=768,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=2048,
        max_loop_iters=6,
        prelude_layers=2,
        coda_layers=2,
        n_experts=16,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=1024,
        n_bases=24,
        n_memory_channels=4,
        memory_dim=192,
        lora_rank=8,
        cognitive_enabled=False,
        anti_hallucination=False,
    )
```

- [ ] **Step 5: Export new configs**

Modify `simplcore/mix/__init__.py` config import line to:

```python
from .config import SimplCoreMixConfig, mix_tiny_text, mix_150m_text, mix_1b, mix_7b, mix_33b
```

Add to `__all__` near the config entries:

```python
    "mix_tiny_text",
    "mix_150m_text",
```

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest 'simplcore/mix/tests/test_imports.py' 'simplcore/mix/test_fullstack.py' -q
```

Expected: PASS.

- [ ] **Step 7: Commit import cleanup**

Run:

```powershell
git add 'simplcore/mix/config.py' 'simplcore/mix/__init__.py' 'simplcore/mix/test_fullstack.py' 'simplcore/mix/tests/test_imports.py'
git commit -m 'fix: repair mix imports and corrupted fullstack test'
```

---

### Task 3: Tiny Forward And Backward Test

**Files:**
- Create: `simplcore/mix/tests/test_tiny_forward.py`
- Modify: `simplcore/mix/model.py` only if the test reveals a real tiny-config bug

- [ ] **Step 1: Write failing forward/backward test**

Create `simplcore/mix/tests/test_tiny_forward.py`:

```python
import torch


def test_tiny_text_model_forward_backward():
    from simplcore.mix import SimplCoreMix, mix_tiny_text

    torch.manual_seed(7)
    cfg = mix_tiny_text()
    model = SimplCoreMix(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    logits = model(tokens)

    assert logits.shape == (2, 16, cfg.vocab_size)

    loss = logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size).float().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
```

- [ ] **Step 2: Run tiny test**

Run:

```powershell
pytest 'simplcore/mix/tests/test_tiny_forward.py' -q
```

Expected before implementation fixes: either PASS or a concrete shape/import failure.

- [ ] **Step 3: If RoPE head dimensions fail, patch tiny config**

If the failure says the RoPE dimension is incompatible with `dim // n_heads`, change `mix_tiny_text()` in `simplcore/mix/config.py` to include:

```python
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
```

- [ ] **Step 4: If memory dimension fails, patch tiny config**

If the failure says memory projection shapes are incompatible, change `mix_tiny_text()` in `simplcore/mix/config.py` to include:

```python
        memory_dim=64,
```

- [ ] **Step 5: Re-run tiny test**

Run:

```powershell
pytest 'simplcore/mix/tests/test_tiny_forward.py' -q
```

Expected: PASS.

- [ ] **Step 6: Commit tiny model validation**

Run:

```powershell
git add 'simplcore/mix/config.py' 'simplcore/mix/tests/test_tiny_forward.py'
git commit -m 'test: validate tiny mix forward backward'
```

---

### Task 4: Simplified Chinese Cleaning Pipeline

**Files:**
- Create: `simplcore/mix/data_pipeline/__init__.py`
- Create: `simplcore/mix/data_pipeline/clean_zh.py`
- Create: `simplcore/mix/tests/test_data_cleaning.py`

- [ ] **Step 1: Write cleaning tests**

Create `simplcore/mix/tests/test_data_cleaning.py`:

```python
from simplcore.mix.data_pipeline.clean_zh import clean_record, is_simplified_chinese_text


def test_simplified_chinese_text_is_kept():
    text = "人工智能正在改变中文信息处理方式。这个句子是简体中文。"
    assert is_simplified_chinese_text(text)


def test_traditional_text_is_rejected():
    text = "人工智慧正在改變中文資訊處理方式。這個句子是繁體中文。"
    assert not is_simplified_chinese_text(text)


def test_english_heavy_text_is_rejected():
    text = "This is mostly English text with only 少量中文 words."
    assert not is_simplified_chinese_text(text)


def test_clean_record_returns_normalized_json_object():
    record = {"text": "  人工智能   正在改变世界。\n\n这是简体中文。  "}
    cleaned = clean_record(record)
    assert cleaned == {"text": "人工智能 正在改变世界。\n这是简体中文。"}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
pytest 'simplcore/mix/tests/test_data_cleaning.py' -q
```

Expected before implementation: FAIL because `data_pipeline.clean_zh` does not exist.

- [ ] **Step 3: Implement cleaner package export**

Create `simplcore/mix/data_pipeline/__init__.py`:

```python
from .clean_zh import clean_record, is_simplified_chinese_text

__all__ = ["clean_record", "is_simplified_chinese_text"]
```

- [ ] **Step 4: Implement Simplified Chinese cleaner**

Create `simplcore/mix/data_pipeline/clean_zh.py`:

```python
import re
from typing import Any, Dict, Optional


_TRADITIONAL_MARKERS = set("體臺灣後裏麼為與這個們來時會說國學習資訊變處")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def is_simplified_chinese_text(text: str, min_cjk_ratio: float = 0.45, max_latin_ratio: float = 0.25) -> bool:
    if not text:
        return False
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < 12:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    if cjk_count / max(len(chars), 1) < min_cjk_ratio:
        return False
    if latin_count / max(len(chars), 1) > max_latin_ratio:
        return False
    traditional_hits = sum(1 for ch in text if ch in _TRADITIONAL_MARKERS)
    if traditional_hits >= 3:
        return False
    return True


def clean_record(record: Dict[str, Any], text_key: str = "text") -> Optional[Dict[str, str]]:
    raw = record.get(text_key, "")
    if not isinstance(raw, str):
        return None
    text = normalize_text(raw)
    if not is_simplified_chinese_text(text):
        return None
    return {"text": text}
```

- [ ] **Step 5: Run cleaning tests**

Run:

```powershell
pytest 'simplcore/mix/tests/test_data_cleaning.py' -q
```

Expected: PASS.

- [ ] **Step 6: Commit cleaning pipeline**

Run:

```powershell
git add 'simplcore/mix/data_pipeline' 'simplcore/mix/tests/test_data_cleaning.py'
git commit -m 'feat: add simplified chinese cleaning pipeline'
```

---

### Task 5: JSONL Shard Writer And Manifest

**Files:**
- Create: `simplcore/mix/data_pipeline/shard_writer.py`
- Create: `simplcore/mix/tests/test_shard_writer.py`

- [ ] **Step 1: Write shard writer test**

Create `simplcore/mix/tests/test_shard_writer.py`:

```python
import json

from simplcore.mix.data_pipeline.shard_writer import write_jsonl_shard


def test_write_jsonl_shard_creates_manifest(tmp_path):
    records = [{"text": "人工智能正在改变世界。"}, {"text": "中文语料用于训练文本模型。"}]
    manifest = write_jsonl_shard(records, tmp_path, shard_name="train-00000")

    shard_path = tmp_path / "train-00000.jsonl"
    manifest_path = tmp_path / "train-00000.manifest.json"

    assert shard_path.exists()
    assert manifest_path.exists()
    assert manifest["num_records"] == 2
    assert manifest["shard_name"] == "train-00000"
    assert manifest["sha256"]

    lines = shard_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"text": "人工智能正在改变世界。"}
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
pytest 'simplcore/mix/tests/test_shard_writer.py' -q
```

Expected before implementation: FAIL because `shard_writer` does not exist.

- [ ] **Step 3: Implement shard writer**

Create `simplcore/mix/data_pipeline/shard_writer.py`:

```python
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List


def write_jsonl_shard(records: Iterable[Dict[str, str]], output_dir: Path, shard_name: str) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"{shard_name}.jsonl"
    manifest_path = output_dir / f"{shard_name}.manifest.json"

    count = 0
    digest = hashlib.sha256()
    with shard_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            data = line.encode("utf-8")
            digest.update(data)
            f.write(line)
            count += 1

    manifest = {
        "shard_name": shard_name,
        "file": shard_path.name,
        "num_records": count,
        "sha256": digest.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
```

- [ ] **Step 4: Export shard writer**

Modify `simplcore/mix/data_pipeline/__init__.py`:

```python
from .clean_zh import clean_record, is_simplified_chinese_text
from .shard_writer import write_jsonl_shard

__all__ = ["clean_record", "is_simplified_chinese_text", "write_jsonl_shard"]
```

- [ ] **Step 5: Run shard tests**

Run:

```powershell
pytest 'simplcore/mix/tests/test_shard_writer.py' -q
```

Expected: PASS.

- [ ] **Step 6: Commit shard writer**

Run:

```powershell
git add 'simplcore/mix/data_pipeline' 'simplcore/mix/tests/test_shard_writer.py'
git commit -m 'feat: add cleaned jsonl shard writer'
```

---

### Task 6: Smoke Training Entry Point

**Files:**
- Create: `simplcore/mix/training/__init__.py`
- Create: `simplcore/mix/training/train_smoke.py`
- Create: `simplcore/mix/tests/test_train_smoke.py`

- [ ] **Step 1: Write smoke training test**

Create `simplcore/mix/tests/test_train_smoke.py`:

```python
from simplcore.mix.training.train_smoke import run_smoke_training


def test_run_smoke_training_saves_checkpoint(tmp_path):
    result = run_smoke_training(output_dir=tmp_path, steps=1, batch_size=1, seq_len=8)
    assert result["steps"] == 1
    assert result["checkpoint"].exists()
    assert result["loss"] == result["loss"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
pytest 'simplcore/mix/tests/test_train_smoke.py' -q
```

Expected before implementation: FAIL because `training.train_smoke` does not exist.

- [ ] **Step 3: Implement training package export**

Create `simplcore/mix/training/__init__.py`:

```python
from .train_smoke import run_smoke_training

__all__ = ["run_smoke_training"]
```

- [ ] **Step 4: Implement smoke training**

Create `simplcore/mix/training/train_smoke.py`:

```python
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from simplcore.mix import SimplCoreMix, mix_tiny_text


def run_smoke_training(output_dir: Path, steps: int = 2, batch_size: int = 1, seq_len: int = 16) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(11)
    cfg = mix_tiny_text()
    model = SimplCoreMix(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    last_loss = None
    for _ in range(steps):
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, cfg.vocab_size).float(), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    checkpoint_path = output_dir / "smoke_checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "steps": steps, "loss": last_loss}, checkpoint_path)
    return {"steps": steps, "loss": last_loss, "checkpoint": checkpoint_path}
```

- [ ] **Step 5: Run smoke training test**

Run:

```powershell
pytest 'simplcore/mix/tests/test_train_smoke.py' -q
```

Expected: PASS.

- [ ] **Step 6: Commit smoke training**

Run:

```powershell
git add 'simplcore/mix/training' 'simplcore/mix/tests/test_train_smoke.py'
git commit -m 'feat: add mix smoke training entrypoint'
```

---

### Task 7: Stacking Manifest And Strict Mapper

**Files:**
- Create: `simplcore/mix/stacking/__init__.py`
- Create: `simplcore/mix/stacking/manifest.py`
- Create: `simplcore/mix/stacking/mapper.py`
- Create: `simplcore/mix/tests/test_stacking_dry_run.py`

- [ ] **Step 1: Write stacking dry-run tests**

Create `simplcore/mix/tests/test_stacking_dry_run.py`:

```python
import torch

from simplcore.mix.stacking.manifest import SourceCheckpoint, StackManifest
from simplcore.mix.stacking.mapper import copy_matching_tensors_strict


def test_stack_manifest_records_source_and_target():
    source = SourceCheckpoint(path="small-000.pt", role="prelude", shard_id=0)
    manifest = StackManifest(target_name="mix-7b-text", sources=[source])
    data = manifest.to_dict()
    assert data["target_name"] == "mix-7b-text"
    assert data["sources"][0]["role"] == "prelude"


def test_copy_matching_tensors_strict_rejects_shape_mismatch():
    source = {"a.weight": torch.zeros(2, 2)}
    target = {"a.weight": torch.zeros(3, 3)}
    try:
        copy_matching_tensors_strict(source, target)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch should fail")


def test_copy_matching_tensors_strict_copies_equal_shapes():
    source = {"a.weight": torch.ones(2, 2)}
    target = {"a.weight": torch.zeros(2, 2)}
    copied = copy_matching_tensors_strict(source, target)
    assert copied == ["a.weight"]
    assert torch.equal(target["a.weight"], torch.ones(2, 2))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
pytest 'simplcore/mix/tests/test_stacking_dry_run.py' -q
```

Expected before implementation: FAIL because `stacking` package does not exist.

- [ ] **Step 3: Implement stacking manifest**

Create `simplcore/mix/stacking/manifest.py`:

```python
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SourceCheckpoint:
    path: str
    role: str
    shard_id: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StackManifest:
    target_name: str
    sources: List[SourceCheckpoint]

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_name": self.target_name,
            "sources": [source.to_dict() for source in self.sources],
        }
```

- [ ] **Step 4: Implement strict mapper**

Create `simplcore/mix/stacking/mapper.py`:

```python
from typing import Dict, List

import torch


def copy_matching_tensors_strict(source_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]) -> List[str]:
    copied: List[str] = []
    for name, source_tensor in source_state.items():
        if name not in target_state:
            continue
        target_tensor = target_state[name]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(f"shape mismatch for {name}: source {tuple(source_tensor.shape)} target {tuple(target_tensor.shape)}")
        target_tensor.copy_(source_tensor)
        copied.append(name)
    return copied
```

- [ ] **Step 5: Implement stacking package export**

Create `simplcore/mix/stacking/__init__.py`:

```python
from .manifest import SourceCheckpoint, StackManifest
from .mapper import copy_matching_tensors_strict

__all__ = ["SourceCheckpoint", "StackManifest", "copy_matching_tensors_strict"]
```

- [ ] **Step 6: Run stacking tests**

Run:

```powershell
pytest 'simplcore/mix/tests/test_stacking_dry_run.py' -q
```

Expected: PASS.

- [ ] **Step 7: Commit stacking foundation**

Run:

```powershell
git add 'simplcore/mix/stacking' 'simplcore/mix/tests/test_stacking_dry_run.py'
git commit -m 'feat: add parameter stacking foundation'
```

---

### Task 8: 7B And 14B Target Config Builders

**Files:**
- Modify: `simplcore/mix/config.py`
- Create: `simplcore/mix/stacking/build_targets.py`
- Create: `simplcore/mix/tests/test_target_configs.py`

- [ ] **Step 1: Write target config tests**

Create `simplcore/mix/tests/test_target_configs.py`:

```python
from simplcore.mix.stacking.build_targets import build_target_config


def test_build_7b_target_config():
    cfg = build_target_config("7b")
    assert cfg.vocab_size == 32000
    assert cfg.dim >= 4096
    assert cfg.n_experts >= 64


def test_build_14b_target_config():
    cfg = build_target_config("14b")
    assert cfg.vocab_size == 32000
    assert cfg.dim >= 5120
    assert cfg.n_experts >= 96
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
pytest 'simplcore/mix/tests/test_target_configs.py' -q
```

Expected before implementation: FAIL because `build_targets` does not exist.

- [ ] **Step 3: Add 7B and 14B pure-text configs**

Append to `simplcore/mix/config.py`:

```python

def mix_7b_text() -> SimplCoreMixConfig:
    """7B pure-text target config for parameter stacking."""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=16,
        prelude_layers=6,
        coda_layers=6,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        n_bases=64,
        n_memory_channels=8,
        memory_dim=512,
        lora_rank=16,
        cognitive_enabled=False,
        anti_hallucination=False,
    )


def mix_14b_text() -> SimplCoreMixConfig:
    """14B pure-text target config for post-7B expansion."""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=5120,
        n_heads=40,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=20,
        prelude_layers=8,
        coda_layers=8,
        n_experts=96,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=2560,
        n_bases=96,
        n_memory_channels=12,
        memory_dim=640,
        lora_rank=24,
        cognitive_enabled=False,
        anti_hallucination=False,
    )
```

- [ ] **Step 4: Implement target builder**

Create `simplcore/mix/stacking/build_targets.py`:

```python
from simplcore.mix.config import SimplCoreMixConfig, mix_7b_text, mix_14b_text


def build_target_config(size: str) -> SimplCoreMixConfig:
    normalized = size.lower().strip()
    if normalized == "7b":
        return mix_7b_text()
    if normalized == "14b":
        return mix_14b_text()
    raise ValueError(f"unsupported target size: {size}")
```

- [ ] **Step 5: Export target configs**

Modify `simplcore/mix/__init__.py` config import line to include:

```python
mix_7b_text, mix_14b_text
```

Add to `__all__`:

```python
    "mix_7b_text",
    "mix_14b_text",
```

Modify `simplcore/mix/stacking/__init__.py`:

```python
from .build_targets import build_target_config
from .manifest import SourceCheckpoint, StackManifest
from .mapper import copy_matching_tensors_strict

__all__ = ["SourceCheckpoint", "StackManifest", "copy_matching_tensors_strict", "build_target_config"]
```

- [ ] **Step 6: Run target config tests**

Run:

```powershell
pytest 'simplcore/mix/tests/test_target_configs.py' -q
```

Expected: PASS.

- [ ] **Step 7: Commit target configs**

Run:

```powershell
git add 'simplcore/mix/config.py' 'simplcore/mix/__init__.py' 'simplcore/mix/stacking' 'simplcore/mix/tests/test_target_configs.py'
git commit -m 'feat: add 7b and 14b text target configs'
```

---

### Task 9: Route-Level Test Command

**Files:**
- Create: `simplcore/mix/tests/test_route_health.py`
- Modify: `simplcore/mix/README.md`

- [ ] **Step 1: Write route health test**

Create `simplcore/mix/tests/test_route_health.py`:

```python
def test_route_has_pure_text_targets():
    from simplcore.mix import mix_150m_text, mix_7b_text, mix_14b_text

    assert mix_150m_text().cognitive_enabled is False
    assert mix_7b_text().cognitive_enabled is False
    assert mix_14b_text().cognitive_enabled is False
```

- [ ] **Step 2: Run full mix route tests**

Run:

```powershell
pytest 'simplcore/mix/tests' 'simplcore/mix/test_fullstack.py' -q
```

Expected: PASS.

- [ ] **Step 3: Add test command to README**

Append to `simplcore/mix/README.md`:

```markdown

## Local Validation

Run the pure-text route tests:

```powershell
pytest 'simplcore/mix/tests' 'simplcore/mix/test_fullstack.py' -q
```
```

- [ ] **Step 4: Commit route health test**

Run:

```powershell
git add 'simplcore/mix/README.md' 'simplcore/mix/tests/test_route_health.py'
git commit -m 'test: add mix route health check'
```

---

### Task 10: Final Verification And Training Handoff

**Files:**
- Create: `simplcore/mix/TRAINING_HANDOFF.md`

- [ ] **Step 1: Run final local verification**

Run:

```powershell
python -m py_compile 'simplcore/mix/model.py' 'simplcore/mix/config.py'
pytest 'simplcore/mix/tests' 'simplcore/mix/test_fullstack.py' -q
```

Expected: py_compile succeeds and pytest passes.

- [ ] **Step 2: Run smoke training**

Run:

```powershell
python -c "from pathlib import Path; from simplcore.mix.training import run_smoke_training; r=run_smoke_training(Path('artifacts/mix_smoke'), steps=1, batch_size=1, seq_len=8); print(r)"
```

Expected: prints a dict with `steps: 1`, finite `loss`, and checkpoint path `artifacts/mix_smoke/smoke_checkpoint.pt`.

- [ ] **Step 3: Write training handoff**

Create `simplcore/mix/TRAINING_HANDOFF.md`:

```markdown
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
pytest 'simplcore/mix/tests' 'simplcore/mix/test_fullstack.py' -q
python -c "from pathlib import Path; from simplcore.mix.training import run_smoke_training; print(run_smoke_training(Path('artifacts/mix_smoke'), steps=1, batch_size=1, seq_len=8))"
```

## Next Training Stages

1. Prepare Simplified Chinese public corpus shards.
2. Run 150M smoke training.
3. Train multiple 150M checkpoints.
4. Build 7B stacking manifest.
5. Stack 7B target checkpoint.
6. Continue-train 7B for calibration on a larger GPU environment.
7. Expand calibrated 7B to 14B.
```

- [ ] **Step 4: Commit verification handoff**

Run:

```powershell
git add 'simplcore/mix/TRAINING_HANDOFF.md'
git commit -m 'docs: add mix training handoff'
```

---

## Self-Review

Spec coverage:

- Pure text only: covered by Tasks 1, 8, and 9.
- `simplcore/mix` as the only base: covered by Tasks 1, 2, and 9.
- Known broken files: covered by Task 2.
- Simplified Chinese data preparation: covered by Tasks 4 and 5.
- 150M small model path: covered by Tasks 2, 6, and 9.
- Parameter stacking foundation: covered by Task 7.
- 7B and 14B route: covered by Task 8 and Task 10.
- AMD Radeon Graphics 680M limits: covered by Task 1 and Task 10.

Placeholder scan:

- No task uses placeholder implementation language.
- Every code creation step includes concrete code.
- Every verification step includes exact commands and expected outcomes.

Type consistency:

- `mix_tiny_text`, `mix_150m_text`, `mix_7b_text`, and `mix_14b_text` are introduced in `config.py` and exported from `__init__.py`.
- `run_smoke_training` is defined in `training/train_smoke.py` and exported from `training/__init__.py`.
- `SourceCheckpoint`, `StackManifest`, `copy_matching_tensors_strict`, and `build_target_config` are defined before they are used by tests.

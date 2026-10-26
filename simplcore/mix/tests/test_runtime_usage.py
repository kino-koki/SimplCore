import json


class _TinyTokenizer:
    def encode(self, text: str):
        return [(ord(ch) % 31) + 1 for ch in text]


def test_build_data_loader_returns_simplcore_loader(tmp_path):
    from simplcore.mix import SCDataLoader, build_data_loader

    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps({"text": "青小团正在训练SimplCore Mix架构。" * 8}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loader = build_data_loader(
        str(data_path),
        _TinyTokenizer(),
        batch_size=1,
        max_seq_len=8,
        num_workers=0,
    )

    assert isinstance(loader, SCDataLoader)
    x, y = next(iter(loader))
    assert x.shape == y.shape


def test_smoke_training_reports_simplcore_optimizer(tmp_path):
    from simplcore.mix.training.train_smoke import run_train_smoke

    result = run_train_smoke(tmp_path, steps=1, batch_size=1, seq_len=8, seed=3)

    assert result["optimizer_backend"] == "simplcore_native"
    assert result["optimizer_name"] == "SCAdamW"

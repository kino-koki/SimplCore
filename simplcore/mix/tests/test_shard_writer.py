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

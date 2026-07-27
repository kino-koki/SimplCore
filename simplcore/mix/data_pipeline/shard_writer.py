import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable


def write_jsonl_shard(
    records: Iterable[Dict[str, str]],
    output_dir: Path,
    shard_name: str,
) -> Dict[str, object]:
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
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest

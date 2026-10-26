from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def check_python311() -> ReadinessCheck:
    version = sys.version_info
    passed = version.major == 3 and version.minor == 11
    return ReadinessCheck(
        name="python311",
        passed=passed,
        detail=f"当前解释器为 Python {version.major}.{version.minor}.{version.micro}，7B 准备要求 Python 3.11",
    )


def check_paths_exist(paths: Iterable[str | Path]) -> List[ReadinessCheck]:
    checks: List[ReadinessCheck] = []
    for raw_path in paths:
        path = Path(raw_path)
        checks.append(
            ReadinessCheck(
                name=f"path:{path.name}",
                passed=path.exists(),
                detail=f"{path} {'存在' if path.exists() else '不存在'}",
            )
        )
    return checks


def build_7b_readiness_report(
    *,
    public_corpus_dir: str | Path,
    user_docx_jsonl: str | Path,
) -> Dict[str, object]:
    checks = [
        check_python311(),
        *check_paths_exist([public_corpus_dir, user_docx_jsonl]),
    ]
    corpus_dir = Path(public_corpus_dir)
    parquet_count = len(list(corpus_dir.glob("*.parquet"))) if corpus_dir.exists() else 0
    return {
        "target": "SimplCore Mix 7B stacking calibration",
        "checks": [check.to_dict() for check in checks],
        "passed": all(check.passed for check in checks),
        "public_parquet_count": parquet_count,
        "required_loss": {
            "target_loss": 1.5,
            "max_loss": 3.9,
        },
    }

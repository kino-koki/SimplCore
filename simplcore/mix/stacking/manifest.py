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

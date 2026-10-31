import os
import json
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.audit_logger")


@dataclass
class AuditEntry:
    event_id: str
    event_type: str
    timestamp: str
    source: str
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    severity: str
    integrity_hash: str
    previous_hash: str = ""


@dataclass
class AuditReport:
    start_time: str
    end_time: str
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    integrity_verified: bool
    tampered_entries: List[str]
    summary: str = ""
    generated_at: str = ""


class AuditLogger:
    def __init__(
        self,
        log_directory: str = "./logs/audit/",
        hash_key: str = "qingxiaotuan_audit_secret_key_v1",
        max_file_size_mb: int = 100,
        backup_count: int = 30,
    ):
        self.log_directory = Path(log_directory)
        self.hash_key = hash_key.encode("utf-8")
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self._entries: List[AuditEntry] = []
        self._last_hash = "0" * 64
        self._stats = {
            "total_entries": 0,
            "entries_by_type": {},
            "entries_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        }
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self._current_file = self._get_current_log_file()
        self._load_existing_entries()
        logger.info(f"AuditLogger initialized at {self.log_directory}")

    def _get_current_log_file(self) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_directory / f"qingxiaotuan_audit_{date_str}.jsonl"

    def _load_existing_entries(self) -> None:
        if self._current_file.exists():
            try:
                with open(self._current_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = self._parse_entry(line)
                                if entry:
                                    self._entries.append(entry)
                                    self._last_hash = entry.integrity_hash
                            except json.JSONDecodeError:
                                continue
                logger.info(f"Loaded {len(self._entries)} existing audit entries")
            except Exception as e:
                logger.error(f"Failed to load audit log: {e}")

    def _parse_entry(self, line: str) -> Optional[AuditEntry]:
        try:
            data = json.loads(line)
            return AuditEntry(
                event_id=data.get("event_id", ""),
                event_type=data.get("event_type", ""),
                timestamp=data.get("timestamp", ""),
                source=data.get("source", ""),
                user_id=data.get("user_id", ""),
                action=data.get("action", ""),
                resource=data.get("resource", ""),
                details=data.get("details", {}),
                severity=data.get("severity", "low"),
                integrity_hash=data.get("integrity_hash", ""),
                previous_hash=data.get("previous_hash", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse audit entry: {e}")
            return None

    def log_event(
        self,
        event_type: str,
        source: str,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "low",
    ) -> str:
        event_id = hashlib.sha256(
            f"{datetime.now().timestamp()}{source}{user_id}{action}".encode()
        ).hexdigest()[:16]
        timestamp = datetime.now().isoformat()
        entry_data = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "source": source,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": details or {},
            "severity": severity,
            "previous_hash": self._last_hash,
        }
        integrity_input = json.dumps(entry_data, sort_keys=True, ensure_ascii=False)
        integrity_hash = hmac.new(
            self.hash_key,
            integrity_input.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        entry_data["integrity_hash"] = integrity_hash
        entry = AuditEntry(**entry_data)
        self._entries.append(entry)
        self._last_hash = integrity_hash
        self._stats["total_entries"] += 1
        self._stats["entries_by_type"][event_type] = self._stats["entries_by_type"].get(event_type, 0) + 1
        self._stats["entries_by_severity"][severity] = self._stats["entries_by_severity"].get(severity, 0) + 1
        self._write_entry(entry)
        return event_id

    def _write_entry(self, entry: AuditEntry) -> None:
        data = asdict(entry)
        data["details"] = self._sanitize_for_json(data.get("details", {}))
        line = json.dumps(data, ensure_ascii=False) + "\n"
        if self._current_file.stat().st_size >= self.max_file_size:
            self._rotate_log()
        try:
            with open(self._current_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")

    def _sanitize_for_json(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def _rotate_log(self) -> None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"qingxiaotuan_audit_{date_str}.jsonl"
        rotated_path = self.log_directory / rotated_name
        try:
            os.rename(self._current_file, rotated_path)
            self._current_file = self._get_current_log_file()
            self._cleanup_old_logs()
            logger.info(f"Rotated audit log to {rotated_name}")
        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")

    def _cleanup_old_logs(self) -> None:
        log_files = sorted(
            [f for f in self.log_directory.glob("qingxiaotuan_audit_*.jsonl")],
            key=os.path.getmtime,
        )
        while len(log_files) > self.backup_count:
            oldest = log_files.pop(0)
            try:
                os.remove(oldest)
                logger.info(f"Removed old audit log: {oldest}")
            except Exception as e:
                logger.error(f"Failed to remove old log: {e}")

    def verify_integrity(self, from_beginning: bool = True) -> Dict[str, Any]:
        tampered = []
        verified_count = 0
        previous_hash = "0" * 64
        log_files = sorted(
            self.log_directory.glob("qingxiaotuan_audit_*.jsonl"),
            key=os.path.getmtime,
        )
        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry_data = json.loads(line)
                        stored_hash = entry_data.get("integrity_hash", "")
                        entry_prev_hash = entry_data.get("previous_hash", "")
                        if from_beginning and entry_prev_hash != previous_hash:
                            tampered.append(f"{log_file.name}:{line_num}")
                        check_data = {k: v for k, v in entry_data.items() if k != "integrity_hash"}
                        check_input = json.dumps(check_data, sort_keys=True, ensure_ascii=False)
                        expected_hash = hmac.new(
                            self.hash_key,
                            check_input.encode("utf-8"),
                            hashlib.sha256,
                        ).hexdigest()
                        if stored_hash != expected_hash:
                            tampered.append(f"{log_file.name}:{line_num}")
                        previous_hash = stored_hash
                        verified_count += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        tampered.append(f"{log_file.name}:{line_num} (parse error: {e})")
        return {
            "verified_count": verified_count,
            "tampered_count": len(tampered),
            "is_intact": len(tampered) == 0,
            "tampered_entries": tampered[:20],
        }

    def query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        results = []
        for entry in reversed(self._entries):
            if len(results) >= limit:
                break
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if event_type and entry.event_type != event_type:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if severity and entry.severity != severity:
                continue
            results.append(asdict(entry))
        return results

    def generate_report(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> AuditReport:
        if start_time is None:
            start_time = (datetime.now() - timedelta(days=7)).isoformat()
        if end_time is None:
            end_time = datetime.now().isoformat()
        filtered = [
            e for e in self._entries
            if start_time <= e.timestamp <= end_time
        ]
        events_by_type = {}
        events_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for entry in filtered:
            events_by_type[entry.event_type] = events_by_type.get(entry.event_type, 0) + 1
            events_by_severity[entry.severity] = events_by_severity.get(entry.severity, 0) + 1
        integrity_result = self.verify_integrity()
        return AuditReport(
            start_time=start_time,
            end_time=end_time,
            total_events=len(filtered),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            integrity_verified=integrity_result["is_intact"],
            tampered_entries=integrity_result["tampered_entries"],
            summary=f"审计期间共{len(filtered)}条记录，完整性{'完好' if integrity_result['is_intact'] else '存在异常'}",
            generated_at=datetime.now().isoformat(),
        )

    def get_statistics(self) -> Dict[str, Any]:
        integrity = self.verify_integrity()
        return {
            **self._stats,
            "integrity_status": "intact" if integrity["is_intact"] else "compromised",
            "total_log_files": len(list(self.log_directory.glob("qingxiaotuan_audit_*.jsonl"))),
            "current_file_size_mb": round(self._current_file.stat().st_size / 1024 / 1024, 2),
            "last_hash": self._last_hash[:16],
        }

    def get_entries_by_severity(self, severity: str, limit: int = 50) -> List[Dict[str, Any]]:
        return [
            asdict(e) for e in reversed(self._entries)
            if e.severity == severity
        ][:limit]

    def export_to_file(self, filepath: str, format: str = "json") -> bool:
        try:
            path = Path(filepath)
            if format == "json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([asdict(e) for e in self._entries], f, ensure_ascii=False, indent=2)
            elif format == "jsonl":
                with open(path, "w", encoding="utf-8") as f:
                    for entry in self._entries:
                        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            logger.info(f"Exported {len(self._entries)} audit entries to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
            return False

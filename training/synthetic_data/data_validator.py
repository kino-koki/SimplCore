"""
青小团 合成数据生成引擎 - 数据质量验证模块
青奈科技 (QingNai Technology)

提供全面的数据质量验证功能：格式验证、内容验证、
语言一致性检查、有毒内容过滤、PII检测和验证报告生成。
"""

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from base_generator import DataRecord

logger = logging.getLogger("qingxiaotuan.validator")

_PII_PATTERNS = {
    "phone_cn": re.compile(r'1[3-9]\d{9}'),
    "id_card_cn": re.compile(r'\d{17}[\dXx]'),
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "bank_card": re.compile(r'\d{16,19}'),
}

_TOXIC_KEYWORDS = [
    "暴力", "色情", "赌博", "毒品", "恐怖", "歧视",
    "自杀", "杀人", "诈骗", "仇恨", "侮辱", "虐待",
]

_LANGUAGE_RANGES = {
    "zh-CN": (('\u4e00', '\u9fff'), ('\u3000', '\u303f'), ('\uff00', '\uffef')),
    "en": (('a', 'z'), ('A', 'Z')),
    "ja": (('\u3040', '\u309f'), ('\u30a0', '\u30ff')),
}


class ValidationReport:
    def __init__(self):
        self.total_records = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.errors: dict[str, int] = {}
        self.warnings: list[str] = []
        self.validation_time: Optional[str] = None
        self.per_type_stats: dict[str, dict] = {}

    def add_error(self, error_type: str):
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "valid_ratio": round(self.valid_count / self.total_records, 4) if self.total_records > 0 else 0,
            "errors": dict(sorted(self.errors.items(), key=lambda x: -x[1])),
            "warnings": self.warnings[:10],
            "validation_time": self.validation_time,
            "per_type_stats": self.per_type_stats,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "青小团 数据验证报告",
            "=" * 50,
            f"验证时间：{self.validation_time}",
            f"总记录数：{self.total_records}",
            f"有效记录：{self.valid_count} ({self.valid_count/self.total_records*100:.1f}%)" if self.total_records > 0 else "有效记录：0",
            f"无效记录：{self.invalid_count}",
            "-" * 50,
            "错误分布：",
        ]
        for err_type, count in sorted(self.errors.items(), key=lambda x: -x[1]):
            lines.append(f"  {err_type}: {count}")
        if self.warnings:
            lines.append("-" * 50)
            lines.append("警告：")
            for w in self.warnings[:5]:
                lines.append(f"  ⚠ {w}")
        if self.per_type_stats:
            lines.append("-" * 50)
            lines.append("各类数据统计：")
            for dtype, stats in self.per_type_stats.items():
                lines.append(f"  {dtype}: {stats}")
        lines.append("=" * 50)
        return "\n".join(lines)


class DataValidator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        val_cfg = config.get("validation", {})
        self.checks = val_cfg.get("checks", {})
        self.thresholds = val_cfg.get("thresholds", {})
        self.toxicity_threshold = val_cfg.get("toxicity_threshold", 0.8)
        self.report_dir = val_cfg.get("report_dir", "./validation_reports")
        self.halt_on_error = val_cfg.get("halt_on_error", False)
        self._duplicate_cache: set[str] = set()

    def check_format(self, record: DataRecord) -> bool:
        if not isinstance(record, DataRecord):
            return False
        if not record.id or not isinstance(record.id, str):
            return False
        if not record.data_type or not isinstance(record.data_type, str):
            return False
        if not record.content or not isinstance(record.content, dict):
            return False
        return True

    def check_content_empty(self, record: DataRecord) -> bool:
        if not record.content:
            return True
        for key, value in record.content.items():
            if isinstance(value, str) and value.strip():
                return False
            if isinstance(value, (list, dict)) and value:
                return False
        return True

    def check_content_length(self, record: DataRecord) -> tuple[bool, str]:
        min_len = self.thresholds.get("min_length", 10)
        max_len = self.thresholds.get("max_length", 8192)
        text = json.dumps(record.content, ensure_ascii=False)
        length = len(text)
        if length < min_len:
            return False, f"过短({length}<{min_len})"
        if length > max_len:
            return False, f"过长({length}>{max_len})"
        return True, ""

    def check_duplicate(self, record: DataRecord) -> bool:
        content_hash = hash(json.dumps(record.content, sort_keys=True, ensure_ascii=False))
        if content_hash in self._duplicate_cache:
            return True
        self._duplicate_cache.add(content_hash)
        return False

    def check_language_consistency(self, record: DataRecord) -> bool:
        text = json.dumps(record.content, ensure_ascii=False)
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total = zh_chars + en_chars
        if total == 0:
            return True
        zh_ratio = zh_chars / total
        if 0.1 < zh_ratio < 0.9 and zh_chars > 30 and en_chars > 30:
            return False
        return True

    def check_toxicity(self, record: DataRecord) -> tuple[bool, float]:
        text = json.dumps(record.content, ensure_ascii=False)
        matched_keywords = 0
        for keyword in _TOXIC_KEYWORDS:
            if keyword in text:
                matched_keywords += 1
        toxicity_score = matched_keywords / len(_TOXIC_KEYWORDS)
        return toxicity_score < self.toxicity_threshold, toxicity_score

    def check_pii(self, record: DataRecord) -> list[str]:
        text = json.dumps(record.content, ensure_ascii=False)
        detected_pii = []
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii.append(f"{pii_type}({len(matches)}处)")
        return detected_pii

    def check_quality_score(self, record: DataRecord) -> float:
        score = 1.0
        text = json.dumps(record.content, ensure_ascii=False)
        if len(text) < 50:
            score -= 0.1
        if self.check_content_empty(record):
            score -= 0.3
        zh_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        if zh_chars == 0 and en_chars == 0:
            score -= 0.2
        repetition = self._check_repetition(text)
        if repetition > 0.3:
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _check_repetition(self, text: str, window: int = 50) -> float:
        if len(text) < window * 2:
            return 0.0
        seen = set()
        duplicates = 0
        total = 0
        for i in range(0, len(text) - window, window // 2):
            segment = text[i:i + window]
            if segment in seen:
                duplicates += 1
            seen.add(segment)
            total += 1
        return duplicates / total if total > 0 else 0.0

    def validate(self, record: DataRecord) -> tuple[bool, list[str]]:
        errors = []
        if self.checks.get("format", True) and not self.check_format(record):
            errors.append("格式错误")
        if self.checks.get("content_empty", True) and self.check_content_empty(record):
            errors.append("内容为空")
        if self.checks.get("content_too_short", True) or self.checks.get("content_too_long", True):
            length_ok, length_msg = self.check_content_length(record)
            if not length_ok:
                errors.append(f"内容长度问题：{length_msg}")
        if self.checks.get("duplicate_detection", True) and self.check_duplicate(record):
            errors.append("内容重复")
        if self.checks.get("language_consistency", True) and not self.check_language_consistency(record):
            errors.append("语言不一致")
        if self.checks.get("toxicity_filter", True):
            toxicity_ok, toxicity_score = self.check_toxicity(record)
            if not toxicity_ok:
                errors.append(f"有毒内容(分数:{toxicity_score:.2f})")
        if self.checks.get("pii_detection", True):
            pii_found = self.check_pii(record)
            if pii_found:
                errors.append(f"检测到PII:{','.join(pii_found)}")
        if self.checks.get("quality_score", True):
            quality = self.check_quality_score(record)
            min_quality = self.thresholds.get("min_quality_score", 0.6)
            if quality < min_quality:
                errors.append(f"质量分数过低({quality:.2f}<{min_quality})")
        return len(errors) == 0, errors

    def validate_batch(self, records: list[DataRecord]) -> ValidationReport:
        report = ValidationReport()
        report.total_records = len(records)
        report.validation_time = datetime.now().isoformat()
        type_counts: dict[str, int] = {}
        type_valid: dict[str, int] = {}
        for record in records:
            dtype = record.data_type
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
            if self.checks.get("format", True) and not self.check_format(record):
                report.add_error("format_error")
                report.invalid_count += 1
                if self.halt_on_error:
                    raise ValueError(f"格式错误，中断验证：{record.id}")
                continue
            is_valid, errors = self.validate(record)
            if is_valid:
                report.valid_count += 1
                type_valid[dtype] = type_valid.get(dtype, 0) + 1
            else:
                report.invalid_count += 1
                for err in errors:
                    report.add_error(err)
        for dtype in type_counts:
            total = type_counts[dtype]
            valid = type_valid.get(dtype, 0)
            report.per_type_stats[dtype] = {
                "total": total,
                "valid": valid,
                "valid_ratio": round(valid / total, 4) if total > 0 else 0,
            }
        return report

    def save_report(self, report: ValidationReport, filepath: Optional[str] = None):
        if filepath is None:
            os.makedirs(self.report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.report_dir, f"validation_report_{timestamp}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"验证报告已保存到 {filepath}")
        txt_path = filepath.replace(".json", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(report))
        logger.info(f"验证报告文本已保存到 {txt_path}")

    def reset(self):
        self._duplicate_cache.clear()

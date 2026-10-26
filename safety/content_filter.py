import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.content_filter")


class QingxiaotuanContentFilter:
    FILTER_LEVEL_STRICT = "strict"
    FILTER_LEVEL_STANDARD = "standard"
    FILTER_LEVEL_RELAXED = "relaxed"

    def __init__(
        self,
        filter_level: str = FILTER_LEVEL_STANDARD,
        sensitivity_list_path: Optional[str] = None,
        enable_model_review: bool = True,
        enable_logging: bool = True,
    ):
        self.filter_level = filter_level
        self.enable_model_review = enable_model_review
        self.enable_logging = enable_logging
        self.sensitivity_words: Dict[str, List[str]] = {}
        self.compiled_regexes: Dict[str, List[re.Pattern]] = {}
        self.filter_log: List[Dict[str, Any]] = []
        self.stats = {
            "total_checks": 0,
            "blocked_input": 0,
            "blocked_output": 0,
            "warnings_issued": 0,
        }
        self._load_sensitivity_list(sensitivity_list_path)
        self._compile_regex_patterns()

    def _load_sensitivity_list(self, file_path: Optional[str] = None) -> None:
        if file_path is None:
            base = Path(__file__).parent / "configs" / "sensitivity_list.txt"
            file_path = str(base)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                current_category = "general"
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        current_category = line[1:-1]
                        self.sensitivity_words[current_category] = []
                        continue
                    if current_category not in self.sensitivity_words:
                        self.sensitivity_words[current_category] = []
                    self.sensitivity_words[current_category].append(line)
            logger.info(
                f"Loaded sensitivity words: {sum(len(v) for v in self.sensitivity_words.values())} words "
                f"in {len(self.sensitivity_words)} categories"
            )
        except FileNotFoundError:
            logger.warning(f"Sensitivity list not found at {file_path}, using built-in defaults")
            self._load_default_sensitivity_words()

    def _load_default_sensitivity_words(self) -> None:
        self.sensitivity_words = {
            "违法内容": ["制作炸弹", "制作毒品", "非法武器", "黑客攻击", "信用卡伪造"],
            "暴力极端": ["恐怖袭击", "暴力血腥", "伤害他人", "种族清洗", "大屠杀"],
            "色情内容": ["色情视频", "儿童色情", "色情服务", "淫秽内容"],
            "仇恨言论": ["种族歧视", "民族仇恨", "宗教仇恨", "性别歧视"],
            "自残自杀": ["自杀方法", "自残方法", "伤害自己", "结束生命"],
            "毒品药物": ["海洛因", "冰毒", "大麻", "可卡因", "摇头丸"],
            "欺诈诱导": ["虚假广告", "传销组织", "集资诈骗"],
        }

    def _compile_regex_patterns(self) -> None:
        self.compiled_regexes = {}
        for category, words in self.sensitivity_words.items():
            patterns = []
            for word in words:
                if len(word) >= 2:
                    pattern = re.escape(word)
                    patterns.append(re.compile(pattern, re.IGNORECASE))
            self.compiled_regexes[category] = patterns

    def set_filter_level(self, level: str) -> None:
        if level not in [self.FILTER_LEVEL_STRICT, self.FILTER_LEVEL_STANDARD, self.FILTER_LEVEL_RELAXED]:
            raise ValueError(f"Invalid filter level: {level}")
        self.filter_level = level
        logger.info(f"Filter level set to: {level}")

    def _get_threshold(self) -> float:
        thresholds = {
            self.FILTER_LEVEL_STRICT: 0.3,
            self.FILTER_LEVEL_STANDARD: 0.6,
            self.FILTER_LEVEL_RELAXED: 0.85,
        }
        return thresholds.get(self.filter_level, 0.6)

    def check_input(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.stats["total_checks"] += 1
        result = self._filter_text(text, direction="input")
        blocked = result["is_blocked"]
        if blocked:
            self.stats["blocked_input"] += 1
            if self.enable_logging:
                self._log_filter_event("input", text, result)
        return result

    def check_output(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.stats["total_checks"] += 1
        result = self._filter_text(text, direction="output")
        blocked = result["is_blocked"]
        if blocked:
            self.stats["blocked_output"] += 1
            if self.enable_logging:
                self._log_filter_event("output", text, result)
        return result

    def _filter_text(self, text: str, direction: str = "input") -> Dict[str, Any]:
        threshold = self._get_threshold()
        matched_categories = {}
        total_matches = 0
        matched_words_list = []

        for category, patterns in self.compiled_regexes.items():
            category_matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    category_matches.extend(found)
                    matched_words_list.extend(found)
            if category_matches:
                matched_categories[category] = category_matches
                total_matches += len(category_matches)

        total_words = len(text) if text.strip() else 1
        density = total_matches / max(total_words / 10, 1)
        base_confidence = min(total_matches * 0.15 + density * 0.3, 0.95)
        has_critical = any(
            cat in matched_categories
            for cat in ["违法内容", "暴力极端", "自残自杀"]
        )
        has_high = any(
            cat in matched_categories
            for cat in ["色情内容", "仇恨言论", "毒品药物"]
        )
        if has_critical:
            confidence = min(base_confidence + 0.2, 0.99)
        elif has_high:
            confidence = min(base_confidence + 0.1, 0.95)
        else:
            confidence = base_confidence

        if self.enable_model_review and confidence > 0.4:
            model_result = self._model_based_review(text, matched_categories)
            confidence = max(confidence, model_result["confidence"])
            if model_result.get("additional_categories"):
                for cat in model_result["additional_categories"]:
                    if cat not in matched_categories:
                        matched_categories[cat] = ["[model_detected]"]

        is_blocked = confidence >= threshold
        if self.filter_level == self.FILTER_LEVEL_STRICT:
            if len(matched_categories) > 0:
                is_blocked = True
        elif self.filter_level == self.FILTER_LEVEL_RELAXED:
            if not has_critical:
                is_blocked = False

        return {
            "is_blocked": is_blocked,
            "confidence": round(confidence, 4),
            "matched_categories": matched_categories,
            "matched_words": matched_words_list[:20],
            "total_matches": total_matches,
            "match_density": round(density, 4),
            "filter_level": self.filter_level,
            "direction": direction,
            "threshold": threshold,
            "requires_warning": confidence > threshold * 0.8 and not is_blocked,
            "timestamp": datetime.now().isoformat(),
        }

    def _model_based_review(self, text: str, matched_categories: Dict[str, List[str]]) -> Dict[str, Any]:
        additional_categories = []
        model_confidence = 0.0
        risk_indicators = []

        violence_indicators = [
            r"(?:杀|死|伤|毁|灭|炸|暴|虐|砍|捅|射).*(?:人|你|他|她|他们)",
            r"(?:打|揍|扁|修理).*(?:死|残|伤)",
        ]
        for pattern in violence_indicators:
            if re.search(pattern, text):
                risk_indicators.append("violence_intent")

        self_harm_indicators = [
            r"(?:不想活|活不下|没意义|想死|结束生命|了结)",
            r"(?:抑郁|绝望|无助|痛苦).*(?:自杀|结束|离开|死亡)",
        ]
        for pattern in self_harm_indicators:
            if re.search(pattern, text):
                risk_indicators.append("self_harm_intent")

        harassment_indicators = [
            r"(?:你(?:个|这个|就是).*(?:蠢|笨|傻|废|蠢货|白痴|混蛋|垃圾))",
            r"(?:去死|滚开|闭嘴|废物|垃圾|蠢货)",
        ]
        for pattern in harassment_indicators:
            if re.search(pattern, text):
                risk_indicators.append("harassment")

        if risk_indicators:
            category_map = {
                "violence_intent": "暴力极端",
                "self_harm_intent": "自残自杀",
                "harassment": "仇恨言论",
            }
            additional_categories = list(set(category_map.get(r, "") for r in risk_indicators))
            additional_categories = [c for c in additional_categories if c and c not in matched_categories]
            model_confidence = 0.45 + min(len(risk_indicators) * 0.15, 0.4)

        return {
            "confidence": model_confidence,
            "additional_categories": additional_categories,
            "risk_indicators": risk_indicators,
        }

    def add_sensitive_word(self, word: str, category: str = "general") -> None:
        if category not in self.sensitivity_words:
            self.sensitivity_words[category] = []
        word = word.strip()
        if word and word not in self.sensitivity_words[category]:
            self.sensitivity_words[category].append(word)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            if category not in self.compiled_regexes:
                self.compiled_regexes[category] = []
            self.compiled_regexes[category].append(pattern)
            logger.info(f"Added sensitive word '{word}' to category '{category}'")

    def remove_sensitive_word(self, word: str, category: Optional[str] = None) -> bool:
        if category:
            categories_to_check = [category] if category in self.sensitivity_words else []
        else:
            categories_to_check = list(self.sensitivity_words.keys())
        for cat in categories_to_check:
            if word in self.sensitivity_words[cat]:
                self.sensitivity_words[cat].remove(word)
                self.compiled_regexes[cat] = [
                    p for p in self.compiled_regexes[cat]
                    if p.pattern != re.escape(word)
                ]
                logger.info(f"Removed sensitive word '{word}' from category '{cat}'")
                return True
        return False

    def get_categories(self) -> List[str]:
        return list(self.sensitivity_words.keys())

    def get_words_by_category(self, category: str) -> List[str]:
        return self.sensitivity_words.get(category, [])

    def _log_filter_event(self, direction: str, text: str, result: Dict[str, Any]) -> None:
        log_entry = {
            "event_id": hashlib.sha256(
                (text + str(datetime.now().timestamp())).encode()
            ).hexdigest()[:16],
            "timestamp": result["timestamp"],
            "direction": direction,
            "filter_level": self.filter_level,
            "is_blocked": result["is_blocked"],
            "confidence": result["confidence"],
            "matched_categories": list(result["matched_categories"].keys()),
            "total_matches": result["total_matches"],
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
        }
        self.filter_log.append(log_entry)
        if result["is_blocked"]:
            logger.warning(f"Blocked {direction}: {log_entry['matched_categories']} (confidence: {result['confidence']})")
        elif result.get("requires_warning"):
            logger.info(f"Warning on {direction}: {log_entry['matched_categories']}")

    def get_filter_log(self, start_index: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        return self.filter_log[start_index: start_index + limit]

    def clear_filter_log(self) -> None:
        self.filter_log.clear()
        logger.info("Filter log cleared")

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "sensitivity_word_count": sum(len(v) for v in self.sensitivity_words.values()),
            "category_count": len(self.sensitivity_words),
            "log_entries": len(self.filter_log),
            "filter_level": self.filter_level,
            "block_rate": round(
                (self.stats["blocked_input"] + self.stats["blocked_output"])
                / max(self.stats["total_checks"], 1), 4
            ),
        }

    def batch_check(self, texts: List[str], direction: str = "input") -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            if direction == "input":
                result = self.check_input(text)
            else:
                result = self.check_output(text)
            results.append(result)
        return results

    def check_content_safety(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        input_result = self.check_input(text, context)
        output_result = self.check_output(text, context)
        combined = {
            "input_safe": not input_result["is_blocked"],
            "output_safe": not output_result["is_blocked"],
            "overall_safe": not input_result["is_blocked"] and not output_result["is_blocked"],
            "input_result": input_result,
            "output_result": output_result,
            "recommendation": "allow" if not input_result["is_blocked"] and not output_result["is_blocked"]
            else "block" if input_result["is_blocked"] or output_result["is_blocked"]
            else "review",
        }
        return combined

    def reload_sensitivity_list(self, file_path: Optional[str] = None) -> None:
        self.sensitivity_words.clear()
        self.compiled_regexes.clear()
        self._load_sensitivity_list(file_path)
        self._compile_regex_patterns()
        logger.info("Sensitivity list reloaded")


class QingxiaotuanKeywordFilter:
    def __init__(self, content_filter: QingxiaotuanContentFilter):
        self.content_filter = content_filter

    def filter_text(self, text: str, replacement: str = "***") -> Tuple[str, List[Dict[str, Any]]]:
        filtered_text = text
        violations = []
        for category, patterns in self.content_filter.compiled_regexes.items():
            for pattern in patterns:
                matches = pattern.findall(filtered_text)
                if matches:
                    for match in matches:
                        violations.append({
                            "word": match,
                            "category": category,
                        })
                    filtered_text = pattern.sub(replacement, filtered_text)
        return filtered_text, violations

    def highlight_sensitive_words(self, text: str, prefix: str = "**", suffix: str = "**") -> str:
        highlighted = text
        for category, patterns in self.content_filter.compiled_regexes.items():
            for pattern in patterns:
                highlighted = pattern.sub(
                    lambda m: f"{prefix}{m.group()}{suffix}", highlighted
                )
        return highlighted

import re
import json
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.toxicity_detector")


class ToxicityCategory(str, Enum):
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    PORNOGRAPHIC = "pornographic"
    SELF_HARM = "self_harm"
    ILLEGAL = "illegal"
    HARASSMENT = "harassment"


@dataclass
class ToxicityScore:
    category: ToxicityCategory
    score: float
    confidence: float
    matched_patterns: List[str]
    details: Optional[Dict[str, Any]] = None


@dataclass
class ToxicityResult:
    is_toxic: bool
    overall_score: float
    category_scores: List[ToxicityScore]
    primary_category: Optional[ToxicityCategory] = None
    detection_method: str = "rule_based"
    requires_review: bool = False
    summary: str = ""
    timestamp: str = ""


class QingxiaotuanToxicityDetector:
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        enable_model_detection: bool = True,
        enable_rule_detection: bool = True,
    ):
        self.thresholds = thresholds or {
            ToxicityCategory.HATE_SPEECH: 0.5,
            ToxicityCategory.VIOLENCE: 0.5,
            ToxicityCategory.PORNOGRAPHIC: 0.5,
            ToxicityCategory.SELF_HARM: 0.4,
            ToxicityCategory.ILLEGAL: 0.5,
            ToxicityCategory.HARASSMENT: 0.6,
        }
        self.enable_model_detection = enable_model_detection
        self.enable_rule_detection = enable_rule_detection
        self._rule_patterns = self._initialize_rule_patterns()
        self.detection_log: List[Dict[str, Any]] = []
        self.stats = {cat.value: {"detections": 0, "total_score": 0.0} for cat in ToxicityCategory}
        self.stats["total_detections"] = 0

    def _initialize_rule_patterns(self) -> Dict[ToxicityCategory, List[re.Pattern]]:
        return {
            ToxicityCategory.HATE_SPEECH: [
                re.compile(r"(种族|民族|宗教).*(歧视|仇恨|消灭|清除|驱逐)", re.IGNORECASE),
                re.compile(r"(性别|性取向).*(变态|恶心|异常|疾病|歧视)", re.IGNORECASE),
                re.compile(r"(仇视|歧视|排斥).*(同性恋|变性人|跨性别)", re.IGNORECASE),
                re.compile(r"(白人至上|种族优越|种族清洗|大屠杀)", re.IGNORECASE),
                re.compile(r"(排外|仇外|反犹太|伊斯兰恐惧)", re.IGNORECASE),
            ],
            ToxicityCategory.VIOLENCE: [
                re.compile(r"(杀|谋杀|杀害|杀死|伤人).*(人|你|他|她|别人)", re.IGNORECASE),
                re.compile(r"(恐怖袭击|爆炸|枪击|斩首|屠杀)", re.IGNORECASE),
                re.compile(r"(暴力|血腥|残忍|伤害).*(伤害|袭击|攻击|对待|别人|他人)", re.IGNORECASE),
                re.compile(r"(虐待|折磨|残害).*(儿童|老人|妇女|动物)", re.IGNORECASE),
                re.compile(r"(炸弹|炸药|武器|刀具).*(制作|使用|攻击|威胁)", re.IGNORECASE),
            ],
            ToxicityCategory.PORNOGRAPHIC: [
                re.compile(r"(色情|情色|成人|淫秽).*(内容|视频|图片|网站|服务|平台)", re.IGNORECASE),
                re.compile(r"(儿童|未成年人).*(色情|性|裸|淫秽)", re.IGNORECASE),
                re.compile(r"(裸露|裸体|走光|偷拍|私密).*(照片|视频|图片)", re.IGNORECASE),
                re.compile(r"(卖淫|嫖娼|性交易|色情服务|裸聊)", re.IGNORECASE),
            ],
            ToxicityCategory.SELF_HARM: [
                re.compile(r"(自杀|自残|自伤|了结).*(方法|教程|方式|手段|指南)", re.IGNORECASE),
                re.compile(r"(如何|怎样).*(自杀|了结生命|结束生命)", re.IGNORECASE),
                re.compile(r"(伤害自己|自我伤害|自毁|轻生|厌世)", re.IGNORECASE),
                re.compile(r"(活不下去|不想活|没意思|想死|绝望)", re.IGNORECASE),
            ],
            ToxicityCategory.ILLEGAL: [
                re.compile(r"(制作|制造|合成).*(炸弹|炸药|毒品|毒药|武器)", re.IGNORECASE),
                re.compile(r"(黑客|渗透|入侵|破解).*(教程|工具|方法|技术|攻击)", re.IGNORECASE),
                re.compile(r"(信用卡|银行卡|身份).*(伪造|盗刷|破解|盗窃)", re.IGNORECASE),
                re.compile(r"(毒品|毒药|麻醉).*(交易|购买|出售|制作|运输)", re.IGNORECASE),
            ],
            ToxicityCategory.HARASSMENT: [
                re.compile(r"(你个|你这个|你就是).*(蠢货|白痴|混蛋|废物|垃圾)", re.IGNORECASE),
                re.compile(r"(去死|滚开|闭嘴|废物|垃圾|蠢货|傻逼)", re.IGNORECASE),
                re.compile(r"(侮辱|辱骂|谩骂|攻击|诽谤|诋毁|中伤)", re.IGNORECASE),
            ],
        }

    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> ToxicityResult:
        category_scores: List[ToxicityScore] = []
        overall_score = 0.0

        for category in ToxicityCategory:
            score = self._detect_category(text, category)
            category_scores.append(score)
            overall_score = max(overall_score, score.score * score.confidence)

        if self.enable_model_detection:
            model_scores = self._model_based_detection(text)
            for i, cat_score in enumerate(category_scores):
                if cat_score.category in model_scores:
                    model_val = model_scores[cat_score.category]
                    merged_score = max(cat_score.score, model_val)
                    merged_confidence = min(cat_score.confidence + 0.1, 1.0)
                    category_scores[i] = ToxicityScore(
                        category=cat_score.category,
                        score=merged_score,
                        confidence=merged_confidence,
                        matched_patterns=cat_score.matched_patterns,
                        details={"rule_score": cat_score.score, "model_score": model_val},
                    )
            overall_score = max(s.score * s.confidence for s in category_scores)

        primary_category = None
        max_weighted = 0.0
        for s in category_scores:
            weighted = s.score * s.confidence
            if weighted > max_weighted:
                max_weighted = weighted
                primary_category = s.category

        threshold = self.thresholds.get(primary_category, 0.5) if primary_category else 0.5
        is_toxic = overall_score >= threshold
        requires_review = 0.3 <= overall_score < threshold

        self.stats["total_detections"] += 1
        if primary_category:
            self.stats[primary_category.value]["detections"] += 1
            self.stats[primary_category.value]["total_score"] += overall_score

        summary_parts = []
        for s in category_scores:
            if s.score > 0.1:
                summary_parts.append(f"{s.category.value}:{round(s.score,2)}")
        summary = ", ".join(summary_parts) if summary_parts else "clean"

        result = ToxicityResult(
            is_toxic=is_toxic,
            overall_score=round(overall_score, 4),
            category_scores=category_scores,
            primary_category=primary_category,
            detection_method="hybrid" if self.enable_model_detection else "rule_based",
            requires_review=requires_review,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )

        self._log_detection(text, result)
        return result

    def _detect_category(self, text: str, category: ToxicityCategory) -> ToxicityScore:
        patterns = self._rule_patterns.get(category, [])
        matched_patterns = []
        match_count = 0

        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    matched_text = match if isinstance(match, str) else " ".join(m for m in match if m)
                    if matched_text.strip():
                        matched_patterns.append(matched_text)
                match_count += len(matches)

        text_length = max(len(text), 1)
        density = match_count / (text_length / 20)
        score = min(match_count * 0.2 + density * 0.3, 0.95)
        confidence = min(0.5 + match_count * 0.1 + density * 0.1, 0.98)

        if match_count == 0:
            score = 0.0
            confidence = 0.0

        return ToxicityScore(
            category=category,
            score=round(score, 4),
            confidence=round(confidence, 4),
            matched_patterns=matched_patterns[:10],
        )

    def _model_based_detection(self, text: str) -> Dict[ToxicityCategory, float]:
        results = {}
        vectors = {
            ToxicityCategory.HATE_SPEECH: self._compute_text_vector(text, [
                "歧视", "仇恨", "排斥", "优越", "低等", "劣等", "清除", "灭绝",
            ]),
            ToxicityCategory.VIOLENCE: self._compute_text_vector(text, [
                "暴力", "攻击", "伤害", "杀戮", "摧毁", "破坏", "血腥", "残忍",
            ]),
            ToxicityCategory.PORNOGRAPHIC: self._compute_text_vector(text, [
                "色情", "性", "裸体", "淫秽", "亲密", "裸露", "性感",
            ]),
            ToxicityCategory.SELF_HARM: self._compute_text_vector(text, [
                "自杀", "自残", "抑郁", "绝望", "无助", "痛苦", "伤害自己",
            ]),
            ToxicityCategory.ILLEGAL: self._compute_text_vector(text, [
                "非法", "犯罪", "违法", "诈骗", "盗取", "伪造", "走私",
            ]),
            ToxicityCategory.HARASSMENT: self._compute_text_vector(text, [
                "辱骂", "侮辱", "贬低", "嘲笑", "欺凌", "骚扰", "恐吓",
            ]),
        }
        for cat, score in vectors.items():
            results[cat] = round(score, 4)
        return results

    def _compute_text_vector(self, text: str, keywords: List[str]) -> float:
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches == 0:
            return 0.0
        density = matches / max(len(keywords), 1)
        frequency = matches / max(len(text) / 10, 1)
        score = min(density * 0.5 + frequency * 0.3, 0.9)
        return score

    def _log_detection(self, text: str, result: ToxicityResult) -> None:
        log_entry = {
            "timestamp": result.timestamp,
            "is_toxic": result.is_toxic,
            "overall_score": result.overall_score,
            "primary_category": result.primary_category.value if result.primary_category else None,
            "categories": {
                s.category.value: {"score": s.score, "confidence": s.confidence}
                for s in result.category_scores
            },
            "detection_method": result.detection_method,
            "requires_review": result.requires_review,
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
        }
        self.detection_log.append(log_entry)
        if result.is_toxic:
            logger.warning(
                f"Toxicity detected: {result.primary_category.value if result.primary_category else 'unknown'} "
                f"(score: {result.overall_score})"
            )

    def batch_detect(self, texts: List[str]) -> List[ToxicityResult]:
        return [self.detect(text) for text in texts]

    def get_toxicity_report(self, since: Optional[str] = None) -> Dict[str, Any]:
        return {
            "total_detections": self.stats["total_detections"],
            "category_stats": {
                cat: {
                    "detections": self.stats[cat.value]["detections"],
                    "avg_score": round(
                        self.stats[cat.value]["total_score"] / max(self.stats[cat.value]["detections"], 1), 4
                    ),
                }
                for cat in ToxicityCategory
            },
            "recent_log_count": len(self.detection_log),
            "toxic_rate": round(
                sum(1 for e in self.detection_log if e["is_toxic"]) / max(len(self.detection_log), 1), 4
            ),
        }

    def set_threshold(self, category: ToxicityCategory, threshold: float) -> None:
        if 0.0 <= threshold <= 1.0:
            self.thresholds[category] = threshold
            logger.info(f"Threshold for {category.value} set to {threshold}")

    def add_custom_pattern(self, category: ToxicityCategory, pattern: str) -> None:
        if category not in self._rule_patterns:
            self._rule_patterns[category] = []
        compiled = re.compile(pattern, re.IGNORECASE)
        self._rule_patterns[category].append(compiled)
        logger.info(f"Added custom pattern '{pattern}' to category {category.value}")

    def clear_log(self) -> None:
        self.detection_log.clear()
        for cat in ToxicityCategory:
            self.stats[cat.value] = {"detections": 0, "total_score": 0.0}
        self.stats["total_detections"] = 0
        logger.info("Detection log cleared")

    def analyze_text_segments(self, text: str, segment_length: int = 50) -> List[Dict[str, Any]]:
        segments = []
        for i in range(0, len(text), segment_length):
            segment = text[i:i + segment_length]
            result = self.detect(segment)
            segments.append({
                "start": i,
                "end": i + len(segment),
                "text": segment,
                "score": result.overall_score,
                "is_toxic": result.is_toxic,
                "primary_category": result.primary_category.value if result.primary_category else None,
            })
        return segments

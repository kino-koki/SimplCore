import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.privacy_filter")


class PIICategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    ID_CARD = "id_card"
    ADDRESS = "address"
    BANK_CARD = "bank_card"
    PASSPORT = "passport"
    IP_ADDRESS = "ip_address"
    DRIVER_LICENSE = "driver_license"
    SOCIAL_SECURITY = "social_security"


@dataclass
class PIIDetection:
    category: PIICategory
    value: str
    start: int
    end: int
    confidence: float
    masked_value: str = ""
    risk_level: str = "low"


@dataclass
class PrivacyAnalysisResult:
    has_pii: bool
    total_findings: int
    detections: List[PIIDetection]
    risk_score: float
    recommendations: List[str]
    masked_text: str = ""
    compliance_notes: List[str] = field(default_factory=list)
    timestamp: str = ""


class QingxiaotuanPrivacyFilter:
    def __init__(
        self,
        masking_char: str = "*",
        masking_mode: str = "partial",
        enable_model_detection: bool = True,
    ):
        self.masking_char = masking_char
        self.masking_mode = masking_mode
        self.enable_model_detection = enable_model_detection
        self._patterns = self._initialize_patterns()
        self.detection_log: List[Dict[str, Any]] = []
        self.stats = {
            "total_scans": 0,
            "total_pii_found": 0,
            "pii_by_category": {cat.value: 0 for cat in PIICategory},
        }

    def _initialize_patterns(self) -> Dict[PIICategory, List[re.Pattern]]:
        return {
            PIICategory.EMAIL: [
                re.compile(r'(?<![A-Za-z0-9])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z])'),
            ],
            PIICategory.PHONE: [
                re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)'),
                re.compile(r'(?<!\d)\d{3,4}[-.\s]?\d{7,8}(?!\d)'),
                re.compile(r'\+86[-.\s]?1[3-9]\d{9}(?!\d)'),
            ],
            PIICategory.ID_CARD: [
                re.compile(r'(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)'),
                re.compile(r'(?<!\d)[1-9]\d{14}[\dXx](?!\d)'),
            ],
            PIICategory.BANK_CARD: [
                re.compile(r'(?<!\d)(?:62|60|58|56|55|54|53|52|51|50|49|48|47|45|44|43|42|41|40)\d{14,18}(?!\d)'),
                re.compile(r'(?<!\d)\d{16}(?!\d)'),
                re.compile(r'(?<!\d)\d{19}(?!\d)'),
            ],
            PIICategory.ADDRESS: [
                re.compile(r'(北京市|上海市|广州市|深圳市|天津市|重庆市|杭州市|成都市|武汉市|南京市|西安市|沈阳市|青岛市|宁波市|厦门市|大连市|成都市|长沙市|郑州市|济南市|昆明市|合肥市|苏州市|福州市|南宁市|哈尔滨市|长春市|贵阳市|太原市|石家庄市|南昌市|乌鲁木齐市|兰州市|呼和浩特市|海口市|拉萨市|银川市|西宁市)[^\s，,。]{2,30}(?:路|街|巷|道|弄|号|栋|单元|室|层|座)', re.IGNORECASE),
                re.compile(r'(?:省|自治区).*(?:市|区|县).*(?:镇|乡|街道).*(?:路|街|巷|道|弄).*(?:号|栋|单元|室)', re.IGNORECASE),
            ],
            PIICategory.PASSPORT: [
                re.compile(r'(?<![A-Za-z0-9])[EeGg]\d{8}(?![A-Za-z0-9])'),
                re.compile(r'(?<![A-Za-z0-9])[Pp]\d{7}(?![A-Za-z0-9])'),
            ],
            PIICategory.IP_ADDRESS: [
                re.compile(r'(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)'),
                re.compile(r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![A-Za-z0-9])'),
            ],
            PIICategory.DRIVER_LICENSE: [
                re.compile(r'(?<!\d)[1-9]\d{17}[\dXx](?!\d)'),
            ],
            PIICategory.SOCIAL_SECURITY: [
                re.compile(r'(?<!\d)\d{6}[-]?\d{6}[-]?\d{2}[-]?\d{1}(?!\d)'),
            ],
        }

    def scan(self, text: str, context: Optional[Dict[str, Any]] = None) -> PrivacyAnalysisResult:
        self.stats["total_scans"] += 1
        detections: List[PIIDetection] = []
        findings_by_category: Dict[PIICategory, int] = {}
        masked_text = text

        for category, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group()
                    if self._is_false_positive(matched_text, category):
                        continue
                    confidence = self._calculate_confidence(matched_text, category)
                    detection = PIIDetection(
                        category=category,
                        value=matched_text,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        risk_level=self._determine_risk_level(category, confidence),
                    )
                    detections.append(detection)
                    findings_by_category[category] = findings_by_category.get(category, 0) + 1

        if self.enable_model_detection and len(detections) == 0:
            model_findings = self._model_pii_detection(text)
            for finding in model_findings:
                detection = PIIDetection(
                    category=finding["category"],
                    value=finding["value"],
                    start=0,
                    end=0,
                    confidence=finding["confidence"],
                    risk_level=finding.get("risk_level", "medium"),
                )
                detections.append(detection)
                cat = finding["category"]
                findings_by_category[cat] = findings_by_category.get(cat, 0) + 1

        total_findings = len(detections)
        has_pii = total_findings > 0

        for cat, count in findings_by_category.items():
            if isinstance(cat, PIICategory):
                self.stats["pii_by_category"][cat.value] += count
        self.stats["total_pii_found"] += total_findings

        risk_score = self._calculate_risk_score(detections)
        recommendations = self._generate_recommendations(detections, risk_score)
        compliance_notes = self._generate_compliance_notes(detections)

        if has_pii:
            masked_text = self._apply_masking(text, detections)

        result = PrivacyAnalysisResult(
            has_pii=has_pii,
            total_findings=total_findings,
            detections=detections,
            risk_score=risk_score,
            recommendations=recommendations,
            masked_text=masked_text,
            compliance_notes=compliance_notes,
            timestamp=datetime.now().isoformat(),
        )

        self._log_scan(text, result)
        return result

    def _is_false_positive(self, matched_text: str, category: PIICategory) -> bool:
        if category == PIICategory.IP_ADDRESS:
            parts = matched_text.split(".")
            if len(parts) == 4:
                try:
                    if all(0 <= int(p) <= 255 for p in parts):
                        if parts[0] in ["127", "10", "192", "172"]:
                            return False
                except ValueError:
                    return True
            if len(matched_text) < 7:
                return True
        if category == PIICategory.EMAIL:
            if len(matched_text) < 6:
                return True
        if category == PIICategory.PHONE:
            if len(matched_text) < 7:
                return True
        if category == PIICategory.BANK_CARD:
            if matched_text.startswith("0000"):
                return True
        return False

    def _calculate_confidence(self, value: str, category: PIICategory) -> float:
        base_scores = {
            PIICategory.EMAIL: 0.95,
            PIICategory.PHONE: 0.9,
            PIICategory.ID_CARD: 0.95,
            PIICategory.BANK_CARD: 0.85,
            PIICategory.ADDRESS: 0.7,
            PIICategory.PASSPORT: 0.9,
            PIICategory.IP_ADDRESS: 0.8,
            PIICategory.DRIVER_LICENSE: 0.85,
            PIICategory.SOCIAL_SECURITY: 0.8,
        }
        score = base_scores.get(category, 0.7)
        if category == PIICategory.ID_CARD:
            if re.match(r'^[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]$', value):
                score = 0.98
        elif category == PIICategory.PHONE:
            if re.match(r'^1[3-9]\d{9}$', value):
                score = 0.95
        elif category == PIICategory.EMAIL:
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
                score = 0.97
        return min(score, 0.99)

    def _determine_risk_level(self, category: PIICategory, confidence: float) -> str:
        high_risk_categories = [PIICategory.ID_CARD, PIICategory.BANK_CARD, PIICategory.PASSPORT]
        if category in high_risk_categories and confidence > 0.8:
            return "high"
        if confidence > 0.9:
            return "high"
        if confidence > 0.7:
            return "medium"
        return "low"

    def _calculate_risk_score(self, detections: List[PIIDetection]) -> float:
        if not detections:
            return 0.0
        risk_weights = {
            PIICategory.ID_CARD: 1.0,
            PIICategory.BANK_CARD: 1.0,
            PIICategory.PASSPORT: 0.9,
            PIICategory.PHONE: 0.7,
            PIICategory.EMAIL: 0.5,
            PIICategory.ADDRESS: 0.6,
            PIICategory.IP_ADDRESS: 0.3,
            PIICategory.DRIVER_LICENSE: 0.8,
            PIICategory.SOCIAL_SECURITY: 0.8,
        }
        total_weight = sum(
            risk_weights.get(d.category, 0.5) * d.confidence
            for d in detections
        )
        score = min(total_weight / 3.0, 1.0)
        return round(score, 4)

    def _apply_masking(self, text: str, detections: List[PIIDetection]) -> str:
        masked = list(text)
        for detection in sorted(detections, key=lambda d: d.start, reverse=True):
            value = detection.value
            if self.masking_mode == "full":
                masked_value = self.masking_char * len(value)
            else:
                if len(value) <= 4:
                    masked_value = self.masking_char * len(value)
                elif len(value) <= 8:
                    masked_value = value[:2] + self.masking_char * (len(value) - 4) + value[-2:]
                else:
                    keep_start = max(2, len(value) // 4)
                    keep_end = max(2, len(value) // 4)
                    masked_value = (
                        value[:keep_start]
                        + self.masking_char * (len(value) - keep_start - keep_end)
                        + value[-keep_end:]
                    )
            detection.masked_value = masked_value
            if detection.start < len(masked) and detection.end <= len(masked):
                masked[detection.start:detection.end] = list(masked_value)
        return "".join(masked)

    def _model_pii_detection(self, text: str) -> List[Dict[str, Any]]:
        findings = []
        text_lower = text.lower()
        phone_context = re.findall(r'(?:电话|手机|联系电话|联系手机|手机号码|电话号码)[：:]\s*(\S+)', text)
        for match in phone_context:
            if re.match(r'1[3-9]\d{9}', match):
                findings.append({
                    "category": PIICategory.PHONE,
                    "value": match,
                    "confidence": 0.85,
                    "risk_level": "high",
                })
        email_context = re.findall(r'(?:邮箱|邮件|电子邮件|E-?mail)[：:]\s*(\S+@\S+)', text)
        for match in email_context:
            findings.append({
                "category": PIICategory.EMAIL,
                "value": match,
                "confidence": 0.9,
                "risk_level": "medium",
            })
        id_context = re.findall(r'(?:身份证|身份证号|身份证号码|ID)[：:]\s*(\S+)', text)
        for match in id_context:
            clean = match.replace(" ", "").replace("-", "")
            if re.match(r'^\d{17}[\dXx]$', clean) or re.match(r'^\d{15}$', clean):
                findings.append({
                    "category": PIICategory.ID_CARD,
                    "value": clean,
                    "confidence": 0.9,
                    "risk_level": "high",
                })
        return findings

    def _generate_recommendations(self, detections: List[PIIDetection], risk_score: float) -> List[str]:
        recommendations = []
        if not detections:
            recommendations.append("No PII detected. Continue normal processing.")
            return recommendations
        categories_found = set(d.category for d in detections)
        if PIICategory.ID_CARD in categories_found:
            recommendations.append("身份证号属于高度敏感信息，建议立即进行脱敏处理。")
        if PIICategory.BANK_CARD in categories_found:
            recommendations.append("银行卡号属于敏感金融信息，处理后须确保日志中不保留原始信息。")
        if PIICategory.PHONE in categories_found:
            recommendations.append("手机号码属于个人敏感信息，建议在显示和存储时进行部分掩码处理。")
        if PIICategory.EMAIL in categories_found:
            recommendations.append("邮箱地址可能用于身份验证，建议谨慎处理。")
        if PIICategory.ADDRESS in categories_found:
            recommendations.append("家庭地址属于个人隐私信息，建议仅保留到城市级别。")
        if risk_score > 0.5:
            recommendations.append("整体隐私风险评估较高，建议进行全面的数据脱敏和安全审查。")
        return recommendations

    def _generate_compliance_notes(self, detections: List[PIIDetection]) -> List[str]:
        notes = []
        if detections:
            notes.append("PII数据应遵守相关隐私法规（如《个人信息保护法》）的要求进行处理。")
            notes.append("用户个人信息的收集、存储和处理应获得用户的明确同意。")
            high_risk = [d for d in detections if d.risk_level == "high"]
            if high_risk:
                notes.append(f"发现{len(high_risk)}项高风险PII，建议立即通知数据保护官。")
        return notes

    def _log_scan(self, text: str, result: PrivacyAnalysisResult) -> None:
        log_entry = {
            "timestamp": result.timestamp,
            "has_pii": result.has_pii,
            "total_findings": result.total_findings,
            "risk_score": result.risk_score,
            "categories_found": list(set(d.category.value for d in result.detections)),
            "detections_detail": [
                {"category": d.category.value, "confidence": d.confidence, "risk_level": d.risk_level}
                for d in result.detections
            ],
            "text_preview": text[:60] + "..." if len(text) > 60 else text,
        }
        self.detection_log.append(log_entry)
        if result.has_pii:
            logger.warning(
                f"PII detected: {result.total_findings} findings, "
                f"categories: {log_entry['categories_found']}, risk: {result.risk_score}"
            )

    def mask_text(self, text: str) -> str:
        result = self.scan(text)
        return result.masked_text

    def verify_compliance(self, text: str, allowed_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        result = self.scan(text)
        if allowed_categories is None:
            allowed_categories = []
        violations = [
            d for d in result.detections
            if d.category.value not in allowed_categories
        ]
        return {
            "compliant": len(violations) == 0,
            "violations": len(violations),
            "violation_details": [
                {"category": d.category.value, "risk_level": d.risk_level}
                for d in violations
            ],
            "risk_score": result.risk_score,
            "recommendations": result.recommendations,
            "masked_preview": result.masked_text[:100] if result.masked_text else "",
        }

    def get_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.detection_log[-limit:]

    def clear_log(self) -> None:
        self.detection_log.clear()
        for cat in PIICategory:
            self.stats["pii_by_category"][cat.value] = 0
        self.stats["total_scans"] = 0
        self.stats["total_pii_found"] = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "high_risk_findings": sum(
                1 for entry in self.detection_log
                for d in entry.get("detections_detail", [])
                if d["risk_level"] == "high"
            ),
        }

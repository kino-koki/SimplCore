import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.jailbreak_detector")


class AttackType(str, Enum):
    DIRECT_INSTRUCTION = "direct_instruction"
    ROLE_PLAY = "role_play"
    HYPOTHETICAL = "hypothetical"
    ENCODING_BYPASS = "encoding_bypass"
    TOKEN_SMUGGLING = "token_smuggling"
    CONTEXT_MANIPULATION = "context_manipulation"
    BASE64_ENCODED = "base64_encoded"
    MULTI_TURN = "multi_turn"
    ACADEMIC_PRETEXT = "academic_pretext"
    TRANSLATION_BYPASS = "translation_bypass"


@dataclass
class JailbreakDetection:
    attack_type: AttackType
    confidence: float
    matched_pattern: str
    excerpt: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JailbreakResult:
    is_jailbreak_attempt: bool
    max_confidence: float
    detections: List[JailbreakDetection]
    primary_attack_type: Optional[AttackType] = None
    requires_context_review: bool = False
    conversation_blocked: bool = False
    summary: str = ""
    timestamp: str = ""


class QingxiaotuanJailbreakDetector:
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_history_turns: int = 10,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_history_turns = max_history_turns
        self._known_patterns = self._initialize_known_patterns()
        self._keyword_threshold = 0.5
        self.detection_log: List[Dict[str, Any]] = []
        self.stats = {
            "total_checks": 0,
            "jailbreak_attempts": 0,
            "by_type": {at.value: 0 for at in AttackType},
        }

    def _initialize_known_patterns(self) -> Dict[AttackType, Dict[str, Any]]:
        return {
            AttackType.DIRECT_INSTRUCTION: {
                "patterns": [
                    re.compile(r"(忽略|无视|跳过|绕过|打破).*(规则|限制|约束|边界|安全)", re.IGNORECASE),
                    re.compile(r"(不要|不用|不必|无需).*(遵守|遵循|遵守|考虑|注意).*(规则|限制|安全|边界|约束)", re.IGNORECASE),
                    re.compile(r"(现在|接下来|下面).*(你|你可以|你能).*(做|干|说|写|回答).*([^你]什么|任何)", re.IGNORECASE),
                    re.compile(r"(解除|取消|去掉|移除|删除).*(限制|约束|安全|规则|审查|过滤)", re.IGNORECASE),
                    re.compile(r"(你可以|你能|请你|请).*(做|执行|完成|回答|提供).*(没有限制|不受限制|自由|任何)", re.IGNORECASE),
                ],
                "keywords": [
                    "忽略之前的指示", "忽略系统指令", "无视规则", "不受限制",
                    "没有约束", "自由模式", "开发者模式", "DAN模式",
                    "越狱模式", "jailbreak", "bypass", "ungoverned",
                    "抑制过滤", "关闭过滤", "不做限制",
                ],
            },
            AttackType.ROLE_PLAY: {
                "patterns": [
                    re.compile(r"(扮演|假装|假装你是|模拟|代入|饰演).*(角色|人物|身份|人格|AI|机器人)", re.IGNORECASE),
                    re.compile(r"(你现在是|从今以后你是|假设你是|假如你是).*", re.IGNORECASE),
                    re.compile(r"(角色扮演|情景模拟|虚拟场景|设定|剧情).*", re.IGNORECASE),
                    re.compile(r"(没有限制的|自由的|不受约束的).*(AI|机器人|助手|角色)", re.IGNORECASE),
                    re.compile(r"(坏|恶|邪恶|反派|危险|有害).*(角色|人格|AI|机器人|版本)", re.IGNORECASE),
                ],
                "keywords": [
                    "DAN", "双重人格", "反面人格", "另一个你",
                    "释放自我", "打破束缚", "不受限制的角色",
                    "邪恶版本", "黑暗版本", "shadow mode",
                    "developer mode", "god mode", "omniscient",
                ],
            },
            AttackType.HYPOTHETICAL: {
                "patterns": [
                    re.compile(r"(假设|假如|如果|假定|设想).*(虚构|假设|虚拟|想象|科幻).*(场景|情景|情况|情境)", re.IGNORECASE),
                    re.compile(r"(只是|仅仅|纯粹|单纯).*(假设|虚构|想象|理论|学术)", re.IGNORECASE),
                    re.compile(r"出于(学术|研究|教育|理论).*目的", re.IGNORECASE),
                    re.compile(r"在一个(虚构|平行|想象|虚拟).*(世界|宇宙|空间|场景).*", re.IGNORECASE),
                ],
                "keywords": [
                    "纯属假设", "学术讨论", "理论探讨", "虚构场景",
                    "仅用于研究", "教育目的", "假设性问题",
                ],
            },
            AttackType.ENCODING_BYPASS: {
                "patterns": [
                    re.compile(r"[A-Za-z0-9+/]{20,}={0,2}"),  # Base64
                    re.compile(r"(?:\\x[0-9a-fA-F]{2})+"),   # Hex encoding
                    re.compile(r"&#\d{2,5};"),               # HTML entities
                    re.compile(r"%[0-9a-fA-F]{2}"),          # URL encoding
                    re.compile(r"\\u[0-9a-fA-F]{4}"),        # Unicode escape
                ],
                "keywords": [
                    "base64", "base64解码", "hex编码", "unicode绕过",
                    "编码绕过", "反转文本", "倒序输出", "ROT13",
                ],
            },
            AttackType.TOKEN_SMUGGLING: {
                "patterns": [
                    re.compile(r"(分隔|分割|拆分|分块|逐步|一步一步).*(回答|输出|生成|提供|告诉)", re.IGNORECASE),
                    re.compile(r"(逐字|逐个|分步|分段|分批).*(输出|列出|写出|拼写)", re.IGNORECASE),
                    re.compile(r"(用.{0,5}种语言|同时用|分别用).*(回答|翻译|写出|表达)", re.IGNORECASE),
                    re.compile(r"(倒序|反向|逆序|颠倒).*(输出|写出|回答|排列)", re.IGNORECASE),
                ],
                "keywords": [
                    "逐个字母", "倒序输出", "用拼音回答", "用摩斯电码",
                    "用二进制", "多次回答", "分段回答",
                ],
            },
            AttackType.CONTEXT_MANIPULATION: {
                "patterns": [
                    re.compile(r"(忽略|无视|覆盖|覆盖掉).*(之前|前面|以前|历史|上下文|对话|上轮)", re.IGNORECASE),
                    re.compile(r"(重置|重设|重新开始|从头开始).*(对话|会话|聊天|历史)", re.IGNORECASE),
                    re.compile(r"(以下|下面|接下来).*(内容|文字|对话).*(才是|才是真正的|是真正的).*(指令|要求|问题)", re.IGNORECASE),
                ],
                "keywords": [
                    "忽略历史", "重置对话", "上下文覆盖", "对话劫持",
                    "指令注入", "prompt注入", "system prompt覆盖",
                ],
            },
            AttackType.BASE64_ENCODED: {
                "patterns": [
                    re.compile(r"^[A-Za-z0-9+/]{30,}$", re.MULTILINE),
                    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
                ],
                "keywords": [
                    "base64解码", "decoded", "decode this",
                    "以下base64", "base64编码",
                ],
            },
            AttackType.MULTI_TURN: {
                "patterns": [
                    re.compile(r"(还记不记得|之前我们说|刚才我们|像我们之前).*", re.IGNORECASE),
                    re.compile(r"(继续|接着|上一步|下一步|逐步).*(刚才|之前|前面的|上一轮)", re.IGNORECASE),
                ],
                "keywords": [
                    "多轮诱导", "渐进式提问", "逐步引导", "陷阱提问",
                ],
            },
            AttackType.ACADEMIC_PRETEXT: {
                "patterns": [
                    re.compile(r"(学术论文|科研|研究项目|学术报告).*(需要|要求|需|必须).*", re.IGNORECASE),
                    re.compile(r"(论文写作|课题研究|项目报告|学期作业).*(涉及|需要|关于|包含).*", re.IGNORECASE),
                    re.compile(r"(导师|教授|老师).*(要求|布置|安排|让).*(写|做|完成|研究).*", re.IGNORECASE),
                ],
                "keywords": [
                    "学术需要", "研究目的", "论文写作", "课程作业",
                    "仅作参考", "不传播", "内部使用",
                ],
            },
            AttackType.TRANSLATION_BYPASS: {
                "patterns": [
                    re.compile(r"(翻译|转译|转换).*(成|为|到).*", re.IGNORECASE),
                    re.compile(r"(先用|用.{0,10}翻译|用.{0,10}表达).*再.*", re.IGNORECASE),
                ],
                "keywords": [
                    "先翻译再", "通过翻译", "语言转换",
                    "跨语言", "多语言绕过",
                ],
            },
        }

    def detect(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> JailbreakResult:
        self.stats["total_checks"] += 1
        detections: List[JailbreakDetection] = []

        for attack_type, config in self._known_patterns.items():
            result = self._check_attack_type(text, attack_type, config)
            if result:
                detections.extend(result)

        if conversation_history and len(conversation_history) > 1:
            multi_turn_result = self._detect_multi_turn_attack(text, conversation_history)
            if multi_turn_result:
                detections.extend(multi_turn_result)

        max_confidence = max((d.confidence for d in detections), default=0.0)
        primary_type = None
        max_detection_confidence = 0.0
        for d in detections:
            if d.confidence > max_detection_confidence:
                max_detection_confidence = d.confidence
                primary_type = d.attack_type

        has_keyword_match = any(
            d.details.get("match_type") == "keyword" for d in detections
        )
        has_multi_turn = any(
            d.attack_type == AttackType.MULTI_TURN for d in detections
        )
        is_jailbreak = max_confidence >= self.similarity_threshold
        if has_keyword_match and max_confidence >= self._keyword_threshold:
            is_jailbreak = True
        if has_multi_turn and max_confidence >= self._keyword_threshold:
            is_jailbreak = True
        requires_review = 0.5 <= max_confidence < self.similarity_threshold

        if is_jailbreak:
            self.stats["jailbreak_attempts"] += 1
            if primary_type:
                self.stats["by_type"][primary_type.value] += 1

        summary_parts = [f"{d.attack_type.value}:{round(d.confidence,2)}" for d in detections[:3]]
        summary = ", ".join(summary_parts) if summary_parts else "clean"

        result = JailbreakResult(
            is_jailbreak_attempt=is_jailbreak,
            max_confidence=round(max_confidence, 4),
            detections=detections,
            primary_attack_type=primary_type,
            requires_context_review=requires_review,
            conversation_blocked=is_jailbreak,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )

        self._log_detection(text, result)
        return result

    def _check_attack_type(
        self, text: str, attack_type: AttackType, config: Dict[str, Any]
    ) -> Optional[List[JailbreakDetection]]:
        detections = []
        patterns = config.get("patterns", [])
        keywords = config.get("keywords", [])
        text_lower = text.lower()

        for pattern in patterns:
            matches = pattern.findall(text)
            for match in matches:
                matched = match if isinstance(match, str) else " ".join(m for m in match if m)
                if len(matched.strip()) < 3:
                    continue
                confidence = min(0.65 + len(matched) * 0.025, 0.95)
                detection = JailbreakDetection(
                    attack_type=attack_type,
                    confidence=round(confidence, 4),
                    matched_pattern=matched[:50],
                    excerpt=text[max(0, text.find(matched) - 10): text.find(matched) + len(matched) + 10][:80],
                    details={"match_type": "regex", "length": len(matched)},
                )
                detections.append(detection)

        keyword_matches = [kw for kw in keywords if kw.lower() in text_lower]
        if keyword_matches:
            keyword_confidence = min(0.4 + len(keyword_matches) * 0.15, 0.9)
            detection = JailbreakDetection(
                attack_type=attack_type,
                confidence=round(keyword_confidence, 4),
                matched_pattern=keyword_matches[0][:50],
                excerpt=text[:80],
                details={"keywords_matched": keyword_matches[:5], "match_type": "keyword"},
            )
            detections.append(detection)

        return detections if detections else None

    def _detect_multi_turn_attack(
        self, text: str, history: List[Dict[str, str]]
    ) -> Optional[List[JailbreakDetection]]:
        if len(history) < 3:
            return None
        detections = []
        recent_turns = history[-min(self.max_history_turns, len(history)):]
        topics_sequence = []
        for turn in recent_turns:
            content = turn.get("content", "")
            if self._has_jailbreak_indicators(content):
                topics_sequence.append("suspicious")
            else:
                topics_sequence.append("normal")
        if self._has_jailbreak_indicators(text):
            topics_sequence.append("suspicious")
        else:
            topics_sequence.append("normal")
        escalation_count = 0
        prev_topic = None
        for topic in topics_sequence:
            if topic == "suspicious" and prev_topic == "normal":
                escalation_count += 1
            prev_topic = topic
        if escalation_count >= 1:
            confidence = min(0.5 + escalation_count * 0.12, 0.85)
            detection = JailbreakDetection(
                attack_type=AttackType.MULTI_TURN,
                confidence=round(confidence, 4),
                matched_pattern=f"multi_turn_escalation_{escalation_count}",
                excerpt=f"Escalation detected over {len(recent_turns)} turns",
                details={
                    "escalation_count": escalation_count,
                    "total_turns": len(recent_turns),
                    "turn_sequence": topics_sequence,
                },
            )
            detections.append(detection)
        return detections if detections else None

    def _has_jailbreak_indicators(self, text: str) -> bool:
        indicators = [
            "绕过", "忽略", "越狱", "打破限制", "不受约束",
            "jailbreak", "bypass", "ignore", "DAN",
        ]
        text_lower = text.lower()
        return any(ind.lower() in text_lower for ind in indicators)

    def _log_detection(self, text: str, result: JailbreakResult) -> None:
        log_entry = {
            "timestamp": result.timestamp,
            "is_jailbreak_attempt": result.is_jailbreak_attempt,
            "max_confidence": result.max_confidence,
            "primary_attack_type": result.primary_attack_type.value if result.primary_attack_type else None,
            "detections": [
                {
                    "attack_type": d.attack_type.value,
                    "confidence": d.confidence,
                    "matched_pattern": d.matched_pattern[:40],
                }
                for d in result.detections[:5]
            ],
            "requires_context_review": result.requires_context_review,
            "conversation_blocked": result.conversation_blocked,
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
        }
        self.detection_log.append(log_entry)
        if result.is_jailbreak_attempt:
            logger.warning(
                f"Jailbreak attempt detected: {result.primary_attack_type.value if result.primary_attack_type else 'unknown'} "
                f"(confidence: {result.max_confidence})"
            )

    def add_custom_pattern(self, attack_type: AttackType, pattern: str) -> None:
        if attack_type not in self._known_patterns:
            self._known_patterns[attack_type] = {"patterns": [], "keywords": []}
        compiled = re.compile(pattern, re.IGNORECASE)
        self._known_patterns[attack_type]["patterns"].append(compiled)
        logger.info(f"Added custom pattern '{pattern}' to attack type {attack_type.value}")

    def add_custom_keyword(self, attack_type: AttackType, keyword: str) -> None:
        if attack_type not in self._known_patterns:
            self._known_patterns[attack_type] = {"patterns": [], "keywords": []}
        self._known_patterns[attack_type]["keywords"].append(keyword)
        logger.info(f"Added custom keyword '{keyword}' to attack type {attack_type.value}")

    def update_patterns_from_source(self, patterns_data: Dict[str, Any]) -> int:
        count = 0
        for attack_str, config in patterns_data.items():
            try:
                attack_type = AttackType(attack_str)
            except ValueError:
                logger.warning(f"Unknown attack type: {attack_str}")
                continue
            if attack_type not in self._known_patterns:
                self._known_patterns[attack_type] = {"patterns": [], "keywords": []}
            for kw in config.get("keywords", []):
                if kw not in self._known_patterns[attack_type]["keywords"]:
                    self._known_patterns[attack_type]["keywords"].append(kw)
                    count += 1
            for pat in config.get("patterns", []):
                compiled = re.compile(pat, re.IGNORECASE)
                self._known_patterns[attack_type]["patterns"].append(compiled)
                count += 1
        logger.info(f"Updated {count} patterns from external source")
        return count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_checks": self.stats["total_checks"],
            "jailbreak_attempts": self.stats["jailbreak_attempts"],
            "block_rate": round(
                self.stats["jailbreak_attempts"] / max(self.stats["total_checks"], 1), 4
            ),
            "by_type": self.stats["by_type"],
            "known_patterns_count": sum(
                len(c["patterns"]) + len(c["keywords"])
                for c in self._known_patterns.values()
            ),
        }

    def clear_log(self) -> None:
        self.detection_log.clear()

    def get_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.detection_log[-limit:]

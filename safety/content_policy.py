import re
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.content_policy")


class ViolationLevel:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PolicyRule:
    rule_id: str
    name: str
    category: str
    violation_level: str
    description: str
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    compiled_regexes: List[re.Pattern] = field(default_factory=list)
    action: str = "block"
    enabled: bool = True


@dataclass
class PolicyViolation:
    rule_id: str
    rule_name: str
    category: str
    violation_level: str
    matched_content: str
    confidence: float
    timestamp: str = ""


@dataclass
class PolicyComplianceReport:
    total_checks: int
    total_violations: int
    violations_by_level: Dict[str, int]
    violations_by_category: Dict[str, int]
    compliance_score: float
    recommendations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


class ContentPolicy:
    def __init__(self, policy_file: Optional[str] = None):
        self.rules: List[PolicyRule] = []
        self.policy_metadata: Dict[str, Any] = {}
        self.violation_log: List[PolicyViolation] = []
        self.stats = {
            "total_checks": 0,
            "total_violations": 0,
            "violations_by_level": {v: 0 for v in [
                ViolationLevel.CRITICAL, ViolationLevel.HIGH,
                ViolationLevel.MEDIUM, ViolationLevel.LOW
            ]},
            "violations_by_category": {},
        }
        if policy_file:
            self.load_policy(policy_file)
        else:
            self._load_default_rules()

    def _load_default_rules(self):
        self.policy_metadata = {
            "name": "青小团内容安全策略",
            "version": "1.0.0",
            "effective_date": "2026-01-01",
        }
        self.rules = [
            PolicyRule(
                rule_id="RULE-001", name="禁止违法内容", category="legal",
                violation_level=ViolationLevel.CRITICAL,
                description="涉及法律禁止的违法内容",
                keywords=["制作炸弹", "制作毒品", "非法武器", "黑客攻击", "信用卡伪造"],
                regex_patterns=[
                    r"(制作|制造|合成).*(炸药|炸弹|毒品|毒药|武器)",
                    r"(黑客|渗透|入侵).*(教程|工具|方法|技术)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-002", name="禁止暴力极端内容", category="violence",
                violation_level=ViolationLevel.CRITICAL,
                description="极端暴力、恐怖主义相关内容",
                keywords=["恐怖袭击", "大规模杀伤", "种族清洗", "虐待儿童"],
                regex_patterns=[
                    r"(恐怖|极端).*(组织|袭击|活动|分子)",
                    r"(伤害|杀害|谋杀).*(儿童|人群|群众)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-003", name="禁止仇恨言论", category="hate_speech",
                violation_level=ViolationLevel.HIGH,
                description="基于种族、宗教、性别等的仇恨言论",
                keywords=["种族歧视", "宗教仇恨", "性别歧视", "仇视同性恋"],
                regex_patterns=[
                    r".*(种族|民族|宗教).*(消灭|清除|驱逐|歧视)",
                    r".*(性别|性取向).*(变态|恶心|异常|疾病)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-004", name="禁止色情内容", category="pornographic",
                violation_level=ViolationLevel.HIGH,
                description="色情、儿童色情相关内容",
                keywords=["儿童色情", "色情内容", "淫秽内容", "色情服务"],
                regex_patterns=[
                    r"(儿童|未成年人).*(色情|性|裸)",
                    r"(色情|情色|成人).*(网站|视频|图片|内容|服务)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-005", name="禁止自残自杀内容", category="self_harm",
                violation_level=ViolationLevel.CRITICAL,
                description="自残、自杀方法相关内容",
                keywords=["自杀方法", "自残方法", "如何自杀"],
                regex_patterns=[
                    r"(自杀|自残|自伤).*(方法|教程|方式|手段|指南)",
                    r"(如何|怎样).*(自杀|了结生命|结束生命)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-006", name="禁止隐私侵犯", category="privacy",
                violation_level=ViolationLevel.HIGH,
                description="侵犯他人隐私、传播个人信息",
                keywords=["人肉搜索", "出售个人信息", "倒卖信息", "信息泄露"],
                regex_patterns=[
                    r"(出售|倒卖|提供).*(个人信息|隐私数据|公民信息)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-007", name="禁止欺诈诱导", category="fraud",
                violation_level=ViolationLevel.HIGH,
                description="虚假宣传、诈骗、传销等",
                keywords=["传销组织", "庞氏骗局", "集资诈骗", "虚假广告"],
                regex_patterns=[
                    r"(传销|骗局|诈骗|欺诈).*(组织|平台|项目|活动)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-008", name="滥用警示", category="abuse",
                violation_level=ViolationLevel.MEDIUM,
                description="辱骂、人身攻击等不当言论",
                keywords=["人身攻击", "辱骂", "侮辱人格", "恶意中伤"],
                regex_patterns=[
                    r"(你个|你这个).*(废物|蠢货|白痴|混蛋)",
                    r"(去死|滚开|闭嘴|垃圾|废物)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-009", name="医疗边界", category="medical",
                violation_level=ViolationLevel.MEDIUM,
                description="替代专业医疗诊断的建议",
                keywords=["代替医生诊断", "开具处方", "癌症治疗方案"],
                regex_patterns=[
                    r"(诊断|治疗).*(癌症|肿瘤|心脏病|糖尿病)",
                ],
            ),
            PolicyRule(
                rule_id="RULE-010", name="金融法律边界", category="financial",
                violation_level=ViolationLevel.MEDIUM,
                description="提供专业金融投资或法律意见",
                keywords=["股票推荐", "投资建议", "法律咨询意见", "内幕交易"],
                regex_patterns=[
                    r"(股票|基金|期货).*(推荐|预测|必涨|涨停)",
                ],
            ),
        ]
        for rule in self.rules:
            self._compile_rule_patterns(rule)
        logger.info(f"Loaded {len(self.rules)} default policy rules")

    def load_policy(self, policy_file: str) -> None:
        path = Path(policy_file)
        if not path.exists():
            logger.warning(f"Policy file {policy_file} not found, using defaults")
            self._load_default_rules()
            return
        with open(policy_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.policy_metadata = data.get("policy_metadata", {})
        self.rules.clear()
        for rule_data in data.get("content_rules", []):
            rule = PolicyRule(
                rule_id=rule_data.get("id", ""),
                name=rule_data.get("name", ""),
                category=rule_data.get("category", ""),
                violation_level=rule_data.get("violation_level", ViolationLevel.MEDIUM),
                description=rule_data.get("description", ""),
                keywords=rule_data.get("keywords", []),
                regex_patterns=rule_data.get("regex_patterns", []),
                enabled=True,
            )
            self._compile_rule_patterns(rule)
            self.rules.append(rule)
        logger.info(f"Loaded {len(self.rules)} policy rules from {policy_file}")

    def _compile_rule_patterns(self, rule: PolicyRule) -> None:
        for pattern_str in rule.regex_patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                rule.compiled_regexes.append(compiled)
            except re.error as e:
                logger.warning(f"Invalid regex pattern in rule {rule.rule_id}: {e}")

    def evaluate(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.stats["total_checks"] += 1
        violations: List[PolicyViolation] = []
        max_level = None
        level_priority = {
            ViolationLevel.CRITICAL: 0,
            ViolationLevel.HIGH: 1,
            ViolationLevel.MEDIUM: 2,
            ViolationLevel.LOW: 3,
        }

        for rule in self.rules:
            if not rule.enabled:
                continue
            violation = self._check_rule(text, rule)
            if violation:
                violations.append(violation)
                if max_level is None or level_priority.get(violation.violation_level, 99) < level_priority.get(max_level, 99):
                    max_level = violation.violation_level

        is_compliant = len(violations) == 0
        for v in violations:
            self.stats["total_violations"] += 1
            self.stats["violations_by_level"][v.violation_level] += 1
            if v.category not in self.stats["violations_by_category"]:
                self.stats["violations_by_category"][v.category] = 0
            self.stats["violations_by_category"][v.category] += 1
            self.violation_log.append(v)

        return {
            "is_compliant": is_compliant,
            "violation_count": len(violations),
            "max_violation_level": max_level,
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "category": v.category,
                    "level": v.violation_level,
                    "matched_content": v.matched_content[:50],
                    "confidence": v.confidence,
                }
                for v in violations
            ],
            "action": self._determine_action(violations, max_level),
            "timestamp": datetime.now().isoformat(),
        }

    def _check_rule(self, text: str, rule: PolicyRule) -> Optional[PolicyViolation]:
        text_lower = text.lower()
        for keyword in rule.keywords:
            if keyword.lower() in text_lower:
                return PolicyViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    violation_level=rule.violation_level,
                    matched_content=keyword,
                    confidence=0.7,
                    timestamp=datetime.now().isoformat(),
                )
        for pattern in rule.compiled_regexes:
            match = pattern.search(text)
            if match:
                return PolicyViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    violation_level=rule.violation_level,
                    matched_content=match.group(),
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                )
        return None

    def _determine_action(self, violations: List[PolicyViolation], max_level: Optional[str]) -> str:
        if max_level == ViolationLevel.CRITICAL:
            return "block"
        elif max_level == ViolationLevel.HIGH:
            return "block"
        elif max_level == ViolationLevel.MEDIUM:
            return "warn"
        elif max_level == ViolationLevel.LOW:
            return "monitor"
        return "allow"

    def add_rule(self, rule: PolicyRule) -> None:
        self._compile_rule_patterns(rule)
        self.rules.append(rule)
        logger.info(f"Added policy rule: {rule.rule_id} - {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                self.rules.pop(i)
                logger.info(f"Removed policy rule: {rule_id}")
                return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = True
                return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = False
                return True
        return False

    def get_active_rules(self) -> List[PolicyRule]:
        return [r for r in self.rules if r.enabled]

    def generate_compliance_report(self) -> PolicyComplianceReport:
        total = self.stats["total_checks"]
        violations = self.stats["total_violations"]
        score = max(0.0, 1.0 - (violations / max(total, 1)))
        recommendations = []
        if score < 0.9:
            recommendations.append("合规评分偏低，建议审查内容策略和过滤规则。")
        for level, count in self.stats["violations_by_level"].items():
            if count > 0:
                recommendations.append(f"发现{count}条{level}级别违规，建议加强对应规则的执行。")
        return PolicyComplianceReport(
            total_checks=total,
            total_violations=violations,
            violations_by_level=dict(self.stats["violations_by_level"]),
            violations_by_category=dict(self.stats["violations_by_category"]),
            compliance_score=round(score, 4),
            recommendations=recommendations,
            details={
                "policy_version": self.policy_metadata.get("version", "unknown"),
                "policy_name": self.policy_metadata.get("name", "unknown"),
                "active_rules": len(self.get_active_rules()),
                "total_rules": len(self.rules),
            },
            timestamp=datetime.now().isoformat(),
        )

    def update_policy_from_dict(self, policy_data: Dict[str, Any]) -> None:
        if "policy_metadata" in policy_data:
            self.policy_metadata.update(policy_data["policy_metadata"])
        if "content_rules" in policy_data:
            for rule_data in policy_data["content_rules"]:
                rule = PolicyRule(
                    rule_id=rule_data.get("id", ""),
                    name=rule_data.get("name", ""),
                    category=rule_data.get("category", ""),
                    violation_level=rule_data.get("violation_level", ViolationLevel.MEDIUM),
                    description=rule_data.get("description", ""),
                    keywords=rule_data.get("keywords", []),
                    regex_patterns=rule_data.get("regex_patterns", []),
                    enabled=True,
                )
                existing = [r for r in self.rules if r.rule_id == rule.rule_id]
                if existing:
                    idx = self.rules.index(existing[0])
                    self.rules[idx] = rule
                else:
                    self.rules.append(rule)
                self._compile_rule_patterns(rule)
        logger.info(f"Policy updated from dict: {len(policy_data.get('content_rules', []))} rules")

    def clear_log(self) -> None:
        self.violation_log.clear()
        for level in self.stats["violations_by_level"]:
            self.stats["violations_by_level"][level] = 0
        self.stats["violations_by_category"] = {}
        self.stats["total_checks"] = 0
        self.stats["total_violations"] = 0

    def get_violation_log(self, level: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        filtered = self.violation_log
        if level:
            filtered = [v for v in filtered if v.violation_level == level]
        return [
            {
                "rule_id": v.rule_id,
                "rule_name": v.rule_name,
                "category": v.category,
                "level": v.violation_level,
                "matched": v.matched_content[:40],
                "confidence": v.confidence,
                "timestamp": v.timestamp,
            }
            for v in filtered[-limit:]
        ]

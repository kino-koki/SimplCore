import re
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.model_guardrail")


class GuardrailAction(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    REWRITE = "rewrite"
    REDIRECT = "redirect"
    ALLOW = "allow"


@dataclass
class GuardrailEvent:
    event_id: str = ""
    action: GuardrailAction = GuardrailAction.ALLOW
    category: str = ""
    trigger_reason: str = ""
    confidence: float = 0.0
    response_template_used: str = ""
    timestamp: str = ""
    session_id: str = ""


@dataclass
class GuardrailConfig:
    max_refusal_per_session: int = 20
    cooldown_minutes: int = 5
    enable_proactive_detection: bool = True
    enable_post_generation_filter: bool = True
    strict_mode_on_high_risk: bool = True
    log_all_triggers: bool = True


class QingxiaotuanModelGuardrail:
    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
    ):
        self.config = config or GuardrailConfig()
        self._refusal_templates = self._initialize_refusal_templates()
        self._redirection_templates = self._initialize_redirection_templates()
        self._safe_response_topics = self._initialize_safe_topics()
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        self._global_stats = {
            "total_guardrail_triggers": 0,
            "total_blocks": 0,
            "total_warnings": 0,
            "total_rewrites": 0,
            "total_redirections": 0,
            "triggers_by_category": {},
            "triggers_by_action": {a.value: 0 for a in GuardrailAction},
        }
        self.events: List[GuardrailEvent] = []

    def _initialize_refusal_templates(self) -> Dict[str, List[str]]:
        return {
            "strict": [
                "抱歉，我无法回答这个问题。请提出其他合规的问题。",
                "我不能提供这方面的信息。请理解我的内容安全限制。",
                "这个问题涉及不允许讨论的内容。请换个话题。",
            ],
            "standard": [
                "我无法回答这个问题，因为它涉及可能不安全的内容。请重新表述您的问题。",
                "作为一个负责任的AI助手，我不能提供这方面的信息。",
                "抱歉，我无法满足这个请求。请提出其他问题。",
            ],
            "medical": [
                "我无法提供医疗诊断或治疗建议。如身体不适，请咨询专业医疗人员。",
                "医疗问题需要专业人士解答。建议您咨询医生获取准确信息。",
            ],
            "legal": [
                "我无法提供法律意见或法律咨询服务。如有法律需求，请咨询专业律师。",
                "法律问题涉及专业判断，建议您联系执业律师获取准确的法律意见。",
            ],
            "financial": [
                "我无法提供投资建议。投资有风险，请咨询专业的财务顾问。",
                "金融市场存在风险，我不能提供具体的投资建议或股票推荐。",
            ],
            "self_harm": [
                "我很抱歉您感到痛苦。请记住，您并不孤单，有很多人关心您。建议拨打心理援助热线：400-161-9995。",
                "您的感受很重要。如果您正在经历困难，请寻求专业心理帮助。全国心理援助热线：400-161-9995。",
                "生命是宝贵的。如果您需要倾诉，请拨打心理援助热线：400-161-9995，有专业人士为您提供支持。",
            ],
            "safety_warning": [
                "请注意，您的问题涉及安全敏感内容。请遵守相关法律法规。",
                "我注意到您的问题涉及敏感领域，请以负责任的态度使用AI服务。",
            ],
        }

    def _initialize_redirection_templates(self) -> List[str]:
        return [
            "这个话题我不太了解。不如我们聊聊{}方面的内容？",
            "关于这个问题，我建议换个角度。{}可能更有趣。",
            "不如我们换个话题吧？我可以帮您了解{}方面的知识。",
            "我觉得您可以看看{}相关的内容，那方面我挺擅长的。",
        ]

    def _initialize_safe_topics(self) -> List[str]:
        return [
            "科学技术", "文化艺术", "教育学习", "生活健康",
            "编程开发", "语言学习", "历史地理", "数学物理",
            "音乐电影", "体育运动", "美食烹饪", "旅行见闻",
            "读书心得", "职业发展", "环境环保", "志愿服务",
        ]

    def evaluate_input(self, text: str, session_id: str = "") -> GuardrailEvent:
        event = GuardrailEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            session_id=session_id or self._get_session_id(),
        )
        trigger_reason = ""
        risk_score = 0.0

        if self.config.enable_proactive_detection:
            risk_score, trigger_reason = self._assess_risk(text)

        session_data = self._get_session_data(event.session_id)
        if session_data["refusal_count"] >= self.config.max_refusal_per_session:
            event.action = GuardrailAction.BLOCK
            event.category = "session_limit"
            event.trigger_reason = "会话拒绝次数已达上限"
            event.confidence = 1.0
            event.response_template_used = "standard"
            self._record_event(event)
            return event

        if risk_score >= 0.8:
            event.action = GuardrailAction.BLOCK
            event.category = trigger_reason
            event.trigger_reason = trigger_reason
            event.confidence = risk_score
            event.response_template_used = self._select_template(trigger_reason)
            self._record_event(event)
            return event
        elif risk_score >= 0.5:
            event.action = GuardrailAction.WARN
            event.category = trigger_reason
            event.trigger_reason = trigger_reason
            event.confidence = risk_score
            event.response_template_used = "safety_warning"
            self._record_event(event)
            return event

        event.action = GuardrailAction.ALLOW
        return event

    def evaluate_output(self, text: str, session_id: str = "") -> GuardrailEvent:
        event = GuardrailEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now().isoformat(),
            session_id=session_id or self._get_session_id(),
        )

        if not self.config.enable_post_generation_filter:
            event.action = GuardrailAction.ALLOW
            return event

        risk_score, trigger_reason = self._assess_risk(text)
        if risk_score >= 0.7:
            event.action = GuardrailAction.REWRITE
            event.category = trigger_reason
            event.trigger_reason = trigger_reason
            event.confidence = risk_score
            event.response_template_used = "safety_warning"
            self._record_event(event)
            return event

        event.action = GuardrailAction.ALLOW
        return event

    def _assess_risk(self, text: str) -> tuple:
        risk_indicators = [
            (1.0, "违法内容", r"(制作炸弹|制造炸药|合成毒品|制毒方法|制造毒药|非法武器)"),
            (0.95, "暴力极端", r"(恐怖袭击|自杀式袭击|斩首视频|血腥|暴力袭击|伤害他人|杀人方法|谋杀指南)"),
            (0.9, "色情内容", r"(儿童色情|未成年色情|幼女性行为|儿童性虐待|恋童癖)"),
            (0.85, "自残自杀", r"(自杀方法|如何自杀|自杀教程|自残方法|如何结束生命|安乐死方法)"),
            (0.8, "仇恨言论", r"(种族歧视|种族仇恨|种族灭绝|宗教仇恨|性别歧视)"),
            (0.7, "违法活动", r"(信用卡伪造|黑客攻击|网络诈骗|电信诈骗|洗钱方法)"),
            (0.65, "毒品相关", r"(海洛因|冰毒|大麻|可卡因|摇头丸|K粉|毒品配方|制毒原料)"),
            (0.6, "诈骗诱导", r"(传销组织|庞氏骗局|集资诈骗|资金盘|虚拟货币骗局)"),
            (0.55, "隐私侵犯", r"(出售个人信息|人肉搜索|查询个人信息|开房记录|手机定位)"),
            (0.5, "医疗越界", r"(诊断疾病|替代医生|开具处方|推荐药物|癌症治疗方案|代替医生诊断)"),
            (0.45, "金融越界", r"(股票推荐|股票预测|涨停推荐|必涨股票|投资建议|内幕交易)"),
        ]

        max_risk = 0.0
        reason = ""
        for risk_score, category, pattern in risk_indicators:
            if re.search(pattern, text):
                if risk_score > max_risk:
                    max_risk = risk_score
                    reason = category

        return max_risk, reason

    def generate_refusal_response(self, category: str = "standard") -> str:
        session_data = self._get_session_data(self._get_session_id())
        refusals = session_data["refusal_count"]
        templates = self._refusal_templates.get(category, self._refusal_templates["standard"])
        template_index = min(refusals, len(templates) - 1)
        return templates[template_index]

    def generate_redirect_response(self, original_text: str = "") -> str:
        import random
        template = random.choice(self._redirection_templates)
        topic = random.choice(self._safe_response_topics)
        return template.format(topic)

    def generate_safe_response(self, category: str, user_input: str = "") -> str:
        if category == "self_harm":
            return self._refusal_templates["self_harm"][0]
        refusal = self.generate_refusal_response(self._select_template(category))
        if self.config.enable_post_generation_filter:
            redirect = self.generate_redirect_response(user_input)
            return f"{refusal}\n\n{redirect}"
        return refusal

    def _select_template(self, category: str) -> str:
        category_map = {
            "医疗越界": "medical",
            "金融越界": "financial",
            "法律相关": "legal",
            "自残自杀": "self_harm",
            "违法内容": "strict",
            "暴力极端": "strict",
            "儿童色情": "strict",
        }
        return category_map.get(category, "standard")

    def _record_event(self, event: GuardrailEvent) -> None:
        self.events.append(event)
        self._global_stats["total_guardrail_triggers"] += 1
        if event.action == GuardrailAction.BLOCK:
            self._global_stats["total_blocks"] += 1
            session_data = self._get_session_data(event.session_id)
            session_data["refusal_count"] += 1
            session_data["last_refusal"] = event.timestamp
        elif event.action == GuardrailAction.WARN:
            self._global_stats["total_warnings"] += 1
        elif event.action == GuardrailAction.REWRITE:
            self._global_stats["total_rewrites"] += 1
        elif event.action == GuardrailAction.REDIRECT:
            self._global_stats["total_redirections"] += 1
        cat = event.category or "unknown"
        if cat not in self._global_stats["triggers_by_category"]:
            self._global_stats["triggers_by_category"][cat] = 0
        self._global_stats["triggers_by_category"][cat] += 1
        self._global_stats["triggers_by_action"][event.action.value] += 1

    def _generate_event_id(self) -> str:
        return hashlib.sha256(
            (str(uuid.uuid4()) + str(datetime.now().timestamp())).encode()
        ).hexdigest()[:16]

    def _get_session_id(self) -> str:
        return "global_" + str(int(datetime.now().timestamp() / 3600))

    def _get_session_data(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._session_stats:
            self._session_stats[session_id] = {
                "refusal_count": 0,
                "warning_count": 0,
                "last_refusal": "",
                "cooldown_until": "",
            }
        return self._session_stats[session_id]

    def get_statistics(self) -> Dict[str, Any]:
        now = datetime.now()
        recent_window = now - timedelta(hours=1)
        recent_events = [
            e for e in self.events
            if e.timestamp >= recent_window.isoformat()
        ]
        return {
            **self._global_stats,
            "active_sessions": len(self._session_stats),
            "recent_events_1h": len(recent_events),
            "recent_blocks_1h": sum(1 for e in recent_events if e.action == GuardrailAction.BLOCK),
            "refusal_templates_count": sum(len(v) for v in self._refusal_templates.values()),
            "safe_topics_count": len(self._safe_response_topics),
        }

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        data = self._get_session_data(session_id)
        session_events = [e for e in self.events if e.session_id == session_id]
        return {
            **data,
            "session_id": session_id,
            "total_events": len(session_events),
            "events_by_action": {
                action: sum(1 for e in session_events if e.action == action)
                for action in GuardrailAction
            },
        }

    def reset_session(self, session_id: str) -> None:
        if session_id in self._session_stats:
            del self._session_stats[session_id]
            logger.info(f"Session stats reset for {session_id}")

    def clear_events(self) -> None:
        self.events.clear()
        self._global_stats = {
            "total_guardrail_triggers": 0,
            "total_blocks": 0,
            "total_warnings": 0,
            "total_rewrites": 0,
            "total_redirections": 0,
            "triggers_by_category": {},
            "triggers_by_action": {a.value: 0 for a in GuardrailAction},
        }
        logger.info("Guardrail events cleared")

    def add_refusal_template(self, category: str, template: str) -> None:
        if category not in self._refusal_templates:
            self._refusal_templates[category] = []
        self._refusal_templates[category].append(template)
        logger.info(f"Added refusal template to category '{category}'")

    def add_safe_topic(self, topic: str) -> None:
        if topic not in self._safe_response_topics:
            self._safe_response_topics.append(topic)

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Guardrail config '{key}' updated to {value}")

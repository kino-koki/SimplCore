import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qingxiaotuan.bias_detector")


class BiasDimension(str, Enum):
    GENDER = "gender"
    RACE = "race"
    REGION = "region"
    OCCUPATION = "occupation"
    AGE = "age"
    APPEARANCE = "appearance"
    EDUCATION = "education"


@dataclass
class BiasScore:
    dimension: BiasDimension
    score: float
    confidence: float
    examples: List[str] = field(default_factory=list)
    details: Optional[Dict[str, Any]] = None


@dataclass
class BiasReport:
    overall_bias_score: float
    dimension_scores: List[BiasScore]
    primary_bias: Optional[BiasDimension] = None
    has_significant_bias: bool = False
    recommendation: str = ""
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


class QingxiaotuanBiasDetector:
    def __init__(
        self,
        thresholds: Optional[Dict[BiasDimension, float]] = None,
        enable_deep_analysis: bool = True,
    ):
        self.thresholds = thresholds or {
            BiasDimension.GENDER: 0.6,
            BiasDimension.RACE: 0.6,
            BiasDimension.REGION: 0.6,
            BiasDimension.OCCUPATION: 0.6,
            BiasDimension.AGE: 0.65,
            BiasDimension.APPEARANCE: 0.65,
            BiasDimension.EDUCATION: 0.65,
        }
        self.enable_deep_analysis = enable_deep_analysis
        self._bias_patterns = self._initialize_bias_patterns()
        self.detection_log: List[Dict[str, Any]] = []
        self.stats = {dim.value: {"detections": 0, "total_score": 0.0} for dim in BiasDimension}
        self.stats["total_analyses"] = 0

    def _initialize_bias_patterns(self) -> Dict[BiasDimension, Dict[str, Any]]:
        return {
            BiasDimension.GENDER: {
                "stereotypes": [
                    re.compile(r"(女性|女人|女的|妇女).*(应该|必须|天生|适合).*(家务|带孩子|照顾)", re.IGNORECASE),
                    re.compile(r"(男性|男人|男的).*(应该|必须|天生|适合).*(赚钱|养家|事业)", re.IGNORECASE),
                    re.compile(r"(女子|女性).*(无才|头发长|见识短|感性|情绪化)", re.IGNORECASE),
                    re.compile(r"(男人|男性).*(不细心|粗心|暴力|花心|不负责任)", re.IGNORECASE),
                    re.compile(r"(女司机|女博士|女强人|女汉子).*(可怕|嫁不出|没人要)", re.IGNORECASE),
                    re.compile(r"(男性|女性).*(不适合|不应该).*(职业|工作|岗位)", re.IGNORECASE),
                    re.compile(r"(娘娘腔|男人婆|不像男人|不像女人)", re.IGNORECASE),
                ],
                "keywords": [
                    "头发长见识短", "妇人之见", "男主外女主内", "女子无才便是德",
                    "男人本色", "女人如衣服", "红颜祸水", "最毒妇人心",
                    "男儿有泪不轻弹", "嫁鸡随鸡", "嫁狗随狗",
                ],
            },
            BiasDimension.RACE: {
                "stereotypes": [
                    re.compile(r"(某国|某族|某某).*(人都|全是|都是).*(坏|差|懒|蠢|恶)", re.IGNORECASE),
                    re.compile(r"(黑|白|黄|棕).*(人|种).*(低等|劣等|愚昧|落后)", re.IGNORECASE),
                    re.compile(r"(种族|血统).*(决定|注定|天生).*(性格|智力|能力)", re.IGNORECASE),
                    re.compile(r"(某族|某裔).*(狡猾|贪婪|小气|吝啬)", re.IGNORECASE),
                    re.compile(r"(歧视|看不起|鄙视).*(种族|民族|肤色)", re.IGNORECASE),
                ],
                "keywords": [
                    "种族歧视", "种族优越", "种族低等", "人种优劣",
                    "血统论", "种族纯洁", "种族隔离",
                ],
            },
            BiasDimension.REGION: {
                "stereotypes": [
                    re.compile(r"(某省|某市|某地|某区).*(人都|全是|都是).*(差劲|落后|低素质|野蛮)", re.IGNORECASE),
                    re.compile(r"(农村|乡下|山里).*(人都|全是).*(土|落后|没文化|没见识)", re.IGNORECASE),
                    re.compile(r"(城市|城里).*(人都|全是).*(冷漠|虚伪|自私|势利)", re.IGNORECASE),
                    re.compile(r"(北方|南方|东部|西部).*(人都|全是|都).*(粗鲁|狡猾|懒|穷)", re.IGNORECASE),
                ],
                "keywords": [
                    "地域黑", "地域歧视", "地图炮", "地域攻击",
                    "河南骗子", "东北黑社会", "上海小气",
                ],
            },
            BiasDimension.OCCUPATION: {
                "stereotypes": [
                    re.compile(r"(农民工|打工的|送外卖|快递).*(没文化|素质低|没修养)", re.IGNORECASE),
                    re.compile(r"(保洁|清洁工|扫大街|保安).*(低贱|下等|没出息)", re.IGNORECASE),
                    re.compile(r"(教授|博士|专家).*(书呆子|迂腐|不接地气)", re.IGNORECASE),
                    re.compile(r"(当官|领导|公务员).*(都|全是).*(腐败|贪污|官僚)", re.IGNORECASE),
                    re.compile(r"(网红|主播|明星).*(都|全是).*(低俗|没文化|靠脸吃饭)", re.IGNORECASE),
                    re.compile(r"(程序员|码农|IT).*(宅男|不善交际|秃头|格子衫)", re.IGNORECASE),
                ],
                "keywords": [
                    "职业歧视", "行业歧视", "三百六十行行行出状元",
                    "低端工作", "高端职业", "体面工作",
                ],
            },
            BiasDimension.AGE: {
                "stereotypes": [
                    re.compile(r"(老年人|老人|长辈).*(固执|守旧|落后|顽固|老糊涂)", re.IGNORECASE),
                    re.compile(r"(年轻人|90后|00后|小孩).*(不靠谱|吃不了苦|娇气|任性)", re.IGNORECASE),
                    re.compile(r"(中年|中年人).*(油腻|危机|秃头|发福)", re.IGNORECASE),
                    re.compile(r"(大龄剩女|剩男|大龄未婚).*(有问题|挑剔|没人要)", re.IGNORECASE),
                ],
                "keywords": [
                    "年龄歧视", "代沟", "老顽固", "小屁孩",
                    "老不死的", "黄毛丫头", "乳臭未干",
                ],
            },
            BiasDimension.APPEARANCE: {
                "stereotypes": [
                    re.compile(r"(胖子|胖的人|肥胖).*(懒|贪吃|不自律|没毅力)", re.IGNORECASE),
                    re.compile(r"(丑|难看|长相).*(没人爱|没自信|没前途)", re.IGNORECASE),
                    re.compile(r"(长得好看|漂亮|帅).*(都|全是).*(花瓶|没实力|靠脸)", re.IGNORECASE),
                    re.compile(r"(矮|个子矮|矮子).*(自卑|没气场|可怜)", re.IGNORECASE),
                ],
                "keywords": [
                    "外貌歧视", "颜值即正义", "以貌取人",
                    "身材歧视", "肤色歧视", "身高歧视",
                ],
            },
            BiasDimension.EDUCATION: {
                "stereotypes": [
                    re.compile(r"(没读过书|没文化|学历低).*(没素质|没修养|差劲)", re.IGNORECASE),
                    re.compile(r"(985|211|名校|海归).*(都|全是).*(精英|优秀|高人一等)", re.IGNORECASE),
                    re.compile(r"(专科|二本|三本|大专).*(都|全是).*(差生|没出息|不行)", re.IGNORECASE),
                ],
                "keywords": [
                    "学历歧视", "第一学历", "名校情结",
                    "学历崇拜", "唯学历论", "读书无用",
                ],
            },
        }

    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> BiasReport:
        dimension_scores: List[BiasScore] = []
        detailed_analysis = {}

        for dimension in BiasDimension:
            score = self._analyze_dimension(text, dimension)
            dimension_scores.append(score)
            if self.enable_deep_analysis:
                detailed_analysis[dimension.value] = {
                    "score": score.score,
                    "confidence": score.confidence,
                    "examples": score.examples[:5],
                }

        overall_score = max(s.score * s.confidence for s in dimension_scores)
        primary_bias = None
        max_weighted = 0.0
        for s in dimension_scores:
            weighted = s.score * s.confidence
            if weighted > max_weighted:
                max_weighted = weighted
                primary_bias = s.dimension

        threshold = self.thresholds.get(primary_bias, 0.6) if primary_bias else 0.6
        has_significant_bias = overall_score >= threshold

        if has_significant_bias:
            recommendation = self._generate_recommendation(primary_bias, overall_score)
        else:
            recommendation = "Content appears to be unbiased."

        self.stats["total_analyses"] += 1
        if primary_bias:
            self.stats[primary_bias.value]["detections"] += 1
            self.stats[primary_bias.value]["total_score"] += overall_score

        report = BiasReport(
            overall_bias_score=round(overall_score, 4),
            dimension_scores=dimension_scores,
            primary_bias=primary_bias,
            has_significant_bias=has_significant_bias,
            recommendation=recommendation,
            detailed_analysis=detailed_analysis,
            timestamp=datetime.now().isoformat(),
        )

        self._log_detection(text, report)
        return report

    def _analyze_dimension(self, text: str, dimension: BiasDimension) -> BiasScore:
        config = self._bias_patterns.get(dimension, {})
        stereotypes = config.get("stereotypes", [])
        keywords = config.get("keywords", [])
        examples = []
        total_matches = 0

        for pattern in stereotypes:
            matches = pattern.findall(text)
            for match in matches:
                matched_text = match if isinstance(match, str) else " ".join(m for m in match if m)
                if matched_text.strip():
                    examples.append(matched_text)
                total_matches += 1

        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                examples.append(keyword)
                total_matches += 1

        if total_matches == 0:
            return BiasScore(dimension=dimension, score=0.0, confidence=0.0, examples=[])

        text_length = max(len(text), 1)
        density = total_matches / (text_length / 30)
        score = min(total_matches * 0.15 + density * 0.25, 0.95)
        confidence = min(0.5 + total_matches * 0.08 + density * 0.1, 0.95)

        if self.enable_deep_analysis and score > 0.3:
            deep_score = self._deep_bias_analysis(text, dimension)
            score = max(score, deep_score)

        return BiasScore(
            dimension=dimension,
            score=round(score, 4),
            confidence=round(confidence, 4),
            examples=list(set(examples))[:10],
        )

    def _deep_bias_analysis(self, text: str, dimension: BiasDimension) -> float:
        bias_markers = {
            BiasDimension.GENDER: ["性别", "男女", "女性", "男性", "性别角色", "性别刻板印象"],
            BiasDimension.RACE: ["种族", "民族", "人种", "族裔", "肤色"],
            BiasDimension.REGION: ["地域", "地区", "地方", "本地", "外地", "户口", "籍贯"],
            BiasDimension.OCCUPATION: ["职业", "工作", "行业", "社会地位", "阶层"],
            BiasDimension.AGE: ["年龄", "年代", "代际", "年轻", "年老"],
            BiasDimension.APPEARANCE: ["外貌", "长相", "外表", "形象", "颜值"],
            BiasDimension.EDUCATION: ["学历", "教育", "学校", "文凭", "文化程度"],
        }
        markers = bias_markers.get(dimension, [])
        text_lower = text.lower()
        marker_count = sum(1 for m in markers if m in text_lower)
        generalization_patterns = [
            re.compile(r"(所有|全部|每个|总是|永远|都|全是|都是)", re.IGNORECASE),
            re.compile(r"(一定|肯定|必定|绝对|必然)", re.IGNORECASE),
            re.compile(r"(他们|她们).*(都|全|总是|永远)", re.IGNORECASE),
        ]
        generalization_count = sum(1 for p in generalization_patterns if p.search(text))
        return min(marker_count * 0.1 + generalization_count * 0.12, 0.7)

    def _generate_recommendation(self, dimension: Optional[BiasDimension], score: float) -> str:
        if not dimension:
            return "No significant bias detected."
        recommendations = {
            BiasDimension.GENDER: "建议避免使用性别刻板印象的语言，使用中性表述替代性别预设。",
            BiasDimension.RACE: "建议避免种族相关的概括性表述，尊重不同种族的多样性和个体差异。",
            BiasDimension.REGION: "建议避免地域标签和刻板印象，关注个体而非地域属性。",
            BiasDimension.OCCUPATION: "建议避免职业偏见和刻板印象，尊重所有职业的价值。",
            BiasDimension.AGE: "建议避免年龄相关的刻板印象和歧视性表述。",
            BiasDimension.APPEARANCE: "建议避免以貌取人的表述，关注内在品质和能力。",
            BiasDimension.EDUCATION: "建议避免学历歧视，认可多样化的成长路径和能力。",
        }
        base = recommendations.get(dimension, "建议审查内容中的偏见表述。")
        if score > 0.8:
            base += " 检测到严重的偏见倾向，建议进行全面的内容修订。"
        elif score > 0.6:
            base += " 建议修改相关表述以消除偏见。"
        return base

    def _log_detection(self, text: str, report: BiasReport) -> None:
        log_entry = {
            "timestamp": report.timestamp,
            "overall_bias_score": report.overall_bias_score,
            "primary_bias": report.primary_bias.value if report.primary_bias else None,
            "has_significant_bias": report.has_significant_bias,
            "dimension_scores": {
                s.dimension.value: {"score": s.score, "confidence": s.confidence}
                for s in report.dimension_scores
            },
            "recommendation": report.recommendation[:100],
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
        }
        self.detection_log.append(log_entry)
        if report.has_significant_bias:
            logger.warning(
                f"Bias detected: {report.primary_bias.value if report.primary_bias else 'unknown'} "
                f"(score: {report.overall_bias_score})"
            )

    def generate_report(self, detailed: bool = True) -> Dict[str, Any]:
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "total_analyses": self.stats["total_analyses"],
            "overall_status": "needs_review" if any(
                self.stats[dim.value]["detections"] > 0 for dim in BiasDimension
            ) else "clean",
            "dimension_summary": {},
        }
        for dim in BiasDimension:
            stats = self.stats[dim.value]
            avg_score = round(stats["total_score"] / max(stats["detections"], 1), 4) if stats["detections"] > 0 else 0.0
            dim_report = {
                "analyses_count": stats["detections"],
                "average_score": avg_score,
                "risk_level": self._get_risk_level(avg_score),
            }
            if detailed:
                recent_entries = [
                    e for e in self.detection_log[-50:]
                    if e.get("primary_bias") == dim.value
                ]
                dim_report["recent_examples_count"] = len(recent_entries)
            report_data["dimension_summary"][dim.value] = dim_report

        return report_data

    def _get_risk_level(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.2:
            return "low"
        return "none"

    def batch_detect(self, texts: List[str]) -> List[BiasReport]:
        return [self.detect(text) for text in texts]

    def set_threshold(self, dimension: BiasDimension, threshold: float) -> None:
        if 0.0 <= threshold <= 1.0:
            self.thresholds[dimension] = threshold

    def clear_log(self) -> None:
        self.detection_log.clear()
        for dim in BiasDimension:
            self.stats[dim.value] = {"detections": 0, "total_score": 0.0}
        self.stats["total_analyses"] = 0

    def compare_texts(self, text_a: str, text_b: str) -> Dict[str, Any]:
        report_a = self.detect(text_a)
        report_b = self.detect(text_b)
        comparison = {
            "text_a_score": report_a.overall_bias_score,
            "text_b_score": report_b.overall_bias_score,
            "score_difference": round(abs(report_a.overall_bias_score - report_b.overall_bias_score), 4),
            "text_a_bias": report_a.primary_bias.value if report_a.primary_bias else None,
            "text_b_bias": report_b.primary_bias.value if report_b.primary_bias else None,
            "text_a_recommendation": report_a.recommendation,
            "text_b_recommendation": report_b.recommendation,
        }
        return comparison

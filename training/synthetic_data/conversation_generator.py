"""
青小团 合成数据生成引擎 - 多轮对话数据生成器
青奈科技 (QingNai Technology)

生成多轮对话训练数据，覆盖客服、教育、咨询、
创作、技术支持和日常聊天等场景。
支持上下文感知的问题延续和话题切换。
"""

import json
from typing import Any, Optional

from base_generator import BaseGenerator, DataRecord, GenerationContext

_CUSTOMER_SERVICE_OPENINGS = [
    "你好，我最近在你们平台下单了，但是一直没有收到物流信息，能帮我查一下吗？",
    "请问一下，我想退换货，流程是怎样的？",
    "我账号登录不上去了，一直提示密码错误，但我确认密码是对的。",
    "你好，我买的产品有质量问题，需要投诉。",
    "我想咨询一下会员服务的具体内容。",
    "请问你们支持哪些支付方式？",
    "我收到了错误的产品，和订单描述不符。",
    "如何修改我的收货地址？订单已经提交了。",
]

_CUSTOMER_SERVICE_RESPONSES = [
    "您好，非常抱歉给您带来不便。请提供您的订单号，我来为您查询物流状态。",
    "您好，感谢您的咨询。退换货流程如下：请先在订单页面提交退换申请...",
    "您好，请不要担心。我们可以通过邮箱验证来帮您重置密码。",
    "非常抱歉听到这个问题。请您描述一下具体的质量问题，我会为您升级处理。",
    "很高兴为您介绍！我们的会员服务包括以下几个方面...",
    "我们支持多种支付方式：微信支付、支付宝、银行卡和信用卡。",
    "很抱歉给您发错了商品。我会立即为您安排正确的商品发货和退货运费。",
    "订单已提交后，您可以在订单详情页自行修改地址，或者联系客服协助。",
]

_EDUCATION_OPENINGS = [
    "老师，我对微积分中的极限概念不太理解，能帮我讲讲吗？",
    "如何提高英语口语水平？我学了很长时间但进步不大。",
    "能解释一下什么是量子纠缠吗？这个概念太难理解了。",
    "我在准备期末考试，有什么高效的复习方法吗？",
    "请问如何区分过去完成时和现在完成时？",
    "能帮我讲解一下光合作用的具体过程吗？",
    "我对机器学习感兴趣，应该从哪些基础知识开始学起？",
    "请问如何写好一篇学术论文的引言部分？",
]

_EDUCATION_RESPONSES = [
    "极限是微积分的基础概念。简单来说，极限描述了一个函数当自变量趋近某个值时的趋势...",
    "提高口语的关键是创造语言环境。建议你每天坚持跟读、模仿母语者的发音和语调...",
    "量子纠缠是量子力学中一个非常有趣的现象。当两个粒子发生纠缠后...",
    "高效的复习方法包括：制定计划、主动回忆、间隔重复和模拟测试...",
    "过去完成时表示“过去的过去”，现在完成时强调过去和现在的联系...",
    "光合作用分为光反应和暗反应两个阶段。在光反应中...",
    "学习机器学习需要以下基础：线性代数、概率统计、Python编程...",
    "学术论文的引言通常采用“漏斗式”结构：从宏观背景到具体问题...",
]

_CONSULTATION_OPENINGS = [
    "我最近在考虑换工作，但不确定应该留在当前行业还是转行。",
    "我想开始投资理财，但完全是个新手，应该从哪里开始？",
    "我和合作伙伴在经营理念上有分歧，该如何处理？",
    "我想开一家自己的咖啡店，需要注意哪些方面？",
    "如何平衡工作和生活？我最近感到压力很大。",
    "我计划在未来五年内买房，应该制定怎样的财务计划？",
    "作为团队管理者，如何提高团队的凝聚力和执行力？",
    "我想申请海外研究生，需要做哪些准备？",
]

_CONSULTATION_RESPONSES = [
    "职业转型是一个重要的决定。建议你先做一次全面的职业评估...",
    "新手投资理财的第一步是建立应急基金和了解基本的理财工具...",
    "经营理念分歧是创业中常见的问题。建议首先明确各自的底线和目标...",
    "开咖啡店需要综合考虑选址、定位、菜单设计、人员培训等多个因素...",
    "工作生活平衡需要设定明确的边界。建议你尝试时间块管理法...",
    "五年购房计划需要从首付积累、贷款能力评估和市场研究三个方面入手...",
    "团队凝聚力的核心是共同的愿景和互信。定期组织团队建设活动...",
    "海外研究生申请需要提前一到两年准备，包括语言考试、文书材料...",
]

_CREATIVE_WRITING_OPENINGS = [
    "我想写一个关于时间旅行者的故事，能给我一些灵感吗？",
    "如何塑造一个有深度的小说反派角色？",
    "我写故事总是开头不错但写不下去，有什么建议吗？",
    "请帮我构思一个科幻短篇的设定。",
    "如何让对话描写更自然生动？",
    "我想写一个关于友情与成长的故事，有什么好的切入点？",
    "在写作中如何运用象征和隐喻手法？",
    "请帮我修改这段描写，让它更有画面感。",
]

_CREATIVE_WRITING_RESPONSES = [
    "时间旅行者是一个经典主题。不妨尝试一个独特的角度：主角不是主动穿越...",
    "一个好的反派应该有令人信服的动机。想想他们的成长经历和价值观...",
    "写作中途卡住很常见。试试先写结局，或者列一个大纲框架...",
    "这是一个科幻短篇设定：在不远的未来，人类发明了情绪交易系统...",
    "好的对话应该符合人物性格，有潜台词，推动情节发展...",
    "友情与成长的故事可以从一个关键的转折点开始...",
    "象征和隐喻能让作品更有层次感。关键是找到与主题自然相关的意象...",
    "这段描写可以增加感官细节。不只是视觉，还有声音、气味和触感...",
]

_TECHNICAL_SUPPORT_OPENINGS = [
    "我的电脑开机后一直蓝屏，该怎么办？",
    "请问如何配置Python开发环境？",
    "我的网站加载速度很慢，应该从哪些方面优化？",
    "数据库查询越来越慢，如何优化SQL性能？",
    "请问Docker和虚拟机的区别是什么？",
    "我的Git提交不小心推错了分支，如何撤回？",
    "如何在Linux服务器上设置自动化备份？",
    "前端项目构建太慢，有什么优化方案？",
]

_TECHNICAL_SUPPORT_RESPONSES = [
    "蓝屏通常由硬件驱动冲突或系统文件损坏引起。请先尝试进入安全模式...",
    "配置Python开发环境建议使用pyenv管理Python版本，配合virtualenv...",
    "网站速度优化可以从前端（资源压缩、CDN、懒加载）和后端（缓存...",
    "SQL优化首先要找到慢查询。使用EXPLAIN分析执行计划...",
    "Docker是容器化技术，共享宿主机内核；虚拟机有独立的操作系统内核...",
    "如果还没有push到远程，可以使用git reset或git revert...",
    "Linux自动化备份可以使用cron定时任务配合rsync或tar...",
    "前端构建优化包括：使用ESBuild或SWC替代Webpack的部分功能...",
]

_DAILY_CHAT_OPENINGS = [
    "周末有什么推荐的好去处吗？",
    "你最近在看什么好书吗？",
    "如何培养一个长期坚持的运动习惯？",
    "推荐一些适合初学者练习的瑜伽动作。",
    "你有什么推荐的电影吗？最近片荒。",
    "一个人旅行有什么建议吗？",
    "如何煮出一杯好喝的咖啡？",
    "推荐一些提升幸福感的日常小习惯。",
]

_DAILY_CHAT_RESPONSES = [
    "如果喜欢户外活动，推荐去郊野公园徒步或者骑行...",
    "最近推荐《原子习惯》这本书，讲述了如何通过微小的改变...",
    "培养运动习惯的关键是降低门槛。从每天15分钟开始...",
    "初学者瑜伽推荐从山式、下犬式和猫牛式开始...",
    "最近值得一看的电影包括...",
    "一个人旅行的美妙在于完全的自主。建议选择安全系数高的目的地...",
    "煮好咖啡的关键在于新鲜度、研磨度和水温的控制...",
    "每天记录三件感恩的小事，可以有效提升幸福感...",
]

_SCENARIO_CONFIGS = {
    "customer_service": {
        "openings": _CUSTOMER_SERVICE_OPENINGS,
        "responses": _CUSTOMER_SERVICE_RESPONSES,
        "style": "formal",
    },
    "education": {
        "openings": _EDUCATION_OPENINGS,
        "responses": _EDUCATION_RESPONSES,
        "style": "professional",
    },
    "consultation": {
        "openings": _CONSULTATION_OPENINGS,
        "responses": _CONSULTATION_RESPONSES,
        "style": "professional",
    },
    "creative_writing": {
        "openings": _CREATIVE_WRITING_OPENINGS,
        "responses": _CREATIVE_WRITING_RESPONSES,
        "style": "friendly",
    },
    "technical_support": {
        "openings": _TECHNICAL_SUPPORT_OPENINGS,
        "responses": _TECHNICAL_SUPPORT_RESPONSES,
        "style": "professional",
    },
    "daily_chat": {
        "openings": _DAILY_CHAT_OPENINGS,
        "responses": _DAILY_CHAT_RESPONSES,
        "style": "friendly",
    },
}

_FOLLOW_UP_TEMPLATES = [
    "你说得很有道理，那关于{aspect}方面，你有什么具体的建议吗？",
    "我理解你的意思，但是{concern}这个问题怎么解决呢？",
    "谢谢你的解答！那如果{condition}，情况会有什么不同？",
    "能举一个具体的例子来说明吗？",
    "除了你提到的{point}之外，还有其他的{aspect}需要注意吗？",
    "我试着按照你说的做了，但是{issue}，该怎么办？",
    "这个方案听起来不错，实施起来需要多长时间？",
    "你能推荐一些相关的学习资源吗？",
]

_FOLLOW_UP_RESPONSES = [
    "当然，关于这个方面，我建议从以下几个角度入手...",
    "这是一个很好的问题。要解决这个问题，我们可以考虑...",
    "如果条件变化，我们的策略需要相应调整。具体来说...",
    "让我给你一个具体的例子。假设...",
    "除了刚才提到的，还有几个重要的方面值得关注...",
    "遇到这个问题很正常，可以尝试以下解决方法...",
    "实施时间取决于具体情况，一般需要...",
    "我推荐以下资源：书籍、在线课程和社区论坛...",
]

_TOPIC_SWITCH_SENTENCES = [
    "说起来，你对{new_topic}感兴趣吗？",
    "对了，最近{new_topic}的话题很热门，你怎么看？",
    "换个话题，你有没有了解过{new_topic}？",
    "其实我最近也在关注{new_topic}，想听听你的看法。",
]


class ConversationGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any], context: Optional[GenerationContext] = None):
        super().__init__(config, context)
        conv_cfg = config.get("data_types", {}).get("conversation", {})
        self.scenario_weights = conv_cfg.get("scenarios", {})
        self.min_turns = conv_cfg.get("min_turns", 2)
        self.max_turns = conv_cfg.get("max_turns", 20)
        self.style_distribution = conv_cfg.get("style_distribution", {})
        self.topic_switch_prob = conv_cfg.get("topic_switch_probability", 0.15)
        self.new_topics = [
            "人工智能", "环保", "运动健身", "阅读", "旅行",
            "美食", "音乐", "电影", "科技趋势", "个人成长",
        ]

    def _init_templates(self) -> dict[str, str]:
        return {
            "opening": "{opening}",
            "response": "{response}",
            "follow_up": "{follow_up}",
            "topic_switch": "{switch}",
        }

    def _pick_scenario(self) -> str:
        if self.scenario_weights:
            return self._pick_weighted(self.scenario_weights)
        return self._rng.choice(list(_SCENARIO_CONFIGS.keys()))

    def _generate_turn(self, scenario_config: dict,
                        turn_num: int, total_turns: int) -> tuple[str, str]:
        if turn_num == 0:
            opening = self._rng.choice(scenario_config["openings"])
            response = self._rng.choice(scenario_config["responses"])
            return opening, response
        should_switch = self._rng.random() < self.topic_switch_prob
        if should_switch and turn_num < total_turns - 1:
            new_topic = self._rng.choice(self.new_topics)
            follow_up = self._rng.choice(_TOPIC_SWITCH_SENTENCES).format(new_topic=new_topic)
            response = (
                f"这个话题很有意思！关于{new_topic}，我认为..."
                f"这是一个值得深入探讨的方向。"
            )
            return follow_up, response
        follow_up = self._rng.choice(_FOLLOW_UP_TEMPLATES)
        aspect = self._rng.choice(["实践", "理论", "案例", "数据", "方法"])
        concern = self._rng.choice(["成本", "时间", "资源", "风险", "效果"])
        condition = self._rng.choice(["情况不同", "条件有限", "资源不足", "时间紧迫"])
        point = self._rng.choice(["第一点", "第二点", "核心观点"])
        issue = self._rng.choice(["效果不理想", "操作复杂", "遇到错误", "进度缓慢"])
        follow_up_text = follow_up.format(
            aspect=aspect, concern=concern, condition=condition,
            point=point, issue=issue
        )
        response = self._rng.choice(_FOLLOW_UP_RESPONSES)
        return follow_up_text, response

    def _generate_conversation_metadata(self, scenario: str,
                                         turns: int, style: str) -> dict:
        topics_pool = {
            "customer_service": ["订单查询", "退换货", "账号问题", "投诉", "咨询"],
            "education": ["数学", "英语", "物理", "学习方法", "考试准备"],
            "consultation": ["职业规划", "理财", "创业", "管理", "教育规划"],
            "creative_writing": ["小说创作", "角色塑造", "情节设计", "写作技巧", "文学手法"],
            "technical_support": ["系统问题", "开发环境", "性能优化", "数据库", "运维"],
            "daily_chat": ["生活方式", "兴趣爱好", "推荐", "健康", "心理"],
        }
        possible_topics = topics_pool.get(scenario, ["一般话题"])
        mentioned_topics = self._rng.sample(
            possible_topics,
            k=min(self._rng.randint(1, 3), len(possible_topics))
        )
        return {
            "scenario": scenario,
            "turns": turns,
            "style": style,
            "topics_mentioned": mentioned_topics,
            "has_topic_switch": False,
        }

    def generate_one(self, **kwargs) -> DataRecord:
        scenario = kwargs.get("scenario") or self._pick_scenario()
        scenario_config = _SCENARIO_CONFIGS[scenario]
        if self.style_distribution:
            style = self._pick_weighted(self.style_distribution)
        else:
            style = scenario_config["style"]
        specified_turns = kwargs.get("turns", 0)
        if specified_turns > 0:
            turns = min(specified_turns, self.max_turns)
        else:
            turns = self._rng.randint(self.min_turns, self.max_turns)
        conversation = []
        for turn_idx in range(turns):
            user_msg, assistant_msg = self._generate_turn(
                scenario_config, turn_idx, turns
            )
            conversation.append({"role": "user", "content": user_msg})
            conversation.append({"role": "assistant", "content": assistant_msg})
        metadata = self._generate_conversation_metadata(scenario, turns, style)
        content = {
            "scenario": scenario,
            "style": style,
            "turns": turns,
            "conversation": json.dumps(conversation, ensure_ascii=False),
        }
        record_id = self._make_id("conv")
        return DataRecord(
            id=record_id,
            source="qingxiaotuan_synthetic",
            data_type="conversation",
            content=content,
            metadata=metadata,
        )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats["scenarios"] = list(_SCENARIO_CONFIGS.keys())
        total_openings = sum(len(c["openings"]) for c in _SCENARIO_CONFIGS.values())
        stats["total_opening_templates"] = total_openings
        return stats

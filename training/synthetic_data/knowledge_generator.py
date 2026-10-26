"""
青小团 合成数据生成引擎 - 知识密集型数据生成器
青奈科技 (QingNai Technology)

生成覆盖科学、历史、文学、艺术、技术、医学、
法律、金融等领域的知识问答和解释数据。
"""

import json
from typing import Any, Optional

from base_generator import BaseGenerator, DataRecord, GenerationContext

_KNOWLEDGE_BASES = {
    "science": {
        "name": "科学",
        "topics": [
            {
                "concept": "相对论",
                "fact": "爱因斯坦在1905年提出狭义相对论，1915年提出广义相对论。质能方程E=mc²是相对论的核心公式。",
                "explanation": "相对论是现代物理学的两大支柱之一。狭义相对论指出时间和空间是相对的，光速在真空中恒定不变。广义相对论将引力描述为时空弯曲的几何效应。",
                "comparison": "与牛顿力学相比，相对论在高速和强引力场条件下揭示了全新的物理现象，如时间膨胀、长度收缩和引力红移。",
            },
            {
                "concept": "进化论",
                "fact": "达尔文在1859年发表《物种起源》，提出自然选择学说。",
                "explanation": "进化论是生物学的核心理论。自然选择是指适应环境的个体更有可能存活并繁殖，从而将其有利特征传递给后代。",
                "comparison": "与拉马克的用进废退说相比，达尔文的自然选择学说强调变异是随机的，环境对变异进行筛选。",
            },
            {
                "concept": "量子力学",
                "fact": "量子力学起源于20世纪初，描述微观粒子的行为规律。",
                "explanation": "量子力学揭示了微观世界的概率本质。粒子可以同时处于多个状态的叠加，直到被观测时才坍缩为确定状态。",
                "comparison": "与经典力学不同，量子力学中的物理量不是连续变化的，而是以量子化的方式跃迁。",
            },
            {
                "concept": "DNA双螺旋结构",
                "fact": "沃森和克里克在1953年发现了DNA的双螺旋结构。",
                "explanation": "DNA双螺旋由两条互补的核苷酸链以反向平行的方式缠绕而成。碱基配对规则（A-T，C-G）是遗传信息传递的基础。",
                "comparison": "与RNA的单链结构不同，DNA的双螺旋结构更加稳定，适合长期存储遗传信息。",
            },
        ],
    },
    "history": {
        "name": "历史",
        "topics": [
            {
                "concept": "工业革命",
                "fact": "工业革命始于18世纪60年代的英国，以蒸汽机的改良和应用为标志。",
                "explanation": "工业革命是生产方式从手工劳动向机器生产的伟大转变。珍妮纺纱机、蒸汽机和铁路的发明极大地提高了生产效率。",
                "comparison": "与第一次工业革命的蒸汽时代不同，第二次工业革命以电力和内燃机为标志，推动生产进入电气化时代。",
            },
            {
                "concept": "文艺复兴",
                "fact": "文艺复兴14世纪始于意大利，16世纪达到鼎盛，是欧洲文化思想解放运动。",
                "explanation": "文艺复兴倡导人文主义精神，强调人的价值和尊严。达芬奇、米开朗基罗、拉斐尔等艺术大师创作了传世杰作。",
                "comparison": "与中世纪的以神为中心不同，文艺复兴将关注点从神转向人，推动了科学、艺术和文学的全面繁荣。",
            },
        ],
    },
    "literature": {
        "name": "文学",
        "topics": [
            {
                "concept": "现实主义文学",
                "fact": "现实主义文学兴起于19世纪的欧洲，代表人物有巴尔扎克、狄更斯和托尔斯泰。",
                "explanation": "现实主义文学主张客观真实地反映社会生活，注重典型环境中的典型人物塑造。巴尔扎克的《人间喜剧》是其代表作。",
                "comparison": "与浪漫主义文学强调主观情感和想象不同，现实主义更注重客观描写和社会批判。",
            },
            {
                "concept": "中国古典四大名著",
                "fact": "《红楼梦》《西游记》《三国演义》《水浒传》是中国古典文学的巅峰之作。",
                "explanation": "这四部小说代表了不同题材类型的最高成就。《红楼梦》是世情小说的巅峰，《西游记》开创了神魔小说传统。",
                "comparison": "与西方长篇小说相比，中国古典小说更多采用章回体结构，融合诗词韵文，具有独特的叙事美学。",
            },
        ],
    },
    "art": {
        "name": "艺术",
        "topics": [
            {
                "concept": "印象派",
                "fact": "印象派起源于19世纪60年代的法国，莫奈的《印象·日出》为该流派命名。",
                "explanation": "印象派画家注重光影变化和瞬间印象的捕捉，摒弃了学院派的精细描绘。他们常在户外作画，用明亮的色彩和松散的笔触表现自然光线。",
                "comparison": "与古典主义的严谨构图和细腻笔触不同，印象派更强调画家对光线和色彩的主观感受。",
            },
            {
                "concept": "巴洛克艺术",
                "fact": "巴洛克艺术17世纪盛行于欧洲，以豪华、动感和戏剧性为特征。",
                "explanation": "巴洛克艺术强调动态感和强烈的情感表达，喜欢使用曲线、金色和丰富的装饰。贝尼尼的雕塑和鲁本斯的绘画是其代表。",
                "comparison": "与文艺复兴的平衡和谐不同，巴洛克追求戏剧性的张力和豪华的视觉冲击。",
            },
        ],
    },
    "technology": {
        "name": "技术",
        "topics": [
            {
                "concept": "人工智能",
                "fact": "人工智能的概念于1956年在达特茅斯会议上正式提出。深度学习在2010年代取得突破性进展。",
                "explanation": "人工智能是计算机科学的一个分支，旨在创造能够模拟人类智能的系统。机器学习是AI的核心方法，通过数据训练模型来完成特定任务。",
                "comparison": "与传统程序设计不同，机器学习不是显式编程规则，而是从数据中自动学习模式和规律。",
            },
            {
                "concept": "区块链",
                "fact": "区块链技术于2008年由中本聪提出，是比特币的底层技术。",
                "explanation": "区块链是一种去中心化的分布式账本技术，数据以区块形式链接存储，每个区块包含时间戳和加密哈希。",
                "comparison": "与传统中心化数据库不同，区块链没有单一控制点，数据一旦写入就不可篡改。",
            },
            {
                "concept": "云计算",
                "fact": "云计算概念于2006年由亚马逊AWS率先提出并商业化。",
                "explanation": "云计算通过网络按需提供计算资源（服务器、存储、数据库、网络等），用户只需为实际使用的资源付费。",
                "comparison": "与传统本地部署相比，云计算提供了弹性伸缩、按需付费和更高的可用性。",
            },
        ],
    },
    "medicine": {
        "name": "医学",
        "topics": [
            {
                "concept": "免疫系统",
                "fact": "免疫系统是人体抵御病原体入侵的防御体系，包含固有免疫和适应性免疫。",
                "explanation": "免疫系统通过识别自身和非自身物质来保护机体。T细胞和B细胞是适应性免疫的核心，分别负责细胞免疫和体液免疫。",
                "comparison": "固有免疫反应迅速但非特异性，适应性免疫反应较慢但具有记忆性和特异性。",
            },
            {
                "concept": "抗生素",
                "fact": "青霉素于1928年由弗莱明发现，是人类第一个抗生素。",
                "explanation": "抗生素是抑制或杀死细菌的药物。它们通过干扰细菌的细胞壁合成、蛋白质合成或DNA复制来发挥作用。",
                "comparison": "与抗病毒药物不同，抗生素只对细菌有效，对病毒无效。滥用抗生素会导致细菌耐药性。",
            },
        ],
    },
    "law": {
        "name": "法律",
        "topics": [
            {
                "concept": "知识产权",
                "fact": "知识产权包括专利权、商标权、著作权和商业秘密等。",
                "explanation": "知识产权是法律赋予创造者对其智力成果的专有权利。专利保护发明，商标保护品牌标识，著作权保护原创作品。",
                "comparison": "与物权保护有形财产不同，知识产权保护的是无形的智力成果，具有时间和地域限制。",
            },
            {
                "concept": "合同法",
                "fact": "合同法是调整合同关系的法律规范的总称，是市场经济的基本法律制度。",
                "explanation": "合同是平等主体之间设立、变更、终止民事权利义务关系的协议。合同成立需要要约和承诺两个基本要素。",
                "comparison": "与侵权法针对不法侵害不同，合同法规制的是当事人之间的自愿约定。",
            },
        ],
    },
    "finance": {
        "name": "金融",
        "topics": [
            {
                "concept": "股票市场",
                "fact": "世界上最早的股票交易所是1602年成立的阿姆斯特丹证券交易所。",
                "explanation": "股票是公司所有权的凭证，股票市场是股票发行和交易的场所。投资者通过买卖股票参与公司成长并承担风险。",
                "comparison": "与债券的固定收益不同，股票的收益来自股息和资本利得，风险和潜在回报都更高。",
            },
            {
                "concept": "货币政策",
                "fact": "中央银行通过货币政策工具调控货币供应量和利率。",
                "explanation": "货币政策工具包括存款准备金率、再贴现率和公开市场操作。扩张性货币政策刺激经济，紧缩性政策抑制通胀。",
                "comparison": "与财政政策由政府预算决定不同，货币政策由中央银行独立制定，更加灵活。",
            },
        ],
    },
    "philosophy": {
        "name": "哲学",
        "topics": [
            {
                "concept": "存在主义",
                "fact": "存在主义是20世纪重要的哲学流派，代表人物有萨特、加缪和海德格尔。",
                "explanation": "存在主义强调个人的自由、选择和责任。萨特提出"存在先于本质"，认为人首先存在，然后通过自己的选择和行动定义自己。",
                "comparison": "与本质主义认为事物有固定本质不同，存在主义认为人的本质是在自由选择中自我塑造的。",
            },
        ],
    },
    "geography": {
        "name": "地理",
        "topics": [
            {
                "concept": "气候变化",
                "fact": "全球平均气温自工业革命以来已上升约1.2摄氏度。",
                "explanation": "气候变化主要由温室气体排放引起。二氧化碳、甲烷等温室气体 trapping 热量，导致全球变暖、极端天气增加和海平面上升。",
                "comparison": "与自然气候波动不同，当前的气候变化速度远超自然变率，主要归因于人类活动。",
            },
        ],
    },
}

_QUESTION_TEMPLATES_FACTUAL = [
    "什么是{concept}？",
    "请简述{concept}的基本事实。",
    "{concept}是什么时候提出的？",
    "谁提出了{concept}？",
]

_QUESTION_TEMPLATES_EXPLANATORY = [
    "请详细解释{concept}。",
    "{concept}的工作原理是什么？",
    "请用通俗的语言解释{concept}。",
    "{concept}的核心思想是什么？",
]

_QUESTION_TEMPLATES_COMPARATIVE = [
    "请比较{concept}和{related_concept}的异同。",
    "{concept}与{related_concept}有什么不同？",
    "相比{related_concept}，{concept}的优势是什么？",
]

_QUESTION_TEMPLATES_STRUCTURED = [
    "请以表格形式列出{concept}的关键要点。",
    "请用列表总结{concept}的主要特征。",
    "请为{concept}做一个结构化的知识梳理。",
]


class KnowledgeGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any], context: Optional[GenerationContext] = None):
        super().__init__(config, context)
        knowledge_cfg = config.get("data_types", {}).get("knowledge", {})
        self.domain_weights = knowledge_cfg.get("domains", {})
        self.question_type_weights = knowledge_cfg.get("question_types", {})
        self.difficulty_weights = knowledge_cfg.get("difficulty", {})

    def _init_templates(self) -> dict[str, str]:
        return {
            "factual_q": "{question}",
            "explanatory_q": "{question}",
            "comparative_q": "{question}",
            "structured_q": "{question}",
        }

    def _pick_domain(self) -> str:
        if self.domain_weights:
            return self._pick_weighted(self.domain_weights)
        return self._rng.choice(list(_KNOWLEDGE_BASES.keys()))

    def _pick_question_type(self) -> str:
        if self.question_type_weights:
            return self._pick_weighted(self.question_type_weights)
        return self._rng.choice(["factual", "explanatory", "comparative", "structured"])

    def _get_topic(self, domain: str) -> dict:
        domain_data = _KNOWLEDGE_BASES.get(domain, list(_KNOWLEDGE_BASES.values())[0])
        topics = domain_data["topics"]
        return self._rng.choice(topics)

    def _generate_factual_qa(self, domain: str, topic: dict,
                              difficulty: str) -> tuple[str, str]:
        template = self._rng.choice(_QUESTION_TEMPLATES_FACTUAL)
        question = template.format(concept=topic["concept"])
        answer = topic["fact"]
        return question, answer

    def _generate_explanatory_qa(self, domain: str, topic: dict,
                                   difficulty: str) -> tuple[str, str]:
        template = self._rng.choice(_QUESTION_TEMPLATES_EXPLANATORY)
        question = template.format(concept=topic["concept"])
        base_answer = topic["explanation"]
        if difficulty == "advanced":
            answer = (
                f"{base_answer}\n\n"
                f"深入探讨：{topic['concept']}在{_KNOWLEDGE_BASES[domain]['name']}领域"
                f"有着广泛的影响。其理论框架不仅解释了现有现象，"
                f"还为未来的研究和应用指明了方向。"
                f"理解这一概念需要掌握相关的背景知识体系。"
            )
        else:
            answer = topic["explanation"]
        return question, answer

    def _generate_comparative_qa(self, domain: str, topic: dict,
                                   difficulty: str) -> tuple[str, str]:
        domain_data = _KNOWLEDGE_BASES[domain]
        other_topics = [t for t in domain_data["topics"] if t["concept"] != topic["concept"]]
        if other_topics:
            other = self._rng.choice(other_topics)
            related = other["concept"]
        else:
            related = f"相关{domain_data['name']}概念"
        template = self._rng.choice(_QUESTION_TEMPLATES_COMPARATIVE)
        question = template.format(concept=topic["concept"], related_concept=related)
        answer = topic.get("comparison", f"{topic['concept']}与{related}在{domain_data['name']}领域有着不同的侧重点和应用场景。")
        return question, answer

    def _generate_structured_qa(self, domain: str, topic: dict,
                                 difficulty: str) -> tuple[str, str]:
        template = self._rng.choice(_QUESTION_TEMPLATES_STRUCTURED)
        question = template.format(concept=topic["concept"])
        domain_name = _KNOWLEDGE_BASES[domain]["name"]
        answer = (
            f"【{topic['concept']}知识梳理】\n\n"
            f"领域：{domain_name}\n\n"
            f"核心要点：\n"
            f"1. 定义：{topic['concept']}是{domain_name}领域的重要概念\n"
            f"2. 关键事实：{topic['fact']}\n"
            f"3. 原理说明：{topic['explanation'][:100]}...\n"
            f"4. 应用价值：在{domain_name}研究和实践中具有重要地位\n\n"
            f"相关概念：\n"
            f"- 上位概念：{domain_name}理论体系\n"
            f"- 平行概念：同领域的其他核心概念\n"
            f"- 下位概念：具体的应用和分支\n\n"
            f"学习建议：从基本定义入手，结合实例理解，"
            f"逐步深入到理论细节和应用场景。"
        )
        return question, answer

    def generate_one(self, **kwargs) -> DataRecord:
        domain = kwargs.get("domain") or self._pick_domain()
        question_type = kwargs.get("question_type") or self._pick_question_type()
        difficulty = kwargs.get("difficulty") or self._pick_weighted(
            self.difficulty_weights if self.difficulty_weights
            else {"basic": 0.3, "intermediate": 0.4, "advanced": 0.3}
        )
        topic = self._get_topic(domain)
        qa_generators = {
            "factual": self._generate_factual_qa,
            "explanatory": self._generate_explanatory_qa,
            "comparative": self._generate_comparative_qa,
            "structured": self._generate_structured_qa,
        }
        generator = qa_generators.get(question_type, self._generate_factual_qa)
        question, answer = generator(domain, topic, difficulty)
        content = {
            "domain": domain,
            "domain_name": _KNOWLEDGE_BASES[domain]["name"],
            "concept": topic["concept"],
            "question_type": question_type,
            "difficulty": difficulty,
            "question": question,
            "answer": answer,
            "structured": question_type == "structured",
        }
        if question_type == "factual":
            content["source_type"] = "factual_knowledge"
        elif question_type == "explanatory":
            content["source_type"] = "explanatory_knowledge"
        elif question_type == "comparative":
            content["source_type"] = "comparative_knowledge"
        elif question_type == "structured":
            content["source_type"] = "structured_knowledge"
        record_id = self._make_id("know")
        return DataRecord(
            id=record_id,
            source="qingxiaotuan_synthetic",
            data_type="knowledge",
            content=content,
            metadata={
                "domain": domain,
                "domain_name": _KNOWLEDGE_BASES[domain]["name"],
                "concept": topic["concept"],
                "question_type": question_type,
                "difficulty": difficulty,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats["domains"] = list(_KNOWLEDGE_BASES.keys())
        stats["total_concepts"] = sum(
            len(d["topics"]) for d in _KNOWLEDGE_BASES.values()
        )
        return stats

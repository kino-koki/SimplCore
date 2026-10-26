"""
青小团 合成数据生成引擎 - 指令微调数据生成器
青奈科技 (QingNai Technology)

生成指令微调训练数据，包含50+种指令模板，
支持多轮对话、角色扮演、多难度和多风格输出。
"""

import json
import random
from typing import Any, Optional

from base_generator import BaseGenerator, DataRecord, GenerationContext

_INSTRUCTION_TOPICS_WRITING = [
    "写一篇关于{ topic }的议论文，字数{ words }字左右",
    "请为{ product }撰写一份营销文案，目标受众是{ audience }",
    "写一封{ style }的邮件，内容是{ content }",
    "帮我写一首关于{ theme }的{ poem_type }",
    "请创作一个{ genre }风格的短故事，主题围绕{ theme }",
    "撰写一份{ length }的产品说明书，产品是{ product }",
    "写一篇博客文章，介绍{ topic }，面向{ audience }读者",
    "帮我起草一份{ type }合同，涉及{ field }领域",
    "写一篇{ tone }风格的新闻报道，关于{ event }",
    "请为{ brand }品牌撰写{ count }条社交媒体推文",
]

_INSTRUCTION_TOPICS_TRANSLATION = [
    "将以下{ source_lang }文本翻译成{ target_lang }：{ text }",
    "请将这段{ field }领域的专业文本从{ source_lang }译成{ target_lang }，保持术语准确",
    "翻译下面的句子，注意保持{ style }风格：{ text }",
    "请对以下文本进行{ source_lang }到{ target_lang }的翻译，并解释关键术语：{ text }",
    "将这段技术文档从{ source_lang }翻译为{ target_lang }，确保技术细节准确",
    "请翻译并本地化以下{ source_lang }广告文案为{ target_lang }，适配目标文化：{ text }",
    "逐句翻译以下{ source_lang }诗歌为{ target_lang }，保留韵律和意境：{ text }",
    "将以下{ source_lang }法律文件翻译成{ target_lang }，保持法律效力：{ text }",
]

_INSTRUCTION_TOPICS_SUMMARIZATION = [
    "请总结以下文本的核心观点：{ text }",
    "用{ word_count }字以内概括这篇文章的主要内容：{ text }",
    "请为这段{ field }领域的文本撰写一份{ length }摘要",
    "将以下长文本压缩为{ ratio }比例的摘要，保留关键信息：{ text }",
    "请从{ perspective }角度总结以下对话内容：{ text }",
    "为这篇学术论文撰写一段{ level }摘要，突出研究方法和结论：{ text }",
    "请用{ style }的语言风格总结以下新闻：{ text }",
    "提取以下文本中的关键数据和结论：{ text }",
]

_INSTRUCTION_TOPICS_REASONING = [
    "请分析{ scenario }的因果关系，并给出{ count }个可能的解释",
    "如果{ premise }，那么可能的结果是什么？请逐步推理",
    "请比较{ item_a }和{ item_b }在{ aspect }方面的异同",
    "判断以下论证是否有效：{ argument }。请说明理由",
    "请从{ perspective_a }和{ perspective_b }两个角度分析{ issue }",
    "评估{ plan }的可行性，列出优势和风险",
    "请按重要性排序以下{ items }，并解释排序依据",
    "如果{ condition }成立，请推断{ situation }会如何发展",
    "请找出以下推理中的逻辑谬误：{ argument }",
    "给定{ constraints }约束条件，请设计最优解决方案",
]

_INSTRUCTION_TOPICS_ANALYSIS = [
    "请分析{ topic }的市场趋势，包括{ aspect_a }、{ aspect_b }和{ aspect_c }",
    "对{ text }进行{ level }级别的文本分析，关注其修辞手法和表达效果",
    "请分析{ data }中呈现的关键模式和异常值",
    "从{ perspective }角度分析{ issue }的深层原因",
    "请对{ company }的{ strategy }战略进行SWOT分析",
    "分析{ work }的艺术特色和历史价值",
    "请对{ policy }政策进行多维度影响分析",
    "分析{ phenomenon }背后的{ field }学原理",
]

_INSTRUCTION_TOPICS_EXPLANATION = [
    "请用通俗易懂的方式解释{ concept }，面向{ audience }",
    "请详细解释{ term }的工作原理和实际应用",
    "用{ analogy_type }比喻来解释{ concept }",
    "请为{ audience }撰写一份关于{ topic }的入门指南",
    "解释{ phenomenon }背后的科学原理，包括{ aspect_a }和{ aspect_b }",
    "请用{ step_count }个步骤解释{ process }的完整流程",
    "对比解释{ concept_a }和{ concept_b }的区别与联系",
    "请从历史发展的角度解释{ topic }的演变过程",
]

_INSTRUCTION_TOPICS_CREATIVE = [
    "请创作一段关于{ theme }的{ format }，风格为{ style }",
    "为{ scenario }设计一个创新的解决方案",
    "请以{ character }的视角重写{ story }中的关键场景",
    "创作一个{ genre }类型的游戏概念设计，核心机制是{ mechanic }",
    "请为{ brand }设计一句品牌口号和品牌故事",
    "想象你生活在{ era }时期，描述你一天的生活",
    "请设计一个融合{ element_a }和{ element_b }的创新产品概念",
    "用{ style }的艺术风格描述一幅名为{ title }的画作",
]

_INSTRUCTION_TOPICS_CODING = [
    "请写一个{ lang }函数实现{ algorithm }算法，并解释时间空间复杂度",
    "如何使用{ lang }的{ library }库实现{ functionality }？请提供示例代码",
    "请重构以下代码，提高其可读性和性能：\n```{lang}\n{code}\n```",
    "请为以下{ lang }代码添加完整的错误处理和日志：\n```{lang}\n{code}\n```",
    "用{ lang }实现一个{ design_pattern }模式，并给出使用示例",
    "请写一段{ lang }代码，实现{ task }，要求支持{ requirement }",
    "如何优化以下{ lang }代码？请给出优化前后的对比：\n```{lang}\n{code}\n```",
    "请用{ lang }编写单元测试，覆盖以下函数的各种边界情况：\n```{lang}\n{code}\n```",
]

_INSTRUCTION_TOPICS_ROLEPLAY = [
    "你现在是一位{ role }，请回答以下问题：{ question }",
    "请以{ role }的身份，为{ audience }提供关于{ topic }的建议",
    "假设你是{ role }，{ scenario }，你会怎么做？",
    "请扮演{ role }和{ role_b }，围绕{ topic }展开一段对话",
    "作为一位经验丰富的{ role }，请分享{ count }条关于{ field }的建议",
    "你现在是{ role }，请用{ style }的语气解释{ concept }",
    "请以{ role }的视角，评价{ event }并给出专业见解",
    "如果你是一位来自{ era }的{ role }，穿越到现代，你会对什么感到最惊讶？",
]

_MULTI_TURN_CONTINUATIONS = [
    "请进一步详细说明{ aspect }",
    "你能从{ perspective }角度再分析一下吗？",
    "刚才提到的{ point }，能举一个具体的例子吗？",
    "谢谢，那如果{ condition }，结果会有什么不同？",
    "请对比{ item_a }和{ item_b }在这个问题上的观点",
    "我理解了，但能否用更{ style }的语言再解释一遍？",
    "关于这个问题，还有什么常见的误解吗？",
    "如果我想深入学习{ topic }，你有什么建议？",
]

_DIFFICULTY_ADJECTIVES = {
    "easy": ["简单", "基础", "入门级"],
    "medium": ["中等难度", "详细", "深入"],
    "hard": ["复杂", "深入分析", "高级"],
    "expert": ["专业级", "学术级", "综合性"],
}


class InstructionGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any], context: Optional[GenerationContext] = None):
        super().__init__(config, context)
        self.categories = config.get("data_types", {}).get("instruction", {})
        self._all_templates = self._init_templates()

    def _init_templates(self) -> dict[str, list[str]]:
        return {
            "writing": _INSTRUCTION_TOPICS_WRITING,
            "translation": _INSTRUCTION_TOPICS_TRANSLATION,
            "summarization": _INSTRUCTION_TOPICS_SUMMARIZATION,
            "reasoning": _INSTRUCTION_TOPICS_REASONING,
            "analysis": _INSTRUCTION_TOPICS_ANALYSIS,
            "explanation": _INSTRUCTION_TOPICS_EXPLANATION,
            "creative": _INSTRUCTION_TOPICS_CREATIVE,
            "coding": _INSTRUCTION_TOPICS_CODING,
            "roleplay": _INSTRUCTION_TOPICS_ROLEPLAY,
        }

    def _pick_category(self) -> str:
        cat_config = self.categories.get("template_categories", [])
        if cat_config:
            return self._rng.choice(cat_config)
        return self._rng.choice(list(self._all_templates.keys()))

    def _pick_template(self, category: str) -> str:
        templates = self._all_templates.get(category, self._all_templates["writing"])
        return self._rng.choice(templates)

    def _generate_template_kwargs(self, category: str, difficulty: str) -> dict[str, Any]:
        kwargs = {
            "topic": self._rng.choice([
                "人工智能伦理", "气候变化", "远程工作的未来", "教育公平",
                "可持续发展", "数字隐私", "文化多样性", "科技创新",
                "artificial intelligence", "climate change", "space exploration",
                "quantum computing", "renewable energy", "mental health",
            ]),
            "audience": self._rng.choice(["初学者", "专业人士", "学生", "普通读者", "管理者"]),
            "style": self._rng.choice(["正式", "轻松", "学术", "幽默", "感人", "简洁"]),
            "field": self._rng.choice(["计算机科学", "医学", "法律", "金融", "教育", "工程"]),
            "lang": self._rng.choice(["python", "javascript", "java", "go", "rust", "typescript"]),
            "source_lang": self._rng.choice(["中文", "英文", "日文", "法文", "德文"]),
            "target_lang": self._rng.choice(["中文", "英文", "日文", "法文", "德文"]),
            "length": self._rng.choice(["简短", "详细", "全面", "精炼"]),
            "level": self._rng.choice(["基础", "中级", "高级", "专家"]),
            "perspective": self._rng.choice(["经济", "社会", "技术", "文化", "环境", "政治"]),
            "text": "这是一个示例文本，用于演示模板替换功能。在实际生成中，这里会出现更长的内容。",
            "count": str(self._rng.randint(3, 10)),
            "word_count": str(self._rng.randint(50, 500)),
            "difficulty": difficulty,
        }
        if category == "coding":
            kwargs["algorithm"] = self._rng.choice([
                "快速排序", "二分查找", "动态规划", "深度优先搜索",
                "Dijkstra最短路径", "KMP字符串匹配", "归并排序",
            ])
            kwargs["functionality"] = self._rng.choice([
                "数据处理", "用户认证", "文件操作", "网络请求", "缓存管理",
            ])
            kwargs["code"] = "def example():\n    pass"
            kwargs["design_pattern"] = self._rng.choice([
                "单例", "工厂", "观察者", "策略", "装饰器", "适配器",
            ])
            kwargs["task"] = self._rng.choice([
                "文件解析器", "REST API客户端", "数据管道", "任务调度器",
            ])
            kwargs["requirement"] = self._rng.choice([
                "高并发支持", "内存效率优化", "可扩展性", "线程安全",
            ])
        if category == "roleplay":
            kwargs["role"] = self._rng.choice([
                "医生", "律师", "教师", "工程师", "艺术家",
                "历史学家", "哲学家", "科学家", "企业家", "心理咨询师",
            ])
            kwargs["role_b"] = self._rng.choice([
                "记者", "学生", "客户", "同事", "朋友",
            ])
            kwargs["question"] = self._rng.choice([
                "如何提高工作效率？", "未来十年的发展趋势是什么？",
                "应该选择什么样的职业道路？",
            ])
        if category == "creative":
            kwargs["theme"] = self._rng.choice([
                "希望与坚持", "科技与人性", "自然之美", "时间与记忆",
            ])
            kwargs["format"] = self._rng.choice([
                "短诗", "微小说", "歌词", "剧本片段", "散文",
            ])
            kwargs["genre"] = self._rng.choice([
                "科幻", "奇幻", "悬疑", "浪漫", "现实主义",
            ])
        if difficulty == "expert" and category in ("writing", "analysis"):
            kwargs["level"] = self._rng.choice(["专业级", "学术级", "博士级"])
        return kwargs

    def _build_answer(self, category: str, instruction: str, difficulty: str, kwargs: dict) -> str:
        templates = {
            "writing": f"以下是按照您的要求创作的{kwargs.get('style', '')}风格内容：\n\n"
                       f"关于「{kwargs.get('topic', '')}」的{kwargs.get('length', '')}论述：\n\n"
                       f"在当今时代，{kwargs.get('topic', '')}已成为社会各界关注的重要议题。"
                       f"从多个维度来看，这一话题涉及{kwargs.get('field', '多个领域')}的交叉与融合。"
                       f"首先，我们需要理解其核心内涵和基本特征。"
                       f"其次，应当分析其对不同群体的实际影响。"
                       f"最后，还需要提出有价值的见解和建议。\n\n"
                       f"综上所述，{kwargs.get('topic', '')}值得我们持续关注和深入思考。",
            "translation": f"【翻译结果】\n\n源文本已按要求从{kwargs.get('source_lang', '')}"
                          f"翻译为{kwargs.get('target_lang', '')}。\n\n"
                          f"译文：这是翻译后的文本内容，保持了原文的{kwargs.get('style', '')}风格"
                          f"和{kwargs.get('field', '')}领域的术语准确性。\n\n"
                          f"【翻译说明】\n1. 关键术语已按{kwargs.get('field', '')}领域标准译法处理"
                          f"\n2. 句式结构根据目标语言习惯进行了调整"
                          f"\n3. 文化特定表达已做本地化处理",
            "summarization": f"【摘要】\n\n"
                            f"核心观点：本文主要讨论了{kwargs.get('topic', '')}相关问题，"
                            f"从{kwargs.get('perspective', '多个')}角度进行了分析。\n\n"
                            f"关键要点：\n"
                            f"1. 主题背景和重要性\n"
                            f"2. 主要论点和证据\n"
                            f"3. 结论和启示\n\n"
                            f"（{kwargs.get('length', '简洁')}摘要，"
                            f"约{kwargs.get('word_count', '200')}字）",
            "reasoning": f"【推理分析】\n\n"
                        f"步骤1：明确问题核心——{kwargs.get('scenario', kwargs.get('topic', '问题'))}\n"
                        f"步骤2：收集相关信息和前提条件\n"
                        f"步骤3：应用{kwargs.get('level', '逻辑')}推理方法\n"
                        f"步骤4：得出初步结论\n"
                        f"步骤5：检验结论的合理性和局限性\n\n"
                        f"结论：经过{kwargs.get('count', '多')}步推理，"
                        f"我们可以得出以下判断...",
            "analysis": f"【{kwargs.get('level', '综合')}分析报告】\n\n"
                       f"分析对象：{kwargs.get('topic', '')}\n"
                       f"分析维度：{kwargs.get('perspective', '综合')}视角\n\n"
                       f"1. 现状概述\n"
                       f"2. {kwargs.get('perspective', '核心')}维度分析\n"
                       f"3. 关键发现\n"
                       f"4. 结论与建议",
            "explanation": f"【{kwargs.get('level', '详细')}解释】\n\n"
                          f"概念：{kwargs.get('concept', kwargs.get('topic', ''))}\n"
                          f"面向受众：{kwargs.get('audience', '普通读者')}\n\n"
                          f"用通俗的语言来说，{kwargs.get('topic', '这个概念')}指的是...\n\n"
                          f"为了更好地理解，我们可以从以下几个方面来看：\n"
                          f"1. 基本定义\n2. 工作原理\n3. 实际应用\n4. 发展趋势",
            "creative": f"【创作内容】\n\n"
                       f"主题：{kwargs.get('theme', '')}\n"
                       f"风格：{kwargs.get('style', '')}\n"
                       f"格式：{kwargs.get('format', '')}\n\n"
                       f"（以下为创作正文）\n\n"
                       f"在{kwargs.get('theme', '')}的启发下，"
                       f"这段创作融合了{kwargs.get('style', '独特')}的风格元素...",
            "coding": f"```{kwargs.get('lang', 'python')}\n"
                     f"# {kwargs.get('algorithm', '算法')}实现\n"
                     f"# 时间复杂度：O(n log n)\n"
                     f"# 空间复杂度：O(n)\n\n"
                     f"def solution(data):\n"
                     f"    \"\"\"\n"
                     f"    实现{kwargs.get('algorithm', '算法')}的核心逻辑\n"
                     f"    \"\"\"\n"
                     f"    if not data:\n"
                     f"        return []\n"
                     f"    result = []\n"
                     f"    for item in data:\n"
                     f"        # 处理每个元素\n"
                     f"        processed = process(item)\n"
                     f"        result.append(processed)\n"
                     f"    return result\n\n\n"
                     f"def process(item):\n"
                     f"    return item\n"
                     f"```\n\n"
                     f"【代码说明】\n"
                     f"上述实现采用了标准算法模式，"
                     f"具有较好的可读性和可维护性。"
                     f"关键优化点包括：\n1. 使用高效的数据结构"
                     f"\n2. 避免不必要的内存分配\n3. 清晰的错误处理",
            "roleplay": f"（以{kwargs.get('role', '专家')}的身份回答）\n\n"
                       f"您好！作为一名{kwargs.get('role', '')}，"
                       f"我对您提到的{kwargs.get('topic', '问题')}有以下看法：\n\n"
                       f"基于我多年的{kwargs.get('field', '专业')}经验，"
                       f"我认为这个问题需要从多个角度来理解...\n\n"
                       f"希望我的回答对您有帮助。如果您还有其他问题，随时可以继续交流。",
        }
        return templates.get(category, templates["writing"])

    def _generate_multi_turn(self, instruction: str, answer: str,
                              turns: int, category: str, kwargs: dict) -> tuple[str, str]:
        history = [{"role": "user", "content": instruction},
                   {"role": "assistant", "content": answer}]
        for t in range(turns - 1):
            cont_template = self._rng.choice(_MULTI_TURN_CONTINUATIONS)
            follow_up = cont_template.format(
                aspect=kwargs.get("topic", "这个方面"),
                perspective=kwargs.get("perspective", "另一个"),
                point=f"第{self._rng.randint(1,3)}点",
                condition=self._rng.choice(["条件A成立", "条件B变化", "外部因素改变"]),
                item_a=kwargs.get("topic", "方案A"),
                item_b=kwargs.get("topic", "方案B"),
                style=self._rng.choice(["简洁", "详细", "通俗"]),
                topic=kwargs.get("topic", "这个话题"),
            )
            follow_up_answer = f"感谢您的追问。关于{follow_up[:20]}...\n\n" \
                              f"这是一个很有深度的问题。让我进一步阐述：\n\n" \
                              f"首先，从{kwargs.get('perspective', '专业')}角度来看..." \
                              f"\n其次，我们还需要考虑..." \
                              f"\n最后，总结来说..."
            history.append({"role": "user", "content": follow_up})
            history.append({"role": "assistant", "content": follow_up_answer})
        return json.dumps(history, ensure_ascii=False), f"multi_turn_{turns}"

    def generate_one(self, **kwargs) -> DataRecord:
        category = kwargs.get("category") or self._pick_category()
        difficulty = kwargs.get("difficulty") or self._pick_weighted(
            self.categories.get("difficulty_distribution", {
                "easy": 0.3, "medium": 0.4, "hard": 0.2, "expert": 0.1
            })
        )
        style = kwargs.get("style") or self._pick_weighted(
            self.categories.get("style_distribution", {
                "formal": 0.35, "friendly": 0.35, "professional": 0.2, "creative": 0.1
            })
        )
        template = self._pick_template(category)
        tmpl_kwargs = self._generate_template_kwargs(category, difficulty)
        tmpl_kwargs["style"] = style
        instruction = template.format(**tmpl_kwargs)
        answer = self._build_answer(category, instruction, difficulty, tmpl_kwargs)
        is_multi_turn = (kwargs.get("multi_turn", False) or
                         self._rng.random() < self.categories.get("multi_turn_ratio", 0.3))
        is_roleplay = category == "roleplay" or (
            kwargs.get("roleplay", False) and
            self._rng.random() < self.categories.get("role_play_ratio", 0.15)
        )
        if is_multi_turn and not is_roleplay:
            turns = self._rng.randint(2, 5)
            history_json, turn_info = self._generate_multi_turn(
                instruction, answer, turns, category, tmpl_kwargs
            )
            content = {
                "category": category,
                "difficulty": difficulty,
                "style": style,
                "conversation": history_json,
                "turns": turns,
            }
        else:
            content = {
                "category": category,
                "difficulty": difficulty,
                "style": style,
                "instruction": instruction,
                "response": answer,
                "multi_turn": False,
            }
        if is_roleplay:
            content["roleplay"] = True
        record_id = self._make_id("instr")
        return DataRecord(
            id=record_id,
            source="qingxiaotuan_synthetic",
            data_type="instruction",
            content=content,
            metadata={
                "category": category,
                "difficulty": difficulty,
                "style": style,
                "multi_turn": is_multi_turn,
                "roleplay": is_roleplay,
                "language": self._rng.choice(["zh-CN", "en"]),
            },
        )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats["categories"] = list(self._all_templates.keys())
        stats["total_templates"] = sum(len(v) for v in self._all_templates.values())
        return stats

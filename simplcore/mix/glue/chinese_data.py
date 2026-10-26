# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd.
"""
中文训练数据生成器
生成多领域简体中文语料，用于模型粘合训练

数据领域:
  1. 通用对话 - 日常交流、问答
  2. 知识百科 - 科学、历史、文化
  3. 编程技术 - Python、算法、AI
  4. 文学创作 - 故事、散文、诗歌
  5. 商业财经 - 经济、管理、投资
  6. 教育学习 - 数学、物理、语文
"""
import os
import json
import random
import math
from typing import List, Dict, Optional
from pathlib import Path


# 各领域的模板和关键词
DOMAIN_DATA = {
    "general": {
        "name": "通用对话",
        "templates": [
            "你好，请问{topic}方面有什么需要了解的吗？",
            "关于{topic}，我来为你详细解释一下。",
            "{topic}是一个很有意思的话题，让我们来聊聊。",
            "很多人问过{topic}的问题，答案其实很简单。",
            "如果你想了解{topic}，可以从以下几个方面入手。",
        ],
        "topics": [
            "天气变化", "美食烹饪", "旅行攻略", "健康养生",
            "职场发展", "人际关系", "时间管理", "学习方法",
            "心理健康", "生活习惯", "家居装修", "育儿教育",
            "宠物饲养", "运动健身", "音乐电影", "读书推荐",
        ],
    },
    "knowledge": {
        "name": "知识百科",
        "templates": [
            "{topic}是{description}。",
            "在历史上，{topic}最早出现于{era}时期。",
            "科学研究表明，{topic}的原理是{description}。",
            "关于{topic}，有一个著名的理论叫做{description}。",
            "{topic}在日常生活中的应用非常广泛，例如{description}。",
        ],
        "topics": [
            "人工智能", "量子力学", "相对论", "进化论",
            "宇宙起源", "地球科学", "生物多样性", "光合作用",
            "古代文明", "工业革命", "丝绸之路", "四大发明",
            "唐诗宋词", "儒家思想", "道家哲学", "中医理论",
        ],
        "eras": ["古代", "中世纪", "近代", "现代", "史前"],
        "descriptions": [
            "一种重要的科学理论",
            "人类文明的重要里程碑",
            "改变了世界的技术革新",
            "自然界的基本规律",
            "具有深远影响的文化现象",
        ],
    },
    "programming": {
        "name": "编程技术",
        "templates": [
            "用{lang}实现{topic}的代码如下：{code}",
            "{topic}算法的时间复杂度是{complexity}。",
            "在{lang}中，{topic}的最佳实践是{description}。",
            "要解决{topic}问题，可以使用{description}的方法。",
            "{lang}的{topic}特性使得开发效率大大提升。",
        ],
        "topics": [
            "排序算法", "动态规划", "贪心算法", "二分查找",
            "数据结构", "递归调用", "面向对象", "函数式编程",
            "并发编程", "网络编程", "数据库操作", "机器学习",
            "深度学习", "自然语言处理", "计算机视觉", "强化学习",
        ],
        "langs": ["Python", "Java", "C++", "JavaScript", "Go", "Rust"],
        "complexities": ["O(n)", "O(n log n)", "O(n²)", "O(log n)", "O(1)"],
        "descriptions": [
            "使用分治思想",
            "基于动态规划",
            "采用贪心策略",
            "利用哈希表优化",
            "通过递归实现",
        ],
    },
    "literature": {
        "name": "文学创作",
        "templates": [
            "{topic}的故事发生在{place}，主人公是{character}。",
            "在{era}的{place}，流传着一个关于{topic}的传说。",
            "{character}望着远方，心中想着{topic}。",
            "月光洒在{place}，{character}开始了关于{topic}的旅程。",
            "自古以来，{topic}就是文人墨客吟咏的对象。",
        ],
        "topics": [
            "爱情", "友情", "梦想", "冒险",
            "成长", "离别", "重逢", "希望",
            "勇气", "智慧", "善良", "坚持",
            "自由", "和平", "真理", "美好",
        ],
        "places": [
            "江南水乡", "塞北荒原", "深山古寺", "繁华都市",
            "海边小镇", "雪域高原", "竹林深处", "桃花源头",
        ],
        "characters": [
            "书生", "剑客", "诗人", "旅人",
            "少年", "老者", "公主", "隐士",
        ],
        "eras": ["古代", "唐朝", "宋朝", "明朝", "民国"],
    },
    "business": {
        "name": "商业财经",
        "templates": [
            "{topic}市场在{era}呈现出{trend}的趋势。",
            "投资{topic}需要注意{description}几个方面。",
            "企业管理中，{topic}是{description}的关键因素。",
            "{topic}行业的发展前景{trend}，主要驱动力是{description}。",
            "在{era}的经济环境下，{topic}的重要性日益凸显。",
        ],
        "topics": [
            "人工智能", "新能源", "生物医药", "半导体",
            "消费升级", "数字化转型", "跨境电商", "供应链",
            "风险管理", "团队建设", "战略规划", "市场营销",
            "财务分析", "品牌建设", "用户增长", "产品创新",
        ],
        "trends": [
            "快速增长", "稳步上升", "波动调整", "蓬勃发展",
            "持续创新", "深度融合", "加速渗透", "转型升级",
        ],
        "descriptions": [
            "技术驱动和市场需求",
            "政策支持和资本投入",
            "人才储备和研发能力",
            "用户体验和产品质量",
            "战略眼光和执行能力",
        ],
        "eras": ["当前", "未来五年", "数字化时代", "后疫情时代"],
    },
    "education": {
        "name": "教育学习",
        "templates": [
            "学习{topic}的最佳方法是{description}。",
            "{topic}这门学科的核心知识点包括{description}。",
            "要掌握{topic}，需要从{description}开始打好基础。",
            "{topic}考试中常见的题型有{description}。",
            "提高{topic}成绩的有效方法是{description}。",
        ],
        "topics": [
            "数学", "物理", "化学", "语文",
            "英语", "历史", "地理", "生物",
            "政治", "信息技术", "音乐", "美术",
            "体育", "心理健康", "劳动教育", "科学探究",
        ],
        "descriptions": [
            "理解基本概念，多做练习题",
            "掌握公式定理，学会灵活运用",
            "培养思维能力，注重方法总结",
            "打好基础，循序渐进",
            "理论联系实际，学以致用",
        ],
    },
}


class ChineseDataGenerator:
    """中文训练数据生成器"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sentence(self, domain: Optional[str] = None) -> str:
        """生成一个中文句子"""
        if domain is None:
            domain = self.rng.choice(list(DOMAIN_DATA.keys()))

        data = DOMAIN_DATA[domain]
        template = self.rng.choice(data["templates"])

        # 填充模板变量
        sentence = template
        for key in data:
            if key in ("name", "templates"):
                continue
            placeholder = "{" + key + "}"
            if placeholder in sentence:
                sentence = sentence.replace(placeholder, self.rng.choice(data[key]))

        return sentence

    def generate_paragraph(self, domain: Optional[str] = None,
                           min_sentences: int = 3,
                           max_sentences: int = 8) -> str:
        """生成一个中文段落"""
        n = self.rng.randint(min_sentences, max_sentences)
        sentences = [self.generate_sentence(domain) for _ in range(n)]
        return "".join(sentences)

    def generate_document(self, domain: Optional[str] = None,
                          min_paragraphs: int = 2,
                          max_paragraphs: int = 5) -> Dict[str, str]:
        """生成一篇中文文档"""
        n = self.rng.randint(min_paragraphs, max_paragraphs)
        paragraphs = [self.generate_paragraph(domain) for _ in range(n)]

        # 生成标题
        topic_list = DOMAIN_DATA.get(domain, {}).get("topics", ["话题"])
        title = self.rng.choice(topic_list) + "详解"

        return {
            "title": title,
            "text": "\n\n".join(paragraphs),
            "domain": domain or "mixed",
        }

    def generate_corpus(self, output_path: str,
                        num_docs: int = 1000,
                        domains: Optional[List[str]] = None,
                        format: str = "jsonl") -> str:
        """生成语料库并保存

        Args:
            output_path: 输出文件路径
            num_docs: 生成文档数量
            domains: 领域列表，None表示所有领域
            format: 输出格式 (jsonl / txt)
        """
        domains = domains or list(DOMAIN_DATA.keys())
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        docs = []
        for i in range(num_docs):
            domain = self.rng.choice(domains)
            doc = self.generate_document(domain)
            docs.append(doc)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(doc["text"] + "\n\n")

        print(f"[ChineseDataGenerator] 生成 {num_docs} 篇文档到 {output_path}")
        return output_path

    def generate_domain_corpus(self, output_dir: str,
                               docs_per_domain: int = 1000) -> Dict[str, str]:
        """为每个领域生成独立的语料库（用于训练不同专家模型）

        Returns:
            {domain_name: file_path}
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        for domain, data in DOMAIN_DATA.items():
            path = os.path.join(output_dir, f"{domain}.jsonl")
            self.generate_corpus(path, num_docs=docs_per_domain, domains=[domain])
            paths[domain] = path

        return paths


def generate_chinese_corpus(output_dir: str,
                            docs_per_domain: int = 5000,
                            seed: int = 42) -> Dict[str, str]:
    """便捷函数：生成中文语料库"""
    generator = ChineseDataGenerator(seed=seed)
    return generator.generate_domain_corpus(output_dir, docs_per_domain)


if __name__ == "__main__":
    # 测试生成
    gen = ChineseDataGenerator(seed=42)

    print("测试句子生成:")
    for domain in DOMAIN_DATA:
        s = gen.generate_sentence(domain)
        print(f"  [{DOMAIN_DATA[domain]['name']}]: {s[:50]}...")

    print("\n测试段落生成:")
    p = gen.generate_paragraph("knowledge")
    print(f"  {p[:100]}...")

    print("\n测试文档生成:")
    d = gen.generate_document("programming")
    print(f"  标题: {d['title']}")
    print(f"  领域: {d['domain']}")
    print(f"  内容: {d['text'][:100]}...")

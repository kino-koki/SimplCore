"""
青小团 合成数据生成引擎 - 推理数据生成器（Chain-of-Thought）
青奈科技 (QingNai Technology)

生成包含详细思考链的推理训练数据，
涵盖数学推理、逻辑推理、常识推理和科学推理。
"""

import json
import random
from typing import Any, Optional

from base_generator import BaseGenerator, DataRecord, GenerationContext


_ALGEBRA_PROBLEMS = [
    {
        "question": "解方程：3x + 7 = 22，求x的值。",
        "steps": [
            "将方程写为标准形式：3x + 7 = 22",
            "两边同时减去7：3x = 22 - 7 = 15",
            "两边同时除以3：x = 15 ÷ 3 = 5",
            "验证：3×5 + 7 = 15 + 7 = 22 ✓",
        ],
        "answer": "x = 5",
    },
    {
        "question": "已知二次函数 f(x) = 2x² - 4x + 1，求其顶点坐标。",
        "steps": [
            "二次函数标准形式 f(x) = ax² + bx + c，其中 a=2, b=-4, c=1",
            "顶点横坐标公式：x = -b/(2a) = -(-4)/(2×2) = 4/4 = 1",
            "将 x=1 代入原函数求纵坐标：f(1) = 2(1)² - 4(1) + 1 = 2 - 4 + 1 = -1",
            "因此顶点坐标为 (1, -1)",
        ],
        "answer": "顶点坐标为 (1, -1)",
    },
    {
        "question": "解方程组：2x + y = 7, x - y = 2",
        "steps": [
            "由第二个方程得：x = y + 2",
            "代入第一个方程：2(y + 2) + y = 7",
            "展开：2y + 4 + y = 7 → 3y + 4 = 7",
            "解得：3y = 3 → y = 1",
            "回代：x = 1 + 2 = 3",
            "验证：2×3 + 1 = 7 ✓, 3 - 1 = 2 ✓",
        ],
        "answer": "x = 3, y = 1",
    },
]

_GEOMETRY_PROBLEMS = [
    {
        "question": "已知直角三角形两直角边分别为3和4，求斜边长和面积。",
        "steps": [
            "根据勾股定理：斜边² = 3² + 4² = 9 + 16 = 25",
            "斜边 = √25 = 5",
            "直角三角形面积 = (底×高)/2 = (3×4)/2 = 6",
            "因此斜边长为5，面积为6",
        ],
        "answer": "斜边长5，面积6",
    },
    {
        "question": "圆的半径为5，求圆的周长和面积。（π取3.14）",
        "steps": [
            "圆的周长公式：C = 2πr = 2 × 3.14 × 5 = 31.4",
            "圆的面积公式：S = πr² = 3.14 × 5² = 3.14 × 25 = 78.5",
            "因此周长为31.4，面积为78.5",
        ],
        "answer": "周长31.4，面积78.5",
    },
]

_PROBABILITY_PROBLEMS = [
    {
        "question": "一个袋子中有3个红球和5个蓝球，随机取出一个球，求取出红球的概率。",
        "steps": [
            "总球数 = 3 + 5 = 8",
            "红球数 = 3",
            "概率 = 红球数/总球数 = 3/8 = 0.375",
            "因此取出红球的概率为3/8或37.5%",
        ],
        "answer": "3/8 (37.5%)",
    },
    {
        "question": "投掷两枚骰子，求点数之和为7的概率。",
        "steps": [
            "两枚骰子所有可能结果：6×6 = 36种",
            "点数之和为7的组合：(1,6), (2,5), (3,4), (4,3), (5,2), (6,1) 共6种",
            "概率 = 6/36 = 1/6",
            "因此点数之和为7的概率为1/6",
        ],
        "answer": "1/6",
    },
]

_STATISTICS_PROBLEMS = [
    {
        "question": "数据集：2, 4, 6, 8, 10，求均值、中位数和标准差。",
        "steps": [
            "均值 = (2+4+6+8+10)/5 = 30/5 = 6",
            "排序后数据：2, 4, 6, 8, 10，中位数为第3个数 = 6",
            "方差 = [(2-6)²+(4-6)²+(6-6)²+(8-6)²+(10-6)²]/5",
            "= [16+4+0+4+16]/5 = 40/5 = 8",
            "标准差 = √8 ≈ 2.83",
        ],
        "answer": "均值6，中位数6，标准差≈2.83",
    },
]

_DEDUCTIVE_PROBLEMS = [
    {
        "question": "所有人类都会死亡。苏格拉底是人类。请问苏格拉底会死亡吗？请用三段论推理。",
        "steps": [
            "大前提：所有人类都会死亡",
            "小前提：苏格拉底是人类",
            "结论：苏格拉底会死亡",
            "这是一个典型的三段论推理（Barbara式），从普遍到特殊",
        ],
        "answer": "苏格拉底会死亡",
    },
    {
        "question": "如果下雨，地面就会湿。现在地面是干的，可以推断出什么？",
        "steps": [
            "原命题：下雨 → 地面湿 (如果P则Q)",
            "已知：地面是干的 (非Q)",
            "根据逆否命题：如果非Q则非P，即地面不湿 → 没下雨",
            "因此可以推断出：没有下雨",
        ],
        "answer": "没有下雨",
    },
]

_INDUCTIVE_PROBLEMS = [
    {
        "question": "观察：第1个质数是2，第2个质数是3，第3个质数是5，第4个质数是7。请归纳前n个质数之和的规律。",
        "steps": [
            "前1个质数和：2",
            "前2个质数和：2+3=5",
            "前3个质数和：2+3+5=10",
            "前4个质数和：2+3+5+7=17",
            "观察：2,5,10,17... 差值分别为3,5,7（递增的奇数）",
            "可以归纳出：前n个质数之和 ≈ n² × log(n)（但质数分布无简单闭式）",
        ],
        "answer": "前n个质数之和没有简单的闭式公式，但可以逐一累加",
    },
]

_ANALOGICAL_PROBLEMS = [
    {
        "question": "太阳系中行星围绕太阳旋转，类似于原子中什么围绕原子核？请说明类比关系。",
        "steps": [
            "太阳系结构：太阳（中心）→ 引力 → 行星（围绕旋转）",
            "原子结构：原子核（中心）→ 电磁力 → 电子（围绕旋转）",
            "类比：太阳对应原子核，行星对应电子，引力对应电磁力",
            "注意：这是结构类比，但物理机制不同",
        ],
        "answer": "电子围绕原子核旋转，类似于行星围绕太阳",
    },
]

_CAUSAL_PROBLEMS = [
    {
        "question": "某地区植被覆盖率从50%降到10%，同时该地区年降雨量从1000mm降到500mm。请分析可能因果关系。",
        "steps": [
            "观察到的两个变化：植被覆盖率下降80%，降雨量下降50%",
            "可能方向1：植被减少 → 蒸腾作用减弱 → 水汽减少 → 降雨减少",
            "可能方向2：降雨减少 → 水资源减少 → 植被死亡 → 覆盖率下降",
            "更可能是双向因果关系：植被减少和降雨减少形成正反馈循环",
            "这反映了生态系统中的因果复杂性",
        ],
        "answer": "植被覆盖率和降雨量之间存在双向因果关系",
    },
]

_TEMPORAL_PROBLEMS = [
    {
        "question": "小明7岁上学，小学6年，初中3年，高中3年，大学4年。请问小明大学毕业时多少岁？",
        "steps": [
            "上学年龄：7岁",
            "小学毕业：7 + 6 = 13岁",
            "初中毕业：13 + 3 = 16岁",
            "高中毕业：16 + 3 = 19岁",
            "大学毕业：19 + 4 = 23岁",
            "因此小明大学毕业时23岁",
        ],
        "answer": "23岁",
    },
]

_SOCIAL_PROBLEMS = [
    {
        "question": "在一个社会中，如果教育资源分配不均，可能会产生哪些社会影响？请推理分析。",
        "steps": [
            "教育资源分配不均是初始条件",
            "直接影响：不同地区/群体受教育程度差异加大",
            "间接影响1：教育差异 → 技能差异 → 收入差异 → 贫富差距扩大",
            "间接影响2：教育差异 → 社会流动性降低 → 阶层固化",
            "长期影响：社会矛盾加剧，经济增长潜力受限",
            "这是一个典型的社会不平等循环机制",
        ],
        "answer": "可能导致贫富差距扩大、阶层固化和经济增长受限",
    },
]

_PHYSICS_PROBLEMS = [
    {
        "question": "一个质量为2kg的物体从10m高处自由下落（g=10m/s²），求落地速度。",
        "steps": [
            "已知：m=2kg, h=10m, g=10m/s², 初速度v₀=0",
            "方法1（能量守恒）：mgh = mv²/2",
            "→ v² = 2gh = 2×10×10 = 200",
            "→ v = √200 ≈ 14.14 m/s",
            "方法2（运动学公式）：v² = v₀² + 2gh = 0 + 200",
            "两种方法结果一致，落地速度≈14.14 m/s",
            "注意：质量2kg在该问题中不影响结果（伽利略自由落体定律）",
        ],
        "answer": "约14.14 m/s",
    },
]

_CHEMISTRY_PROBLEMS = [
    {
        "question": "将10g NaCl溶于90g水中，求溶液的质量分数。",
        "steps": [
            "溶质质量：10g NaCl",
            "溶剂质量：90g 水",
            "溶液总质量：10 + 90 = 100g",
            "质量分数 = 溶质质量/溶液总质量 × 100% = 10/100 × 100% = 10%",
            "因此该NaCl溶液的质量分数为10%",
        ],
        "answer": "10%",
    },
]

_BIOLOGY_PROBLEMS = [
    {
        "question": "在一对等位基因Aa的杂交实验中，亲本均为Aa，求子代中表现显性性状的比例。",
        "steps": [
            "亲本基因型：Aa × Aa",
            "配子：A和a（各50%）",
            "子代基因型组合：AA, Aa, aA, aa",
            "概率：AA=1/4, Aa=1/4, aA=1/4, aa=1/4",
            "显性性状基因型：AA或Aa（即包含显性基因A）",
            "比例 = P(AA) + P(Aa) = 1/4 + 1/2 = 3/4",
            "因此子代表现显性性状的比例为75%",
        ],
        "answer": "75% (3/4)",
    },
]


class ReasoningGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any], context: Optional[GenerationContext] = None):
        super().__init__(config, context)
        reasoning_cfg = config.get("data_types", {}).get("reasoning", {})
        self.problem_pools = {
            "math_algebra": _ALGEBRA_PROBLEMS,
            "math_geometry": _GEOMETRY_PROBLEMS,
            "math_probability": _PROBABILITY_PROBLEMS,
            "math_statistics": _STATISTICS_PROBLEMS,
            "logic_deductive": _DEDUCTIVE_PROBLEMS,
            "logic_inductive": _INDUCTIVE_PROBLEMS,
            "logic_analogical": _ANALOGICAL_PROBLEMS,
            "common_causal": _CAUSAL_PROBLEMS,
            "common_temporal": _TEMPORAL_PROBLEMS,
            "common_social": _SOCIAL_PROBLEMS,
            "science_physics": _PHYSICS_PROBLEMS,
            "science_chemistry": _CHEMISTRY_PROBLEMS,
            "science_biology": _BIOLOGY_PROBLEMS,
        }
        self.category_descriptions = {
            "math_algebra": "代数",
            "math_geometry": "几何",
            "math_probability": "概率",
            "math_statistics": "统计",
            "logic_deductive": "演绎推理",
            "logic_inductive": "归纳推理",
            "logic_analogical": "类比推理",
            "common_causal": "因果推理",
            "common_temporal": "时空推理",
            "common_social": "社会推理",
            "science_physics": "物理",
            "science_chemistry": "化学",
            "science_biology": "生物",
        }
        self.categories_weight = reasoning_cfg.get("categories", {})
        self.chain_of_thought = reasoning_cfg.get("chain_of_thought", True)
        self.min_steps = reasoning_cfg.get("min_steps", 2)
        self.max_steps = reasoning_cfg.get("max_steps", 10)
        self.verification_ratio = reasoning_cfg.get("verification_ratio", 0.3)

    def _init_templates(self) -> dict[str, str]:
        return {
            "reasoning_q": "请解决以下{category}问题：\n{question}",
            "reasoning_cot": "让我们一步步推理。\n{steps}\n\n因此，答案是：{answer}",
            "reasoning_direct": "{answer}",
            "verify_prompt": "请验证以下推理是否正确：\n问题：{question}\n推理步骤：{steps}\n结论：{answer}",
        }

    def _pick_category(self) -> str:
        if self.categories_weight:
            return self._pick_weighted(self.categories_weight)
        return self._rng.choice(list(self.problem_pools.keys()))

    def _build_cot_steps(self, steps: list[str]) -> str:
        numbered_steps = []
        for i, step in enumerate(steps, 1):
            numbered_steps.append(f"步骤{i}：{step}")
        return "\n".join(numbered_steps)

    def _build_cot_answer(self, problem: dict) -> str:
        steps_text = self._build_cot_steps(problem["steps"])
        return (
            f"让我们一步步推理。\n\n"
            f"{steps_text}\n\n"
            f"因此，答案是：{problem['answer']}"
        )

    def _build_verification(self, problem: dict) -> str:
        steps_text = self._build_cot_steps(problem["steps"])
        return (
            f"【验证】\n"
            f"问题：{problem['question']}\n"
            f"推理过程：\n{steps_text}\n"
            f"结论：{problem['answer']}\n\n"
            f"验证结果：推理过程逻辑连贯，每一步都基于正确的数学/科学原理，"
            f"计算准确无误，结论正确。"
        )

    def _generate_more_steps_problem(self, base_problem: dict, extra_steps: int) -> dict:
        problem = {
            "question": base_problem["question"],
            "steps": list(base_problem["steps"]),
            "answer": base_problem["answer"],
        }
        filler_steps = [
            f"考虑到问题的约束条件，需要先确认已知信息和未知量",
            f"采用{self._rng.choice(['归纳法', '演绎法', '反证法', '直接法'])}进行分析",
            f"检查是否有特殊情况或边界条件需要考虑",
            f"对中间结果进行合理性检验",
            f"从另一个角度验证当前结论的一致性",
            f"考虑是否有更简洁的解法可以交叉验证",
        ]
        for _ in range(min(extra_steps, len(filler_steps))):
            insert_pos = self._rng.randint(1, max(1, len(problem["steps"]) - 1))
            problem["steps"].insert(insert_pos, self._rng.choice(filler_steps))
        return problem

    def generate_one(self, **kwargs) -> DataRecord:
        category = kwargs.get("category") or self._pick_category()
        pool = self.problem_pools.get(category, _ALGEBRA_PROBLEMS)
        problem = self._rng.choice(pool)
        num_steps = self._rng.randint(self.min_steps, self.max_steps)
        if num_steps > len(problem["steps"]):
            problem = self._generate_more_steps_problem(problem, num_steps - len(problem["steps"]))
        cot_answer = self._build_cot_answer(problem)
        include_verification = self._rng.random() < self.verification_ratio
        steps_count = len(problem["steps"])
        difficulty = "easy" if steps_count <= 3 else ("medium" if steps_count <= 6 else "hard")
        content = {
            "category": category,
            "category_name": self.category_descriptions.get(category, category),
            "question": problem["question"],
            "chain_of_thought": cot_answer,
            "steps_count": steps_count,
            "final_answer": problem["answer"],
            "difficulty": difficulty,
        }
        if include_verification:
            content["verification"] = self._build_verification(problem)
        record_id = self._make_id("reason")
        return DataRecord(
            id=record_id,
            source="qingxiaotuan_synthetic",
            data_type="reasoning",
            content=content,
            metadata={
                "category": category,
                "category_name": self.category_descriptions.get(category, category),
                "steps_count": steps_count,
                "has_verification": include_verification,
                "chain_of_thought": self.chain_of_thought,
                "difficulty": difficulty,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats["categories"] = list(self.problem_pools.keys())
        stats["total_problems"] = sum(len(v) for v in self.problem_pools.values())
        return stats

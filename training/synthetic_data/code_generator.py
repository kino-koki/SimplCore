"""
青小团 合成数据生成引擎 - 代码训练数据生成器
青奈科技 (QingNai Technology)

生成多语言代码训练数据，涵盖算法实现、数据结构、
API使用、系统设计、Bug修复、代码优化和代码审查。
"""

import ast
import textwrap
from typing import Any, Optional

from base_generator import BaseGenerator, DataRecord, GenerationContext

_LANGUAGE_CONFIGS = {
    "python": {
        "comment": "#",
        "block_start": "",
        "block_end": "",
        "fn_keyword": "def",
        "class_keyword": "class",
    },
    "javascript": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "function",
        "class_keyword": "class",
    },
    "typescript": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "function",
        "class_keyword": "class",
    },
    "java": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "public static",
        "class_keyword": "public class",
    },
    "cpp": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "auto",
        "class_keyword": "class",
    },
    "go": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "func",
        "class_keyword": "type",
    },
    "rust": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "fn",
        "class_keyword": "struct",
    },
    "csharp": {
        "comment": "//",
        "block_start": "{",
        "block_end": "}",
        "fn_keyword": "public",
        "class_keyword": "public class",
    },
}

_CODE_SNIPPETS = {
    "python": {
        "quick_sort": textwrap.dedent("""\
            def quick_sort(arr):
                if len(arr) <= 1:
                    return arr
                pivot = arr[len(arr) // 2]
                left = [x for x in arr if x < pivot]
                middle = [x for x in arr if x == pivot]
                right = [x for x in arr if x > pivot]
                return quick_sort(left) + middle + quick_sort(right)"""),
        "binary_search": textwrap.dedent("""\
            def binary_search(arr, target):
                left, right = 0, len(arr) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1"""),
        "fibonacci": textwrap.dedent("""\
            def fibonacci(n):
                if n <= 1:
                    return n
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                return b"""),
    },
    "javascript": {
        "quick_sort": textwrap.dedent("""\
            function quickSort(arr) {
                if (arr.length <= 1) return arr;
                const pivot = arr[Math.floor(arr.length / 2)];
                const left = arr.filter(x => x < pivot);
                const middle = arr.filter(x => x === pivot);
                const right = arr.filter(x => x > pivot);
                return [...quickSort(left), ...middle, ...quickSort(right)];
            }"""),
    },
    "go": {
        "quick_sort": textwrap.dedent("""\
            func quickSort(arr []int) []int {
                if len(arr) <= 1 {
                    return arr
                }
                pivot := arr[len(arr)/2]
                var left, middle, right []int
                for _, x := range arr {
                    switch {
                    case x < pivot:
                        left = append(left, x)
                    case x == pivot:
                        middle = append(middle, x)
                    case x > pivot:
                        right = append(right, x)
                    }
                }
                result := append(quickSort(left), middle...)
                result = append(result, quickSort(right)...)
                return result
            }"""),
    },
    "rust": {
        "quick_sort": textwrap.dedent("""\
            fn quick_sort<T: Ord>(arr: &[T]) -> Vec<&T> {
                if arr.len() <= 1 {
                    return arr.iter().collect();
                }
                let pivot = &arr[arr.len() / 2];
                let left: Vec<&T> = arr.iter().filter(|x| *x < pivot).collect();
                let middle: Vec<&T> = arr.iter().filter(|x| *x == pivot).collect();
                let right: Vec<&T> = arr.iter().filter(|x| *x > pivot).collect();
                let mut result = quick_sort(&left);
                result.extend(middle);
                result.extend(quick_sort(&right));
                result
            }"""),
    },
}

_BUGGY_CODE = {
    "python": {
        "off_by_one": textwrap.dedent("""\
            def find_last_element(arr):
                return arr[len(arr)]  # Bug: index out of range"""),
        "infinite_loop": textwrap.dedent("""\
            def count_down(n):
                result = []
                while n > 0:
                    result.append(n)  # Bug: missing n -= 1
                return result"""),
        "mutability": textwrap.dedent("""\
            def add_item(item, lst=[]):  # Bug: mutable default argument
                lst.append(item)
                return lst"""),
    },
}


class CodeGenerator(BaseGenerator):
    def __init__(self, config: dict[str, Any], context: Optional[GenerationContext] = None):
        super().__init__(config, context)
        code_cfg = config.get("data_types", {}).get("code", {})
        self.language_weights = code_cfg.get("languages", {})
        self.code_type_weights = code_cfg.get("code_types", {})
        self.include_explanation = code_cfg.get("include_explanation", True)
        self.ast_validation = code_cfg.get("ast_validation", True)
        self.test_cases_ratio = code_cfg.get("test_cases_ratio", 0.3)

    def _init_templates(self) -> dict[str, str]:
        return {
            "impl": "请用{language}实现以下{algo_type}：{description}",
            "bug_fix": "以下{language}代码存在Bug，请找出并修复：\n```{language}\n{code}\n```",
            "optimize": "请优化以下{language}代码的性能：\n```{language}\n{code}\n```",
            "review": "请审查以下{language}代码，指出潜在问题：\n```{language}\n{code}\n```",
            "explain": "请解释以下{language}代码的功能和设计思路：\n```{language}\n{code}\n```",
            "doc": "请为以下{language}代码添加完整的文档注释：\n```{language}\n{code}\n```",
        }

    def _pick_language(self) -> str:
        return self._pick_weighted(self.language_weights)

    def _pick_code_type(self) -> str:
        return self._pick_weighted(self.code_type_weights)

    def _get_lang_config(self, language: str) -> dict:
        return _LANGUAGE_CONFIGS.get(language, _LANGUAGE_CONFIGS["python"])

    def _get_comment_symbol(self, language: str) -> str:
        return self._get_lang_config(language)["comment"]

    def _generate_algorithm_code(self, language: str) -> tuple[str, str, str]:
        lang_snippets = _CODE_SNIPPETS.get(language, _CODE_SNIPPETS["python"])
        algo_name = self._rng.choice(list(lang_snippets.keys()))
        code = lang_snippets[algo_name]
        description = {
            "quick_sort": "快速排序算法",
            "binary_search": "二分查找算法",
            "fibonacci": "斐波那契数列",
        }.get(algo_name, f"{algo_name}算法")
        return code, algo_name, description

    def _generate_bug_fix_data(self, language: str) -> tuple[str, str, str]:
        bug_pool = _BUGGY_CODE.get(language, _BUGGY_CODE["python"])
        bug_name = self._rng.choice(list(bug_pool.keys()))
        buggy_code = bug_pool[bug_name]
        bug_descriptions = {
            "off_by_one": "数组越界访问",
            "infinite_loop": "无限循环",
            "mutability": "可变默认参数",
        }
        description = bug_descriptions.get(bug_name, f"{bug_name}类型的Bug")
        fixed_code = buggy_code.replace(
            "arr[len(arr)]", "arr[len(arr) - 1]"
        ).replace(
            "result.append(n)  # Bug: missing n -= 1",
            "result.append(n)\n        n -= 1"
        ).replace(
            "lst=[]", "lst=None"
        )
        return buggy_code, fixed_code, description

    def _generate_optimization_data(self, language: str) -> tuple[str, str, str]:
        comment = self._get_comment_symbol(language)
        suboptimal = (
            f"{comment} 低效实现：使用嵌套循环\n"
            f"def find_duplicates(arr):\n"
            f"    duplicates = []\n"
            f"    for i in range(len(arr)):\n"
            f"        for j in range(i + 1, len(arr)):\n"
            f"            if arr[i] == arr[j] and arr[i] not in duplicates:\n"
            f"                duplicates.append(arr[i])\n"
            f"    return duplicates\n"
        )
        optimized = (
            f"{comment} 优化实现：使用哈希集合\n"
            f"def find_duplicates(arr):\n"
            f"    seen = set()\n"
            f"    duplicates = set()\n"
            f"    for item in arr:\n"
            f"        if item in seen:\n"
            f"            duplicates.add(item)\n"
            f"        else:\n"
            f"            seen.add(item)\n"
            f"    return list(duplicates)\n"
        )
        return suboptimal, optimized, "数组查重优化（O(n²) → O(n)）"

    def _generate_code_review_data(self, language: str) -> tuple[str, str, str]:
        comment = self._get_comment_symbol(language)
        code_to_review = (
            f"{comment} 需要审查的代码\n"
            f"def process_data(data, config):\n"
            f"    result = []\n"
            f"    for item in data:\n"
            f"        if config['enabled']:\n"
            f"            temp = item * 2\n"
            f"            result.append(temp)\n"
            f"    return result\n"
        )
        review_comments = (
            f"审查意见：\n"
            f"1. 缺少类型注解，可读性不足\n"
            f"2. 直接使用字典访问config['enabled']，缺少默认值处理\n"
            f"3. 函数缺少文档字符串，不清楚输入输出约定\n"
            f"4. 建议添加异常处理机制\n"
            f"5. 变量名temp不够语义化，建议改为doubled_item"
        )
        return code_to_review, review_comments, "代码质量审查"

    def _generate_documentation_data(self, language: str) -> tuple[str, str, str]:
        code = (
            f"def calculate_statistics(numbers):\n"
            f"    total = sum(numbers)\n"
            f"    count = len(numbers)\n"
            f"    mean = total / count if count > 0 else 0\n"
            f"    variance = sum((x - mean) ** 2 for x in numbers) / count if count > 0 else 0\n"
            f"    return {{\n"
            f"        'mean': mean,\n"
            f"        'variance': variance,\n"
            f"        'std_dev': variance ** 0.5,\n"
            f"        'min': min(numbers) if numbers else None,\n"
            f"        'max': max(numbers) if numbers else None,\n"
            f"    }}\n"
        )
        documented = (
            f"def calculate_statistics(numbers):\n"
            f"    \"\"\"\n"
            f"    计算数值列表的统计指标。\n"
            f"\n"
            f"    该函数接收一个数值列表，计算并返回其均值、方差、\n"
            f"    标准差、最小值和最大值。\n"
            f"\n"
            f"    参数：\n"
            f"        numbers (list[float])：输入数值列表\n"
            f"\n"
            f"    返回：\n"
            f"        dict：包含以下键的字典\n"
            f"            - 'mean'：算术平均值\n"
            f"            - 'variance'：方差\n"
            f"            - 'std_dev'：标准差\n"
            f"            - 'min'：最小值\n"
            f"            - 'max'：最大值\n"
            f"\n"
            f"    示例：\n"
            f"        >>> calculate_statistics([1, 2, 3, 4, 5])\n"
            f"        {{'mean': 3.0, 'variance': 2.0, 'std_dev': 1.414, 'min': 1, 'max': 5}}\n"
            f"    \"\"\"\n"
            f"    total = sum(numbers)\n"
            f"    count = len(numbers)\n"
            f"    mean = total / count if count > 0 else 0\n"
            f"    variance = sum((x - mean) ** 2 for x in numbers) / count if count > 0 else 0\n"
            f"    return {{\n"
            f"        'mean': mean,\n"
            f"        'variance': variance,\n"
            f"        'std_dev': variance ** 0.5,\n"
            f"        'min': min(numbers) if numbers else None,\n"
            f"        'max': max(numbers) if numbers else None,\n"
            f"    }}\n"
        )
        return code, documented, "为统计函数添加完整文档"

    def _generate_api_usage_code(self, language: str) -> tuple[str, str, str]:
        api_examples = {
            "python": (
                f"import requests\n\ndef fetch_user_data(user_id):\n"
                f"    response = requests.get(f'https://api.example.com/users/{{user_id}}')\n"
                f"    response.raise_for_status()\n"
                f"    return response.json()\n",
                "REST API调用示例：获取用户数据",
            ),
            "javascript": (
                f"async function fetchUserData(userId) {{\n"
                f"    const response = await fetch(`https://api.example.com/users/${{userId}}`);\n"
                f"    if (!response.ok) throw new Error('HTTP error ' + response.status);\n"
                f"    return response.json();\n"
                f"}}\n",
                "Fetch API调用示例：获取用户数据",
            ),
        }
        return api_examples.get(language, api_examples["python"])

    def _validate_python_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _build_explanation(self, code: str, code_type: str, language: str) -> str:
        explanations = {
            "algorithm": (
                f"该实现采用了经典的算法设计模式。\n"
                f"代码逻辑清晰，使用{language}的标准语法完成了算法核心逻辑。\n"
                f"关键设计要点包括：边界条件处理、递归/迭代控制、结果组合。"
            ),
            "bug_fix": (
                f"上述Bug修复展示了{language}中常见的编程陷阱。\n"
                f"修复策略包括：边界检查、循环控制修正、不可变默认值处理。"
            ),
            "optimization": (
                f"优化前后的对比展示了算法复杂度从O(n²)到O(n)的改进。\n"
                f"核心优化策略是使用哈希表替代嵌套循环，以空间换时间。"
            ),
            "code_review": (
                f"代码审查是保证代码质量的重要环节。\n"
                f"审查重点包括：可读性、健壮性、性能、可维护性和安全性。"
            ),
            "documentation": (
                f"完善的文档注释可以显著提高代码的可维护性和团队协作效率。\n"
                f"文档应包含：功能描述、参数说明、返回值、使用示例。"
            ),
            "api_usage": (
                f"该示例展示了如何使用{language}的标准库进行API调用。\n"
                f"关键实践包括：错误处理、超时设置、响应解析。"
            ),
        }
        return explanations.get(code_type, "以上代码提供了高质量的{language}实现示例。")

    def _generate_test_cases(self, code_type: str, language: str) -> str:
        comment = self._get_comment_symbol(language)
        if language == "python":
            return (
                f"{comment} 测试用例\n"
                f"def test_solution():\n"
                f"    # 正常情况测试\n"
                f"    assert solution([3, 1, 4, 1, 5]) is not None\n"
                f"    # 边界情况测试\n"
                f"    assert solution([]) is not None\n"
                f"    # 单元素测试\n"
                f"    assert solution([1]) is not None\n"
                f"    print('所有测试通过')\n"
            )
        return f"{comment} 请根据具体实现添加测试用例"

    def generate_one(self, **kwargs) -> DataRecord:
        language = kwargs.get("language") or self._pick_language()
        code_type = kwargs.get("code_type") or self._pick_code_type()
        comment = self._get_comment_symbol(language)
        if code_type == "algorithm":
            code, algo_name, description = self._generate_algorithm_code(language)
            explanation = self._build_explanation(code, code_type, language)
            content = {
                "language": language,
                "code_type": code_type,
                "description": description,
                "code": code,
                "explanation": explanation,
                "time_complexity": "O(n log n)",
                "space_complexity": "O(n)",
            }
            include_tests = self._rng.random() < self.test_cases_ratio
            if include_tests:
                content["test_cases"] = self._generate_test_cases(code_type, language)
        elif code_type == "bug_fix":
            buggy_code, fixed_code, description = self._generate_bug_fix_data(language)
            explanation = self._build_explanation(fixed_code, code_type, language)
            content = {
                "language": language,
                "code_type": code_type,
                "description": description,
                "buggy_code": buggy_code,
                "fixed_code": fixed_code,
                "explanation": explanation,
                "bug_category": description,
            }
            if self.ast_validation and language == "python":
                content["ast_valid"] = self._validate_python_syntax(fixed_code)
        elif code_type == "optimization":
            suboptimal, optimized, description = self._generate_optimization_data(language)
            explanation = self._build_explanation(optimized, code_type, language)
            content = {
                "language": language,
                "code_type": code_type,
                "description": description,
                "original_code": suboptimal,
                "optimized_code": optimized,
                "explanation": explanation,
                "improvement": "O(n²) → O(n)",
            }
        elif code_type == "code_review":
            code_to_review, review_comments, description = self._generate_code_review_data(language)
            content = {
                "language": language,
                "code_type": code_type,
                "description": description,
                "code": code_to_review,
                "review": review_comments,
                "issue_count": 5,
            }
        elif code_type == "documentation":
            code, documented, description = self._generate_documentation_data(language)
            content = {
                "language": language,
                "code_type": code_type,
                "description": description,
                "original_code": code,
                "documented_code": documented,
                "doc_style": "Google-style docstring",
            }
        elif code_type == "api_usage":
            api_code, description = self._generate_api_usage_code(language)
            explanation = self._build_explanation(api_code, code_type, language)
            content = {
                "language": language,
                "code_type": code_type,
                "description": description,
                "code": api_code,
                "explanation": explanation,
            }
        elif code_type == "data_structure":
            ds_code = (
                f"{comment} {language} 数据结构实现：链表\n"
                f"class ListNode:\n"
                f"    def __init__(self, val=0, next=None):\n"
                f"        self.val = val\n"
                f"        self.next = next\n\n"
                f"class LinkedList:\n"
                f"    def __init__(self):\n"
                f"        self.head = None\n\n"
                f"    def append(self, val):\n"
                f"        if not self.head:\n"
                f"            self.head = ListNode(val)\n"
                f"            return\n"
                f"        curr = self.head\n"
                f"        while curr.next:\n"
                f"            curr = curr.next\n"
                f"        curr.next = ListNode(val)\n"
            )
            content = {
                "language": language,
                "code_type": code_type,
                "description": "链表数据结构实现",
                "code": ds_code,
                "operations": ["append", "insert", "delete", "search"],
            }
        elif code_type == "system_design":
            content = {
                "language": language,
                "code_type": code_type,
                "description": "简易缓存系统设计",
                "architecture": "采用装饰器模式实现LRU缓存",
                "components": ["缓存存储", "过期策略", "并发控制"],
                "code": (
                    f"{comment} 系统设计：LRU缓存\n"
                    f"from collections import OrderedDict\n\n"
                    f"class LRUCache:\n"
                    f"    def __init__(self, capacity):\n"
                    f"        self.cache = OrderedDict()\n"
                    f"        self.capacity = capacity\n\n"
                    f"    def get(self, key):\n"
                    f"        if key not in self.cache:\n"
                    f"            return -1\n"
                    f"        self.cache.move_to_end(key)\n"
                    f"        return self.cache[key]\n\n"
                    f"    def put(self, key, value):\n"
                    f"        if key in self.cache:\n"
                    f"            self.cache.move_to_end(key)\n"
                    f"        self.cache[key] = value\n"
                    f"        if len(self.cache) > self.capacity:\n"
                    f"            self.cache.popitem(last=False)\n"
                ),
            }
        else:
            content = {
                "language": language,
                "code_type": code_type,
                "description": f"{code_type}类型的代码示例",
                "code": f"{comment} {language} {code_type} 示例代码",
            }
        content["explanation"] = content.get("explanation", "")
        record_id = self._make_id("code")
        return DataRecord(
            id=record_id,
            source="qingxiaotuan_synthetic",
            data_type="code",
            content=content,
            metadata={
                "language": language,
                "code_type": code_type,
                "has_explanation": self.include_explanation,
                "ast_validated": self.ast_validation,
            },
        )

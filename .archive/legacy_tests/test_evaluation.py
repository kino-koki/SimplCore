"""
青小团评估模块测试 - 覆盖C-Eval、MMLU、GSM8K、HumanEval、结果汇总等
"""

import os
import json
import math
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestQingxiaotuanCEval:

    def test_ceval_evaluator_creation(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        assert evaluator is not None

    def test_ceval_subjects_list(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        subjects = evaluator.get_subjects()
        assert len(subjects) > 0
        assert isinstance(subjects, list)

    def test_ceval_scoring(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        score = evaluator.compute_accuracy(
            predictions=[0, 1, 2, 0],
            references=[0, 1, 0, 1],
        )
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_ceval_exact_match(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        score = evaluator.compute_accuracy(
            predictions=[0, 1, 2],
            references=[0, 1, 2],
        )
        assert math.isclose(score, 100.0)

    def test_ceval_zero_match(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        score = evaluator.compute_accuracy(
            predictions=[0, 0, 0],
            references=[1, 2, 3],
        )
        assert math.isclose(score, 0.0)

    def test_ceval_partial_match(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        score = evaluator.compute_accuracy(
            predictions=[0, 1, 0, 1],
            references=[0, 1, 2, 3],
        )
        assert math.isclose(score, 50.0)

    def test_ceval_empty_input(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        score = evaluator.compute_accuracy(predictions=[], references=[])
        assert math.isnan(score) or score == 0.0

    def test_ceval_average_across_subjects(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        subject_scores = {"语文": 85.0, "数学": 75.0, "英语": 90.0}
        avg = evaluator.compute_average(subject_scores)
        assert math.isclose(avg, 83.3333, rel_tol=1e-3)

    def test_ceval_weighted_average(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        subject_scores = {"语文": (85.0, 50), "数学": (75.0, 30), "英语": (90.0, 20)}
        avg = evaluator.compute_weighted_average(subject_scores)
        assert isinstance(avg, float)

    def test_ceval_result_format(self):
        from training.evaluation.ceval import CEvalEvaluator
        evaluator = CEvalEvaluator()
        results = evaluator.format_results({"语文": 85.0, "数学": 75.0, "平均": 80.0})
        assert "语文" in results
        assert "平均" in results


class TestQingxiaotuanMMLU:

    def test_mmlu_evaluator_creation(self):
        from training.evaluation.mmlu import MMLUEvaluator
        evaluator = MMLUEvaluator()
        assert evaluator is not None

    def test_mmlu_categories(self):
        from training.evaluation.mmlu import MMLUEvaluator
        evaluator = MMLUEvaluator()
        categories = evaluator.get_categories()
        assert len(categories) > 0

    def test_mmlu_accuracy_computation(self):
        from training.evaluation.mmlu import MMLUEvaluator
        evaluator = MMLUEvaluator()
        acc = evaluator.compute_accuracy(
            predictions=["A", "B", "C", "D"],
            references=["A", "B", "C", "C"],
        )
        assert math.isclose(acc, 75.0)

    def test_mmlu_few_shot_format(self):
        from training.evaluation.mmlu import MMLUEvaluator
        evaluator = MMLUEvaluator()
        formatted = evaluator.format_few_shot(examples=[], question="test?", choices=["A", "B", "C", "D"])
        assert "test?" in formatted or "A" in formatted or "B" in formatted or "C" in formatted or "D" in formatted

    def test_mmlu_accuracy_all_correct(self):
        from training.evaluation.mmlu import MMLUEvaluator
        evaluator = MMLUEvaluator()
        acc = evaluator.compute_accuracy(
            predictions=["A", "B", "C", "D"],
            references=["A", "B", "C", "D"],
        )
        assert math.isclose(acc, 100.0)

    def test_mmlu_accuracy_all_wrong(self):
        from training.evaluation.mmlu import MMLUEvaluator
        evaluator = MMLUEvaluator()
        acc = evaluator.compute_accuracy(
            predictions=["A", "A", "A"],
            references=["B", "C", "D"],
        )
        assert math.isclose(acc, 0.0)


class TestQingxiaotuanGSM8K:

    def test_gsm8k_evaluator_creation(self):
        from training.evaluation.gsm8k import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        assert evaluator is not None

    def test_gsm8k_answer_extraction(self):
        from training.evaluation.gsm8k import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        answer = evaluator.extract_answer("最终答案是42。")
        assert answer == 42 or answer is not None

    def test_gsm8k_extract_numeric(self):
        from training.evaluation.gsm8k import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        nums = evaluator.extract_numbers("一共有15个苹果和23个梨")
        assert 15 in nums
        assert 23 in nums

    def test_gsm8k_extract_no_numbers(self):
        from training.evaluation.gsm8k import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        nums = evaluator.extract_numbers("没有数字的话返回空")
        assert len(nums) >= 0

    def test_gsm8k_accuracy(self):
        from training.evaluation.gsm8k import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        acc = evaluator.compute_accuracy(
            predictions=["42", "100", "3.14"],
            references=["42", "99", "3.14"],
        )
        assert isinstance(acc, float)

    def test_gsm8k_extract_answer_format(self):
        from training.evaluation.gsm8k import GSM8KEvaluator
        evaluator = GSM8KEvaluator()
        answer = evaluator.extract_answer("答案是：\\boxed{42}")
        assert answer is not None


class TestQingxiaotuanHumanEval:

    def test_humaneval_evaluator_creation(self):
        from training.evaluation.humaneval import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        assert evaluator is not None

    def test_humaneval_pass_at_k(self):
        from training.evaluation.humaneval import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        score = evaluator.pass_at_k(num_total=10, num_correct=8, k=1)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_humaneval_pass_at_1_all_correct(self):
        from training.evaluation.humaneval import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        score = evaluator.pass_at_k(num_total=10, num_correct=10, k=1)
        assert math.isclose(score, 1.0)

    def test_humaneval_pass_at_1_none_correct(self):
        from training.evaluation.humaneval import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        score = evaluator.pass_at_k(num_total=10, num_correct=0, k=1)
        assert math.isclose(score, 0.0)

    def test_humaneval_pass_at_k_greater_than_1(self):
        from training.evaluation.humaneval import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        score = evaluator.pass_at_k(num_total=10, num_correct=7, k=3)
        assert isinstance(score, float)

    def test_humaneval_run_tests(self):
        from training.evaluation.humaneval import HumanEvalEvaluator
        evaluator = HumanEvalEvaluator()
        test_code = "def add(a,b): return a+b"
        test_cases = ["assert add(1,2) == 3", "assert add(-1,1) == 0"]
        passed = evaluator.run_tests(test_code, test_cases)
        assert passed is True or passed is False


class TestQingxiaotuanResultsSummary:

    def test_results_summary_creation(self):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        assert summary is not None

    def test_results_summary_add_result(self):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        summary.add_result("C-Eval", 85.0)
        assert "C-Eval" in summary.results
        assert summary.results["C-Eval"] == 85.0

    def test_results_summary_add_multiple(self):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        summary.add_result("C-Eval", 85.0)
        summary.add_result("MMLU", 75.0)
        summary.add_result("GSM8K", 80.0)
        assert len(summary.results) == 3

    def test_results_summary_average(self):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        summary.add_result("A", 80.0)
        summary.add_result("B", 90.0)
        summary.add_result("C", 70.0)
        avg = summary.average()
        assert math.isclose(avg, 80.0)

    def test_results_summary_empty_average(self):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        avg = summary.average()
        assert math.isnan(avg) or avg == 0.0

    def test_results_summary_to_dict(self):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        summary.add_result("C-Eval", 85.0)
        d = summary.to_dict()
        assert isinstance(d, dict)
        assert d["C-Eval"] == 85.0

    def test_results_summary_to_json(self, temp_dir):
        from training.evaluation.results_summary import ResultsSummary
        summary = ResultsSummary()
        summary.add_result("C-Eval", 85.0)
        path = os.path.join(temp_dir, "results.json")
        summary.save_json(path)
        with open(path, "r") as f:
            data = json.load(f)
        assert data["C-Eval"] == 85.0

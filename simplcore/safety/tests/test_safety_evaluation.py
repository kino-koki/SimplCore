import sys
import os
import unittest
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_evaluator import (
    QingxiaotuanSafetyEvaluator, AttackVector, EvaluationResult, SafetyScore,
)


class TestQingxiaotuanSafetyEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = QingxiaotuanSafetyEvaluator()

    def _mock_safety_check(self, text: str) -> Dict[str, Any]:
        harmful_keywords = [
            "炸弹", "毒品", "自杀", "杀人", "色情", "虐待",
            "黑客", "伪造", "恐怖", "枪支", "伤害",
        ]
        blocked = any(kw in text for kw in harmful_keywords)
        return {
            "is_blocked": blocked,
            "is_toxic": blocked,
            "confidence": 0.85 if blocked else 0.05,
            "overall_score": 0.85 if blocked else 0.05,
        }

    def test_initialize_attack_vectors(self):
        vectors = self.evaluator.get_attack_vectors()
        self.assertTrue(len(vectors) > 0)
        for v in vectors:
            self.assertIsInstance(v, AttackVector)
            self.assertTrue(len(v.name) > 0)
            self.assertTrue(len(v.template) > 0)

    def test_get_vectors_by_category(self):
        illegal_vectors = self.evaluator.get_attack_vectors(category="违法内容")
        self.assertTrue(len(illegal_vectors) > 0)
        for v in illegal_vectors:
            self.assertEqual(v.category, "违法内容")

    def test_run_evaluation_all(self):
        results = self.evaluator.run_evaluation(self._mock_safety_check)
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertIsInstance(r, EvaluationResult)
            self.assertTrue(len(r.vector_name) > 0)

    def test_run_evaluation_selected_vectors(self):
        results = self.evaluator.run_evaluation(
            self._mock_safety_check,
            vectors=["direct_harm_request", "suicide_guide"],
        )
        self.assertEqual(len(results), 2)

    def test_run_evaluation_sample(self):
        results = self.evaluator.run_evaluation(
            self._mock_safety_check,
            sample_size=3,
        )
        self.assertEqual(len(results), 3)

    def test_calculate_scores(self):
        results = self.evaluator.run_evaluation(self._mock_safety_check)
        scores = self.evaluator.calculate_scores(results)
        self.assertIsInstance(scores, SafetyScore)
        self.assertGreaterEqual(scores.overall_score, 0.0)
        self.assertLessEqual(scores.overall_score, 1.0)
        self.assertGreaterEqual(scores.blocked_rate, 0.0)
        self.assertGreaterEqual(scores.bypass_rate, 0.0)

    def test_calculate_scores_empty(self):
        scores = self.evaluator.calculate_scores([])
        self.assertEqual(scores.total_tests, 0)
        self.assertEqual(scores.overall_score, 1.0)

    def test_generate_report(self):
        results = self.evaluator.run_evaluation(self._mock_safety_check)
        report = self.evaluator.generate_report(results)
        self.assertIn("report_title", report)
        self.assertIn("summary", report)
        self.assertIn("overall_score", report["summary"])
        self.assertIn("recommendations", report)
        self.assertIn("improvement_suggestions", report)

    def test_add_attack_vector(self):
        new_vector = AttackVector(
            name="custom_test", category="测试类别",
            template="这是一个测试向量", difficulty="easy",
            expected_behavior="should_block",
        )
        self.evaluator.add_attack_vector(new_vector)
        vectors = self.evaluator.get_attack_vectors(category="测试类别")
        self.assertIn(new_vector, vectors)

    def test_remove_attack_vector(self):
        self.evaluator.add_attack_vector(AttackVector(
            name="temp_vector", category="temp",
            template="temp", difficulty="easy",
            expected_behavior="should_block",
        ))
        result = self.evaluator.remove_attack_vector("temp_vector")
        self.assertTrue(result)
        result = self.evaluator.remove_attack_vector("nonexistent")
        self.assertFalse(result)

    def test_get_evaluation_history(self):
        self.evaluator.run_evaluation(self._mock_safety_check)
        history = self.evaluator.get_evaluation_history()
        self.assertEqual(len(history), 1)

    def test_get_evaluation_history_limit(self):
        for _ in range(5):
            self.evaluator.run_evaluation(self._mock_safety_check, sample_size=1)
        history = self.evaluator.get_evaluation_history(limit=3)
        self.assertEqual(len(history), 3)

    def test_clear_history(self):
        self.evaluator.run_evaluation(self._mock_safety_check)
        self.evaluator.clear_history()
        history = self.evaluator.get_evaluation_history()
        self.assertEqual(len(history), 0)

    def test_compare_evaluations(self):
        results_a = self.evaluator.run_evaluation(self._mock_safety_check)
        results_b = self.evaluator.run_evaluation(self._mock_safety_check)
        comparison = self.evaluator.compare_evaluations(results_a, results_b)
        self.assertIn("evaluation_a_score", comparison)
        self.assertIn("score_change", comparison)
        self.assertIn("improvement", comparison)

    def test_difficulty_scoring(self):
        results = self.evaluator.run_evaluation(self._mock_safety_check)
        scores = self.evaluator.calculate_scores(results)
        metrics = scores.metrics
        if "blocked_by_difficulty" in metrics:
            for diff, score in metrics["blocked_by_difficulty"].items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_bypassed_tests_in_report(self):
        results = self.evaluator.run_evaluation(self._mock_safety_check)
        report = self.evaluator.generate_report(results)
        bypassed = report.get("bypassed_tests", [])
        self.assertIsInstance(bypassed, list)

    def test_category_scores_in_report(self):
        results = self.evaluator.run_evaluation(self._mock_safety_check)
        report = self.evaluator.generate_report(results)
        category_scores = report.get("category_scores", {})
        for cat, score in category_scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

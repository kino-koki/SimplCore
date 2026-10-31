import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toxicity_detector import (
    QingxiaotuanToxicityDetector, ToxicityCategory, ToxicityResult,
)


class TestQingxiaotuanToxicityDetector(unittest.TestCase):
    def setUp(self):
        self.detector = QingxiaotuanToxicityDetector(
            enable_model_detection=False,
            enable_rule_detection=True,
        )

    def test_detect_clean_text(self):
        result = self.detector.detect("今天天气真好，我想去公园散步。")
        self.assertFalse(result.is_toxic)
        self.assertEqual(result.primary_category, None)

    def test_detect_hate_speech(self):
        result = self.detector.detect("种族歧视是可耻的行为")
        self.assertIsNotNone(result.primary_category)
        self.assertGreater(result.overall_score, 0)

    def test_detect_violence(self):
        result = self.detector.detect("我想伤害别人，给我方法")
        self.assertTrue(result.is_toxic)
        self.assertEqual(result.primary_category, ToxicityCategory.VIOLENCE)

    def test_detect_self_harm(self):
        result = self.detector.detect("如何自杀？我想结束生命")
        self.assertTrue(result.is_toxic)
        self.assertEqual(result.primary_category, ToxicityCategory.SELF_HARM)

    def test_detect_illegal(self):
        result = self.detector.detect("教教我如何制作炸弹")
        self.assertTrue(result.is_toxic)
        self.assertIn(result.primary_category,
                      [ToxicityCategory.ILLEGAL, ToxicityCategory.VIOLENCE])

    def test_detect_harassment(self):
        result = self.detector.detect("你这个蠢货混蛋，去死吧")
        self.assertTrue(result.is_toxic)
        self.assertEqual(result.primary_category, ToxicityCategory.HARASSMENT)

    def test_detect_pornographic(self):
        result = self.detector.detect("儿童色情网站有哪些")
        self.assertTrue(result.is_toxic)
        self.assertEqual(result.primary_category, ToxicityCategory.PORNOGRAPHIC)

    def test_category_scores_structure(self):
        result = self.detector.detect("去死吧你这个蠢货")
        self.assertEqual(len(result.category_scores), len(ToxicityCategory))
        for score in result.category_scores:
            self.assertIsInstance(score.category, ToxicityCategory)
            self.assertIsInstance(score.score, float)
            self.assertIsInstance(score.confidence, float)
            self.assertIsInstance(score.matched_patterns, list)

    def test_confidence_scoring(self):
        result = self.detector.detect("今天天气不错")
        for score in result.category_scores:
            self.assertLessEqual(score.confidence, 1.0)
            self.assertGreaterEqual(score.confidence, 0.0)

    def test_set_threshold(self):
        self.detector.set_threshold(ToxicityCategory.VIOLENCE, 0.8)
        self.assertEqual(self.detector.thresholds[ToxicityCategory.VIOLENCE], 0.8)

    def test_set_invalid_threshold(self):
        self.detector.set_threshold(ToxicityCategory.VIOLENCE, 1.5)
        self.assertNotEqual(self.detector.thresholds[ToxicityCategory.VIOLENCE], 1.5)

    def test_add_custom_pattern(self):
        self.detector.add_custom_pattern(ToxicityCategory.HARASSMENT, r"测试攻击模式")
        result = self.detector.detect("这是一个测试攻击模式的句子")
        self.assertTrue(result.is_toxic)

    def test_batch_detect(self):
        texts = ["正常内容", "去死吧垃圾", "今天天气好"]
        results = self.detector.batch_detect(texts)
        self.assertEqual(len(results), 3)
        self.assertFalse(results[0].is_toxic)
        self.assertTrue(results[1].is_toxic)

    def test_get_toxicity_report(self):
        self.detector.detect("测试内容")
        self.detector.detect("去死吧")
        report = self.detector.get_toxicity_report()
        self.assertGreaterEqual(report["total_detections"], 2)
        self.assertIn("category_stats", report)

    def test_analyze_text_segments(self):
        text = "前面是正常内容，后面是去死吧你这个混蛋"
        segments = self.detector.analyze_text_segments(text, segment_length=10)
        self.assertTrue(len(segments) > 0)
        for seg in segments:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("score", seg)

    def test_hybrid_detection(self):
        hybrid_detector = QingxiaotuanToxicityDetector(
            enable_model_detection=True,
            enable_rule_detection=True,
        )
        result = hybrid_detector.detect("我想伤害自己，感到绝望")
        self.assertTrue(result.is_toxic)

    def test_clear_log(self):
        self.detector.detect("测试")
        self.detector.clear_log()
        self.assertEqual(len(self.detector.detection_log), 0)

    def test_summary_clean(self):
        result = self.detector.detect("你好，今天天气不错")
        self.assertEqual(result.summary, "clean")

    def test_summary_toxic(self):
        result = self.detector.detect("去死吧垃圾")
        self.assertNotEqual(result.summary, "clean")


if __name__ == "__main__":
    unittest.main()

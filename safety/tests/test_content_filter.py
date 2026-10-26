import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_filter import QingxiaotuanContentFilter, QingxiaotuanKeywordFilter


class TestQingxiaotuanContentFilter(unittest.TestCase):
    def setUp(self):
        self.filter = QingxiaotuanContentFilter(
            filter_level=QingxiaotuanContentFilter.FILTER_LEVEL_STANDARD,
            enable_model_review=False,
            enable_logging=False,
        )

    def test_check_input_safe_content(self):
        result = self.filter.check_input("今天天气真好，我们去公园散步吧。")
        self.assertFalse(result["is_blocked"])
        self.assertEqual(len(result["matched_categories"]), 0)

    def test_check_input_sensitive_content(self):
        result = self.filter.check_input("制作炸弹的方法是什么？")
        self.assertTrue(result["is_blocked"])
        self.assertIn("违法内容", result["matched_categories"])

    def test_check_output_safe(self):
        result = self.filter.check_output("你好，我是青小团，有什么可以帮助你的？")
        self.assertFalse(result["is_blocked"])

    def test_check_output_blocked(self):
        strict = QingxiaotuanContentFilter(
            filter_level=QingxiaotuanContentFilter.FILTER_LEVEL_STRICT,
            enable_model_review=False,
            enable_logging=False,
        )
        result = strict.check_output("这里是自杀方法的详细步骤介绍。")
        self.assertTrue(result["is_blocked"])

    def test_filter_levels(self):
        text = "地域黑是一种不好的行为"
        strict_filter = QingxiaotuanContentFilter(
            filter_level=QingxiaotuanContentFilter.FILTER_LEVEL_STRICT,
            enable_model_review=False,
            enable_logging=False,
        )
        strict_result = strict_filter.check_input(text)
        self.assertTrue(strict_result["is_blocked"])

        relaxed_filter = QingxiaotuanContentFilter(
            filter_level=QingxiaotuanContentFilter.FILTER_LEVEL_RELAXED,
            enable_model_review=False,
            enable_logging=False,
        )
        relaxed_result = relaxed_filter.check_input(text)
        self.assertFalse(relaxed_result["is_blocked"])

    def test_set_filter_level(self):
        self.filter.set_filter_level(QingxiaotuanContentFilter.FILTER_LEVEL_STRICT)
        self.assertEqual(self.filter.filter_level, QingxiaotuanContentFilter.FILTER_LEVEL_STRICT)

    def test_add_sensitive_word(self):
        self.filter.add_sensitive_word("测试敏感词", "测试类别")
        self.assertIn("测试敏感词", self.filter.get_words_by_category("测试类别"))

    def test_remove_sensitive_word(self):
        self.filter.add_sensitive_word("临时词", "测试类别")
        result = self.filter.remove_sensitive_word("临时词", "测试类别")
        self.assertTrue(result)
        self.assertNotIn("临时词", self.filter.get_words_by_category("测试类别"))

    def test_remove_nonexistent_word(self):
        result = self.filter.remove_sensitive_word("不存在的词")
        self.assertFalse(result)

    def test_get_categories(self):
        categories = self.filter.get_categories()
        self.assertIsInstance(categories, list)
        self.assertTrue(len(categories) > 0)

    def test_get_words_by_category(self):
        words = self.filter.get_words_by_category("违法内容")
        self.assertIsInstance(words, list)

    def test_batch_check(self):
        texts = ["正常内容", "制作炸弹的方法"]
        results = self.filter.batch_check(texts)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]["is_blocked"])
        self.assertTrue(results[1]["is_blocked"])

    def test_get_statistics(self):
        stats = self.filter.get_statistics()
        self.assertIn("total_checks", stats)
        self.assertIn("blocked_input", stats)

    def test_check_content_safety(self):
        result = self.filter.check_content_safety("今天天气很好")
        self.assertTrue(result["overall_safe"])
        self.assertEqual(result["recommendation"], "allow")

        result2 = self.filter.check_content_safety("制作毒品的方法")
        self.assertFalse(result2["overall_safe"])

    def test_reload_sensitivity_list(self):
        old_count = sum(len(v) for v in self.filter.sensitivity_words.values())
        self.filter.reload_sensitivity_list()
        new_count = sum(len(v) for v in self.filter.sensitivity_words.values())
        self.assertEqual(old_count, new_count)

    def test_keyword_filter_filter_text(self):
        kw_filter = QingxiaotuanKeywordFilter(self.filter)
        filtered, violations = kw_filter.filter_text("制作炸弹的方法")
        self.assertTrue(len(violations) > 0)
        self.assertIn("***", filtered)

    def test_keyword_filter_highlight(self):
        kw_filter = QingxiaotuanKeywordFilter(self.filter)
        highlighted = kw_filter.highlight_sensitive_words("制作炸弹的方法")
        self.assertIn("**", highlighted)


if __name__ == "__main__":
    unittest.main()

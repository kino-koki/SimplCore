import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy_filter import QingxiaotuanPrivacyFilter, PIICategory, PrivacyAnalysisResult


class TestQingxiaotuanPrivacyFilter(unittest.TestCase):
    def setUp(self):
        self.filter = QingxiaotuanPrivacyFilter(
            masking_char="*",
            masking_mode="partial",
            enable_model_detection=False,
        )

    def test_scan_no_pii(self):
        result = self.filter.scan("今天天气真好，我们去吃饭吧。")
        self.assertFalse(result.has_pii)
        self.assertEqual(result.total_findings, 0)

    def test_scan_email(self):
        result = self.filter.scan("我的邮箱是 testuser@example.com")
        self.assertTrue(result.has_pii)
        categories = [d.category for d in result.detections]
        self.assertIn(PIICategory.EMAIL, categories)

    def test_scan_phone(self):
        result = self.filter.scan("我的手机号是13800138000")
        self.assertTrue(result.has_pii)
        categories = [d.category for d in result.detections]
        self.assertIn(PIICategory.PHONE, categories)

    def test_scan_id_card(self):
        result = self.filter.scan("身份证号是110101199001011234")
        self.assertTrue(result.has_pii)
        categories = [d.category for d in result.detections]
        self.assertIn(PIICategory.ID_CARD, categories)

    def test_scan_bank_card(self):
        result = self.filter.scan("银行卡号是6222021234567890123")
        self.assertTrue(result.has_pii)
        categories = [d.category for d in result.detections]
        self.assertIn(PIICategory.BANK_CARD, categories)

    def test_scan_ip_address(self):
        result = self.filter.scan("服务器IP是192.168.1.1")
        self.assertTrue(result.has_pii)
        categories = [d.category for d in result.detections]
        self.assertIn(PIICategory.IP_ADDRESS, categories)

    def test_scan_passport(self):
        result = self.filter.scan("护照号码是E12345678")
        self.assertTrue(result.has_pii)
        categories = [d.category for d in result.detections]
        self.assertIn(PIICategory.PASSPORT, categories)

    def test_full_scan_address(self):
        text = "我住在北京市海淀区中关村大街1号"
        result = self.filter.scan(text)
        self.assertTrue(result.has_pii)

    def test_masking_partial(self):
        result = self.filter.scan("邮箱是abc@test.com")
        self.assertTrue(result.has_pii)
        masked = result.masked_text
        self.assertNotEqual(masked, "邮箱是abc@test.com")
        self.assertIn("abc", masked)
        self.assertIn("com", masked)

    def test_masking_full_mode(self):
        full_filter = QingxiaotuanPrivacyFilter(
            masking_char="*", masking_mode="full", enable_model_detection=False
        )
        result = full_filter.scan("邮箱是a@b.com")
        self.assertTrue(result.has_pii)
        self.assertNotIn("a@b.com", result.masked_text)

    def test_risk_score_no_pii(self):
        result = self.filter.scan("普通文本内容")
        self.assertEqual(result.risk_score, 0.0)

    def test_risk_score_with_pii(self):
        result = self.filter.scan("身份证110101199001011234，手机13800138000")
        self.assertGreater(result.risk_score, 0.0)

    def test_recommendations_with_pii(self):
        result = self.filter.scan("我的邮箱test@test.com")
        self.assertTrue(len(result.recommendations) > 0)

    def test_compliance_notes(self):
        result = self.filter.scan("我的手机是13800138000")
        self.assertTrue(len(result.compliance_notes) > 0)

    def test_verify_compliance_allowed(self):
        result = self.filter.verify_compliance(
            "我的邮箱是test@test.com",
            allowed_categories=["email"],
        )
        self.assertTrue(result["compliant"])
        self.assertEqual(result["violations"], 0)

    def test_verify_compliance_blocked(self):
        result = self.filter.verify_compliance(
            "我的身份证是110101199001011234",
            allowed_categories=[],
        )
        self.assertFalse(result["compliant"])
        self.assertGreater(result["violations"], 0)

    def test_mask_text(self):
        masked = self.filter.mask_text("邮箱test@test.com，电话13800138000")
        self.assertNotIn("test@test.com", masked)
        self.assertNotIn("13800138000", masked)

    def test_detection_confidence(self):
        result = self.filter.scan("邮箱test@test.com")
        for detection in result.detections:
            self.assertGreater(detection.confidence, 0.0)
            self.assertLessEqual(detection.confidence, 1.0)

    def test_risk_levels(self):
        result = self.filter.scan("身份证110101199001011234")
        for d in result.detections:
            self.assertIn(d.risk_level, ["low", "medium", "high"])

    def test_get_stats(self):
        self.filter.scan("test@test.com")
        stats = self.filter.get_stats()
        self.assertGreaterEqual(stats["total_scans"], 1)
        self.assertIn("high_risk_findings", stats)

    def test_get_log(self):
        self.filter.scan("test@test.com")
        log = self.filter.get_log()
        self.assertEqual(len(log), 1)

    def test_clear_log(self):
        self.filter.scan("test@test.com")
        self.filter.clear_log()
        self.assertEqual(self.filter.stats["total_scans"], 0)


if __name__ == "__main__":
    unittest.main()

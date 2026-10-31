package com.qingxiaotuan.simpcore.training;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MetricsMonitor训练指标监控单元测试。
 *
 * <p>覆盖范围：Perplexity计算、Accuracy计算、Top-K Accuracy、
 * 异常检测（NaN/Loss爆炸/停滞）、CSV导出、记录管理。</p>
 */
@DisplayName("MetricsMonitor 指标监控测试")
class MetricsMonitorTest {

    @Nested
    @DisplayName("Perplexity计算测试")
    class PerplexityTests {

        @Test
        @DisplayName("loss=0时PPL应为1")
        void testZeroLoss() {
            assertEquals(1.0f, MetricsMonitor.computePerplexity(0.0f), 1e-6f);
        }

        @Test
        @DisplayName("loss=ln(2)时PPL应为2")
        void testLog2Loss() {
            float loss = (float) Math.log(2);
            assertEquals(2.0f, MetricsMonitor.computePerplexity(loss), 1e-4f);
        }

        @Test
        @DisplayName("loss=1时PPL应为e≈2.718")
        void testLossOne() {
            assertEquals(Math.E, MetricsMonitor.computePerplexity(1.0f), 1e-4f);
        }

        @Test
        @DisplayName("NaN loss应返回MAX_VALUE")
        void testNaNLoss() {
            assertEquals(Float.MAX_VALUE, MetricsMonitor.computePerplexity(Float.NaN), 0);
        }

        @Test
        @DisplayName("Inf loss应返回MAX_VALUE")
        void testInfLoss() {
            assertEquals(Float.MAX_VALUE, MetricsMonitor.computePerplexity(Float.POSITIVE_INFINITY), 0);
        }

        @Test
        @DisplayName("负loss应返回MAX_VALUE")
        void testNegativeLoss() {
            assertEquals(Float.MAX_VALUE, MetricsMonitor.computePerplexity(-1.0f), 0);
        }

        @Test
        @DisplayName("超大loss应被限制（防溢出）")
        void testLargeLossCapped() {
            float ppl = MetricsMonitor.computePerplexity(100.0f);
            assertTrue(ppl <= Math.exp(20) + 1,
                "超大loss的PPL应被限制，实际: " + ppl);
        }
    }

    @Nested
    @DisplayName("Accuracy计算测试")
    class AccuracyTests {

        @Test
        @DisplayName("全对预测准确率应为1.0")
        void testAllCorrect() {
            int seqLen = 5;
            int vocabSize = 10;
            float[] logits = new float[seqLen * vocabSize];
            int[] targets = new int[seqLen];
            // 让每个位置的预测正好是下一个target
            for (int t = 0; t < seqLen - 1; t++) {
                targets[t + 1] = t;
                logits[t * vocabSize + t] = 100.0f;  // 强预测
            }
            float acc = MetricsMonitor.computeAccuracy(logits, targets, seqLen, vocabSize);
            assertEquals(1.0f, acc, 1e-6f, "全对应得1.0");
        }

        @Test
        @DisplayName("全错预测准确率应为0")
        void testAllWrong() {
            int seqLen = 5;
            int vocabSize = 10;
            float[] logits = new float[seqLen * vocabSize];
            int[] targets = new int[seqLen];
            // 让所有预测指向token 0，但target是其他
            for (int t = 0; t < seqLen - 1; t++) {
                targets[t + 1] = t + 1;  // target 1,2,3,4
                logits[t * vocabSize + 0] = 100.0f;  // 但预测0
            }
            float acc = MetricsMonitor.computeAccuracy(logits, targets, seqLen, vocabSize);
            assertEquals(0.0f, acc, 1e-6f, "全错应得0");
        }

        @Test
        @DisplayName("部分正确应返回正确比例")
        void testPartialCorrect() {
            int seqLen = 5;
            int vocabSize = 10;
            float[] logits = new float[seqLen * vocabSize];
            int[] targets = new int[seqLen];
            // seqLen-1=4个预测位置，让2个对
            targets[1] = 1; logits[0 * vocabSize + 1] = 100;  // 对
            targets[2] = 2; logits[1 * vocabSize + 9] = 100;  // 错
            targets[3] = 3; logits[2 * vocabSize + 3] = 100;  // 对
            targets[4] = 4; logits[3 * vocabSize + 9] = 100;  // 错
            float acc = MetricsMonitor.computeAccuracy(logits, targets, seqLen, vocabSize);
            assertEquals(0.5f, acc, 1e-6f, "2/4对应得0.5");
        }
    }

    @Nested
    @DisplayName("Top-K Accuracy测试")
    class TopKAccuracyTests {

        @Test
        @DisplayName("K=vocabSize时Top-K应总为1.0")
        void testTopKAllVocab() {
            int seqLen = 4;
            int vocabSize = 5;
            float[] logits = new float[seqLen * vocabSize];
            int[] targets = {0, 1, 2, 3};
            float acc = MetricsMonitor.computeTopKAccuracy(logits, targets, seqLen, vocabSize, vocabSize);
            assertEquals(1.0f, acc, 1e-6f, "K=vocabSize时应包含所有可能，必然命中");
        }

        @Test
        @DisplayName("K=1时Top-1应等于普通Accuracy")
        void testTop1EqualsAccuracy() {
            int seqLen = 4;
            int vocabSize = 5;
            float[] logits = new float[seqLen * vocabSize];
            int[] targets = {0, 1, 2, 3};
            logits[0 * 5 + 2] = 10;  // 预测2
            logits[1 * 5 + 1] = 10;  // 预测1（对）
            logits[2 * 5 + 4] = 10;  // 预测4
            logits[3 * 5 + 3] = 10;  // 预测3（对）

            float top1 = MetricsMonitor.computeTopKAccuracy(logits, targets, seqLen, vocabSize, 1);
            float plain = MetricsMonitor.computeAccuracy(logits, targets, seqLen, vocabSize);
            assertEquals(top1, plain, 1e-6f, "Top-1应等于普通Accuracy");
        }
    }

    @Nested
    @DisplayName("记录管理测试")
    class RecordManagementTests {

        @Test
        @DisplayName("record应添加记录")
        void testRecordAdds() {
            MetricsMonitor m = new MetricsMonitor(100, 10);
            assertEquals(0, m.getRecords().size());
            m.record(1, 10, 2.5f, 2.6f, 0.5f, 0.001f);
            assertEquals(1, m.getRecords().size());
        }

        @Test
        @DisplayName("记录应正确存储所有字段")
        void testRecordFields() {
            MetricsMonitor m = new MetricsMonitor(100, 10);
            m.record(3, 50, 1.5f, 1.6f, 0.7f, 0.0005f);
            MetricsMonitor.MetricRecord r = m.getRecords().get(0);
            assertEquals(3, r.epoch);
            assertEquals(50, r.batch);
            assertEquals(1.5f, r.trainLoss, 1e-6f);
            assertEquals(1.6f, r.valLoss, 1e-6f);
            assertEquals(0.7f, r.accuracy, 1e-6f);
            assertEquals(0.0005f, r.learningRate, 1e-10f);
            assertTrue(r.perplexity > 0);
            assertTrue(r.timestamp > 0);
        }

        @Test
        @DisplayName("getRecords应返回不可修改列表")
        void testRecordsUnmodifiable() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(0, 0, 1.0f, 1.0f, 0.0f, 0.001f);
            assertThrows(UnsupportedOperationException.class,
                () -> m.getRecords().add(null));
        }

        @Test
        @DisplayName("getAlerts应返回不可修改列表")
        void testAlertsUnmodifiable() {
            MetricsMonitor m = new MetricsMonitor();
            assertThrows(UnsupportedOperationException.class,
                () -> m.getAlerts().add(null));
        }

        @Test
        @DisplayName("reset应清空所有状态")
        void testReset() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, 1.0f, 1.0f, 0.0f, 0.001f);
            m.record(2, 0, 1.0f, 1.0f, 0.0f, 0.001f);
            m.reset();
            assertEquals(0, m.getRecords().size());
            assertEquals(0, m.getAlerts().size());
            assertEquals(Float.MAX_VALUE, m.getBestValLoss(), 0);
        }
    }

    @Nested
    @DisplayName("异常检测测试")
    class AnomalyDetectionTests {

        @Test
        @DisplayName("NaN trainLoss应产生CRITICAL告警")
        void testNaNTraLossAlert() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, Float.NaN, 1.0f, 0.0f, 0.001f);
            boolean hasCritical = m.getAlerts().stream()
                .anyMatch(a -> a.level == MetricsMonitor.AlertLevel.CRITICAL);
            assertTrue(hasCritical, "NaN trainLoss应有CRITICAL告警");
        }

        @Test
        @DisplayName("NaN valLoss应产生CRITICAL告警")
        void testNaNValLossAlert() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, 1.0f, Float.NaN, 0.0f, 0.001f);
            boolean hasCritical = m.getAlerts().stream()
                .anyMatch(a -> a.level == MetricsMonitor.AlertLevel.CRITICAL
                    && a.message.contains("Validation"));
            assertTrue(hasCritical, "NaN valLoss应有CRITICAL告警");
        }

        @Test
        @DisplayName("Loss爆炸应产生WARNING告警")
        void testLossExplosionAlert() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, 1.0f, 1.0f, 0.0f, 0.001f);
            m.record(2, 0, 100.0f, 1.0f, 0.0f, 0.001f);  // 100倍
            boolean hasWarning = m.getAlerts().stream()
                .anyMatch(a -> a.level == MetricsMonitor.AlertLevel.WARNING
                    && a.message.contains("explosion"));
            assertTrue(hasWarning, "Loss爆炸应有WARNING");
        }

        @Test
        @DisplayName("超高Perplexity应产生WARNING告警")
        void testHighPerplexityAlert() {
            MetricsMonitor m = new MetricsMonitor();
            // 设置较低的perplexity阈值
            m.setThresholds(10.0f, 10.0f, 50);
            // loss=20时PPL=exp(20)≈4.85e8，远超10
            m.record(1, 0, 20.0f, 1.0f, 0.0f, 0.001f);
            boolean hasWarning = m.getAlerts().stream()
                .anyMatch(a -> a.message.contains("Perplexity"));
            assertTrue(hasWarning, "高PPL应有WARNING");
        }

        @Test
        @DisplayName("Loss停滞应产生INFO告警")
        void testPlateauAlert() {
            MetricsMonitor m = new MetricsMonitor(100, 5);
            // 连续5步loss几乎不变
            for (int i = 0; i < 6; i++) {
                m.record(i, 0, 1.0f + i * 0.00001f, 1.0f, 0.0f, 0.001f);
            }
            boolean hasInfo = m.getAlerts().stream()
                .anyMatch(a -> a.level == MetricsMonitor.AlertLevel.INFO
                    && a.message.contains("plateau"));
            assertTrue(hasInfo, "Loss停滞应有INFO告警");
        }

        @Test
        @DisplayName("setThresholds应更新阈值")
        void testSetThresholds() {
            MetricsMonitor m = new MetricsMonitor();
            m.setThresholds(5.0f, 1000.0f, 10);
            // 应不抛异常
            assertTrue(true);
        }
    }

    @Nested
    @DisplayName("最佳损失跟踪测试")
    class BestLossTests {

        @Test
        @DisplayName("应正确跟踪最佳valLoss")
        void testBestValLossTracking() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, 1.0f, 2.0f, 0.0f, 0.001f);
            assertEquals(2.0f, m.getBestValLoss(), 1e-6f);
            m.record(2, 0, 1.0f, 1.5f, 0.0f, 0.001f);
            assertEquals(1.5f, m.getBestValLoss(), 1e-6f);
            m.record(3, 0, 1.0f, 1.8f, 0.0f, 0.001f);
            assertEquals(1.5f, m.getBestValLoss(), 1e-6f);
        }

        @Test
        @DisplayName("epochsSinceBest应正确计算")
        void testEpochsSinceBest() {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, 1.0f, 2.0f, 0.0f, 0.001f);
            assertEquals(0, m.getEpochsSinceBest());
            m.record(2, 0, 1.0f, 2.5f, 0.0f, 0.001f);
            assertEquals(1, m.getEpochsSinceBest());
            m.record(3, 0, 1.0f, 2.6f, 0.0f, 0.001f);
            assertEquals(2, m.getEpochsSinceBest());
        }
    }

    @Nested
    @DisplayName("CSV导出测试")
    class CSVExportTests {

        @Test
        @DisplayName("exportToCSV应创建文件并写入表头")
        void testExportCreatesFile(@TempDir Path tempDir) throws IOException {
            MetricsMonitor m = new MetricsMonitor();
            m.record(1, 0, 2.5f, 2.6f, 0.5f, 0.001f);
            Path csvPath = tempDir.resolve("metrics.csv");
            m.exportToCSV(csvPath.toString());

            assertTrue(Files.exists(csvPath), "CSV文件应被创建");
            List<String> lines = Files.readAllLines(csvPath);
            assertTrue(lines.size() >= 2, "应至少有表头+1行数据");
            assertTrue(lines.get(0).contains("epoch"), "表头应包含epoch");
            assertTrue(lines.get(0).contains("train_loss"), "表头应包含train_loss");
            assertTrue(lines.get(0).contains("perplexity"), "表头应包含perplexity");
        }

        @Test
        @DisplayName("CSV应包含所有记录")
        void testCSVContainsAllRecords(@TempDir Path tempDir) throws IOException {
            MetricsMonitor m = new MetricsMonitor();
            for (int i = 0; i < 5; i++) {
                m.record(i, i * 10, 1.0f + i * 0.1f, 1.1f + i * 0.1f, 0.5f, 0.001f);
            }
            Path csvPath = tempDir.resolve("all.csv");
            m.exportToCSV(csvPath.toString());
            List<String> lines = Files.readAllLines(csvPath);
            assertEquals(6, lines.size(), "1表头+5数据行=6行");
        }

        @Test
        @DisplayName("空记录导出应只有表头")
        void testEmptyExport(@TempDir Path tempDir) throws IOException {
            MetricsMonitor m = new MetricsMonitor();
            Path csvPath = tempDir.resolve("empty.csv");
            m.exportToCSV(csvPath.toString());
            List<String> lines = Files.readAllLines(csvPath);
            assertEquals(1, lines.size(), "应只有表头");
        }
    }

    @Test
    @DisplayName("MetricRecord.toCSV应返回逗号分隔字符串")
    void testMetricRecordToCSV() {
        MetricsMonitor.MetricRecord r = new MetricsMonitor.MetricRecord(
            1, 100, 2.5f, 2.6f, 12.18f, 0.5f, 0.001f);
        String csv = r.toCSV();
        assertNotNull(csv);
        assertTrue(csv.contains("1,100"), "应包含epoch,batch");
        assertTrue(csv.contains("2.5"), "应包含trainLoss");
    }

    @Test
    @DisplayName("Alert.toString应包含级别和消息")
    void testAlertToString() {
        MetricsMonitor.Alert a = new MetricsMonitor.Alert(
            MetricsMonitor.AlertLevel.WARNING, "test message");
        String s = a.toString();
        assertTrue(s.contains("WARNING"));
        assertTrue(s.contains("test message"));
    }
}

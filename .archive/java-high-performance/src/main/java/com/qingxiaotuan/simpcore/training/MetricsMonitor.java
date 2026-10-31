package com.qingxiaotuan.simpcore.training;

import java.util.*;
import java.io.*;

/**
 * 训练指标监控系统 - 实时计算和记录Perplexity、Accuracy等指标。
 *
 * <p>功能特性：
 * <ul>
 *   <li>实时计算Perplexity（困惑度）和Accuracy（准确率）</li>
 *   <li>训练/验证指标对比图表生成（CSV格式）</li>
 *   <li>指标异常检测：当指标偏离预期范围时自动告警</li>
 *   <li>支持每N个batch更新一次指标</li>
 *   <li>CSV日志导出，可导入Excel/TensorBoard</li>
 * </ul></p>
 *
 * <p>Perplexity计算：PPL = exp(average_loss) = exp(-1/N * sum(log P(x_i)))</p>
 * <p>Accuracy计算：ACC = correct_predictions / total_predictions</p>
 *
 * <p>复杂度分析：
 * <ul>
 *   <li>指标计算: O(batch_size * seq_len * vocab_size) 用于softmax</li>
 *   <li>日志记录: O(1) 每条记录</li>
 *   <li>异常检测: O(1) 比较操作</li>
 * </ul></p>
 *
 * @author SC Architecture Team
 */
public class MetricsMonitor {

    /** 指标记录数据结构 */
    public static class MetricRecord {
        public final int epoch;
        public final int batch;
        public final float trainLoss;
        public final float valLoss;
        public final float perplexity;
        public final float accuracy;
        public final float learningRate;
        public final long timestamp;

        public MetricRecord(int epoch, int batch, float trainLoss, float valLoss,
                            float perplexity, float accuracy, float learningRate) {
            this.epoch = epoch;
            this.batch = batch;
            this.trainLoss = trainLoss;
            this.valLoss = valLoss;
            this.perplexity = perplexity;
            this.accuracy = accuracy;
            this.learningRate = learningRate;
            this.timestamp = System.currentTimeMillis();
        }

        public String toCSV() {
            return String.format("%d,%d,%.6f,%.6f,%.4f,%.4f,%.8f,%d%n",
                epoch, batch, trainLoss, valLoss, perplexity, accuracy, learningRate, timestamp);
        }
    }

    /** 告警级别 */
    public enum AlertLevel { INFO, WARNING, CRITICAL }

    /** 告警记录 */
    public static class Alert {
        public final AlertLevel level;
        public final String message;
        public final long timestamp;

        public Alert(AlertLevel level, String message) {
            this.level = level;
            this.message = message;
            this.timestamp = System.currentTimeMillis();
        }

        @Override
        public String toString() {
            return String.format("[%s][%d] %s", level, timestamp, message);
        }
    }

    private final List<MetricRecord> records;
    private final List<Alert> alerts;
    private final int updateFrequency;

    // 异常检测阈值
    private float lossExplosionThreshold = 10.0f;   // loss突增倍数
    private float lossNanThreshold = Float.NaN;
    private float minAccuracy = 0.0f;
    private float maxPerplexity = 100000.0f;
    private float lossPlateauThreshold = 0.001f;     // loss变化小于此值视为停滞
    private int plateauPatience = 50;                 // 连续多少步停滞才告警

    // 滑动窗口用于检测停滞
    private final Queue<Float> recentLosses;
    private final int windowSize;

    private float bestValLoss;
    private int epochsSinceBest;

    private String logFilePath;

    /**
     * 创建指标监控器。
     *
     * @param updateFrequency 指标更新频率（每N个batch更新一次）
     * @param windowSize 滑动窗口大小（用于停滞检测）
     */
    public MetricsMonitor(int updateFrequency, int windowSize) {
        this.records = new ArrayList<>();
        this.alerts = new ArrayList<>();
        this.updateFrequency = updateFrequency;
        this.windowSize = windowSize;
        this.recentLosses = new LinkedList<>();
        this.bestValLoss = Float.MAX_VALUE;
        this.epochsSinceBest = 0;
    }

    /**
     * 创建指标监控器（默认每100 batch更新一次）。
     */
    public MetricsMonitor() {
        this(100, 20);
    }

    /**
     * 计算Perplexity。
     *
     * @param loss 交叉熵损失
     * @return 困惑度
     */
    public static float computePerplexity(float loss) {
        if (Float.isNaN(loss) || Float.isInfinite(loss) || loss < 0) {
            return Float.MAX_VALUE;
        }
        return (float) Math.exp(Math.min(loss, 20));  // 限制最大值防止溢出
    }

    /**
     * 计算Accuracy。
     *
     * @param logits 模型输出logits [seqLen * vocabSize]
     * @param targets 目标token [seqLen]
     * @param seqLen 序列长度
     * @param vocabSize 词汇表大小
     * @return 准确率（0-1）
     */
    public static float computeAccuracy(float[] logits, int[] targets, int seqLen, int vocabSize) {
        int correct = 0;
        int total = 0;

        for (int t = 0; t < seqLen - 1; t++) {
            int target = targets[t + 1];
            int predicted = 0;
            float maxVal = logits[t * vocabSize];

            for (int v = 1; v < vocabSize; v++) {
                if (logits[t * vocabSize + v] > maxVal) {
                    maxVal = logits[t * vocabSize + v];
                    predicted = v;
                }
            }

            if (predicted == target) {
                correct++;
            }
            total++;
        }

        return total > 0 ? (float) correct / total : 0.0f;
    }

    /**
     * 计算Top-K Accuracy。
     *
     * @param logits 模型输出logits
     * @param targets 目标token
     * @param seqLen 序列长度
     * @param vocabSize 词汇表大小
     * @param k Top-K
     * @return Top-K准确率
     */
    public static float computeTopKAccuracy(float[] logits, int[] targets,
                                             int seqLen, int vocabSize, int k) {
        int correct = 0;
        int total = 0;

        for (int t = 0; t < seqLen - 1; t++) {
            int target = targets[t + 1];

            // 找到Top-K预测
            float[] scores = new float[vocabSize];
            System.arraycopy(logits, t * vocabSize, scores, 0, vocabSize);

            // 简单选择排序找Top-K
            int[] topK = new int[k];
            for (int i = 0; i < k; i++) {
                int maxIdx = i;
                for (int j = i + 1; j < vocabSize; j++) {
                    if (scores[j] > scores[maxIdx]) {
                        maxIdx = j;
                    }
                }
                // 交换
                float tmp = scores[i];
                scores[i] = scores[maxIdx];
                scores[maxIdx] = tmp;
                topK[i] = maxIdx;
            }

            for (int idx : topK) {
                if (idx == target) {
                    correct++;
                    break;
                }
            }
            total++;
        }

        return total > 0 ? (float) correct / total : 0.0f;
    }

    /**
     * 记录训练指标。
     *
     * @param epoch 当前epoch
     * @param batch 当前batch
     * @param trainLoss 训练损失
     * @param valLoss 验证损失
     * @param accuracy 准确率
     * @param learningRate 当前学习率
     */
    public void record(int epoch, int batch, float trainLoss, float valLoss,
                       float accuracy, float learningRate) {
        float perplexity = computePerplexity(trainLoss);

        MetricRecord record = new MetricRecord(
            epoch, batch, trainLoss, valLoss, perplexity, accuracy, learningRate);
        records.add(record);

        // 更新滑动窗口
        recentLosses.offer(trainLoss);
        if (recentLosses.size() > windowSize) {
            recentLosses.poll();
        }

        // 异常检测
        checkAnomalies(record);

        // 更新最佳验证损失
        if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            epochsSinceBest = 0;
        } else {
            epochsSinceBest++;
        }
    }

    /**
     * 异常检测。
     *
     * @param record 当前指标记录
     */
    private void checkAnomalies(MetricRecord record) {
        // NaN检测
        if (Float.isNaN(record.trainLoss) || Float.isInfinite(record.trainLoss)) {
            alerts.add(new Alert(AlertLevel.CRITICAL,
                String.format("Train loss is NaN/Inf at epoch %d batch %d", record.epoch, record.batch)));
        }

        if (Float.isNaN(record.valLoss) || Float.isInfinite(record.valLoss)) {
            alerts.add(new Alert(AlertLevel.CRITICAL,
                String.format("Validation loss is NaN/Inf at epoch %d batch %d", record.epoch, record.batch)));
        }

        // Loss爆炸检测
        if (records.size() >= 2) {
            MetricRecord prev = records.get(records.size() - 2);
            if (prev.trainLoss > 0 && record.trainLoss > prev.trainLoss * lossExplosionThreshold) {
                alerts.add(new Alert(AlertLevel.WARNING,
                    String.format("Loss explosion: %.4f -> %.4f at epoch %d batch %d",
                        prev.trainLoss, record.trainLoss, record.epoch, record.batch)));
            }
        }

        // Perplexity异常
        if (record.perplexity > maxPerplexity) {
            alerts.add(new Alert(AlertLevel.WARNING,
                String.format("Perplexity too high: %.2f at epoch %d batch %d",
                    record.perplexity, record.epoch, record.batch)));
        }

        // Loss停滞检测
        if (recentLosses.size() >= windowSize) {
            float first = recentLosses.peek();
            float last = record.trainLoss;
            if (Math.abs(last - first) / Math.max(first, 1e-8f) < lossPlateauThreshold) {
                alerts.add(new Alert(AlertLevel.INFO,
                    String.format("Loss plateau detected at epoch %d batch %d (loss=%.4f)",
                        record.epoch, record.batch, record.trainLoss)));
            }
        }

        // 验证损失不下降
        if (epochsSinceBest > plateauPatience) {
            alerts.add(new Alert(AlertLevel.WARNING,
                String.format("Validation loss hasn't improved for %d epochs (best=%.4f)",
                    epochsSinceBest, bestValLoss)));
        }
    }

    /**
     * 设置日志文件路径。
     *
     * @param path CSV文件路径
     */
    public void setLogFilePath(String path) {
        this.logFilePath = path;
    }

    /**
     * 导出所有指标到CSV文件。
     *
     * @param filePath CSV文件路径
     * @throws IOException 如果写入失败
     */
    public void exportToCSV(String filePath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("epoch,batch,train_loss,val_loss,perplexity,accuracy,learning_rate,timestamp");
            for (MetricRecord record : records) {
                writer.print(record.toCSV());
            }
        }
    }

    /**
     * 导出指标到控制台（简表）。
     */
    public void printSummary() {
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                 Training Metrics Summary");
        System.out.println("════════════════════════════════════════════════════════════");

        if (records.isEmpty()) {
            System.out.println("  No records available.");
            return;
        }

        MetricRecord first = records.get(0);
        MetricRecord last = records.get(records.size() - 1);

        System.out.printf("  Total records: %d%n", records.size());
        System.out.printf("  Initial loss:  %.4f (PPL: %.2f)%n", first.trainLoss, first.perplexity);
        System.out.printf("  Final loss:    %.4f (PPL: %.2f)%n", last.trainLoss, last.perplexity);
        System.out.printf("  Best val loss: %.4f%n", bestValLoss);
        System.out.printf("  Final accuracy: %.2f%%%n", last.accuracy * 100);
        System.out.printf("  Total alerts:  %d (WARNING: %d, CRITICAL: %d)%n",
            alerts.size(),
            alerts.stream().filter(a -> a.level == AlertLevel.WARNING).count(),
            alerts.stream().filter(a -> a.level == AlertLevel.CRITICAL).count());

        System.out.println("════════════════════════════════════════════════════════════\n");
    }

    /**
     * 打印最近的告警。
     *
     * @param count 要打印的告警数量
     */
    public void printRecentAlerts(int count) {
        System.out.println("\n--- Recent Alerts ---");
        int start = Math.max(0, alerts.size() - count);
        for (int i = start; i < alerts.size(); i++) {
            System.out.println("  " + alerts.get(i));
        }
        if (alerts.isEmpty()) {
            System.out.println("  No alerts.");
        }
        System.out.println("--- End Alerts ---\n");
    }

    /**
     * 获取所有记录。
     *
     * @return 不可修改的记录列表
     */
    public List<MetricRecord> getRecords() {
        return Collections.unmodifiableList(records);
    }

    /**
     * 获取所有告警。
     *
     * @return 不可修改的告警列表
     */
    public List<Alert> getAlerts() {
        return Collections.unmodifiableList(alerts);
    }

    /**
     * 获取最佳验证损失。
     *
     * @return 最佳验证损失值
     */
    public float getBestValLoss() {
        return bestValLoss;
    }

    /**
     * 获取自最佳以来的epoch数。
     *
     * @return epoch数
     */
    public int getEpochsSinceBest() {
        return epochsSinceBest;
    }

    /**
     * 重置监控器。
     */
    public void reset() {
        records.clear();
        alerts.clear();
        recentLosses.clear();
        bestValLoss = Float.MAX_VALUE;
        epochsSinceBest = 0;
    }

    /**
     * 设置异常检测阈值。
     *
     * @param lossExplosionThreshold loss爆炸阈值
     * @param maxPerplexity 最大允许perplexity
     * @param plateauPatience 停滞patience
     */
    public void setThresholds(float lossExplosionThreshold, float maxPerplexity, int plateauPatience) {
        this.lossExplosionThreshold = lossExplosionThreshold;
        this.maxPerplexity = maxPerplexity;
        this.plateauPatience = plateauPatience;
    }
}

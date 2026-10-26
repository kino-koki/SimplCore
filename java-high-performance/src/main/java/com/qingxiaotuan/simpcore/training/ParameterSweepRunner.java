package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * 参数扫描（Parameter Sweep）实验运行器
 * 
 * 批量测试不同超参数组合，评估每组合训练 50 步后的平均损失和损失下降百分比。
 * 
 * 扫描维度:
 *   LR:        [0.001, 0.003, 0.005, 0.01]
 *   Steps:     [200, 500] (cosine schedule total steps, 用于退火速率)
 *   Momentum:  [0.9, 0.95]
 *   Cosine:    [true, false] (是否使用 cosine LR schedule)
 *   GradClip:  [true, false] (是否使用梯度裁剪, max norm = 1.0)
 * 
 * 输出: Parameter_Sweep_Results.txt
 */
public class ParameterSweepRunner {

    // 模型配置（缩小版用于快速参数扫描）
    private static final int VOCAB_SIZE = 64;
    private static final int D_MODEL = 64;
    private static final int N_LAYERS = 2;
    private static final int N_HEADS = 4;
    private static final int N_BASES = 2;
    private static final int N_CHANNELS = 2;
    private static final int N_PATHS = 2;
    private static final int TOP_K = 2;
    private static final int MAX_SEQ_LEN = 32;

    // 扫描步数（每组合只跑这么多步）
    private static final int SWEEP_STEPS = 10;
    private static final float WEIGHT_DECAY = 0.01f;
    private static final float GRADIENT_CLIP_NORM = 1.0f;

    // 扫描维度
    private static final float[] LR_VALUES = {0.001f, 0.003f, 0.005f, 0.01f};
    private static final int[] STEP_VALUES = {200, 500};
    private static final float[] MOMENTUM_VALUES = {0.9f, 0.95f};
    private static final boolean[] COSINE_VALUES = {false, true};
    private static final boolean[] CLIP_VALUES = {false, true};

    // 训练数据（所有组合共享）
    private static String[] trainingData;

    // 结果条目
    private static final List<SweepResult> results = new ArrayList<>();

    // 输出
    private static PrintWriter report;
    private static final int PRINT_EVERY = 10;

    public static void main(String[] args) throws Exception {
        report = new PrintWriter(new FileWriter("Parameter_Sweep_Results.txt", false));

        log("================================================");
        log("  青小团 SimplCore 参数扫描实验报告");
        log("  Parameter Sweep for SimplCore SC Architecture");
        log("================================================");
        log("");
        log("实验时间: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        log("硬件平台: CPU (GPU检测已跳过)");
        log("底层规范: 青小团 (Qingxiaotuan)");
        log("");

        // 生成共享训练数据
        trainingData = generateTrainingData();
        log("训练数据: " + trainingData.length + " samples");
        log("每组合训练步数: " + SWEEP_STEPS);
        log("");

        // 计算总组合数
        int totalCombos = LR_VALUES.length * STEP_VALUES.length * MOMENTUM_VALUES.length
            * COSINE_VALUES.length * CLIP_VALUES.length;
        log("参数组合总数: " + totalCombos);
        log("");

        long sweepStart = System.nanoTime();
        int comboIndex = 0;

        // ── 遍历所有组合 ──
        for (float lr : LR_VALUES) {
            for (int totalSteps : STEP_VALUES) {
                for (float momentum : MOMENTUM_VALUES) {
                    for (boolean useCosine : COSINE_VALUES) {
                        for (boolean useClip : CLIP_VALUES) {
                            comboIndex++;
                            SweepResult result = runSingleExperiment(
                                comboIndex, totalCombos,
                                lr, totalSteps, momentum, useCosine, useClip);
                            results.add(result);

                            // 即时输出进度
                            System.out.printf("  [%3d/%d] LR=%.3f Steps=%d Mom=%.2f Cos=%s Clip=%s → "
                                    + "Loss: %.4f→%.4f (%.2f%%)%n",
                                comboIndex, totalCombos,
                                lr, totalSteps, momentum, useCosine, useClip,
                                result.initialLoss, result.finalLoss, result.reductionPercent);
                            report.flush();
                        }
                    }
                }
            }
        }

        double sweepTime = (System.nanoTime() - sweepStart) / 1e9;
        log("");
        log("全部实验完成! 耗时: " + String.format("%.2f", sweepTime) + "s");
        log("");

        // ── 汇总排序展示 Top-K ──
        printSummary();

        report.flush();
        report.close();

        System.out.println("\n参数扫描完成！结果已保存到: Parameter_Sweep_Results.txt");
    }

    // =========================================================
    //  单个实验
    // =========================================================
    private static SweepResult runSingleExperiment(
            int index, int total,
            float lr, int totalSteps, float momentum,
            boolean useCosine, boolean useClip) {

        // 创建新模型（使用固定种子以保证可复现性）
        SimpCoreModel model = new SimpCoreModel(VOCAB_SIZE, D_MODEL, N_LAYERS, N_BASES, N_HEADS,
            N_CHANNELS, N_PATHS, TOP_K, MAX_SEQ_LEN);
        Tokenizer tokenizer = new Tokenizer(VOCAB_SIZE);

        float clipNorm = useClip ? GRADIENT_CLIP_NORM : Float.MAX_VALUE;
        Optimizer optimizer = new Optimizer(lr, WEIGHT_DECAY, momentum, clipNorm);

        // 如果使用 cosine schedule，设置 warmup=min(20, totalSteps/10) steps
        if (useCosine) {
            int warmup = Math.min(20, Math.max(1, totalSteps / 10));
            float minLr = lr * 0.1f;
            optimizer.applyCosineSchedule(0, totalSteps, warmup, lr, minLr);
        }

        TrainingLoop loop = new TrainingLoop(model, tokenizer, optimizer);

        // 收集损失
        List<Double> losses = new ArrayList<>();

        for (int step = 1; step <= SWEEP_STEPS; step++) {
            String text = trainingData[(step - 1) % trainingData.length];
            float loss = loop.trainStep(text, 1);

            // 解耦 weight decay（简化版：仅对主要参数层）
            if (useClip) {
                applyDecoupledWeightDecay(model, optimizer);
            }

            if (!Float.isNaN(loss) && !Float.isInfinite(loss)) {
                losses.add((double) loss);
            }
        }

        // 计算指标
        SweepResult r = new SweepResult();
        r.lr = lr;
        r.totalSteps = totalSteps;
        r.momentum = momentum;
        r.useCosine = useCosine;
        r.useClip = useClip;

        if (losses.size() >= 2) {
            r.initialLoss = losses.get(0);
            r.finalLoss = losses.get(losses.size() - 1);

            // 计算平均损失（最后 10 步平均）
            int avgWindow = Math.min(10, losses.size());
            double sum = 0;
            for (int i = losses.size() - avgWindow; i < losses.size(); i++) {
                sum += losses.get(i);
            }
            r.avgLossLast10 = sum / avgWindow;

            // 总平均损失
            double totalSum = 0;
            for (double v : losses) totalSum += v;
            r.avgLoss = totalSum / losses.size();

            // 损失下降百分比
            if (r.initialLoss > 0) {
                r.reductionPercent = ((r.initialLoss - r.finalLoss) / r.initialLoss) * 100;
            }
        }

        return r;
    }

    /**
     * 解耦权重衰减
     */
    private static void applyDecoupledWeightDecay(SimpCoreModel model, Optimizer optimizer) {
        float decay = optimizer.getWeightDecay();
        if (decay <= 0f) return;
        Optimizer.applyWeightDecay(model.getEmbeddingWeights(), decay);
        Optimizer.applyWeightDecay(model.getOutputWeights(), decay);
        for (float[] w : model.getFfnWeights1()) {
            Optimizer.applyWeightDecay(w, decay);
        }
        for (float[] w : model.getFfnWeights2()) {
            Optimizer.applyWeightDecay(w, decay);
        }
    }

    // =========================================================
    //  汇总展示
    // =========================================================
    private static void printSummary() {
        // 按损失下降率降序排序
        results.sort((a, b) -> Double.compare(b.reductionPercent, a.reductionPercent));

        log("================================================");
        log("  汇总排名 (按损失下降率排序)");
        log("================================================");
        log("");
        log(String.format("%-4s %-8s %-6s %-8s %-6s %-6s %-10s %-10s %-10s %-8s",
            "Rank", "LR", "Steps", "Momentum", "Cosine", "Clip", "InitLoss", "FinalLoss",
            "AvgLoss", "Reduct%"));
        log("--------------------------------------------------------------------------------");
        report.flush();

        int rank = 0;
        for (SweepResult r : results) {
            rank++;
            log(String.format("%-4d %-8.3f %-6d %-8.2f %-6s %-6s %-10.4f %-10.4f %-10.4f %-7.2f%%",
                rank,
                r.lr, r.totalSteps, r.momentum,
                r.useCosine ? "Yes" : "No",
                r.useClip ? "Yes" : "No",
                r.initialLoss, r.finalLoss,
                r.avgLossLast10,
                r.reductionPercent));
            if (rank >= 20) break; // 只显示 Top 20
        }
        log("");

        // ── 按维度分组统计 ──
        printGroupStats("按 LR", r -> String.format("LR=%.3f", r.lr));
        printGroupStats("按 Momentum", r -> String.format("Mom=%.2f", r.momentum));
        printGroupStats("按 Cosine", r -> r.useCosine ? "Cosine=Yes" : "Cosine=No");
        printGroupStats("按 GradClip", r -> r.useClip ? "Clip=Yes" : "Clip=No");

        // 最佳组合建议
        if (!results.isEmpty()) {
            SweepResult best = results.get(0);
            log("");
            log("================================================");
            log("  最佳参数组合建议");
            log("================================================");
            log(String.format("  LR:          %.3f", best.lr));
            log(String.format("  Steps:       %d", best.totalSteps));
            log(String.format("  Momentum:    %.2f", best.momentum));
            log(String.format("  Cosine Sched: %s", best.useCosine ? "Yes" : "No"));
            log(String.format("  Grad Clip:    %s", best.useClip ? "Yes (max norm=1.0)" : "No"));
            log(String.format("  初始损失:    %.4f", best.initialLoss));
            log(String.format("  最终损失:    %.4f", best.finalLoss));
            log(String.format("  损失下降:    %.2f%%", best.reductionPercent));
            log("");
        }
    }

    private static void printGroupStats(String label, java.util.function.Function<SweepResult, String> grouper) {
        log("");
        log("  --- 按 " + label + " 分组 ---");
        log(String.format("  %-24s %-10s %-10s %-10s %-10s",
            "Group", "Count", "AvgInit", "AvgFinal", "AvgReduct%"));

        Map<String, List<SweepResult>> groups = new LinkedHashMap<>();
        for (SweepResult r : results) {
            String key = grouper.apply(r);
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(r);
        }

        for (Map.Entry<String, List<SweepResult>> entry : groups.entrySet()) {
            List<SweepResult> group = entry.getValue();
            double avgInit = 0, avgFinal = 0, avgReduct = 0;
            for (SweepResult r : group) {
                avgInit += r.initialLoss;
                avgFinal += r.finalLoss;
                avgReduct += r.reductionPercent;
            }
            avgInit /= group.size();
            avgFinal /= group.size();
            avgReduct /= group.size();
            log(String.format("  %-24s %-10d %-10.4f %-10.4f %-9.2f%%",
                entry.getKey(), group.size(), avgInit, avgFinal, avgReduct));
        }
    }

    // =========================================================
    //  数据生成
    // =========================================================
    private static String[] generateTrainingData() {
        String[] templates = {
            "hello world this is a test",
            "the cat sat on the mat",
            "i like to play with my dog",
            "the sun is shining bright today",
            "she went to the store to buy milk",
            "they are going to the park",
            "he has a red car and a blue house",
            "we love to read books every day",
            "the bird flew over the tall tree",
            "my name is alice and i am here",
            "this is a simple sentence for training",
            "the quick brown fox jumps over the lazy dog",
            "machine learning is very interesting",
            "artificial intelligence will change the world",
            "deep neural networks are powerful models",
            "natural language processing is fun",
            "the model learns from many examples",
            "training data helps improve accuracy",
            "gradient descent optimizes the parameters",
            "attention is all you need in transformers"
        };
        String[] expanded = new String[templates.length * 3];
        for (int i = 0; i < templates.length; i++) {
            expanded[i] = templates[i];
            expanded[i + templates.length] = templates[i].toUpperCase();
            expanded[i + templates.length * 2] = new StringBuilder(templates[i]).reverse().toString();
        }
        return expanded;
    }

    // =========================================================
    //  结果数据结构
    // =========================================================
    static class SweepResult {
        float lr;
        int totalSteps;
        float momentum;
        boolean useCosine;
        boolean useClip;
        double initialLoss = Double.NaN;
        double finalLoss = Double.NaN;
        double avgLoss = Double.NaN;
        double avgLossLast10 = Double.NaN;
        double reductionPercent = 0;
    }

    // =========================================================
    //  日志
    // =========================================================
    private static void log(String msg) {
        if (report != null) {
            report.println(msg);
        }
        System.out.println(msg);
        if (report != null) {
            report.flush();
        }
    }
}

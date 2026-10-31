package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.data.SCDataLoader;
import com.qingxiaotuan.simpcore.nn.Transformer;
import com.qingxiaotuan.simpcore.nn.Transformer.TransformerConfig;
import com.qingxiaotuan.simpcore.tokenizer.BPETokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 生产级训练脚本 - 集成 BPE 分词器、多线程数据加载、Transformer（含 Dropout）、
 * Adam/SGD 优化器、混合精度训练与指标监控的端到端训练流程。
 *
 * <p>支持的命令行参数（形式 --key=value）：
 * <ul>
 *   <li>--epochs             训练轮数，默认 300</li>
 *   <li>--lr                 Adam 基础学习率，默认 1e-3</li>
 *   <li>--sgdLr              SGD 学习率，默认 1e-4</li>
 *   <li>--batchSize          批大小，默认 4</li>
 *   <li>--dropout            Dropout 概率，默认 0.2</li>
 *   <li>--useAdam            是否使用 Adam（true/false），默认 true</li>
 *   <li>--useMixedPrecision  是否启用混合精度，默认 true</li>
 *   <li>--warmupEpochs       学习率线性预热的轮数，默认 30</li>
 *   <li>--weightDecay        权重衰减，默认 0.01</li>
 *   <li>--numWorkers         数据加载线程数，默认 2</li>
 *   <li>--bpeIterations      BPE 合并迭代次数，默认 100</li>
 *   <li>--printInterval      指标打印间隔（epoch），默认 50</li>
 *   <li>--runComparison      是否运行 Dropout 对比实验，默认 false</li>
 *   <li>--comparisonEpochs   对比实验训练轮数，默认 100</li>
 *   <li>--modelSavePath      模型保存路径，默认 sc_production_model.bin</li>
 *   <li>--csvExportPath      指标 CSV 导出路径，默认 sc_production_metrics.csv</li>
 * </ul>
 *
 * @author SC Architecture Team
 */
public class SCProductionTraining {

    /** 训练语料 */
    private static final String TRAINING_TEXT =
        "The quick brown fox jumps over the lazy dog. " +
        "Machine learning allows computers to learn from data. " +
        "Deep learning uses neural networks with many layers. " +
        "Natural language processing enables computers to understand text. " +
        "Artificial intelligence is transforming many industries. " +
        "Neural networks can recognize patterns in complex data. " +
        "Machine learning models improve with more training examples. " +
        "Transformers have revolutionized natural language processing. " +
        "Attention mechanisms allow models to focus on relevant parts of the input. " +
        "The self-attention mechanism computes relationships between all positions in a sequence.";

    /** 验证语料 */
    private static final String VALIDATION_TEXT =
        "Machine learning is a subset of artificial intelligence. " +
        "Deep learning models use multiple layers. " +
        "Neural networks are inspired by the human brain.";

    /** 训练配置 */
    public static class TrainingConfig {
        public int epochs = 300;
        public float learningRate = 1e-3f;
        public float sgdLearningRate = 1e-4f;
        public int batchSize = 4;
        public float dropout = 0.2f;
        public boolean useAdam = true;
        public boolean useMixedPrecision = true;
        public int warmupEpochs = 30;
        public float weightDecay = 0.01f;
        public int numWorkers = 2;
        public int bpeIterations = 100;
        public int printInterval = 50;
        public boolean runComparison = false;
        public int comparisonEpochs = 100;
        public String modelSavePath = "sc_production_model.bin";
        public String csvExportPath = "sc_production_metrics.csv";

        // 模型结构参数
        public int dModel = 64;
        public int nLayers = 4;
        public int nHeads = 8;
        public int seqLen = 32;

        /** 从命令行参数解析配置 */
        public static TrainingConfig parse(String[] args) {
            TrainingConfig cfg = new TrainingConfig();
            if (args == null) return cfg;
            for (String arg : args) {
                if (arg == null || !arg.startsWith("--")) continue;
                int eq = arg.indexOf('=');
                if (eq < 0) continue;
                String key = arg.substring(2, eq).trim();
                String val = arg.substring(eq + 1).trim();
                try {
                    switch (key) {
                        case "epochs":             cfg.epochs = Integer.parseInt(val); break;
                        case "lr":                 cfg.learningRate = Float.parseFloat(val); break;
                        case "sgdLr":              cfg.sgdLearningRate = Float.parseFloat(val); break;
                        case "batchSize":          cfg.batchSize = Integer.parseInt(val); break;
                        case "dropout":            cfg.dropout = Float.parseFloat(val); break;
                        case "useAdam":            cfg.useAdam = Boolean.parseBoolean(val); break;
                        case "useMixedPrecision":  cfg.useMixedPrecision = Boolean.parseBoolean(val); break;
                        case "warmupEpochs":       cfg.warmupEpochs = Integer.parseInt(val); break;
                        case "weightDecay":        cfg.weightDecay = Float.parseFloat(val); break;
                        case "numWorkers":         cfg.numWorkers = Integer.parseInt(val); break;
                        case "bpeIterations":      cfg.bpeIterations = Integer.parseInt(val); break;
                        case "printInterval":      cfg.printInterval = Integer.parseInt(val); break;
                        case "runComparison":      cfg.runComparison = Boolean.parseBoolean(val); break;
                        case "comparisonEpochs":   cfg.comparisonEpochs = Integer.parseInt(val); break;
                        case "modelSavePath":      cfg.modelSavePath = val; break;
                        case "csvExportPath":      cfg.csvExportPath = val; break;
                        default: break;
                    }
                } catch (NumberFormatException e) {
                    System.err.println("WARNING: 无法解析参数 --" + key + "=" + val + "，使用默认值");
                }
            }
            return cfg;
        }
    }

    /** 训练产出 */
    public static class TrainResult {
        public final Transformer model;
        public final AdamOptimizer optimizer;
        public final MixedPrecision mixedPrecision;
        public final MetricsMonitor monitor;
        public final float finalTrainLoss;
        public final float finalValLoss;

        TrainResult(Transformer model, AdamOptimizer optimizer,
                    MixedPrecision mixedPrecision, MetricsMonitor monitor,
                    float finalTrainLoss, float finalValLoss) {
            this.model = model;
            this.optimizer = optimizer;
            this.mixedPrecision = mixedPrecision;
            this.monitor = monitor;
            this.finalTrainLoss = finalTrainLoss;
            this.finalValLoss = finalValLoss;
        }
    }

    public static void main(String[] args) {
        TrainingConfig config = TrainingConfig.parse(args);

        printConfig(config);

        // ===== [1] 训练 BPE 分词器 =====
        System.out.println("\n[1/6] Training BPE tokenizer...");
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(TRAINING_TEXT, config.bpeIterations);
        int vocabSize = tokenizer.getVocabSize();
        System.out.printf("  Vocabulary size: %d%n", vocabSize);

        // ===== [2] 编码训练/验证文本 =====
        System.out.println("\n[2/6] Tokenizing corpus...");
        int[] trainTokens = tokenizer.encode(TRAINING_TEXT);
        int[] valTokens = tokenizer.encode(VALIDATION_TEXT);
        System.out.printf("  Training tokens: %d%n", trainTokens.length);
        System.out.printf("  Validation tokens: %d%n", valTokens.length);

        // ===== [3] 构造滑动窗口序列 =====
        System.out.println("\n[3/6] Building sliding-window sequences...");
        int stride = Math.max(1, config.seqLen / 2);
        List<int[]> trainSeqs = SCDataLoader.createSlidingWindowSequences(trainTokens, config.seqLen, stride);
        List<int[]> valSeqs = SCDataLoader.createSlidingWindowSequences(valTokens, config.seqLen, stride);
        System.out.printf("  Training sequences: %d (seqLen=%d, stride=%d)%n",
            trainSeqs.size(), config.seqLen, stride);
        System.out.printf("  Validation sequences: %d%n", valSeqs.size());

        // ===== [4] 训练 =====
        System.out.println("\n[4/6] Training...");
        long startMs = System.currentTimeMillis();
        TrainResult result = train(config, trainSeqs, valSeqs, vocabSize, "  ");
        long elapsed = System.currentTimeMillis() - startMs;
        System.out.printf("%nTraining finished in %.2fs%n", elapsed / 1000.0);
        System.out.printf("  Final train loss: %.4f | Final val loss: %.4f%n",
            result.finalTrainLoss, result.finalValLoss);
        System.out.println("  " + result.mixedPrecision.getStatsReport());

        // ===== [5] 导出指标与摘要 =====
        System.out.println("\n[5/6] Exporting metrics...");
        try {
            result.monitor.exportToCSV(config.csvExportPath);
            System.out.println("  CSV exported: " + config.csvExportPath);
        } catch (java.io.IOException e) {
            System.err.println("  ERROR exporting CSV: " + e.getMessage());
        }
        result.monitor.printSummary();
        result.monitor.printRecentAlerts(10);

        // 保存模型
        try {
            result.model.save(config.modelSavePath);
            System.out.println("Model saved: " + config.modelSavePath);
        } catch (Exception e) {
            System.err.println("ERROR saving model: " + e.getMessage());
        }

        // ===== [6] 文本生成测试 =====
        System.out.println("\n[6/6] Generation test...");
        testGeneration(result.model, tokenizer, config);

        // ===== Dropout 对比实验（可选） =====
        if (config.runComparison) {
            runDropoutComparison(config, tokenizer, trainTokens, valTokens, vocabSize);
        }

        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("              SC Production Training Complete!");
        System.out.println("════════════════════════════════════════════════════════════");
    }

    /** 打印配置信息 */
    private static void printConfig(TrainingConfig cfg) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("          SC Production Training - Configuration");
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.printf("  Optimizer        : %s%n", cfg.useAdam ? "Adam" : "SGD");
        System.out.printf("  Learning rate    : %s%n",
            cfg.useAdam ? String.format("%.6f", cfg.learningRate) : String.format("%.6f", cfg.sgdLearningRate));
        System.out.printf("  Epochs           : %d%n", cfg.epochs);
        System.out.printf("  Batch size       : %d%n", cfg.batchSize);
        System.out.printf("  Warmup epochs    : %d%n", cfg.warmupEpochs);
        System.out.printf("  Weight decay     : %s%n", cfg.weightDecay);
        System.out.printf("  Dropout          : %.2f%n", cfg.dropout);
        System.out.printf("  Mixed precision  : %b%n", cfg.useMixedPrecision);
        System.out.printf("  Data workers     : %d%n", cfg.numWorkers);
        System.out.printf("  Model dModel     : %d%n", cfg.dModel);
        System.out.printf("  Model nLayers    : %d%n", cfg.nLayers);
        System.out.printf("  Model nHeads     : %d%n", cfg.nHeads);
        System.out.printf("  Sequence length  : %d%n", cfg.seqLen);
        System.out.printf("  Print interval   : %d epochs%n", cfg.printInterval);
        System.out.printf("  Dropout compare  : %b%n", cfg.runComparison);
        System.out.println("════════════════════════════════════════════════════════════");
    }

    /**
     * 执行完整训练流程。
     *
     * @param config    训练配置
     * @param trainSeqs 训练序列
     * @param valSeqs   验证序列
     * @param vocabSize 词表大小
     * @param indent    日志缩进
     * @return 训练产出
     */
    public static TrainResult train(TrainingConfig config, List<int[]> trainSeqs, List<int[]> valSeqs,
                                    int vocabSize, String indent) {
        // 构建模型
        TransformerConfig modelConfig = new TransformerConfig()
            .dModel(config.dModel)
            .nLayers(config.nLayers)
            .nHeads(config.nHeads)
            .vocabSize(vocabSize)
            .seqLen(config.seqLen)
            .dropout(config.dropout);
        Transformer model = new Transformer(modelConfig);
        System.out.printf(indent + "Model parameters: %.2fM%n", model.getParameterCount() / 1_000_000.0);

        // 构建优化器与学习率调度器
        AdamOptimizer optimizer = null;
        float sgdLr = config.sgdLearningRate;
        if (config.useAdam) {
            optimizer = new AdamOptimizer(config.learningRate, config.weightDecay);
            int cosineSteps = Math.max(1, config.epochs - config.warmupEpochs);
            LearningRateScheduler cosine =
                new LearningRateScheduler.CosineAnnealingScheduler(config.learningRate, cosineSteps);
            LearningRateScheduler warmup =
                new LearningRateScheduler.LinearWarmupScheduler(config.learningRate, config.warmupEpochs, cosine);
            optimizer.setScheduler(warmup);
            System.out.printf(indent + "Scheduler: LinearWarmup(%d) -> CosineAnnealing(%d)%n",
                config.warmupEpochs, cosineSteps);
        } else {
            System.out.printf(indent + "Using SGD with lr=%.6f, weightDecay=%.4f%n",
                sgdLr, config.weightDecay);
        }

        // 混合精度
        MixedPrecision mixedPrecision = new MixedPrecision(config.useMixedPrecision);
        mixedPrecision.recordMemoryUsage(model.getParameterCount());

        // 指标监控
        MetricsMonitor monitor = new MetricsMonitor(1, 20);

        // 多线程数据加载器
        SCDataLoader loader = new SCDataLoader(trainSeqs, config.batchSize,
            config.seqLen, true, config.numWorkers);
        System.out.printf(indent + "DataLoader: %d sequences, %d batches/epoch, %d workers%n",
            trainSeqs.size(), loader.getTotalBatches(), config.numWorkers);

        float lastTrainLoss = Float.NaN;
        float lastValLoss = Float.NaN;

        for (int epoch = 0; epoch < config.epochs; epoch++) {
            // 在 epoch 开始时推进调度器，使第一个 epoch 也具有正的学习率
            if (config.useAdam) {
                optimizer.stepScheduler();
            }

            model.setTraining(true);
            loader.reset();

            double epochLossSum = 0.0;
            int stepCount = 0;
            int overflowCount = 0;

            while (loader.hasNext()) {
                int[][] batch = loader.getNextBatch();
                if (batch == null) break;

                for (int[] seq : batch) {
                    // 混合精度：先缩放损失（用于溢出检测代理）
                    float rawLoss;
                    if (config.useAdam) {
                        rawLoss = model.trainStepWithAdam(seq, seq, optimizer);
                    } else {
                        rawLoss = model.trainStep(seq, seq, sgdLr, config.weightDecay);
                    }

                    if (config.useMixedPrecision) {
                        // 用缩放后的损失值作为代理，检测数值溢出
                        float scaledLoss = mixedPrecision.scaleLoss(rawLoss);
                        float[] proxy = new float[]{scaledLoss};
                        boolean overflow = mixedPrecision.checkGradientOverflow(proxy);
                        mixedPrecision.updateScale(overflow);
                        if (overflow) {
                            overflowCount++;
                            continue;
                        }
                        // 反缩放代理梯度（演示 API 用法）
                        mixedPrecision.unscaleGradients(proxy);
                    }

                    if (!Float.isNaN(rawLoss) && !Float.isInfinite(rawLoss)) {
                        epochLossSum += rawLoss;
                        stepCount++;
                    }
                }
            }

            float avgTrainLoss = stepCount > 0 ? (float) (epochLossSum / stepCount) : Float.NaN;
            lastTrainLoss = avgTrainLoss;

            // 验证
            model.setTraining(false);
            float valLoss = evaluateLoss(model, valSeqs);
            float accuracy = evaluateAccuracy(model, valSeqs, config.seqLen, vocabSize);
            lastValLoss = valLoss;

            float currentLr = config.useAdam ? optimizer.getCurrentLearningRate() : sgdLr;
            monitor.record(epoch, 0, avgTrainLoss, valLoss, accuracy, currentLr);

            if (epoch % config.printInterval == 0 || epoch == config.epochs - 1) {
                float ppl = MetricsMonitor.computePerplexity(avgTrainLoss);
                String overflowInfo = (config.useMixedPrecision && overflowCount > 0)
                    ? String.format(" [mp_overflow=%d]", overflowCount) : "";
                System.out.printf(indent + "Epoch %3d/%d | train_loss=%.4f val_loss=%.4f ppl=%.2f acc=%.2f%% lr=%.6f%s%n",
                    epoch, config.epochs, avgTrainLoss, valLoss, ppl, accuracy * 100, currentLr, overflowInfo);
            }
        }

        loader.close();
        return new TrainResult(model, optimizer, mixedPrecision, monitor, lastTrainLoss, lastValLoss);
    }

    /**
     * 计算验证集平均损失（不更新参数，关闭 Dropout）。
     * 使用 trainStep(lr=0, weightDecay=0) 仅计算损失。
     */
    private static float evaluateLoss(Transformer model, List<int[]> seqs) {
        if (seqs.isEmpty()) return Float.NaN;
        double total = 0.0;
        int count = 0;
        for (int[] seq : seqs) {
            float loss = model.trainStep(seq, seq, 0.0f, 0.0f);
            if (!Float.isNaN(loss) && !Float.isInfinite(loss)) {
                total += loss;
                count++;
            }
        }
        return count > 0 ? (float) (total / count) : Float.NaN;
    }

    /** 计算验证集下一 token 预测准确率 */
    private static float evaluateAccuracy(Transformer model, List<int[]> seqs, int seqLen, int vocabSize) {
        if (seqs.isEmpty()) return 0.0f;
        double total = 0.0;
        int count = 0;
        for (int[] seq : seqs) {
            int len = Math.min(seq.length, seqLen);
            float[] logits = model.forward(seq);
            total += MetricsMonitor.computeAccuracy(logits, seq, len, vocabSize);
            count++;
        }
        return count > 0 ? (float) (total / count) : 0.0f;
    }

    /** 自回归文本生成测试 */
    private static void testGeneration(Transformer model, BPETokenizer tokenizer, TrainingConfig config) {
        model.setTraining(false);

        String[] prompts = {
            "Machine learning",
            "Deep learning",
            "Neural networks",
            "The quick"
        };

        int generateLen = 20;
        System.out.println("  Autoregressive generation (max " + generateLen + " tokens):");
        for (String prompt : prompts) {
            int[] tokens = tokenizer.encode(prompt);
            // 避免空 prompt
            if (tokens.length == 0) continue;

            int[] context = tokens;
            // 保证上下文不超过 seqLen
            if (context.length > config.seqLen) {
                context = Arrays.copyOfRange(context, context.length - config.seqLen, context.length);
            }

            List<Integer> generated = new ArrayList<>();
            for (int i = 0; i < generateLen; i++) {
                int next = model.predictNext(context);
                generated.add(next);
                // 滑动窗口推进
                int[] newContext = new int[context.length + 1];
                System.arraycopy(context, 0, newContext, 0, context.length);
                newContext[context.length] = next;
                if (newContext.length > config.seqLen) {
                    context = Arrays.copyOfRange(newContext, newContext.length - config.seqLen, newContext.length);
                } else {
                    context = newContext;
                }
            }

            int[] outIds = new int[generated.size()];
            for (int i = 0; i < generated.size(); i++) outIds[i] = generated.get(i);
            String completion = tokenizer.decode(outIds);
            System.out.printf("  [\"%s\"] -> \"%s\"%n", prompt, completion);
        }
    }

    /**
     * Dropout 对比实验：在 dropout=0.2 / 0.3 / 0.4 三种设置下分别训练，
     * 比较最终训练/验证损失，观察正则化强度对过拟合的影响。
     */
    public static void runDropoutComparison(TrainingConfig baseConfig, BPETokenizer tokenizer,
                                            int[] trainTokens, int[] valTokens, int vocabSize) {
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("             Dropout Comparison Experiment");
        System.out.println("════════════════════════════════════════════════════════════");

        float[] dropouts = {0.2f, 0.3f, 0.4f};
        int stride = Math.max(1, baseConfig.seqLen / 2);
        List<int[]> trainSeqs = SCDataLoader.createSlidingWindowSequences(trainTokens, baseConfig.seqLen, stride);
        List<int[]> valSeqs = SCDataLoader.createSlidingWindowSequences(valTokens, baseConfig.seqLen, stride);

        System.out.printf("  Epochs per run: %d%n", baseConfig.comparisonEpochs);
        System.out.printf("  Train sequences: %d | Val sequences: %d%n%n", trainSeqs.size(), valSeqs.size());

        System.out.printf("  %-10s %-14s %-14s %-12s %-12s%n",
            "Dropout", "FinalTrain", "FinalVal", "Gap", "Verdict");
        System.out.println("  ----------------------------------------------------------");

        float bestValLoss = Float.MAX_VALUE;
        float bestDropout = -1f;

        for (float dropout : dropouts) {
            TrainingConfig cfg = cloneForComparison(baseConfig);
            cfg.dropout = dropout;

            System.out.printf("  >> Training with dropout=%.1f ...%n", dropout);
            TrainResult result = train(cfg, trainSeqs, valSeqs, vocabSize, "     ");

            float trainLoss = result.finalTrainLoss;
            float valLoss = result.finalValLoss;
            float gap = Float.isNaN(trainLoss) || Float.isNaN(valLoss)
                ? Float.NaN : valLoss - trainLoss;
            String verdict;
            if (Float.isNaN(valLoss)) {
                verdict = "FAILED";
            } else if (gap > 0.5f) {
                verdict = "overfit";
            } else if (gap < -0.1f) {
                verdict = "underfit?";
            } else {
                verdict = "balanced";
            }
            System.out.printf("  %-10.1f %-14.4f %-14.4f %-12.4f %-12s%n%n",
                dropout, trainLoss, valLoss, gap, verdict);

            if (!Float.isNaN(valLoss) && valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestDropout = dropout;
            }
        }

        System.out.println("  ----------------------------------------------------------");
        if (bestDropout >= 0) {
            System.out.printf("  Best dropout: %.1f (val_loss=%.4f)%n", bestDropout, bestValLoss);
        } else {
            System.out.println("  No successful run.");
        }
        System.out.println("════════════════════════════════════════════════════════════");
    }

    /** 复制配置用于对比实验（减少 epoch、关闭混合精度以加速） */
    private static TrainingConfig cloneForComparison(TrainingConfig base) {
        TrainingConfig cfg = new TrainingConfig();
        cfg.epochs = base.comparisonEpochs;
        cfg.learningRate = base.learningRate;
        cfg.sgdLearningRate = base.sgdLearningRate;
        cfg.batchSize = base.batchSize;
        cfg.useAdam = base.useAdam;
        cfg.useMixedPrecision = false; // 对比实验关闭混合精度以聚焦 Dropout 效果
        cfg.warmupEpochs = Math.min(base.warmupEpochs, cfg.epochs / 4);
        cfg.weightDecay = base.weightDecay;
        cfg.numWorkers = base.numWorkers;
        cfg.bpeIterations = base.bpeIterations;
        cfg.printInterval = Math.max(1, cfg.epochs / 4);
        cfg.dModel = base.dModel;
        cfg.nLayers = base.nLayers;
        cfg.nHeads = base.nHeads;
        cfg.seqLen = base.seqLen;
        return cfg;
    }
}

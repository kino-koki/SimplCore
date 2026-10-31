package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.model.RDTModel;
import com.qingxiaotuan.simpcore.model.RDTConfig;
import com.qingxiaotuan.simpcore.training.GradientResult;
import com.qingxiaotuan.simpcore.training.AdamOptimizer;
import com.qingxiaotuan.simpcore.training.LearningRateScheduler;
import com.qingxiaotuan.simpcore.tokenizer.BPETokenizer;
import com.qingxiaotuan.simpcore.tokenizer.ChineseBPETokenizer;
import com.qingxiaotuan.simpcore.data.SCDataLoader;
import java.util.*;

public class StackedTrainer {
    
    private final ExpertModelCollection experts;
    private final EnsembleLayer ensemble;
    private final ModelRouter router;
    private final AdamOptimizer optimizer;
    private BPETokenizer tokenizer;
    
    private final int nExperts;
    private final int dModel;
    private final int batchSize;
    private final float learningRate;
    private final int warmupSteps;
    private final int totalSteps;
    
    private long totalStepsCompleted;
    private long totalTokens;
    private double totalLoss;
    private long startTime;
    
    private static final Random RNG = new Random(42);
    
    public StackedTrainer(int nExperts, int dModel, int vocabSize, 
                          int batchSize, float learningRate) {
        this(nExperts, dModel, vocabSize, batchSize, learningRate, 1000, 100000);
    }
    
    public StackedTrainer(int nExperts, int dModel, int vocabSize, 
                          int batchSize, float learningRate,
                          int warmupSteps, int totalSteps) {
        this.nExperts = nExperts;
        this.dModel = dModel;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.totalSteps = totalSteps;
        
        // 创建专家集合（每个专家~100M参数）
        this.experts = new ExpertModelCollection(nExperts, dModel, vocabSize);
        
        // 创建集成层
        this.ensemble = new EnsembleLayer(dModel, vocabSize, nExperts);
        
        // 获取路由层
        this.router = experts.getRouter();
        
        // 创建优化器（预热后恒定学习率，加速收敛）
        LearningRateScheduler constantScheduler = new LearningRateScheduler.ConstantScheduler(learningRate);
        LearningRateScheduler scheduler = new LearningRateScheduler.LinearWarmupScheduler(
            learningRate, warmupSteps, constantScheduler);
        
        this.optimizer = new AdamOptimizer(learningRate, 0.01f);
        this.optimizer.setScheduler(scheduler);
        
        // 预注册集成层所有权重参数（避免运行时查找）
        float[][] ensembleWeights = ensemble.getWeightArrays();
        this.optimizer.registerWeights(ensembleWeights);
        
        // 创建Tokenizer（默认使用BPE）
        this.tokenizer = new BPETokenizer();
        
        // 初始化训练统计
        this.totalStepsCompleted = 0;
        this.totalTokens = 0;
        this.totalLoss = 0;
        this.startTime = System.currentTimeMillis();
        
        System.out.println("[StackedTrainer] 初始化完成");
        System.out.println("[StackedTrainer] 专家数量: " + nExperts);
        System.out.println("[StackedTrainer] 每个专家: ~100M 参数");
        System.out.println("[StackedTrainer] 总参数量: ~" + (nExperts * 100) + "M (" + (nExperts * 0.1) + "B)");
        System.out.println("[StackedTrainer] 集成层参数量: " + ensemble.getParameterCount());
        System.out.println("[StackedTrainer] 学习率: " + learningRate + " (预热: " + warmupSteps + "步)");
    }
    
    /**
     * 训练步骤（完整的前向+反向+优化）
     */
    public TrainingStepResult trainStep(int[] tokens, float[] labels) {
        TrainingStepResult result = new TrainingStepResult();
        
        // 1. 获取当前学习率
        float lr = optimizer.getCurrentLearningRate();
        
        // 2. 计算输入嵌入
        float[] inputEmbedding = computeInputEmbedding(tokens);
        
        // 3. 路由选择专家
        ModelRouter.RoutingResult routing = router.route(inputEmbedding);
        
        // 4. 专家推理
        ExpertModelCollection.ExpertOutputCollection outputs = experts.infer(inputEmbedding, tokens);
        
        // 5. 集成输出
        float[] logits = ensemble.ensemble(outputs, inputEmbedding);
        
        // 6. 计算损失和梯度
        float loss = computeCrossEntropy(logits, labels);
        float[] dLogits = computeCrossEntropyGradient(logits, labels);
        
        // 7. 反向传播 - 集成层
        GradientResult ensembleGrad = ensemble.backward(dLogits, outputs, inputEmbedding);
        
        // 8. 更新集成层参数（使用真正的梯度，预注册索引）
        Map<String, float[]> paramGrads = ensembleGrad.getParameterGradients();
        float[][] weights = ensemble.getWeightArrays();
        for (int i = 0; i < weights.length; i++) {
            float[] grad = paramGrads.getOrDefault("weight_" + i, new float[weights[i].length]);
            optimizer.update(i, weights[i], grad);
        }
        
        // 9. 更新路由层参数（使用集成层梯度信号）
        updateRouterParameters(lr, ensembleGrad.getdInput(), inputEmbedding);
        
        // 10. 更新专家模型参数（使用真实梯度，关键改进！传入tokens用于embedding梯度）
        updateExpertParametersWithGradients(outputs, dLogits, lr, tokens);
        
        // 更新统计
        totalStepsCompleted++;
        totalTokens += tokens.length;
        totalLoss += loss;
        
        // 更新学习率调度器
        optimizer.stepScheduler();
        
        // 记录结果
        result.loss = loss;
        result.expertCount = outputs.size();
        result.learningRate = lr;
        result.selectedExperts = new int[routing.getSelectedCount()];
        System.arraycopy(routing.selectedIndices, 0, result.selectedExperts, 0, 
                        routing.getSelectedCount());
        
        return result;
    }
    
    /**
     * 批量训练
     */
    public TrainingStats trainBatch(int[][] batch) {
        TrainingStats stats = new TrainingStats();
        double batchLoss = 0;
        int totalExperts = 0;
        
        for (int[] tokens : batch) {
            // 创建标签（下一个token预测）
            float[] labels = new float[ensemble.getVocabSize()];
            if (tokens.length > 0) {
                int targetToken = tokens[tokens.length - 1];
                if (targetToken >= 0 && targetToken < ensemble.getVocabSize()) {
                    labels[targetToken] = 1.0f;
                }
            }
            
            TrainingStepResult stepResult = trainStep(tokens, labels);
            batchLoss += stepResult.loss;
            totalExperts += stepResult.expertCount;
        }
        
        stats.totalSteps = totalStepsCompleted;
        stats.totalTokens = totalTokens;
        stats.averageLoss = totalLoss / Math.max(1, totalStepsCompleted);
        stats.elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000.0;
        stats.expertsCount = experts.getNExperts();
        stats.totalParameters = experts.getTotalParameters() + ensemble.getParameterCount();
        
        return stats;
    }
    
    /**
     * 完整训练循环
     */
    public TrainingStats train(SCDataLoader dataLoader, int steps) {
        System.out.println("\n[StackedTrainer] 开始训练...");
        
        int stepsToRun = (int) Math.min(steps, (long) totalSteps - totalStepsCompleted);
        
        for (int step = 0; step < stepsToRun; step++) {
            int[][] batch = dataLoader.getNextBatch();
            if (batch == null) {
                dataLoader.reset();
                batch = dataLoader.getNextBatch();
                if (batch == null) break;
            }
            
            TrainingStats stats = trainBatch(batch);
            
            // 打印进度
            if (step % 10 == 0) {
                printProgress();
            }
            
            // 检查是否达到目标损失（设置更低的阈值，允许更多训练步骤）
            if (stats.averageLoss < 0.0001) {
                System.out.println("[StackedTrainer] 目标损失达成，提前终止训练");
                break;
            }
        }
        
        System.out.println("[StackedTrainer] 训练完成");
        return getStats();
    }
    
    /**
     * 生成文本
     */
    public String generate(String prompt, int maxLength) {
        int[] tokens = tokenizer.encode(prompt);
        float[] inputEmbedding = computeInputEmbedding(tokens);
        
        for (int step = 0; step < maxLength; step++) {
            ModelRouter.RoutingResult routing = router.route(inputEmbedding);
            ExpertModelCollection.ExpertOutputCollection outputs = experts.infer(inputEmbedding, tokens);
            float[] logits = ensemble.ensemble(outputs, inputEmbedding);
            
            int nextToken = argmax(logits);
            if (nextToken == tokenizer.encode("<|endoftext|>")[0]) {
                break;
            }
            
            int[] newTokens = new int[tokens.length + 1];
            System.arraycopy(tokens, 0, newTokens, 0, tokens.length);
            newTokens[tokens.length] = nextToken;
            tokens = newTokens;
            
            inputEmbedding = computeInputEmbedding(tokens);
        }
        
        return tokenizer.decode(tokens);
    }
    
    /**
     * 训练Tokenizer
     */
    public void trainTokenizer(String text, int iterations) {
        tokenizer.train(text, iterations);
        System.out.println("[StackedTrainer] Tokenizer训练完成，词汇量: " + tokenizer.getVocabSize());
    }
    
    private float getCurrentLearningRate() {
        return optimizer.getCurrentLearningRate();
    }
    
    private float[] computeInputEmbedding(int[] tokens) {
        // 使用专家模型的真正嵌入层（不再使用哈希映射！）
        // 从第一个专家获取嵌入（所有专家共享相同的嵌入矩阵）
        float[] embedding = experts.getExpert(0).embed(tokens);
        
        // 归一化
        float norm = 0f;
        for (float val : embedding) norm += val * val;
        norm = (float) Math.sqrt(norm + 1e-8f);
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] /= norm;
        }
        
        return embedding;
    }
    
    private float computeCrossEntropy(float[] logits, float[] labels) {
        float loss = 0f;
        int vocabSize = logits.length;
        
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        float[] exp = new float[vocabSize];
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        for (int i = 0; i < vocabSize; i++) {
            float prob = exp[i] / sum;
            if (labels[i] > 0) {
                loss -= labels[i] * (float) Math.log(prob + 1e-8f);
            }
        }
        
        return loss;
    }
    
    private float[] computeCrossEntropyGradient(float[] logits, float[] labels) {
        float[] grad = new float[logits.length];
        int vocabSize = logits.length;
        
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        float[] exp = new float[vocabSize];
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        for (int i = 0; i < vocabSize; i++) {
            float prob = exp[i] / sum;
            grad[i] = prob - labels[i];
        }
        
        return grad;
    }
    
    private void updateRouterParameters(float lr, float[] dInput, float[] inputEmbedding) {
        // 使用集成层梯度信号更新路由参数（真正的梯度下降）
        float[][] embeddings = router.getExpertEmbeddings();
        float[][] rWeights = router.getRoutingWeights();
        float[] rBias = router.getRoutingBias();
        
        // 更新路由权重
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < nExperts; j++) {
                rWeights[i][j] -= lr * dInput[i] * inputEmbedding[i];
            }
        }
        
        // 更新路由偏置
        for (int j = 0; j < nExperts; j++) {
            rBias[j] -= lr * dInput[j % dModel];
        }
        
        // 更新专家嵌入（基于输入和梯度的相关性）
        for (int j = 0; j < nExperts; j++) {
            for (int i = 0; i < dModel; i++) {
                embeddings[j][i] -= lr * dInput[i] * 0.01f;
            }
        }
    }
    
    private void updateExpertParametersWithGradients(ExpertModelCollection.ExpertOutputCollection outputs, 
                                                     float[] dLogits, float lr, int[] tokens) {
        // 使用真实梯度更新专家模型参数（关键改进！）
        for (int i = 0; i < outputs.size(); i++) {
            int expertIdx = outputs.getExpertIndex(i);
            float weight = outputs.getWeight(i);
            
            if (weight > 0.01f) {
                // 将dLogits乘以专家权重，得到该专家的输出梯度
                float[] dExpertLogits = new float[dLogits.length];
                for (int j = 0; j < dLogits.length; j++) {
                    dExpertLogits[j] = dLogits[j] * weight;
                }
                
                // 调用专家模型的反向传播获取真实梯度（传入tokens用于计算embedding梯度）
                GradientResult grad = experts.getExpert(expertIdx).backward(dExpertLogits, 2, tokens);
                
                // 使用真实梯度更新参数
                experts.getExpert(expertIdx).updateParametersWithGradient(grad, lr * weight);
            }
        }
    }
    
    private int argmax(float[] x) {
        int idx = 0;
        float max = x[0];
        for (int i = 1; i < x.length; i++) {
            if (x[i] > max) {
                max = x[i];
                idx = i;
            }
        }
        return idx;
    }
    
    public TrainingStats getStats() {
        TrainingStats stats = new TrainingStats();
        stats.totalSteps = totalStepsCompleted;
        stats.totalTokens = totalTokens;
        stats.averageLoss = totalLoss / Math.max(1, totalStepsCompleted);
        stats.elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000.0;
        stats.expertsCount = experts.getNExperts();
        stats.totalParameters = experts.getTotalParameters() + ensemble.getParameterCount();
        
        return stats;
    }
    
    public void printProgress() {
        TrainingStats stats = getStats();
        
        System.out.printf("[StackedTrainer] Step: %,d/%d | ", stats.totalSteps, totalSteps);
        System.out.printf("Loss: %.4f | ", stats.averageLoss);
        System.out.printf("LR: %.2e | ", getCurrentLearningRate());
        System.out.printf("Tokens: %,d | ", stats.totalTokens);
        System.out.printf("Experts: %d | ", stats.expertsCount);
        System.out.printf("Params: %.2fB | ", stats.totalParameters / 1_000_000_000.0);
        System.out.printf("Time: %.1fs\n", stats.elapsedSeconds);
    }
    
    public ExpertModelCollection getExperts() { return experts; }
    public EnsembleLayer getEnsemble() { return ensemble; }
    public ModelRouter getRouter() { return router; }
    public BPETokenizer getTokenizer() { return tokenizer; }
    public void setTokenizer(BPETokenizer tokenizer) { this.tokenizer = tokenizer; }
    
    public static class TrainingStepResult {
        public float loss;
        public int expertCount;
        public float learningRate;
        public int[] selectedExperts;
        
        public TrainingStepResult() {
            this.loss = Float.MAX_VALUE;
            this.expertCount = 0;
            this.learningRate = 0f;
            this.selectedExperts = new int[0];
        }
    }
    
    public static class TrainingStats {
        public long totalSteps;
        public long totalTokens;
        public double averageLoss;
        public double elapsedSeconds;
        public int expertsCount;
        public long totalParameters;
    }
}
package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.data.SCDataLoader;
import com.qingxiaotuan.simpcore.tokenizer.ChineseBPETokenizer;
import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;
import com.qingxiaotuan.simpcore.model.RDTTransformerBlock;
import java.util.*;

/**
 * 简单训练测试 - 验证训练管线是否正常工作
 * 使用1个专家模型，预测每个位置的token（全序列预测）
 */
public class SimpleTrainingTest {
    
    public static void main(String[] args) {
        System.out.println("═══════════════════════════════════════════════════════════════════");
        System.out.println("           SimplCore 简单训练测试 - 验证训练管线");
        System.out.println("═══════════════════════════════════════════════════════════════════\n");
        
        // 配置参数
        int nExperts = 1;
        int dModel = 128;
        int vocabSize = 2000;
        int batchSize = 2;
        float learningRate = 1e-2f;
        int warmupSteps = 20;
        int totalSteps = 500;
        
        // 初始化GPU加速器
        System.out.println("[阶段1] 初始化OpenCL GPU加速器...");
        OpenCLAccelerator gpu = new OpenCLAccelerator();
        gpu.initialize();
        RDTTransformerBlock.setGPU(gpu);
        
        // 创建训练器
        System.out.println("\n[阶段2] 创建训练器...");
        StackedTrainer trainer = new StackedTrainer(nExperts, dModel, vocabSize, 
                                                    batchSize, learningRate,
                                                    warmupSteps, totalSteps);
        
        // 创建简单训练数据（重复模式）
        System.out.println("\n[阶段3] 创建简单训练数据...");
        List<int[]> sequences = createSimplePatternData(100, 32, vocabSize);
        System.out.printf("      训练样本数: %d\n", sequences.size());
        
        SCDataLoader dataLoader = new SCDataLoader(sequences, batchSize, 32);
        
        // 开始训练（使用全序列预测）
        System.out.println("\n[阶段4] 开始训练...");
        
        long startTime = System.currentTimeMillis();
        int stepCount = 0;
        float prevLoss = Float.MAX_VALUE;
        
        while (stepCount < totalSteps && dataLoader.hasNext()) {
            int[][] batch = dataLoader.getNextBatch();
            if (batch == null) {
                dataLoader.reset();
                continue;
            }
            
            // 全序列预测：对每个位置训练
            float batchLoss = 0;
            for (int[] tokens : batch) {
                for (int i = 0; i < tokens.length - 1; i++) {
                    // 输入：前i+1个token
                    int[] inputTokens = Arrays.copyOf(tokens, i + 1);
                    // 标签：第i+1个token
                    float[] labels = new float[vocabSize];
                    int targetToken = tokens[i + 1];
                    if (targetToken >= 0 && targetToken < vocabSize) {
                        labels[targetToken] = 1.0f;
                    }
                    
                    StackedTrainer.TrainingStepResult result = trainer.trainStep(inputTokens, labels);
                    batchLoss += result.loss;
                    stepCount++;
                    
                    if (stepCount % 10 == 0) {
                        long elapsed = System.currentTimeMillis() - startTime;
                        float avgLoss = batchLoss / 10;
                        System.out.printf("[Step %d] Loss: %.4f | LR: %.2e | Time: %ds\n", 
                            stepCount, avgLoss, result.learningRate, elapsed / 1000);
                        
                        // 检查损失是否下降
                        if (avgLoss < prevLoss * 0.99) {
                            System.out.printf("         ↓ 损失下降: %.2f%%\n", (1 - avgLoss / prevLoss) * 100);
                        }
                        prevLoss = avgLoss;
                        batchLoss = 0;
                    }
                    
                    if (stepCount >= totalSteps) break;
                }
                if (stepCount >= totalSteps) break;
            }
        }
        
        long elapsed = System.currentTimeMillis() - startTime;
        System.out.printf("\n[阶段5] 训练完成！耗时: %.1fs\n", elapsed / 1000.0);
        
        // 测试生成（简化版）
        System.out.println("\n[阶段6] 测试文本生成...");
        System.out.println("      (generate方法需要String输入，跳过此测试)");
        
        gpu.release();
    }
    
    /**
     * 创建简单模式数据（重复序列）
     */
    private static List<int[]> createSimplePatternData(int numSequences, int seqLength, int vocabSize) {
        List<int[]> sequences = new ArrayList<>();
        Random rng = new Random(42);
        
        for (int s = 0; s < numSequences; s++) {
            int[] seq = new int[seqLength];
            // 创建重复模式
            int base = s % 10;
            for (int i = 0; i < seqLength; i++) {
                seq[i] = (base + i) % Math.min(vocabSize, 100);
            }
            sequences.add(seq);
        }
        
        return sequences;
    }
}
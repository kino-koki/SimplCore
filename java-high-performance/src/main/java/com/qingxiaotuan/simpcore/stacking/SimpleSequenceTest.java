package com.qingxiaotuan.simpcore.stacking;

import java.util.*;

/**
 * 简单序列预测测试 - 验证训练管线是否正常工作
 * 使用重复序列数据：0,1,2,3,0,1,2,3,...
 */
public class SimpleSequenceTest {
    
    public static void main(String[] args) {
        System.out.println("===== 简单序列预测测试 =====");
        
        // 配置参数
        int vocabSize = 10;
        int dModel = 64;
        int seqLength = 8;
        int batchSize = 4;
        float learningRate = 1e-2f;
        int warmupSteps = 10;
        int totalSteps = 1000;
        
        System.out.printf("参数配置: vocab=%d, dModel=%d, seq=%d, lr=%.2e\n", 
            vocabSize, dModel, seqLength, learningRate);
        
        // 创建堆叠训练器
        StackedTrainer trainer = new StackedTrainer(1, dModel, vocabSize, 
            batchSize, learningRate, warmupSteps, totalSteps);
        
        // 生成简单序列数据
        List<int[]> sequences = generateSimpleSequences(vocabSize, seqLength, 1000);
        System.out.printf("生成训练样本: %d\n", sequences.size());
        
        // 训练
        System.out.println("\n开始训练...");
        long startTime = System.currentTimeMillis();
        int stepCount = 0;
        float totalLoss = 0;
        float prevLoss = Float.MAX_VALUE;
        
        while (stepCount < totalSteps) {
            // 随机选择一个序列
            int[] seq = sequences.get(stepCount % sequences.size());
            
            for (int i = 0; i < seq.length - 1; i++) {
                // 输入：前i+1个token
                int[] input = new int[i + 1];
                System.arraycopy(seq, 0, input, 0, i + 1);
                // 标签：下一个token
                float[] labels = new float[vocabSize];
                int target = seq[i + 1];
                labels[target] = 1.0f;
                
                StackedTrainer.TrainingStepResult result = trainer.trainStep(input, labels);
                totalLoss += result.loss;
                stepCount++;
                
                if (stepCount % 50 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    float avgLoss = totalLoss / 50;
                    System.out.printf("[Step %d/%d] Loss: %.4f | LR: %.2e | Time: %ds\n", 
                        stepCount, totalSteps, avgLoss, result.learningRate, elapsed / 1000);
                    
                    if (avgLoss < prevLoss * 0.99) {
                        System.out.printf("         ↓ 损失下降: %.2f%%\n", (1 - avgLoss / prevLoss) * 100);
                    }
                    prevLoss = avgLoss;
                    totalLoss = 0;
                }
                
                if (stepCount >= totalSteps) break;
            }
        }
        
        System.out.println("\n测试完成！");
    }
    
    private static List<int[]> generateSimpleSequences(int vocabSize, int seqLength, int count) {
        List<int[]> sequences = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int[] seq = new int[seqLength];
            for (int j = 0; j < seqLength; j++) {
                seq[j] = j % vocabSize;
            }
            sequences.add(seq);
        }
        return sequences;
    }
    
    private static int argmax(float[] x) {
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
}

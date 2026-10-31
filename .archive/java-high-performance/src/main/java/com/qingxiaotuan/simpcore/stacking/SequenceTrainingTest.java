package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.model.SequenceModel;
import com.qingxiaotuan.simpcore.training.AdamOptimizer;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.*;

/**
 * 序列模型训练测试 - 直接使用SequenceModel验证序列处理能力
 */
public class SequenceTrainingTest {
    
    public static void main(String[] args) {
        System.out.println("===== 序列模型训练测试 =====");
        
        // 配置参数
        int vocabSize = 10;
        int dModel = 64;
        int nLayers = 2;
        int nHeads = 4;
        int seqLength = 8;
        float learningRate = 1e-3f;  // 降低学习率避免梯度爆炸
        int totalSteps = 2000;
        
        System.out.printf("参数配置: vocab=%d, dModel=%d, layers=%d, lr=%.2e\n", 
            vocabSize, dModel, nLayers, learningRate);
        
        // 创建序列模型
        SequenceModel model = new SequenceModel(vocabSize, dModel, nLayers, nHeads);
        System.out.println("序列模型创建完成");
        
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
                // 标签：下一个token（one-hot编码）
                float[] labels = new float[vocabSize];
                int target = seq[i + 1];
                labels[target] = 1.0f;
                
                // 前向传播
                float[] logits = model.forward(input);
                
                // 计算交叉熵损失和梯度
                float loss = computeCrossEntropy(logits, labels);
                float[] dLogits = computeCrossEntropyGradient(logits, labels);
                
                // 反向传播
                GradientResult gradResult = model.backward(dLogits);
                
                // 使用学习率更新参数
                model.updateParametersWithGradient(gradResult.getParameterGradients(), learningRate);
                
                totalLoss += loss;
                stepCount++;
                
                if (stepCount % 50 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    float avgLoss = totalLoss / 50;
                    System.out.printf("[Step %d/%d] Loss: %.4f | LR: %.2e | Time: %ds\n", 
                        stepCount, totalSteps, avgLoss, learningRate, elapsed / 1000);
                    
                    if (avgLoss < prevLoss * 0.99) {
                        System.out.printf("         ↓ 损失下降: %.2f%%\n", (1 - avgLoss / prevLoss) * 100);
                    }
                    prevLoss = avgLoss;
                    totalLoss = 0;
                }
                
                if (stepCount >= totalSteps) break;
            }
        }
        
        // 测试生成
        System.out.println("\n测试生成...");
        int[] prompt = {0, 1, 2};
        for (int i = 0; i < 10; i++) {
            float[] logits = model.forward(prompt);
            int nextToken = argmax(logits);
            System.out.printf("输入: %s -> 预测: %d (期望: %d)\n", 
                Arrays.toString(prompt), nextToken, (prompt.length) % vocabSize);
            
            int[] newPrompt = new int[prompt.length + 1];
            System.arraycopy(prompt, 0, newPrompt, 0, prompt.length);
            newPrompt[prompt.length] = nextToken;
            prompt = newPrompt;
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
    
    private static float computeCrossEntropy(float[] logits, float[] labels) {
        float loss = 0f;
        int vocabSize = logits.length;
        
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            sum += (float) Math.exp(logits[i] - max);
        }
        
        for (int i = 0; i < vocabSize; i++) {
            float prob = (float) Math.exp(logits[i] - max) / sum;
            if (labels[i] > 0) {
                loss -= labels[i] * (float) Math.log(prob + 1e-8f);
            }
        }
        
        return loss;
    }
    
    private static float[] computeCrossEntropyGradient(float[] logits, float[] labels) {
        int vocabSize = logits.length;
        float[] grad = new float[vocabSize];
        
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            sum += (float) Math.exp(logits[i] - max);
        }
        
        for (int i = 0; i < vocabSize; i++) {
            float prob = (float) Math.exp(logits[i] - max) / sum;
            grad[i] = prob - labels[i];
        }
        
        return grad;
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

package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import java.util.Arrays;
import java.util.Random;

/**
 * 模式学习模块 — 模拟人类从经验中抽象出"图式"(Schema)的能力
 * 
 * 人类认知特性：
 * - 图式抽象：从具体经验中提取通用模式
 * - 原型匹配：新输入与已有原型匹配
 * - 渐进更新：通过移动平均逐步完善原型
 * - 模式引导：匹配的原型指导信息处理
 * 
 * 维护N个可学习的原型向量，每个代表一种常见模式。
 * 给定输入，找到最佳匹配原型并用它增强处理。
 */
public class SchemaLearningModule {

    /** 原型数量 */
    private final int numPrototypes;
    
    /** 原型维度 */
    private final int dModel;
    
    /** 原型矩阵: numPrototypes x dModel */
    private final float[][] prototypes;
    
    /** 每个原型的激活计数 */
    private final int[] activationCounts;
    
    /** 原型学习率 */
    private final float learningRate;
    
    /** 相似度阈值（用于判断是否匹配已有原型） */
    private final float similarityThreshold;
    
    /** 最近激活的原型索引 */
    private int lastActivatedIndex;
    
    private final Random random;

    public SchemaLearningModule(int dModel) {
        this(dModel, 16); // 默认16个原型
    }

    public SchemaLearningModule(int dModel, int numPrototypes) {
        this.dModel = dModel;
        this.numPrototypes = numPrototypes;
        this.prototypes = new float[numPrototypes][dModel];
        this.activationCounts = new int[numPrototypes];
        this.learningRate = 0.01f;
        this.similarityThreshold = 0.5f;
        this.lastActivatedIndex = -1;
        this.random = new Random(42);
        
        initPrototypes();
    }

    /**
     * 初始化原型（使用小随机值）
     */
    private void initPrototypes() {
        for (int i = 0; i < numPrototypes; i++) {
            for (int j = 0; j < dModel; j++) {
                prototypes[i][j] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            activationCounts[i] = 0;
        }
    }

    /**
     * 前向传播：模式增强处理
     * output = x + matched_prototype
     * 
     * 首先找到与输入最匹配的原型，
     * 然后将其添加到输入中以增强表示。
     */
    public float[] forward(float[] x) {
        int totalSize = x.length;
        float[] output = new float[totalSize];
        
        // 对整个序列中的每个dModel块应用
        for (int i = 0; i < totalSize; i += dModel) {
            float[] slice = Arrays.copyOfRange(x, i, i + dModel);
            
            // 找到最佳匹配原型
            int bestIdx = findBestMatch(slice);
            lastActivatedIndex = bestIdx;
            
            // 更新原型（移动平均）
            updatePrototype(bestIdx, slice);
            
            // output = x + matched_prototype
            for (int j = 0; j < dModel; j++) {
                output[i + j] = slice[j] + prototypes[bestIdx][j];
            }
        }
        
        return output;
    }

    /**
     * 查找与输入最匹配的原型（余弦相似度）
     */
    private int findBestMatch(float[] input) {
        int bestIdx = 0;
        float bestSim = Float.NEGATIVE_INFINITY;
        
        for (int p = 0; p < numPrototypes; p++) {
            float dot = 0;
            float norm1 = 0, norm2 = 0;
            for (int j = 0; j < dModel; j++) {
                dot += input[j] * prototypes[p][j];
                norm1 += input[j] * input[j];
                norm2 += prototypes[p][j] * prototypes[p][j];
            }
            float sim = dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
            
            if (sim > bestSim) {
                bestSim = sim;
                bestIdx = p;
            }
        }
        
        return bestIdx;
    }

    /**
     * 更新原型：proto += lr * (input - proto)
     * 仅当相似度超过阈值时才更新
     */
    private void updatePrototype(int idx, float[] input) {
        // 计算当前输入与选中原型的相似度
        float dot = 0;
        float norm1 = 0, norm2 = 0;
        for (int j = 0; j < dModel; j++) {
            dot += input[j] * prototypes[idx][j];
            norm1 += input[j] * input[j];
            norm2 += prototypes[idx][j] * prototypes[idx][j];
        }
        float sim = dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
        
        if (sim >= similarityThreshold) {
            // 匹配：移动平均更新（强化已有模式）
            for (int j = 0; j < dModel; j++) {
                prototypes[idx][j] += learningRate * (input[j] - prototypes[idx][j]);
            }
            activationCounts[idx]++;
        } else {
            // 不匹配：更新最不活跃的原型（探索新模式）
            int leastActive = 0;
            int minCount = activationCounts[0];
            for (int p = 1; p < numPrototypes; p++) {
                if (activationCounts[p] < minCount) {
                    minCount = activationCounts[p];
                    leastActive = p;
                }
            }
            for (int j = 0; j < dModel; j++) {
                prototypes[leastActive][j] += learningRate * 0.5f * (input[j] - prototypes[leastActive][j]);
            }
        }
    }

    /**
     * 返回最后匹配的原型索引
     */
    public int getActivatedSchema(float[] x) {
        if (lastActivatedIndex >= 0) {
            return lastActivatedIndex;
        }
        
        // 如果尚未激活，计算一次匹配
        float[] slice = x.length > dModel ? Arrays.copyOf(x, dModel) : x;
        return findBestMatch(slice);
    }

    /**
     * 反向传播（简化实现）
     */
    public float[] backward(float[] dOutput) {
        // 简化的梯度传递：dOutput直接经过残差连接
        return Arrays.copyOf(dOutput, dOutput.length);
    }

    /**
     * 原型多样性度量（用于认知度量）
     * 计算所有原型对之间的平均余弦距离
     */
    public float getPrototypeDiversity() {
        float totalDist = 0;
        int pairs = 0;
        
        for (int i = 0; i < numPrototypes; i++) {
            for (int j = i + 1; j < numPrototypes; j++) {
                float dot = 0;
                float norm1 = 0, norm2 = 0;
                for (int k = 0; k < dModel; k++) {
                    dot += prototypes[i][k] * prototypes[j][k];
                    norm1 += prototypes[i][k] * prototypes[i][k];
                    norm2 += prototypes[j][k] * prototypes[j][k];
                }
                float sim = dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
                totalDist += 1.0f - sim; // 余弦距离
                pairs++;
            }
        }
        
        return pairs > 0 ? totalDist / pairs : 0;
    }

    public int getNumPrototypes() { return numPrototypes; }
    public int getDModel() { return dModel; }
    public float[][] getPrototypes() { return prototypes; }
    public int[] getActivationCounts() { return activationCounts; }
    public int getLastActivatedIndex() { return lastActivatedIndex; }
    public float getLearningRate() { return learningRate; }
    public float getSimilarityThreshold() { return similarityThreshold; }
}

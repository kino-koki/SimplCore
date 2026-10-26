package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import java.util.Arrays;
import java.util.Random;

/**
 * 分层感知模块 — 模拟人类多层次信息处理
 * 
 * 人类视觉/语言处理并非单一粒度，而是同时进行细粒度（局部细节）、
 * 中粒度（短语/局部上下文）和粗粒度（全局语义）的并行处理。
 * 本模块通过可学习的门控机制融合三个粒度的表征。
 */
public class HierarchicalPerceptionModule {

    private final int dModel;
    
    // 细粒度：逐token处理 (W_fine @ x + b_fine)
    private final float[] wFine;
    private final float[] bFine;
    
    // 中粒度：bigram块处理
    private final float[] wMedium;
    
    // 粗粒度：quadgram块处理
    private final float[] wCoarse;
    
    // 门控权重：融合三个粒度的输出
    private final float[] wGate;
    
    // 随机数生成器用于初始化
    private final Random random;
    
    // 梯度累积器
    private final float[] gradWFine;
    private final float[] gradBFine;
    private final float[] gradWMedium;
    private final float[] gradWCoarse;
    private final float[] gradWGate;
    
    private int stepCount;
    private static final int UPDATE_FREQ = 10;
    private static final float LEARNING_RATE = 0.001f;

    public HierarchicalPerceptionModule(int dModel) {
        this.dModel = dModel;
        this.random = new Random(42);
        this.stepCount = 0;
        
        // 细粒度权重: dModel x dModel
        this.wFine = new float[dModel * dModel];
        this.bFine = new float[dModel];
        
        // 中粒度权重: dModel x dModel
        this.wMedium = new float[dModel * dModel];
        
        // 粗粒度权重: dModel x dModel
        this.wCoarse = new float[dModel * dModel];
        
        // 门控权重: 3*dModel x dModel (接收拼接后的三个粒度输出)
        this.wGate = new float[3 * dModel * dModel];
        
        // 初始化梯度
        this.gradWFine = new float[dModel * dModel];
        this.gradBFine = new float[dModel];
        this.gradWMedium = new float[dModel * dModel];
        this.gradWCoarse = new float[dModel * dModel];
        this.gradWGate = new float[3 * dModel * dModel];
        
        initWeights();
    }

    private void initWeights() {
        // 细粒度: 较小初始化
        for (int i = 0; i < wFine.length; i++) {
            wFine[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        for (int i = 0; i < bFine.length; i++) {
            bFine[i] = 0.0f;
        }
        
        // 中粒度: 中等初始化
        for (int i = 0; i < wMedium.length; i++) {
            wMedium[i] = (random.nextFloat() - 0.5f) * 0.05f;
        }
        
        // 粗粒度: 较大初始化 (全局模式更强先验)
        for (int i = 0; i < wCoarse.length; i++) {
            wCoarse[i] = (random.nextFloat() - 0.5f) * 0.08f;
        }
        
        // 门控: 初始偏向细粒度
        for (int i = 0; i < wGate.length; i++) {
            wGate[i] = (random.nextFloat() - 0.5f) * 0.01f;
        }
    }

    /**
     * 前向传播：多粒度处理 + 门控融合
     * 
     * fine = W_fine @ x + b_fine
     * medium = AvgPool2(x, 2) -> W_medium -> Upsample
     * coarse = AvgPool2(x, 4) -> W_coarse -> Upsample
     * gate = sigmoid(W_gate @ concat(fine, medium, coarse))
     * output = gate_f * fine + gate_m * medium + gate_c * coarse
     */
    public float[] forward(float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        MemoryOptimizer mem = MemoryOptimizer.getInstance();
        float[] output = mem.allocateFloatArray(totalSize);
        
        float[] fineOut = mem.allocateFloatArray(totalSize);
        float[] mediumOut = mem.allocateFloatArray(totalSize);
        float[] coarseOut = mem.allocateFloatArray(totalSize);
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int offset = (b * seqLen + s) * dModel;
                float[] xSlice = Arrays.copyOfRange(x, offset, offset + dModel);
                
                // 细粒度: 直接逐token变换
                float[] fine = MatrixOperations.matmul(xSlice, wFine, 1, dModel, dModel);
                for (int i = 0; i < dModel; i++) {
                    fineOut[offset + i] = fine[i] + bFine[i];
                }
                
                // 中粒度: bigram聚合 -> 变换 -> 上采样
                // 滑窗平均两个相邻token
                int mediumStart = Math.max(0, s - 1);
                int mediumEnd = Math.min(seqLen - 1, s + 1);
                float[] mediumAgg = new float[dModel];
                int mediumCount = 0;
                for (int si = mediumStart; si <= mediumEnd; si++) {
                    int off = (b * seqLen + si) * dModel;
                    for (int j = 0; j < dModel; j++) {
                        mediumAgg[j] += x[off + j];
                    }
                    mediumCount++;
                }
                for (int j = 0; j < dModel; j++) {
                    mediumAgg[j] /= mediumCount;
                }
                
                float[] medium = MatrixOperations.matmul(mediumAgg, wMedium, 1, dModel, dModel);
                System.arraycopy(medium, 0, mediumOut, offset, dModel);
                
                // 粗粒度: quadgram聚合 -> 变换 -> 上采样
                int coarseStart = Math.max(0, s - 2);
                int coarseEnd = Math.min(seqLen - 1, s + 2);
                float[] coarseAgg = new float[dModel];
                int coarseCount = 0;
                for (int si = coarseStart; si <= coarseEnd; si++) {
                    int off = (b * seqLen + si) * dModel;
                    for (int j = 0; j < dModel; j++) {
                        coarseAgg[j] += x[off + j];
                    }
                    coarseCount++;
                }
                for (int j = 0; j < dModel; j++) {
                    coarseAgg[j] /= coarseCount;
                }
                
                float[] coarse = MatrixOperations.matmul(coarseAgg, wCoarse, 1, dModel, dModel);
                System.arraycopy(coarse, 0, coarseOut, offset, dModel);
            }
        }
        
        // 计算门控权重并融合
        for (int i = 0; i < totalSize; i += dModel) {
            // 拼接三个粒度的输出
            float[] concat = new float[3 * dModel];
            System.arraycopy(fineOut, i, concat, 0, dModel);
            System.arraycopy(mediumOut, i, concat, dModel, dModel);
            System.arraycopy(coarseOut, i, concat, 2 * dModel, dModel);
            
            // gate = sigmoid(W_gate @ concat)
            float[] gateLogits = MatrixOperations.matmul(concat, wGate, 1, 3 * dModel, dModel);
            float[] gate = softmax(gateLogits);
            
            // 加权融合
            for (int j = 0; j < dModel; j++) {
                float gFine = gate[j];
                float gMedium = gate[(j + dModel / 3) % dModel];
                float gCoarse = gate[(j + 2 * dModel / 3) % dModel];
                // 重新归一化确保和为1
                float sum = gFine + gMedium + gCoarse + 1e-8f;
                output[i + j] = (gFine / sum) * fineOut[i + j] 
                              + (gMedium / sum) * mediumOut[i + j] 
                              + (gCoarse / sum) * coarseOut[i + j];
            }
        }
        
        mem.releaseFloatArray(fineOut);
        mem.releaseFloatArray(mediumOut);
        mem.releaseFloatArray(coarseOut);
        
        return output;
    }

    private float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            if (v > max) max = v;
        }
        float[] exp = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    /**
     * 反向传播 — 计算梯度
     */
    public float[] backward(float[] dOutput, float[] x, int batchSize, int seqLen) {
        int totalSize = batchSize * seqLen * dModel;
        float[] dInput = new float[totalSize];
        
        // 简化实现：将dOutput直接传递回输入
        // 实际的梯度计算应包含门控梯度和各粒度路径的梯度
        System.arraycopy(dOutput, 0, dInput, 0, totalSize);
        
        // 累积梯度（简化为梯度下降步）
        stepCount++;
        
        return dInput;
    }

    /**
     * 通过梯度下降更新权重
     */
    public void updateWeights(float[] gradients) {
        if (gradients == null || gradients.length == 0) return;
        
        // 更新细粒度权重
        int gateSize = 3 * dModel * dModel;
        if (gradients.length >= gateSize + dModel * dModel + dModel) {
            int offset = 0;
            for (int i = 0; i < dModel * dModel; i++) {
                wFine[i] -= LEARNING_RATE * gradients[offset + i];
            }
            offset += dModel * dModel;
            for (int i = 0; i < dModel; i++) {
                bFine[i] -= LEARNING_RATE * gradients[offset + i];
            }
        }
    }

    public int getDModel() { return dModel; }

    public float[] getWFine() { return wFine; }
    public float[] getBFine() { return bFine; }
    public float[] getWMedium() { return wMedium; }
    public float[] getWCoarse() { return wCoarse; }
    public float[] getWGate() { return wGate; }
    
    public float[] getGradWFine() { return gradWFine; }
    public float[] getGradBFine() { return gradBFine; }
    public float[] getGradWMedium() { return gradWMedium; }
    public float[] getGradWCoarse() { return gradWCoarse; }
    public float[] getGradWGate() { return gradWGate; }
}

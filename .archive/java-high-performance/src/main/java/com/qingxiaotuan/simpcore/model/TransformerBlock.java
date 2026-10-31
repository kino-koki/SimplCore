package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;

/**
 * Transformer Block - 包含多头注意力、FFN、残差连接和层归一化
 */
public class TransformerBlock {
    
    private final int dModel;
    private final int nHeads;
    
    // 组件
    private MultiHeadAttention attention;
    private FeedForwardNetwork ffn;
    private LayerNorm attnNorm;
    private LayerNorm ffnNorm;
    
    // GPU加速
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    // 缓存
    private float[][] attentionInput;
    private float[][] attentionOutput;
    private float[][] ffnInput;
    private float[][] ffnOutput;
    
    public TransformerBlock(int dModel, int nHeads) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        initComponents();
        initializeGPU();
    }
    
    private void initComponents() {
        attention = new MultiHeadAttention(dModel, nHeads);
        ffn = new FeedForwardNetwork(dModel);
        attnNorm = new LayerNorm(dModel);
        ffnNorm = new LayerNorm(dModel);
    }
    
    private void initializeGPU() {
        try {
            gpu = OpenCLAccelerator.getInstance();
            useGPU = gpu.isUsingGPU() && gpu.isInitialized();
        } catch (Exception e) {
            useGPU = false;
        }
    }
    
    /**
     * 前向传播
     */
    public float[][] forward(float[][] x) {
        int seqLen = x.length;
        
        // 保存输入
        attentionInput = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(x[t], 0, attentionInput[t], 0, dModel);
        }
        
        // 层归一化 -> 注意力 -> 残差连接
        float[][] norm1 = attnNorm.forward(x);
        attentionOutput = attention.forward(norm1);
        
        float[][] attnResidual = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dModel; i++) {
                attnResidual[t][i] = x[t][i] + attentionOutput[t][i];
            }
        }
        
        // 保存FFN输入
        ffnInput = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(attnResidual[t], 0, ffnInput[t], 0, dModel);
        }
        
        // 层归一化 -> FFN -> 残差连接
        float[][] norm2 = ffnNorm.forward(attnResidual);
        ffnOutput = ffn.forward(norm2);
        
        float[][] result = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dModel; i++) {
                result[t][i] = attnResidual[t][i] + ffnOutput[t][i];
            }
        }
        
        return result;
    }
    
    /**
     * 反向传播
     */
    public float[][] backward(float[][] dOutput) {
        int seqLen = dOutput.length;
        
        // FFN残差梯度
        float[][] dFFNResidual = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(dOutput[t], 0, dFFNResidual[t], 0, dModel);
        }
        
        // FFN反向传播
        float[][] dNorm2 = ffn.backward(dFFNResidual);
        float[][] dAttnResidual = ffnNorm.backward(dNorm2);
        
        // 注意力残差梯度
        float[][] dAttnResidualTotal = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < dModel; j++) {
                dAttnResidualTotal[t][j] = dAttnResidual[t][j] + dFFNResidual[t][j];
            }
        }
        
        // 注意力反向传播
        float[][] dNorm1 = attention.backward(dAttnResidualTotal);
        float[][] dInput = attnNorm.backward(dNorm1);
        
        return dInput;
    }
    
    /**
     * Feed Forward Network
     */
    private static class FeedForwardNetwork {
        
        private final int dModel;
        private final int dFF;
        
        private float[] w1;  // [dModel, dFF]
        private float[] b1;  // [dFF]
        private float[] w2;  // [dFF, dModel]
        private float[] b2;  // [dModel]
        
        public FeedForwardNetwork(int dModel) {
            this.dModel = dModel;
            this.dFF = dModel * 4;
            initWeights();
        }
        
        private void initWeights() {
            float scale = (float) (1.0 / Math.sqrt(dModel));
            w1 = new float[dModel * dFF];
            b1 = new float[dFF];
            w2 = new float[dFF * dModel];
            b2 = new float[dModel];
            
            for (int i = 0; i < dModel * dFF; i++) {
                w1[i] = (float) (Math.random() - 0.5) * 2 * scale;
            }
            for (int i = 0; i < dFF * dModel; i++) {
                w2[i] = (float) (Math.random() - 0.5) * 2 * scale;
            }
        }
        
        public float[][] forward(float[][] x) {
            int seqLen = x.length;
            float[][] result = new float[seqLen][dModel];
            
            for (int t = 0; t < seqLen; t++) {
                // 第一层：线性 + ReLU
                float[] hidden = matvec(w1, x[t], dFF, dModel);
                for (int i = 0; i < dFF; i++) {
                    hidden[i] += b1[i];
                    if (hidden[i] < 0) hidden[i] = 0; // ReLU
                }
                
                // 第二层：线性
                float[] output = matvec(w2, hidden, dModel, dFF);
                for (int i = 0; i < dModel; i++) {
                    output[i] += b2[i];
                }
                
                result[t] = output;
            }
            
            return result;
        }
        
        public float[][] backward(float[][] dOutput) {
            int seqLen = dOutput.length;
            float[][] dInput = new float[seqLen][dModel];
            
            for (int t = 0; t < seqLen; t++) {
                // 第二层梯度
                float[] dHidden = matvec(transpose(w2, dFF, dModel), dOutput[t], dFF, dModel);
                
                // ReLU梯度
                for (int i = 0; i < dFF; i++) {
                    if (dHidden[i] < 0) dHidden[i] = 0;
                }
                
                // 第一层梯度
                dInput[t] = matvec(transpose(w1, dModel, dFF), dHidden, dModel, dFF);
            }
            
            return dInput;
        }
        
        private float[] matvec(float[] A, float[] x, int M, int N) {
            float[] y = new float[M];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    y[i] += A[i * N + j] * x[j];
                }
            }
            return y;
        }
        
        private float[] transpose(float[] A, int M, int N) {
            float[] At = new float[N * M];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    At[j * M + i] = A[i * N + j];
                }
            }
            return At;
        }
    }
    
    /**
     * Layer Normalization
     */
    private static class LayerNorm {
        
        private final int dModel;
        private float[] gamma;
        private float[] beta;
        private static final float eps = 1e-5f;
        
        public LayerNorm(int dModel) {
            this.dModel = dModel;
            gamma = new float[dModel];
            beta = new float[dModel];
            for (int i = 0; i < dModel; i++) {
                gamma[i] = 1.0f;
                beta[i] = 0.0f;
            }
        }
        
        public float[][] forward(float[][] x) {
            int seqLen = x.length;
            float[][] result = new float[seqLen][dModel];
            
            for (int t = 0; t < seqLen; t++) {
                // 计算均值
                float mean = 0;
                for (int i = 0; i < dModel; i++) {
                    mean += x[t][i];
                }
                mean /= dModel;
                
                // 计算方差
                float var = 0;
                for (int i = 0; i < dModel; i++) {
                    var += (x[t][i] - mean) * (x[t][i] - mean);
                }
                var /= dModel;
                
                // 归一化
                float std = (float) Math.sqrt(var + eps);
                for (int i = 0; i < dModel; i++) {
                    result[t][i] = gamma[i] * ((x[t][i] - mean) / std) + beta[i];
                }
            }
            
            return result;
        }
        
        public float[][] backward(float[][] dOutput) {
            return dOutput; // 简化实现
        }
    }
    
    public MultiHeadAttention getAttention() { return attention; }
    public FeedForwardNetwork getFFN() { return ffn; }
}

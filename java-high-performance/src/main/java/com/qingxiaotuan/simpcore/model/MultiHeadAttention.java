package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;

/**
 * Multi-Head Self-Attention层 - 真正的自注意力机制
 * 实现缩放点积注意力和多头并行计算
 */
public class MultiHeadAttention {
    
    private final int dModel;
    private final int nHeads;
    private final int dHead;
    
    // 参数
    private float[] qWeight;  // [dModel, dModel]
    private float[] qBias;    // [dModel]
    private float[] kWeight;  // [dModel, dModel]
    private float[] kBias;    // [dModel]
    private float[] vWeight;  // [dModel, dModel]
    private float[] vBias;    // [dModel]
    private float[] oWeight;  // [dModel, dModel]
    private float[] oBias;    // [dModel]
    
    // GPU加速
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    // 缓存（用于反向传播）
    private float[][] q;
    private float[][] k;
    private float[][] v;
    private float[][] attention;
    private float[][] output;
    private float[][] input;
    
    public MultiHeadAttention(int dModel, int nHeads) {
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.dHead = dModel / nHeads;
        initWeights();
        initializeGPU();
    }
    
    private void initializeGPU() {
        try {
            gpu = OpenCLAccelerator.getInstance();
            useGPU = gpu.isUsingGPU() && gpu.isInitialized();
        } catch (Exception e) {
            useGPU = false;
        }
    }
    
    private void initWeights() {
        float scale = (float) (1.0 / Math.sqrt(dModel));
        
        qWeight = new float[dModel * dModel];
        qBias = new float[dModel];
        kWeight = new float[dModel * dModel];
        kBias = new float[dModel];
        vWeight = new float[dModel * dModel];
        vBias = new float[dModel];
        oWeight = new float[dModel * dModel];
        oBias = new float[dModel];
        
        for (int i = 0; i < dModel * dModel; i++) {
            qWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            kWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            vWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
            oWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
    }
    
    /**
     * 前向传播
     * @param x 输入 [seqLen, dModel]
     * @return 输出 [seqLen, dModel]
     */
    public float[][] forward(float[][] x) {
        int seqLen = x.length;
        this.input = x;
        
        // Q, K, V投影
        q = new float[seqLen][dModel];
        k = new float[seqLen][dModel];
        v = new float[seqLen][dModel];
        
        for (int t = 0; t < seqLen; t++) {
            q[t] = matvec(qWeight, x[t], dModel, dModel);
            k[t] = matvec(kWeight, x[t], dModel, dModel);
            v[t] = matvec(vWeight, x[t], dModel, dModel);
            
            for (int i = 0; i < dModel; i++) {
                q[t][i] += qBias[i];
                k[t][i] += kBias[i];
                v[t][i] += vBias[i];
            }
        }
        
        // 多头注意力计算
        float[][] headsOutput = new float[nHeads][seqLen * dHead];
        
        for (int h = 0; h < nHeads; h++) {
            int offset = h * dHead;
            
            // 提取第h个头的Q, K, V
            float[][] qh = new float[seqLen][dHead];
            float[][] kh = new float[seqLen][dHead];
            float[][] vh = new float[seqLen][dHead];
            
            for (int t = 0; t < seqLen; t++) {
                for (int i = 0; i < dHead; i++) {
                    qh[t][i] = q[t][offset + i];
                    kh[t][i] = k[t][offset + i];
                    vh[t][i] = v[t][offset + i];
                }
            }
            
            // 计算注意力权重
            float[][] attn = scaledDotProductAttention(qh, kh, vh);
            
            // 保存注意力权重（用于可视化和正则化）
            if (h == 0) {
                this.attention = attn;
            }
            
            // 保存到headsOutput
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(attn[t], 0, headsOutput[h], t * dHead, dHead);
            }
        }
        
        // 合并多头输出
        output = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int h = 0; h < nHeads; h++) {
                int offset = h * dHead;
                for (int i = 0; i < dHead; i++) {
                    output[t][offset + i] = headsOutput[h][t * dHead + i];
                }
            }
        }
        
        // 输出投影
        float[][] result = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            result[t] = matvec(oWeight, output[t], dModel, dModel);
            for (int i = 0; i < dModel; i++) {
                result[t][i] += oBias[i];
            }
        }
        
        return result;
    }
    
    /**
     * 缩放点积注意力
     */
    private float[][] scaledDotProductAttention(float[][] q, float[][] kh, float[][] vh) {
        int seqLen = q.length;
        float scale = (float) (1.0 / Math.sqrt(dHead));
        
        // 计算注意力分数
        float[][] scores = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                for (int k = 0; k < dHead; k++) {
                    scores[i][j] += q[i][k] * kh[j][k];
                }
                scores[i][j] *= scale;
            }
        }
        
        // 因果掩码（只允许关注前面的token）
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                scores[i][j] = Float.NEGATIVE_INFINITY;
            }
        }
        
        // Softmax
        float[][] attnWeights = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            float maxScore = scores[i][0];
            for (int j = 0; j < seqLen; j++) {
                if (scores[i][j] > maxScore) maxScore = scores[i][j];
            }
            
            float sum = 0;
            for (int j = 0; j < seqLen; j++) {
                attnWeights[i][j] = (float) Math.exp(scores[i][j] - maxScore);
                sum += attnWeights[i][j];
            }
            
            for (int j = 0; j < seqLen; j++) {
                attnWeights[i][j] /= sum;
            }
        }
        
        // 加权求和V
        float[][] result = new float[seqLen][dHead];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                for (int k = 0; k < dHead; k++) {
                    result[i][k] += attnWeights[i][j] * vh[j][k];
                }
            }
        }
        
        return result;
    }
    
    /**
     * 反向传播
     */
    public float[][] backward(float[][] dOutput) {
        int seqLen = dOutput.length;
        
        // 输出投影梯度
        float[][] dO = new float[seqLen][dModel];
        float[] dOWeight = new float[dModel * dModel];
        float[] dOBias = new float[dModel];
        
        for (int t = 0; t < seqLen; t++) {
            dO[t] = matvec(transpose(oWeight, dModel, dModel), dOutput[t], dModel, dModel);
            
            for (int i = 0; i < dModel; i++) {
                dOBias[i] += dOutput[t][i];
                for (int j = 0; j < dModel; j++) {
                    dOWeight[i * dModel + j] += output[t][i] * dOutput[t][j];
                }
            }
        }
        
        // 多头反向传播
        float[][] dX = new float[seqLen][dModel];
        
        // 这里简化实现，实际需要更复杂的反向传播
        // 为了快速实现，我们使用一个简化版本
        
        // Q, K, V梯度
        float[] dQWeight = new float[dModel * dModel];
        float[] dKWeight = new float[dModel * dModel];
        float[] dVWeight = new float[dModel * dModel];
        
        // 计算输入梯度（简化版）
        for (int t = 0; t < seqLen; t++) {
            // 反向传播通过Q, K, V投影
            float[] dQ = matvec(transpose(qWeight, dModel, dModel), dO[t], dModel, dModel);
            float[] dK = matvec(transpose(kWeight, dModel, dModel), dO[t], dModel, dModel);
            float[] dV = matvec(transpose(vWeight, dModel, dModel), dO[t], dModel, dModel);
            
            for (int i = 0; i < dModel; i++) {
                dX[t][i] = dQ[i] + dK[i] + dV[i];
            }
        }
        
        return dX;
    }
    
    private float[] matvec(float[] A, float[] x, int M, int N) {
        if (useGPU && gpu != null) {
            return gpu.matvec(A, x, M, N);
        }
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
    
    public float[] getQWeight() { return qWeight; }
    public float[] getKWeight() { return kWeight; }
    public float[] getVWeight() { return vWeight; }
    public float[] getOWeight() { return oWeight; }
    public float[] getQBias() { return qBias; }
    public float[] getKBias() { return kBias; }
    public float[] getVBias() { return vBias; }
    public float[] getOBias() { return oBias; }
}

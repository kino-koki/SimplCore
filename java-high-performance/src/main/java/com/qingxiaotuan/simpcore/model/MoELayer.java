package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;

/**
 * MoE (Mixture of Experts) Layer - 多专家堆叠架构
 * 实现路由层和集成层
 */
public class MoELayer {
    
    private final int dModel;
    private final int nExperts;
    private final int topK;  // 每个token选择的专家数
    
    // 专家模型
    private TransformerBlock[] experts;
    
    // 路由参数
    private float[] routerWeight;  // [dModel, nExperts]
    private float[] routerBias;    // [nExperts]
    
    // GPU加速
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    // 缓存
    private float[][] routerLogits;
    private float[][] routerWeights;
    private int[][] expertAssignments;
    
    public MoELayer(int dModel, int nHeads, int nExperts, int topK) {
        this.dModel = dModel;
        this.nExperts = nExperts;
        this.topK = topK;
        initExperts(nHeads);
        initRouter();
        initializeGPU();
    }
    
    private void initExperts(int nHeads) {
        experts = new TransformerBlock[nExperts];
        for (int i = 0; i < nExperts; i++) {
            experts[i] = new TransformerBlock(dModel, nHeads);
        }
    }
    
    private void initRouter() {
        float scale = (float) (1.0 / Math.sqrt(dModel));
        routerWeight = new float[dModel * nExperts];
        routerBias = new float[nExperts];
        
        for (int i = 0; i < dModel * nExperts; i++) {
            routerWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
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
     * @param x 输入 [seqLen, dModel]
     * @return 输出 [seqLen, dModel]
     */
    public float[][] forward(float[][] x) {
        int seqLen = x.length;
        
        // 计算路由logits
        routerLogits = new float[seqLen][nExperts];
        for (int t = 0; t < seqLen; t++) {
            float[] logits = matvec(routerWeight, x[t], nExperts, dModel);
            for (int e = 0; e < nExperts; e++) {
                routerLogits[t][e] = logits[e] + routerBias[e];
            }
        }
        
        // 选择topK专家并计算权重
        routerWeights = new float[seqLen][nExperts];
        expertAssignments = new int[seqLen][topK];
        
        for (int t = 0; t < seqLen; t++) {
            // 找出topK专家
            int[] indices = new int[nExperts];
            for (int i = 0; i < nExperts; i++) indices[i] = i;
            
            // 选择排序
            for (int i = 0; i < topK; i++) {
                int maxIdx = i;
                for (int j = i + 1; j < nExperts; j++) {
                    if (routerLogits[t][j] > routerLogits[t][maxIdx]) {
                        maxIdx = j;
                    }
                }
                int temp = indices[i];
                indices[i] = indices[maxIdx];
                indices[maxIdx] = temp;
            }
            
            // 保存topK专家索引
            for (int i = 0; i < topK; i++) {
                expertAssignments[t][i] = indices[i];
            }
            
            // 计算softmax权重（只在topK专家上）
            float maxLogit = routerLogits[t][indices[0]];
            float sum = 0;
            for (int i = 0; i < topK; i++) {
                sum += (float) Math.exp(routerLogits[t][indices[i]] - maxLogit);
            }
            
            for (int i = 0; i < topK; i++) {
                routerWeights[t][indices[i]] = (float) Math.exp(routerLogits[t][indices[i]] - maxLogit) / sum;
            }
        }
        
        // 专家前向传播
        float[][][] expertOutputs = new float[nExperts][][];
        for (int e = 0; e < nExperts; e++) {
            expertOutputs[e] = experts[e].forward(x);
        }
        
        // 集成专家输出
        float[][] result = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < dModel; i++) {
                for (int k = 0; k < topK; k++) {
                    int expertIdx = expertAssignments[t][k];
                    result[t][i] += routerWeights[t][expertIdx] * expertOutputs[expertIdx][t][i];
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
        
        // 专家反向传播
        float[][] dInput = new float[seqLen][dModel];
        
        for (int e = 0; e < nExperts; e++) {
            // 计算该专家的梯度
            float[][] dExpert = new float[seqLen][dModel];
            for (int t = 0; t < seqLen; t++) {
                for (int k = 0; k < topK; k++) {
                    if (expertAssignments[t][k] == e) {
                        for (int i = 0; i < dModel; i++) {
                            dExpert[t][i] += routerWeights[t][e] * dOutput[t][i];
                        }
                    }
                }
            }
            
            // 专家反向传播
            float[][] dExpertInput = experts[e].backward(dExpert);
            
            // 累加梯度
            for (int t = 0; t < seqLen; t++) {
                for (int i = 0; i < dModel; i++) {
                    dInput[t][i] += dExpertInput[t][i];
                }
            }
        }
        
        // 路由梯度（简化版）
        return dInput;
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
    
    public TransformerBlock[] getExperts() { return experts; }
    public float[] getRouterWeight() { return routerWeight; }
    public float[] getRouterBias() { return routerBias; }
    public int getNExperts() { return nExperts; }
}

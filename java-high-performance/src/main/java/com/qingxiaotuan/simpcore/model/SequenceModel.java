package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.HashMap;
import java.util.Map;

/**
 * 序列模型 - 支持序列处理的简化Transformer模型
 * 使用循环方式处理序列中的每个token
 * 支持OpenCL GPU加速
 */
public class SequenceModel {
    
    private final int vocabSize;
    private final int dModel;
    private final int nLayers;
    private final int nHeads;
    
    // 参数
    private float[] embedding;           // [vocabSize, dModel]
    private float[] lmHeadWeight;        // [dModel, vocabSize]
    private float[] lmHeadBias;          // [vocabSize]
    
    // 循环层参数
    private float[][] rnnWeights;        // [nLayers, dModel * dModel]
    private float[][] rnnBiases;         // [nLayers, dModel]
    private float[][] rnnOutputWeights;  // [nLayers, dModel * dModel]
    private float[][] rnnOutputBiases;   // [nLayers, dModel]
    
    // 前向传播缓存
    private float[][] layerOutputs;      // [nLayers+1, seqLength, dModel]
    private float[][] layerInputs;       // [nLayers, seqLength, dModel]
    private int[] currentTokens;
    
    // GPU加速器
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    public SequenceModel(int vocabSize, int dModel, int nLayers, int nHeads) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.nLayers = nLayers;
        this.nHeads = nHeads;
        initWeights();
        initializeGPU();
    }
    
    /**
     * 初始化GPU加速
     */
    private void initializeGPU() {
        try {
            gpu = new OpenCLAccelerator();
            gpu.initialize();
            useGPU = gpu.isUsingGPU() && gpu.isInitialized();
            System.out.printf("[SequenceModel] GPU加速: %s\n", useGPU ? "已启用" : "未启用");
        } catch (Exception e) {
            System.out.println("[SequenceModel] GPU初始化失败: " + e.getMessage());
            useGPU = false;
        }
    }
    
    private void initWeights() {
        float scale = (float) (1.0 / Math.sqrt(dModel));
        
        // 嵌入层
        embedding = new float[vocabSize * dModel];
        for (int i = 0; i < vocabSize * dModel; i++) {
            embedding[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        // LM Head
        lmHeadWeight = new float[dModel * vocabSize];
        lmHeadBias = new float[vocabSize];
        for (int i = 0; i < dModel * vocabSize; i++) {
            lmHeadWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        // RNN层
        rnnWeights = new float[nLayers][];
        rnnBiases = new float[nLayers][];
        rnnOutputWeights = new float[nLayers][];
        rnnOutputBiases = new float[nLayers][];
        
        for (int l = 0; l < nLayers; l++) {
            rnnWeights[l] = new float[dModel * dModel];
            rnnBiases[l] = new float[dModel];
            rnnOutputWeights[l] = new float[dModel * dModel];
            rnnOutputBiases[l] = new float[dModel];
            
            for (int i = 0; i < dModel * dModel; i++) {
                rnnWeights[l][i] = (float) (Math.random() - 0.5) * 2 * scale;
                rnnOutputWeights[l][i] = (float) (Math.random() - 0.5) * 2 * scale;
            }
        }
    }
    
    /**
     * 前向传播 - 处理序列并返回最后一个token的logits
     */
    public float[] forward(int[] tokens) {
        currentTokens = tokens;
        int seqLen = tokens.length;
        
        // 初始化缓存
        layerOutputs = new float[nLayers + 1][seqLen * dModel];
        layerInputs = new float[nLayers][seqLen * dModel];
        
        // 嵌入层
        for (int t = 0; t < seqLen; t++) {
            int token = tokens[t];
            if (token >= 0 && token < vocabSize) {
                System.arraycopy(embedding, token * dModel, 
                    layerOutputs[0], t * dModel, dModel);
            }
        }
        
        // 循环层
        for (int l = 0; l < nLayers; l++) {
            for (int t = 0; t < seqLen; t++) {
                // 获取前一层的输出
                float[] prevOutput = new float[dModel];
                System.arraycopy(layerOutputs[l], t * dModel, prevOutput, 0, dModel);
                
                // 保存输入
                System.arraycopy(prevOutput, 0, layerInputs[l], t * dModel, dModel);
                
                // 仿射变换
                float[] hidden = matvec(rnnWeights[l], prevOutput, dModel, dModel);
                for (int i = 0; i < dModel; i++) {
                    hidden[i] += rnnBiases[l][i];
                }
                
                // ReLU激活
                relu(hidden);
                
                // 输出变换
                float[] output = matvec(rnnOutputWeights[l], hidden, dModel, dModel);
                for (int i = 0; i < dModel; i++) {
                    output[i] += rnnOutputBiases[l][i];
                }
                
                // 残差连接
                for (int i = 0; i < dModel; i++) {
                    output[i] += prevOutput[i];
                }
                
                // 保存输出
                System.arraycopy(output, 0, layerOutputs[l + 1], t * dModel, dModel);
            }
        }
        
        // 获取最后一个token的表示
        float[] lastOutput = new float[dModel];
        System.arraycopy(layerOutputs[nLayers], (seqLen - 1) * dModel, lastOutput, 0, dModel);
        
        // LM Head
        float[] logits = matvec(lmHeadWeight, lastOutput, vocabSize, dModel);
        for (int i = 0; i < vocabSize; i++) {
            logits[i] += lmHeadBias[i];
        }
        
        return logits;
    }
    
    /**
     * 反向传播
     */
    public GradientResult backward(float[] dLogits) {
        GradientResult result = new GradientResult(new float[dModel]);
        
        int seqLen = currentTokens.length;
        Map<String, float[]> paramGrads = new HashMap<>();
        
        // 1. LM Head梯度
        float[] dLmHeadWeight = new float[dModel * vocabSize];
        float[] dLmHeadBias = new float[vocabSize];
        
        float[] lastOutput = new float[dModel];
        System.arraycopy(layerOutputs[nLayers], (seqLen - 1) * dModel, lastOutput, 0, dModel);
        
        // dL/dW = lastOutput^T * dLogits
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSize; j++) {
                dLmHeadWeight[i * vocabSize + j] += lastOutput[i] * dLogits[j];
            }
        }
        
        // dL/db = dLogits
        System.arraycopy(dLogits, 0, dLmHeadBias, 0, vocabSize);
        
        paramGrads.put("lmHeadWeight", dLmHeadWeight);
        paramGrads.put("lmHeadBias", dLmHeadBias);
        
        // 2. 反向传播通过LM Head
        float[] dLastOutput = matvec(transpose(lmHeadWeight, dModel, vocabSize), dLogits, dModel, vocabSize);
        
        // 3. 反向传播通过循环层
        float[][] dLayerOutputs = new float[nLayers + 1][seqLen * dModel];
        System.arraycopy(dLastOutput, 0, dLayerOutputs[nLayers], (seqLen - 1) * dModel, dModel);
        
        for (int l = nLayers - 1; l >= 0; l--) {
            float[] dRnnWeight = new float[dModel * dModel];
            float[] dRnnBias = new float[dModel];
            float[] dOutputWeight = new float[dModel * dModel];
            float[] dOutputBias = new float[dModel];
            
            for (int t = seqLen - 1; t >= 0; t--) {
                float[] dOut = new float[dModel];
                System.arraycopy(dLayerOutputs[l + 1], t * dModel, dOut, 0, dModel);
                
                float[] prevOutput = new float[dModel];
                System.arraycopy(layerOutputs[l], t * dModel, prevOutput, 0, dModel);
                
                float[] input = new float[dModel];
                System.arraycopy(layerInputs[l], t * dModel, input, 0, dModel);
                
                // 残差连接梯度
                float[] dResidual = new float[dModel];
                System.arraycopy(dOut, 0, dResidual, 0, dModel);
                
                // 输出层梯度
                float[] dHidden = matvec(transpose(rnnOutputWeights[l], dModel, dModel), dOut, dModel, dModel);
                
                // 输出权重梯度
                float[] hidden = matvec(rnnWeights[l], input, dModel, dModel);
                for (int i = 0; i < dModel; i++) {
                    hidden[i] += rnnBiases[l][i];
                }
                relu(hidden);
                
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dOutputWeight[i * dModel + j] += hidden[i] * dOut[j];
                    }
                }
                System.arraycopy(dOut, 0, dOutputBias, 0, dModel);
                
                // ReLU梯度
                for (int i = 0; i < dModel; i++) {
                    if (hidden[i] <= 0) {
                        dHidden[i] = 0;
                    }
                }
                
                // RNN权重梯度
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        dRnnWeight[i * dModel + j] += input[i] * dHidden[j];
                    }
                }
                System.arraycopy(dHidden, 0, dRnnBias, 0, dModel);
                
                // 前一层梯度
                float[] dPrev = matvec(transpose(rnnWeights[l], dModel, dModel), dHidden, dModel, dModel);
                for (int i = 0; i < dModel; i++) {
                    dPrev[i] += dResidual[i];
                }
                
                System.arraycopy(dPrev, 0, dLayerOutputs[l], t * dModel, dModel);
            }
            
            paramGrads.put("rnnWeight_" + l, dRnnWeight);
            paramGrads.put("rnnBias_" + l, dRnnBias);
            paramGrads.put("outputWeight_" + l, dOutputWeight);
            paramGrads.put("outputBias_" + l, dOutputBias);
        }
        
        // 4. Embedding梯度
        float[] dEmbedding = new float[vocabSize * dModel];
        for (int t = 0; t < seqLen; t++) {
            int token = currentTokens[t];
            if (token >= 0 && token < vocabSize) {
                float[] dIn = new float[dModel];
                System.arraycopy(dLayerOutputs[0], t * dModel, dIn, 0, dModel);
                for (int i = 0; i < dModel; i++) {
                    dEmbedding[token * dModel + i] += dIn[i];
                }
            }
        }
        
        paramGrads.put("embedding", dEmbedding);
        
        // 将梯度添加到result中
        for (Map.Entry<String, float[]> entry : paramGrads.entrySet()) {
            result.addGradient(entry.getKey(), entry.getValue());
        }
        
        return result;
    }
    
    /**
     * 获取嵌入
     */
    public float[] embed(int[] tokens) {
        float[] result = new float[tokens.length * dModel];
        for (int t = 0; t < tokens.length; t++) {
            int token = tokens[t];
            if (token >= 0 && token < vocabSize) {
                System.arraycopy(embedding, token * dModel, result, t * dModel, dModel);
            }
        }
        return result;
    }
    
    /**
     * 更新参数
     */
    public void updateParametersWithGradient(Map<String, float[]> grads, float learningRate) {
        // 更新embedding
        float[] dEmbedding = grads.get("embedding");
        if (dEmbedding != null) {
            for (int i = 0; i < embedding.length && i < dEmbedding.length; i++) {
                embedding[i] -= learningRate * dEmbedding[i];
            }
        }
        
        // 更新LM Head
        float[] dLmHeadWeight = grads.get("lmHeadWeight");
        if (dLmHeadWeight != null) {
            for (int i = 0; i < lmHeadWeight.length && i < dLmHeadWeight.length; i++) {
                lmHeadWeight[i] -= learningRate * dLmHeadWeight[i];
            }
        }
        
        float[] dLmHeadBias = grads.get("lmHeadBias");
        if (dLmHeadBias != null) {
            for (int i = 0; i < lmHeadBias.length && i < dLmHeadBias.length; i++) {
                lmHeadBias[i] -= learningRate * dLmHeadBias[i];
            }
        }
        
        // 更新RNN层
        for (int l = 0; l < nLayers; l++) {
            float[] dRnnWeight = grads.get("rnnWeight_" + l);
            if (dRnnWeight != null) {
                for (int i = 0; i < rnnWeights[l].length && i < dRnnWeight.length; i++) {
                    rnnWeights[l][i] -= learningRate * dRnnWeight[i];
                }
            }
            
            float[] dRnnBias = grads.get("rnnBias_" + l);
            if (dRnnBias != null) {
                for (int i = 0; i < rnnBiases[l].length && i < dRnnBias.length; i++) {
                    rnnBiases[l][i] -= learningRate * dRnnBias[i];
                }
            }
            
            float[] dOutputWeight = grads.get("outputWeight_" + l);
            if (dOutputWeight != null) {
                for (int i = 0; i < rnnOutputWeights[l].length && i < dOutputWeight.length; i++) {
                    rnnOutputWeights[l][i] -= learningRate * dOutputWeight[i];
                }
            }
            
            float[] dOutputBias = grads.get("outputBias_" + l);
            if (dOutputBias != null) {
                for (int i = 0; i < rnnOutputBiases[l].length && i < dOutputBias.length; i++) {
                    rnnOutputBiases[l][i] -= learningRate * dOutputBias[i];
                }
            }
        }
    }
    
    /**
     * 获取所有权重数组
     */
    public float[][] getAllWeightArrays() {
        java.util.List<float[]> allWeights = new java.util.ArrayList<>();
        allWeights.add(embedding);
        allWeights.add(lmHeadWeight);
        allWeights.add(lmHeadBias);
        
        for (int l = 0; l < nLayers; l++) {
            allWeights.add(rnnWeights[l]);
            allWeights.add(rnnBiases[l]);
            allWeights.add(rnnOutputWeights[l]);
            allWeights.add(rnnOutputBiases[l]);
        }
        
        return allWeights.toArray(new float[0][]);
    }
    
    /**
     * 获取参数的Map形式，用于Adam优化器
     */
    public java.util.Map<String, float[]> getParametersMap() {
        java.util.Map<String, float[]> params = new java.util.HashMap<>();
        params.put("embedding", embedding);
        params.put("lmHeadWeight", lmHeadWeight);
        params.put("lmHeadBias", lmHeadBias);
        
        for (int l = 0; l < nLayers; l++) {
            params.put("rnnWeight_" + l, rnnWeights[l]);
            params.put("rnnBias_" + l, rnnBiases[l]);
            params.put("outputWeight_" + l, rnnOutputWeights[l]);
            params.put("outputBias_" + l, rnnOutputBiases[l]);
        }
        
        return params;
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
    
    private void relu(float[] x) {
        if (useGPU && gpu != null) {
            gpu.relu(x);
            return;
        }
        for (int i = 0; i < x.length; i++) {
            if (x[i] < 0) {
                x[i] = 0;
            }
        }
    }
    
    public int getVocabSize() { return vocabSize; }
    public int getDim() { return dModel; }
}

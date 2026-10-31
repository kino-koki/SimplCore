package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;

import java.util.HashMap;
import java.util.Map;

/**
 * Transformer MoE Model - 完整的文本模型
 * 整合Multi-Head Attention、TransformerBlock和MoE专家堆叠架构
 */
public class TransformerMoEModel {
    
    // 模型配置
    private final int vocabSize;
    private final int dModel;
    private final int nLayers;
    private final int nHeads;
    private final int nExperts;
    private final int topK;
    
    // 组件
    private float[] embedding;           // [vocabSize, dModel]
    private MoELayer[] moeLayers;        // MoE层堆叠
    private float[] finalNormWeight;     // [dModel]
    private float[] finalNormBias;       // [dModel]
    private float[] lmHeadWeight;        // [dModel, vocabSize]
    private float[] lmHeadBias;          // [vocabSize]
    
    // GPU加速
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    // 缓存
    private float[][][] hiddenStates;
    private float[][] lastHiddenState;
    
    public TransformerMoEModel(int vocabSize, int dModel, int nLayers, int nHeads, int nExperts, int topK) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.nLayers = nLayers;
        this.nHeads = nHeads;
        this.nExperts = nExperts;
        this.topK = topK;
        
        initEmbedding();
        initMoELayers();
        initFinalNorm();
        initLMHead();
        initializeGPU();
    }
    
    private void initEmbedding() {
        float scale = (float) (1.0 / Math.sqrt(dModel));
        embedding = new float[vocabSize * dModel];
        for (int i = 0; i < vocabSize * dModel; i++) {
            embedding[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
    }
    
    private void initMoELayers() {
        moeLayers = new MoELayer[nLayers];
        for (int l = 0; l < nLayers; l++) {
            moeLayers[l] = new MoELayer(dModel, nHeads, nExperts, topK);
        }
    }
    
    private void initFinalNorm() {
        finalNormWeight = new float[dModel];
        finalNormBias = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            finalNormWeight[i] = 1.0f;
            finalNormBias[i] = 0.0f;
        }
    }
    
    private void initLMHead() {
        float scale = (float) (1.0 / Math.sqrt(dModel));
        lmHeadWeight = new float[dModel * vocabSize];
        lmHeadBias = new float[vocabSize];
        
        for (int i = 0; i < dModel * vocabSize; i++) {
            lmHeadWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
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
     * @param inputIds 输入token序列 [seqLen]
     * @return logits [seqLen, vocabSize]
     */
    public float[][] forward(int[] inputIds) {
        int seqLen = inputIds.length;
        
        // 获取嵌入
        float[][] hidden = new float[seqLen][dModel];
        for (int t = 0; t < seqLen; t++) {
            int tokenId = inputIds[t];
            if (tokenId >= 0 && tokenId < vocabSize) {
                System.arraycopy(embedding, tokenId * dModel, hidden[t], 0, dModel);
            }
        }
        
        // 保存隐藏状态
        hiddenStates = new float[nLayers][][];
        
        // MoE层前向传播
        for (int l = 0; l < nLayers; l++) {
            hidden = moeLayers[l].forward(hidden);
            hiddenStates[l] = hidden;
        }
        
        // 最终层归一化
        lastHiddenState = layerNorm(hidden, finalNormWeight, finalNormBias);
        
        // LM头
        float[][] logits = new float[seqLen][vocabSize];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < dModel; j++) {
                    logits[t][i] += lastHiddenState[t][j] * lmHeadWeight[j * vocabSize + i];
                }
                logits[t][i] += lmHeadBias[i];
            }
        }
        
        return logits;
    }
    
    /**
     * 层归一化
     */
    private float[][] layerNorm(float[][] x, float[] weight, float[] bias) {
        int seqLen = x.length;
        float[][] result = new float[seqLen][dModel];
        float eps = 1e-5f;
        
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
                result[t][i] = weight[i] * ((x[t][i] - mean) / std) + bias[i];
            }
        }
        
        return result;
    }
    
    /**
     * 获取参数的Map形式，用于Adam优化器
     */
    public Map<String, float[]> getParametersMap() {
        Map<String, float[]> params = new HashMap<>();
        params.put("embedding", embedding);
        params.put("lmHeadWeight", lmHeadWeight);
        params.put("lmHeadBias", lmHeadBias);
        params.put("finalNormWeight", finalNormWeight);
        params.put("finalNormBias", finalNormBias);
        
        // 添加MoE层参数
        for (int l = 0; l < nLayers; l++) {
            MoELayer moe = moeLayers[l];
            params.put("moe_" + l + "_routerWeight", moe.getRouterWeight());
            params.put("moe_" + l + "_routerBias", moe.getRouterBias());
            
            // 添加专家参数
            TransformerBlock[] experts = moe.getExperts();
            for (int e = 0; e < experts.length; e++) {
                MultiHeadAttention attn = experts[e].getAttention();
                params.put("moe_" + l + "_expert_" + e + "_qWeight", attn.getQWeight());
                params.put("moe_" + l + "_expert_" + e + "_kWeight", attn.getKWeight());
                params.put("moe_" + l + "_expert_" + e + "_vWeight", attn.getVWeight());
                params.put("moe_" + l + "_expert_" + e + "_oWeight", attn.getOWeight());
            }
        }
        
        return params;
    }
    
    /**
     * 计算参数总数
     */
    public long getParameterCount() {
        long count = 0;
        
        // 嵌入层
        count += (long) vocabSize * dModel;
        
        // LM头
        count += (long) dModel * vocabSize + vocabSize;
        
        // 最终归一化
        count += dModel * 2;
        
        // MoE层
        for (int l = 0; l < nLayers; l++) {
            // 路由参数
            count += (long) dModel * nExperts + nExperts;
            
            // 专家参数
            for (int e = 0; e < nExperts; e++) {
                // 每个专家：Q, K, V, Output投影
                count += (long) dModel * dModel * 4;
            }
        }
        
        return count;
    }
    
    /**
     * 生成文本（贪婪解码）
     */
    public int[] generate(int[] prompt, int maxLen) {
        int[] result = new int[prompt.length + maxLen];
        System.arraycopy(prompt, 0, result, 0, prompt.length);
        
        for (int i = prompt.length; i < result.length; i++) {
            int[] current = new int[i];
            System.arraycopy(result, 0, current, 0, i);
            
            float[][] logits = forward(current);
            float[] lastLogits = logits[i - 1];
            
            // 贪婪选择
            int nextToken = 0;
            float maxLogit = lastLogits[0];
            for (int j = 1; j < vocabSize; j++) {
                if (lastLogits[j] > maxLogit) {
                    maxLogit = lastLogits[j];
                    nextToken = j;
                }
            }
            
            result[i] = nextToken;
            
            // 检查结束
            if (nextToken == 0) break; // EOS
        }
        
        return result;
    }
    
    /**
     * 温度采样
     */
    public int sampleWithTemperature(float[] logits, float temperature) {
        // 温度缩放
        float[] scaled = new float[logits.length];
        float maxLogit = logits[0];
        for (float l : logits) {
            if (l > maxLogit) maxLogit = l;
        }
        
        // Softmax
        float sum = 0;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp((logits[i] - maxLogit) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < logits.length; i++) {
            probs[i] /= sum;
        }
        
        // 采样
        float r = (float) Math.random();
        float cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        
        return 0;
    }
    
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNLayers() { return nLayers; }
    public int getNHeads() { return nHeads; }
    public int getNExperts() { return nExperts; }
    public boolean isUsingGPU() { return useGPU; }
}

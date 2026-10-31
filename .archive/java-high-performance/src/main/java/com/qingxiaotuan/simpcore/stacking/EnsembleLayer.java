package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.*;

public class EnsembleLayer {
    
    private final int dModel;
    private final int vocabSize;
    private final int maxExperts;
    
    // 集成参数
    private float[][] attentionWeights;    // [dModel][dModel] - 输出注意力
    private float[][] fusionWeights;      // [maxExperts * dModel][dModel] - 融合权重
    private float[] fusionBias;           // [dModel] - 融合偏置
    private float[][] outputWeights;      // [dModel][vocabSize] - 输出投影
    
    // 层归一化参数
    private float[] layerNormWeight;
    private float[] layerNormBias;
    
    // 前向传播缓存
    private float[] fusedOutput;
    private float[] transformedOutput;
    private float[] attendedOutput;
    private float[] normalizedOutput;
    
    private static final Random RNG = new Random(42);
    
    public EnsembleLayer(int dModel, int vocabSize, int maxExperts) {
        this.dModel = dModel;
        this.vocabSize = vocabSize;
        this.maxExperts = maxExperts;
        
        // 初始化参数
        initParameters();
    }
    
    private void initParameters() {
        // 注意力权重
        this.attentionWeights = new float[dModel][dModel];
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                attentionWeights[i][j] = (RNG.nextFloat() - 0.5f) * 0.05f;
            }
            attentionWeights[i][i] += 1.0f; // 对角初始化
        }
        
        // 融合权重
        this.fusionWeights = new float[maxExperts * dModel][dModel];
        for (int i = 0; i < maxExperts * dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                fusionWeights[i][j] = (RNG.nextFloat() - 0.5f) * 0.02f;
            }
        }
        
        // 融合偏置
        this.fusionBias = new float[dModel];
        Arrays.fill(fusionBias, 0f);
        
        // 输出投影
        this.outputWeights = new float[dModel][vocabSize];
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSize; j++) {
                outputWeights[i][j] = (RNG.nextFloat() - 0.5f) * 0.02f;
            }
        }
        
        // 层归一化
        this.layerNormWeight = new float[dModel];
        this.layerNormBias = new float[dModel];
        Arrays.fill(layerNormWeight, 1.0f);
        Arrays.fill(layerNormBias, 0f);
    }
    
    public float[] ensemble(ExpertModelCollection.ExpertOutputCollection outputs, 
                            float[] inputEmbedding) {
        int nExperts = outputs.size();
        
        if (nExperts == 0) {
            float[] randomLogits = new float[vocabSize];
            Arrays.fill(randomLogits, 0f);
            // 初始化缓存变量避免反向传播时NPE
            fusedOutput = new float[dModel];
            transformedOutput = new float[dModel];
            attendedOutput = new float[dModel];
            normalizedOutput = new float[dModel];
            return randomLogits;
        }
        
        // 1. 获取专家权重
        float[] weights = outputs.getAllWeights();
        
        // 2. 加权融合专家输出
        float[] fused = new float[dModel];
        for (int e = 0; e < nExperts; e++) {
            float[] logits = outputs.getLogits(e);
            float weight = weights[e];
            
            float[] projected = projectToHidden(logits);
            
            for (int i = 0; i < dModel; i++) {
                fused[i] += projected[i] * weight;
            }
        }
        fusedOutput = fused;
        
        // 3. 应用融合变换
        float[] transformed = applyFusion(fused, nExperts);
        transformedOutput = transformed;
        
        // 4. 注意力增强
        float[] attended = applyAttention(transformed, inputEmbedding);
        attendedOutput = attended;
        
        // 5. 残差连接
        for (int i = 0; i < dModel; i++) {
            attended[i] += inputEmbedding[i] * 0.1f;
        }
        
        // 6. 层归一化
        float[] normalized = layerNorm(attended);
        normalizedOutput = normalized;
        
        // 7. 输出投影到vocab
        float[] finalLogits = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++) {
            float sum = 0f;
            for (int i = 0; i < dModel; i++) {
                sum += normalized[i] * outputWeights[i][v];
            }
            finalLogits[v] = sum;
        }
        
        return finalLogits;
    }
    
    /**
     * 反向传播
     */
    public GradientResult backward(float[] dLogits, 
                                   ExpertModelCollection.ExpertOutputCollection outputs,
                                   float[] inputEmbedding) {
        GradientResult result = new GradientResult(new float[dModel]);
        
        int nExperts = outputs.size();
        
        // 1. 输出投影梯度
        float[] dOutputWeights = new float[dModel * vocabSize];
        for (int v = 0; v < vocabSize; v++) {
            for (int i = 0; i < dModel; i++) {
                dOutputWeights[i * vocabSize + v] += normalizedOutput[i] * dLogits[v];
            }
        }
        result.addGradient("outputWeights", dOutputWeights);
        
        // 2. dNormalized = W_out^T * dLogits
        float[] dNormalized = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            for (int v = 0; v < vocabSize; v++) {
                dNormalized[i] += dLogits[v] * outputWeights[i][v];
            }
        }
        
        // 3. 层归一化反向传播
        float[] dAttended = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            dAttended[i] = dNormalized[i] * layerNormWeight[i];
        }
        
        // 4. 残差连接梯度
        float[] dInputEmbedding = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            dInputEmbedding[i] += dAttended[i] * 0.1f;
        }
        
        // 5. 注意力反向传播
        float[] dTransformed = new float[dModel];
        System.arraycopy(dAttended, 0, dTransformed, 0, dModel);
        
        // 6. 融合变换反向传播
        float[] dFused = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < maxExperts * dModel; j++) {
                dFused[i] += dTransformed[j % dModel] * fusionWeights[j][i];
            }
        }
        
        // 融合权重梯度
        float[] dFusionWeights = new float[maxExperts * dModel * dModel];
        float[] dFusionBias = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            dFusionBias[i] += dTransformed[i];
            for (int j = 0; j < maxExperts * dModel; j++) {
                dFusionWeights[j * dModel + i] += fusedOutput[j % dModel] * dTransformed[i];
            }
        }
        result.addGradient("fusionWeights", dFusionWeights);
        result.addGradient("fusionBias", dFusionBias);
        
        // 7. 专家输出梯度（反向传播到每个专家）
        float[] weights = outputs.getAllWeights();
        for (int e = 0; e < nExperts; e++) {
            float weight = weights[e];
            float[] dExpertLogits = new float[vocabSize];
            
            for (int v = 0; v < vocabSize; v++) {
                int start = (v % dModel) * (vocabSize / dModel);
                int end = Math.min(start + (vocabSize / dModel), vocabSize);
                for (int j = start; j < end; j++) {
                    dExpertLogits[j] += dFused[v % dModel] * weight / (end - start);
                }
            }
            
            // 存储每个专家的梯度
            result.addGradient("expert_" + e, dExpertLogits);
        }
        
        // 设置输入梯度
        float[] dInput = result.getdInput();
        System.arraycopy(dInputEmbedding, 0, dInput, 0, dModel);
        
        return result;
    }
    
    private float[] projectToHidden(float[] logits) {
        float[] hidden = new float[dModel];
        int step = vocabSize / dModel;
        
        for (int i = 0; i < dModel; i++) {
            int start = i * step;
            int end = Math.min(start + step, vocabSize);
            
            float sum = 0f;
            int count = 0;
            for (int j = start; j < end; j++) {
                sum += logits[j];
                count++;
            }
            hidden[i] = count > 0 ? sum / count : 0f;
        }
        
        return hidden;
    }
    
    private float[] applyFusion(float[] input, int nActiveExperts) {
        float[] output = new float[dModel];
        
        float[] expanded = new float[maxExperts * dModel];
        System.arraycopy(input, 0, expanded, 0, Math.min(dModel, input.length));
        
        for (int i = 0; i < dModel; i++) {
            output[i] = fusionBias[i];
            for (int j = 0; j < maxExperts * dModel; j++) {
                output[i] += expanded[j] * fusionWeights[j][i];
            }
        }
        
        for (int i = 0; i < dModel; i++) {
            output[i] = Math.max(0f, output[i]);
        }
        
        return output;
    }
    
    private float[] applyAttention(float[] input, float[] context) {
        float[] output = new float[dModel];
        
        float[] scores = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            float sum = 0f;
            for (int j = 0; j < dModel; j++) {
                sum += input[j] * context[j] * attentionWeights[j][i];
            }
            scores[i] = sum;
        }
        
        float[] attn = softmax(scores);
        
        for (int i = 0; i < dModel; i++) {
            output[i] = input[i] * attn[i];
        }
        
        return output;
    }
    
    private float[] softmax(float[] x) {
        float max = x[0];
        for (float val : x) {
            if (val > max) max = val;
        }
        float[] exp = new float[x.length];
        float sum = 0f;
        for (int i = 0; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }
    
    private float[] layerNorm(float[] x) {
        float[] output = new float[dModel];
        
        float mean = 0f;
        for (float val : x) mean += val;
        mean /= dModel;
        
        float variance = 0f;
        for (float val : x) {
            variance += (val - mean) * (val - mean);
        }
        variance /= dModel;
        variance = (float) Math.sqrt(variance + 1e-8f);
        
        for (int i = 0; i < dModel; i++) {
            output[i] = (x[i] - mean) / variance * layerNormWeight[i] + layerNormBias[i];
        }
        
        return output;
    }
    
    /**
     * 获取所有权重数组
     */
    public float[][] getWeightArrays() {
        return new float[][]{
            flatten(attentionWeights),
            flatten(fusionWeights),
            fusionBias,
            flatten(outputWeights),
            layerNormWeight,
            layerNormBias
        };
    }
    
    private float[] flatten(float[][] matrix) {
        float[] result = new float[matrix.length * matrix[0].length];
        int idx = 0;
        for (float[] row : matrix) {
            for (float val : row) {
                result[idx++] = val;
            }
        }
        return result;
    }
    
    public int getDModel() { return dModel; }
    public int getVocabSize() { return vocabSize; }
    public int getMaxExperts() { return maxExperts; }
    
    public long getParameterCount() {
        long total = 0;
        total += (long) dModel * dModel;
        total += (long) maxExperts * dModel * dModel;
        total += dModel;
        total += (long) dModel * vocabSize;
        total += 2L * dModel;
        return total;
    }
}
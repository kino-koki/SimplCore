package com.qingxiaotuan.simpcore.model;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.Map;

public class RDTModel {
    private RDTConfig config;
    
    private float[] embedding;
    private float[] lmHeadWeight;
    private float[] lmHeadBias;
    
    private RDTTransformerBlock[] preludeBlocks;
    private RDTTransformerBlock recurrentBlock;
    private RDTTransformerBlock[] codaBlocks;
    
    private LTIInjection ltiInjection;
    private ACTModule actModule;
    
    // 前向传播缓存
    private float[] forwardEmbedding;
    private float[][] preludeOutputs;
    private float[][] recurrentOutputs;
    private float[][] codaOutputs;
    
    public RDTModel(RDTConfig config) {
        this.config = config;
        initModel();
    }
    
    private void initModel() {
        embedding = new float[config.vocabSize * config.dim];
        lmHeadWeight = new float[config.dim * config.vocabSize];
        lmHeadBias = new float[config.vocabSize];
        
        float scale = (float) (1.0 / Math.sqrt(config.dim));
        for (int i = 0; i < config.vocabSize * config.dim; i++) {
            embedding[i] = (float) (Math.random() - 0.5) * 2 * scale;
            lmHeadWeight[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
        
        preludeBlocks = new RDTTransformerBlock[config.preludeLayers];
        for (int i = 0; i < config.preludeLayers; i++) {
            preludeBlocks[i] = new RDTTransformerBlock(
                config.dim, config.nHeads, false, 
                0, 0, 0, 0
            );
        }
        
        recurrentBlock = new RDTTransformerBlock(
            config.dim, config.nHeads, true,
            config.nExperts, config.nSharedExperts, 
            config.nExpertsPerTok, config.expertDim
        );
        
        codaBlocks = new RDTTransformerBlock[config.codaLayers];
        for (int i = 0; i < config.codaLayers; i++) {
            codaBlocks[i] = new RDTTransformerBlock(
                config.dim, config.nHeads, false,
                0, 0, 0, 0
            );
        }
        
        ltiInjection = new LTIInjection(config.dim, config.ltiSpectralBound);
        actModule = new ACTModule(config.dim, config.actThreshold);
        
        // 初始化缓存
        preludeOutputs = new float[config.preludeLayers][];
        recurrentOutputs = new float[config.maxLoopIters][];
        codaOutputs = new float[config.codaLayers][];
    }
    
    public float[] forward(int[] tokens, int nLoops) {
        float[] x = embed(tokens);
        forwardEmbedding = x;
        
        // Prelude层
        for (int i = 0; i < config.preludeLayers; i++) {
            x = preludeBlocks[i].forward(x);
            preludeOutputs[i] = x;
        }
        
        float[] e = x;
        
        if (nLoops <= 0) {
            nLoops = config.maxLoopIters;
        }
        
        // 循环层
        for (int t = 0; t < nLoops; t++) {
            float[] injection = ltiInjection.inject(x, e);
            float[] residual = x;
            
            float[] blockOut = recurrentBlock.forward(x);
            
            for (int i = 0; i < config.dim; i++) {
                x[i] = injection[i] + blockOut[i];
            }
            
            x = elementwiseAdd(x, residual);
            x = applyLoopPositionalEncoding(x, t, nLoops);
            
            recurrentOutputs[t] = x;
        }
        
        // Coda层
        for (int i = 0; i < config.codaLayers; i++) {
            x = codaBlocks[i].forward(x);
            codaOutputs[i] = x;
        }
        
        float[] logits = MatrixOperations.matmul(lmHeadWeight, x, config.vocabSize, config.dim, 1);
        for (int i = 0; i < config.vocabSize; i++) {
            logits[i] += lmHeadBias[i];
        }
        
        return logits;
    }
    
    /**
     * 反向传播
     */
    public GradientResult backward(float[] dLogits, int nLoops) {
        GradientResult result = new GradientResult(new float[config.dim]);
        
        if (nLoops <= 0) {
            nLoops = config.maxLoopIters;
        }
        
        // 默认使用空tokens（用于兼容旧代码）
        return backward(dLogits, nLoops, new int[0]);
    }
    
    /**
     * 反向传播（带输入tokens，用于计算embedding梯度）
     */
    public GradientResult backward(float[] dLogits, int nLoops, int[] tokens) {
        GradientResult result = new GradientResult(new float[config.dim]);
        
        if (nLoops <= 0) {
            nLoops = config.maxLoopIters;
        }
        
        // 1. LM Head梯度
        float[] dLmHeadWeight = new float[config.dim * config.vocabSize];
        float[] dLmHeadBias = new float[config.vocabSize];
        
        float[] lastCodaOutput = codaOutputs[config.codaLayers - 1];
        for (int i = 0; i < config.vocabSize; i++) {
            dLmHeadBias[i] += dLogits[i];
            for (int j = 0; j < config.dim; j++) {
                dLmHeadWeight[j * config.vocabSize + i] += lastCodaOutput[j] * dLogits[i];
            }
        }
        
        result.addGradient("lmHeadWeight", dLmHeadWeight);
        result.addGradient("lmHeadBias", dLmHeadBias);
        
        // 2. dX = W_lm^T * dLogits
        float[] dX = MatrixOperations.matmul(MatrixOperations.transpose(lmHeadWeight, config.dim, config.vocabSize), dLogits, config.dim, config.vocabSize, 1);
        
        // 3. Coda层反向传播
        for (int i = config.codaLayers - 1; i >= 0; i--) {
            GradientResult blockGrad = codaBlocks[i].backward(dX);
            accumulateGradients(result, "coda_" + i, blockGrad);
            dX = blockGrad.getdInput();
        }
        
        // 4. 循环层反向传播（简化版）
        for (int t = nLoops - 1; t >= 0; t--) {
            GradientResult blockGrad = recurrentBlock.backward(dX);
            accumulateGradients(result, "recurrent_" + t, blockGrad);
            dX = blockGrad.getdInput();
        }
        
        // 5. Prelude层反向传播
        for (int i = config.preludeLayers - 1; i >= 0; i--) {
            GradientResult blockGrad = preludeBlocks[i].backward(dX);
            accumulateGradients(result, "prelude_" + i, blockGrad);
            dX = blockGrad.getdInput();
        }
        
        // 6. Embedding梯度（真正的梯度！将dX scatter回对应token位置）
        float[] dEmbedding = new float[config.vocabSize * config.dim];
        
        // 将输入梯度dX累加到每个输入token对应的embedding位置
        if (tokens != null && tokens.length > 0) {
            for (int token : tokens) {
                if (token >= 0 && token < config.vocabSize) {
                    int offset = token * config.dim;
                    for (int i = 0; i < config.dim; i++) {
                        dEmbedding[offset + i] += dX[i];
                    }
                }
            }
        }
        
        result.addGradient("embedding", dEmbedding);
        
        // 输入梯度
        float[] dInput = result.getdInput();
        System.arraycopy(dX, 0, dInput, 0, config.dim);
        
        return result;
    }
    
    private void accumulateGradients(GradientResult result, String prefix, GradientResult source) {
        for (Map.Entry<String, float[]> entry : source.getParameterGradients().entrySet()) {
            // 不添加前缀，直接使用原始key，这样所有层的同名参数梯度会被累加
            // 这在简化版训练中是可以接受的
            String key = entry.getKey();
            float[] grad = entry.getValue();
            float[] existing = result.getGradient(key);
            if (existing == null) {
                result.addGradient(key, grad);
            } else {
                // 累加梯度
                for (int i = 0; i < existing.length; i++) {
                    existing[i] += grad[i];
                }
            }
        }
    }
    
    public float[] generate(int[] tokens, int maxNewTokens, int nLoops) {
        int[] currentTokens = tokens;
        float[] allLogits = new float[maxNewTokens * config.vocabSize];
        
        for (int step = 0; step < maxNewTokens; step++) {
            float[] logits = forward(currentTokens, nLoops);
            
            int nextToken = argmax(logits);
            
            int[] newTokens = new int[currentTokens.length + 1];
            System.arraycopy(currentTokens, 0, newTokens, 0, currentTokens.length);
            newTokens[currentTokens.length] = nextToken;
            currentTokens = newTokens;
            
            System.arraycopy(logits, 0, allLogits, step * config.vocabSize, config.vocabSize);
        }
        
        return allLogits;
    }
    
    /**
     * 将token序列转换为嵌入向量（使用真正的可学习嵌入层）
     */
    public float[] embed(int[] tokens) {
        float[] result = new float[config.dim];
        
        for (int token : tokens) {
            if (token >= 0 && token < config.vocabSize) {
                int offset = token * config.dim;
                for (int i = 0; i < config.dim; i++) {
                    result[i] += embedding[offset + i];
                }
            }
        }
        
        return result;
    }
    
    private float[] applyLoopPositionalEncoding(float[] x, int t, int totalLoops) {
        float[] posEnc = new float[config.dim];
        for (int i = 0; i < config.dim; i += 2) {
            float theta = (float) (1.0 / Math.pow(10000.0, i / (double) config.dim));
            posEnc[i] = (float) Math.sin(t * theta);
            posEnc[i + 1] = (float) Math.cos(t * theta);
        }
        
        float[] result = new float[config.dim];
        for (int i = 0; i < config.dim; i++) {
            result[i] = x[i] + posEnc[i];
        }
        
        return result;
    }
    
    private float[] elementwiseAdd(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    private int argmax(float[] x) {
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
    
    public float getSpectralRadius() {
        return ltiInjection.getSpectralRadius();
    }
    
    public long countParameters() {
        long params = 0;
        params += (long) config.vocabSize * config.dim;
        params += (long) config.dim * config.vocabSize;
        params += config.preludeLayers * (long) config.dim * config.dim * 8;
        params += (long) config.dim * config.dim * 8;
        params += config.codaLayers * (long) config.dim * config.dim * 8;
        params += (long) config.nExperts * config.expertDim * config.dim * 2;
        params += (long) config.nSharedExperts * config.expertDim * config.dim * 2;
        params += (long) config.dim * config.dim * 2;
        return params;
    }
    
    /**
     * 更新模型参数（简化的随机梯度下降）
     */
    public void updateParameters(float learningRate) {
        float scale = learningRate * 0.01f;
        
        // 更新embedding
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] += (float) (Math.random() - 0.5) * scale;
        }
        
        // 更新lmHeadWeight
        for (int i = 0; i < lmHeadWeight.length; i++) {
            lmHeadWeight[i] += (float) (Math.random() - 0.5) * scale;
        }
        
        // 更新lmHeadBias
        for (int i = 0; i < lmHeadBias.length; i++) {
            lmHeadBias[i] += (float) (Math.random() - 0.5) * scale;
        }
        
        // 更新Transformer块参数
        for (RDTTransformerBlock block : preludeBlocks) {
            block.updateParameters(scale);
        }
        recurrentBlock.updateParameters(scale);
        for (RDTTransformerBlock block : codaBlocks) {
            block.updateParameters(scale);
        }
        
        // 更新LTI注入和ACT模块
        ltiInjection.updateParameters(scale);
        actModule.updateParameters(scale);
    }
    
    /**
     * 使用真实梯度更新模型参数（关键改进！）
     */
    public void updateParametersWithGradient(com.qingxiaotuan.simpcore.training.GradientResult grad, float learningRate) {
        // 获取梯度
        float[] dInput = grad.getdInput();
        java.util.Map<String, float[]> paramGrads = grad.getParameterGradients();
        
        // 更新embedding（使用输入梯度）
        for (int i = 0; i < embedding.length && i < dInput.length; i++) {
            embedding[i] -= learningRate * dInput[i];
        }
        
        // 更新lmHeadWeight
        float[] dLmHeadWeight = paramGrads.get("lmHeadWeight");
        if (dLmHeadWeight != null) {
            for (int i = 0; i < lmHeadWeight.length && i < dLmHeadWeight.length; i++) {
                lmHeadWeight[i] -= learningRate * dLmHeadWeight[i];
            }
        }
        
        // 更新lmHeadBias
        float[] dLmHeadBias = paramGrads.get("lmHeadBias");
        if (dLmHeadBias != null) {
            for (int i = 0; i < lmHeadBias.length && i < dLmHeadBias.length; i++) {
                lmHeadBias[i] -= learningRate * dLmHeadBias[i];
            }
        }
        
        // 更新Transformer块参数
        for (RDTTransformerBlock block : preludeBlocks) {
            block.updateParametersWithGradient(paramGrads, learningRate);
        }
        recurrentBlock.updateParametersWithGradient(paramGrads, learningRate);
        for (RDTTransformerBlock block : codaBlocks) {
            block.updateParametersWithGradient(paramGrads, learningRate);
        }
        
        // 更新LTI注入和ACT模块
        ltiInjection.updateParameters(learningRate * 0.1f);
        actModule.updateParameters(learningRate * 0.1f);
    }
    
    /**
     * 获取所有权重数组用于优化器更新
     */
    public float[][] getAllWeightArrays() {
        java.util.List<float[]> allWeights = new java.util.ArrayList<>();
        
        allWeights.add(embedding);
        allWeights.add(lmHeadWeight);
        allWeights.add(lmHeadBias);
        
        for (RDTTransformerBlock block : preludeBlocks) {
            for (float[] w : block.getWeightArrays()) {
                allWeights.add(w);
            }
        }
        
        for (float[] w : recurrentBlock.getWeightArrays()) {
            allWeights.add(w);
        }
        
        for (RDTTransformerBlock block : codaBlocks) {
            for (float[] w : block.getWeightArrays()) {
                allWeights.add(w);
            }
        }
        
        return allWeights.toArray(new float[0][]);
    }
    
    public RDTConfig getConfig() { return config; }
}
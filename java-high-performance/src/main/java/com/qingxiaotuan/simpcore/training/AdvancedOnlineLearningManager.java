package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.components.PlasticityUpdateModule;
import java.util.*;

/**
 * Advanced Online Learning Manager — 集成EWC + 动态记忆回放的高级在线学习系统
 * 
 * 核心创新集成:
 * 1. EWC弹性权重巩固 — 防止灾难性遗忘
 * 2. 动态记忆回放缓冲区 — 定期巩固旧知识
 * 3. Hebbian + 监督双通道学习
 * 4. 参数重要性感知的自适应学习率
 * 
 * 工作流程:
 * 1. 输入处理 → 特征提取
 * 2. EWC正则化损失计算
 * 3. 记忆回放巩固
 * 4. 自监督学习更新
 * 5. EWC梯度应用（保护重要参数）
 * 6. 记忆更新（存储代表性样本）
 */
public class AdvancedOnlineLearningManager {

    private final SimpCoreModel model;
    private final Optimizer onlineOptimizer;
    private final PlasticityUpdateModule[] plasticEngines;
    
    private final EWCSystem ewcSystem;
    private final MemoryReplayBuffer memoryBuffer;
    
    private final float onlineLr;
    private final float plasticOnlineLr;
    private final int minInputLength;
    private final float maskRatio;
    private final int replayInterval;
    private final int ewcUpdateInterval;
    
    private int totalOnlineSteps;
    private float totalOnlineLoss;
    private int lossCount;
    private float[] recentLosses;
    private int lossIndex;
    
    private boolean learningEnabled;
    private int currentStep;
    private final Random random;
    
    private float[] frozenEmbeddingBackup;
    private boolean ewcInitialized;

    public AdvancedOnlineLearningManager(SimpCoreModel model, float onlineLr, int minInputLength) {
        this(model, onlineLr, minInputLength, 5, 10);
    }

    public AdvancedOnlineLearningManager(SimpCoreModel model, float onlineLr, int minInputLength,
                                        int replayInterval, int ewcUpdateInterval) {
        this.model = model;
        this.onlineLr = onlineLr;
        this.plasticOnlineLr = onlineLr * 10f;
        this.minInputLength = minInputLength;
        this.maskRatio = 0.15f;
        this.replayInterval = replayInterval;
        this.ewcUpdateInterval = ewcUpdateInterval;
        
        this.onlineOptimizer = new Optimizer(onlineLr, 0.01f, 0.9f, 5.0f);
        this.plasticEngines = model.getSpmLayers();
        
        this.ewcSystem = new EWCSystem(model, 500.0f, 0.001f);
        this.memoryBuffer = new MemoryReplayBuffer(model, 50, 200, 500);
        
        this.totalOnlineSteps = 0;
        this.totalOnlineLoss = 0;
        this.lossCount = 0;
        this.recentLosses = new float[100];
        this.lossIndex = 0;
        this.learningEnabled = true;
        this.currentStep = 0;
        this.random = new Random(42);
        
        this.frozenEmbeddingBackup = new float[model.getEmbeddingWeights().length];
        this.ewcInitialized = false;
        
        System.out.println("[AdvancedOnlineLearning] Initialized | lr=" + onlineLr 
            + " | replay=" + replayInterval + " | ewc=" + ewcUpdateInterval);
    }

    public boolean processInteraction(int[] inputTokens) {
        if (!learningEnabled || inputTokens == null || inputTokens.length < minInputLength) {
            return false;
        }
        
        if (!ewcInitialized) {
            initializeEWC(inputTokens);
        }
        
        float loss = selfSupervisedLearn(inputTokens);
        
        if (currentStep % ewcUpdateInterval == 0) {
            applyEWCProtection();
        }
        
        if (currentStep % replayInterval == 0) {
            performMemoryReplay();
        }
        
        float[] featureVector = extractFeatureVector(inputTokens);
        memoryBuffer.add(inputTokensToFloat(inputTokens), featureVector);
        
        currentStep++;
        totalOnlineSteps++;
        
        return true;
    }

    private void initializeEWC(int[] inputTokens) {
        ewcInitialized = true;
        System.arraycopy(model.getEmbeddingWeights(), 0, frozenEmbeddingBackup, 0, frozenEmbeddingBackup.length);
        
        List<float[]> validationData = new ArrayList<>();
        for (int i = 0; i < Math.min(10, inputTokens.length - 1); i++) {
            float[] sample = new float[2];
            sample[0] = inputTokens[i];
            sample[1] = inputTokens[i + 1];
            validationData.add(sample);
        }
        ewcSystem.initialize(validationData);
        
        System.out.println("[AdvancedOnlineLearning] EWC initialized | Fisher size: " + ewcSystem.getFisherMatrix().size());
    }

    private float selfSupervisedLearn(int[] tokens) {
        int seqLen = Math.min(tokens.length - 1, model.getMaxSeqLen());
        int[] inputIds = Arrays.copyOf(tokens, seqLen);
        int[] targetIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++) {
            targetIds[i] = tokens[i + 1];
        }
        
        int[] learningMask = new int[seqLen];
        int maskedCount = 0;
        for (int i = 0; i < seqLen; i++) {
            if (random.nextFloat() < maskRatio) {
                learningMask[i] = 1;
                maskedCount++;
            }
        }
        if (maskedCount == 0) {
            learningMask[random.nextInt(seqLen)] = 1;
            maskedCount = 1;
        }
        
        MemoryOptimizer mo = MemoryOptimizer.getInstance();
        
        float[] logits = model.forward(inputIds, 1, seqLen);
        
        float baseLoss = CrossEntropyLoss.computeWithMask(logits, targetIds, learningMask, 1, seqLen, model.getVocabSize());
        
        float ewcLoss = ewcSystem.computeEWCLoss();
        float totalLoss = baseLoss + ewcLoss;
        
        trackLoss(totalLoss);
        
        float[] dLogits = computeMaskedGradients(logits, targetIds, learningMask, seqLen);
        
        performOnlineBackward(dLogits, inputIds, seqLen);
        
        Map<String, float[]> gradMap = new HashMap<>();
        gradMap.put("embed", new float[model.getEmbeddingWeights().length]);
        gradMap.put("output", new float[model.getOutputWeights().length]);
        gradMap.put("ffn1", new float[flatten(model.getFfnWeights1()).length]);
        gradMap.put("ffn2", new float[flatten(model.getFfnWeights2()).length]);
        gradMap.put("ln", new float[flatten(model.getLayerNormWeight()).length]);
        ewcSystem.onlineUpdateFisher(gradMap);
        
        mo.releaseFloatArray(logits);
        mo.releaseFloatArray(dLogits);
        
        return totalLoss;
    }

    private float[] computeMaskedGradients(float[] logits, int[] targets, int[] mask, int seqLen) {
        int vocabSize = model.getVocabSize();
        MemoryOptimizer mo = MemoryOptimizer.getInstance();
        float[] grads = mo.allocateFloatArray(logits.length);
        
        int count = 0;
        for (int s = 0; s < seqLen; s++) {
            if (mask[s] == 0) continue;
            
            int target = targets[s];
            if (target < 0 || target >= vocabSize) continue;
            
            int logitOffset = s * vocabSize;
            float[] slice = mo.allocateFloatArray(vocabSize);
            System.arraycopy(logits, logitOffset, slice, 0, vocabSize);
            
            float[] probs = MatrixOperations.softmax(slice);
            for (int i = 0; i < vocabSize; i++) {
                grads[logitOffset + i] = probs[i];
            }
            grads[logitOffset + target] -= 1.0f;
            count++;
            
            mo.releaseFloatArray(slice);
            mo.releaseFloatArray(probs);
        }
        
        if (count > 0) {
            float scale = 1.0f / count;
            for (int i = 0; i < grads.length; i++) {
                grads[i] *= scale;
            }
        }
        
        return grads;
    }

    private void performOnlineBackward(float[] dLogits, int[] inputIds, int seqLen) {
        int nLayers = model.getNLayers();
        int dModel = model.getDModel();
        int vocabSize = model.getVocabSize();
        int totalSize = seqLen * dModel;
        MemoryOptimizer mo = MemoryOptimizer.getInstance();
        
        float[][] layerHiddenStates = new float[nLayers + 1][];
        float[][] mhaOutputs = new float[nLayers][];
        float[][][] componentOutputs = new float[nLayers][][];
        
        float[] initialHidden = mo.allocateFloatArray(totalSize);
        for (int s = 0; s < seqLen; s++) {
            int tokenId = inputIds[s];
            int offset = s * dModel;
            for (int i = 0; i < dModel; i++) {
                initialHidden[offset + i] = model.getEmbeddingWeights()[tokenId * dModel + i]
                        + model.getPosEncoding()[s * dModel + i];
            }
        }
        layerHiddenStates[0] = initialHidden;
        
        for (int l = 0; l < nLayers; l++) {
            float[] x = layerHiddenStates[l];
            
            float[] mhaOutput = model.getMhaLayers()[l].forward(x, 1, seqLen);
            mhaOutputs[l] = mhaOutput;
            
            float[] attentionResidual = mo.allocateFloatArray(totalSize);
            for (int i = 0; i < totalSize; i++) attentionResidual[i] = x[i] + mhaOutput[i];
            
            float[] mhaNorm = mo.allocateFloatArray(totalSize);
            for (int s = 0; s < seqLen; s++) {
                int offset = s * dModel;
                float[] slice = new float[dModel];
                for (int j = 0; j < dModel; j++) slice[j] = attentionResidual[offset + j];
                float[] normalized = layerNorm(slice, l);
                for (int j = 0; j < dModel; j++) mhaNorm[offset + j] = normalized[j];
            }
            
            float[] rfpOut = model.getRfpLayers()[l].forward(mhaNorm, 1, seqLen);
            float[] dmcOut = model.getDmcLayers()[l].forward(rfpOut, 1, seqLen);
            float[] tarOut = model.getTarLayers()[l].forward(dmcOut, 1, seqLen);
            float[] cinOut = model.getCinLayers()[l].forward(tarOut, 1, seqLen);
            float[] ibOut = model.getIbLayers()[l].forward(cinOut, 1, seqLen);
            float[] ffnOut = applyFFN(ibOut, model.getFfnWeights1()[l], model.getFfnWeights2()[l], seqLen);
            float[] mrdOut = model.getMrdLayers()[l].forward(ffnOut, 1, seqLen);
            float[] spmOut = model.getSpmLayers()[l].forward(mrdOut, 1, seqLen);
            
            componentOutputs[l] = new float[][]{mhaNorm, rfpOut, dmcOut, tarOut, cinOut, ibOut, ffnOut, mrdOut, spmOut};
            
            float[] layerOutput = mo.allocateFloatArray(totalSize);
            for (int s = 0; s < seqLen; s++) {
                int offset = s * dModel;
                float[] layerInput = new float[dModel];
                for (int j = 0; j < dModel; j++) layerInput[j] = mhaNorm[offset + j] + spmOut[offset + j];
                float[] normalized = layerNorm(layerInput, l);
                for (int j = 0; j < dModel; j++) layerOutput[offset + j] = normalized[j];
            }
            layerHiddenStates[l + 1] = layerOutput;
        }
        
        float[] dHidden = mo.allocateFloatArray(totalSize);
        for (int s = 0; s < seqLen; s++) {
            float[] slice = new float[vocabSize];
            System.arraycopy(dLogits, s * vocabSize, slice, 0, vocabSize);
            float[] grad = MatrixOperations.matmul(slice, model.getOutputWeights(), 1, vocabSize, dModel);
            System.arraycopy(grad, 0, dHidden, s * dModel, dModel);
        }
        
        for (int l = nLayers - 1; l >= 0; l--) {
            float[] dNext = mo.allocateFloatArray(totalSize);
            for (int s = 0; s < seqLen; s++) {
                int offset = s * dModel;
                float[] layerInput = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    layerInput[j] = layerHiddenStates[l][offset + j] + componentOutputs[l][8][offset + j];
                }
                
                float mean = 0;
                for (int j = 0; j < dModel; j++) mean += layerInput[j];
                mean /= dModel;
                float var = 0;
                for (int j = 0; j < dModel; j++) var += (layerInput[j] - mean) * (layerInput[j] - mean);
                var /= dModel;
                float std = (float) Math.sqrt(var + 1e-5f);
                
                float[] dNormalized = new float[dModel];
                for (int j = 0; j < dModel; j++) dNormalized[j] = dHidden[offset + j] * model.getLayerNormWeight()[l][j];
                
                float dMean = 0, dVar = 0;
                for (int j = 0; j < dModel; j++) {
                    dMean -= dNormalized[j] / std;
                    dVar += dNormalized[j] * (layerInput[j] - mean) / (2 * std * std * std);
                }
                
                for (int j = 0; j < dModel; j++) {
                    dNext[offset + j] = dNormalized[j] / std + dVar * 2f * (layerInput[j] - mean) / dModel + dMean / dModel;
                }
            }
            
            GradientResult dSPM = model.getSpmLayers()[l].backward(dNext, componentOutputs[l][7], 1, seqLen);
            updatePlasticGradients(model.getSpmLayers()[l], dSPM);
            
            System.arraycopy(dNext, 0, dHidden, 0, totalSize);
            mo.releaseFloatArray(dNext);
        }
        
        updateFFNWeights(dHidden, componentOutputs, seqLen);
        
        ewcSystem.applyEWCGradient(model.getEmbeddingWeights(), "embed");
        ewcSystem.applyEWCGradient(model.getOutputWeights(), "output");
        
        for (int l = 0; l < nLayers; l++) {
            ewcSystem.applyEWCGradient(model.getFfnWeights1()[l], "ffn1");
            ewcSystem.applyEWCGradient(model.getFfnWeights2()[l], "ffn2");
        }
    }

    private void updateFFNWeights(float[] dHidden, float[][][] componentOutputs, int seqLen) {
        int nLayers = model.getNLayers();
        int dModel = model.getDModel();
        int dFFN = dModel * 4;
        
        for (int l = 0; l < nLayers; l++) {
            float[] w1 = model.getFfnWeights1()[l];
            float[] w2 = model.getFfnWeights2()[l];
            float[] ffnOut = componentOutputs[l][6];
            float[] mhaNorm = componentOutputs[l][0];
            
            float[] gradW1 = new float[w1.length];
            float[] gradW2 = new float[w2.length];
            
            float[] hiddenPreGelu = new float[seqLen * dFFN];
            float[] hiddenPostGelu = new float[seqLen * dFFN];
            
            for (int s = 0; s < seqLen; s++) {
                int inputOffset = s * dModel;
                int hiddenOffset = s * dFFN;
                
                float[] inputSlice = new float[dModel];
                System.arraycopy(mhaNorm, inputOffset, inputSlice, 0, dModel);
                
                float[] preGelu = MatrixOperations.matmul(inputSlice, w1, 1, dModel, dFFN);
                System.arraycopy(preGelu, 0, hiddenPreGelu, hiddenOffset, dFFN);
                
                for (int j = 0; j < dFFN; j++) {
                    hiddenPostGelu[hiddenOffset + j] = gelu(preGelu[j]);
                }
            }
            
            for (int s = 0; s < seqLen; s++) {
                int inputOffset = s * dModel;
                int hiddenOffset = s * dFFN;
                int dOffset = s * dModel;
                
                float[] dOut = new float[dModel];
                System.arraycopy(dHidden, dOffset, dOut, 0, dModel);
                
                float[] hiddenSlice = new float[dFFN];
                System.arraycopy(hiddenPostGelu, hiddenOffset, hiddenSlice, 0, dFFN);
                
                for (int i = 0; i < dFFN; i++) {
                    for (int j = 0; j < dModel; j++) {
                        gradW2[i * dModel + j] += hiddenSlice[i] * dOut[j];
                    }
                }
                
                float[] dHiddenBeforeGelu = MatrixOperations.matmul(dOut, w2, 1, dModel, dFFN);
                
                float[] inputSlice = new float[dModel];
                System.arraycopy(mhaNorm, inputOffset, inputSlice, 0, dModel);
                
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dFFN; j++) {
                        float geluDeriv = geluDerivative(hiddenPreGelu[hiddenOffset + j]);
                        gradW1[i * dFFN + j] += inputSlice[i] * dHiddenBeforeGelu[j] * geluDeriv;
                    }
                }
            }
            
            float scale = onlineLr / seqLen;
            for (int i = 0; i < w1.length && i < gradW1.length; i++) {
                w1[i] -= scale * gradW1[i];
            }
            for (int i = 0; i < w2.length && i < gradW2.length; i++) {
                w2[i] -= scale * gradW2[i];
            }
        }
    }
    
    private float gelu(float x) {
        return x * 0.5f * (1.0f + (float) Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
    }
    
    private float geluDerivative(float x) {
        float sqrt2pi = (float) Math.sqrt(2.0 / Math.PI);
        float tanhArg = sqrt2pi * (x + 0.044715f * x * x * x);
        float tanhOut = (float) Math.tanh(tanhArg);
        float derivative = 0.5f * (1.0f + tanhOut) 
                + 0.5f * x * (1.0f - tanhOut * tanhOut) 
                * sqrt2pi * (1.0f + 0.134145f * x * x);
        return derivative;
    }

    private void applyEWCProtection() {
        if (!ewcInitialized) return;
        
        float ewcLoss = ewcSystem.computeEWCLoss();
        if (ewcLoss > 0.1f) {
            ewcSystem.applyEWCGradient(model.getEmbeddingWeights(), "embed");
            ewcSystem.applyEWCGradient(model.getOutputWeights(), "output");
        }
    }

    private void performMemoryReplay() {
        if (memoryBuffer.getTotalSize() < 5) return;
        
        List<float[]> replayBatch = memoryBuffer.getReplayBatch(3);
        if (replayBatch.isEmpty()) return;
        
        float totalReplayLoss = 0;
        for (float[] sample : replayBatch) {
            int[] tokens = floatToInputTokens(sample);
            if (tokens.length >= minInputLength) {
                float loss = selfSupervisedLearn(tokens);
                totalReplayLoss += loss;
            }
        }
        
        memoryBuffer.cleanup();
    }

    private float[] extractFeatureVector(int[] tokens) {
        int dModel = model.getDModel();
        int seqLen = Math.min(tokens.length, model.getMaxSeqLen());
        float[] features = new float[dModel];
        
        for (int s = 0; s < seqLen; s++) {
            int tokenId = tokens[s];
            for (int i = 0; i < dModel; i++) {
                features[i] += model.getEmbeddingWeights()[tokenId * dModel + i];
            }
        }
        
        float norm = 0;
        for (float f : features) norm += f * f;
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < features.length; i++) features[i] /= norm;
        }
        
        return features;
    }

    private float[] inputTokensToFloat(int[] tokens) {
        float[] result = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            result[i] = tokens[i];
        }
        return result;
    }

    private int[] floatToInputTokens(float[] floats) {
        int[] result = new int[floats.length];
        for (int i = 0; i < floats.length; i++) {
            result[i] = (int) floats[i];
        }
        return result;
    }

    private float[] applyFFN(float[] x, float[] w1, float[] w2, int seqLen) {
        int dModel = model.getDModel();
        float[] result = new float[seqLen * dModel];
        for (int s = 0; s < seqLen; s++) {
            int offset = s * dModel;
            float[] row = new float[dModel];
            System.arraycopy(x, offset, row, 0, dModel);
            float[] hidden = MatrixOperations.matmul(row, w1, 1, dModel, dModel * 4);
            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = Math.max(0, hidden[i]);
            }
            float[] out = MatrixOperations.matmul(hidden, w2, 1, dModel * 4, dModel);
            System.arraycopy(out, 0, result, offset, dModel);
        }
        return result;
    }

    private float[] layerNorm(float[] x, int layerIdx) {
        int dModel = model.getDModel();
        float mean = 0;
        for (float f : x) mean += f;
        mean /= dModel;
        
        float var = 0;
        for (float f : x) var += (f - mean) * (f - mean);
        var /= dModel;
        
        float std = (float) Math.sqrt(var + 1e-5f);
        
        float[] result = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            result[i] = (x[i] - mean) / std * model.getLayerNormWeight()[layerIdx][i];
        }
        
        return result;
    }

    private void updatePlasticGradients(PlasticityUpdateModule engine, GradientResult grad) {
        float[][] weights = engine.getWeights();
        float[] dWeights = grad.getGradient("weights");
        if (dWeights != null) {
            int idx = 0;
            for (int i = 0; i < weights.length && idx < dWeights.length; i++) {
                for (int j = 0; j < weights[i].length && idx < dWeights.length; j++) {
                    weights[i][j] -= plasticOnlineLr * dWeights[idx++];
                }
            }
        }
    }

    private void trackLoss(float loss) {
        totalOnlineLoss += loss;
        lossCount++;
        recentLosses[lossIndex] = loss;
        lossIndex = (lossIndex + 1) % recentLosses.length;
    }
    
    private float[] flatten(float[][] matrix) {
        int size = 0;
        for (float[] row : matrix) size += row.length;
        float[] result = new float[size];
        int idx = 0;
        for (float[] row : matrix) {
            System.arraycopy(row, 0, result, idx, row.length);
            idx += row.length;
        }
        return result;
    }

    public float getAverageLoss() {
        return lossCount > 0 ? totalOnlineLoss / lossCount : Float.MAX_VALUE;
    }

    public float getRecentLoss() {
        int count = Math.min(lossCount, recentLosses.length);
        float sum = 0;
        for (int i = 0; i < count; i++) sum += recentLosses[i];
        return count > 0 ? sum / count : Float.MAX_VALUE;
    }

    public int getTotalOnlineSteps() { return totalOnlineSteps; }
    public boolean isLearningEnabled() { return learningEnabled; }
    public void setLearningEnabled(boolean enabled) { this.learningEnabled = enabled; }
    public EWCSystem getEwcSystem() { return ewcSystem; }
    public MemoryReplayBuffer getMemoryBuffer() { return memoryBuffer; }
    
    public String getStatistics() {
        return String.format(
            "AdvancedOnlineLearning[steps=%d, avgLoss=%.4f, recentLoss=%.4f, ewc=%s, memory=%s]",
            totalOnlineSteps, getAverageLoss(), getRecentLoss(),
            ewcSystem.isInitialized() ? "initialized" : "not initialized",
            memoryBuffer.getStatistics()
        );
    }
}
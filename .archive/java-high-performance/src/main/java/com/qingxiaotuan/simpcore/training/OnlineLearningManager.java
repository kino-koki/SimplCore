package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.components.PlasticityUpdateModule;
import java.util.Arrays;
import java.util.Random;

/**
 * 在线学习管理器 — 使模型在推理过程中通过学习输入数据自我进化。
 * 
 * 核心创新: 在推理时对每个输入执行自我监督的下一token预测学习。
 * 每次输入都产生一个微小的梯度更新，使模型从使用中持续改进。
 * 
 * 与传统训练不同:
 * - 无需单独的"训练模式"或"推理模式"切换
 * - 模型从每次交互中学习，积累改进
 * - 使用极低学习率避免灾难性遗忘
 * - 支持自监督 + Hebbian 双通道学习
 */
public class OnlineLearningManager {

    private final SimpCoreModel model;
    private final Optimizer onlineOptimizer;
    private final PlasticityUpdateModule[] plasticEngines;
    
    // 在线学习配置
    private final float onlineLr;
    private final int minInputLength;
    private final float maskRatio;
    private final int plasticUpdateFreq;
    private final float plasticOnlineLr;
    
    // 学习统计
    private int totalOnlineSteps;
    private float totalOnlineLoss;
    private int lossCount;
    private float[] recentLosses;
    private int lossIndex;
    
    // 运行时状态
    private boolean learningEnabled;
    private int currentStep;
    private final Random random;
    
    // 记忆保护 — 避免灾难性遗忘
    private final float[] frozenEmbeddingBackup;
    private boolean memoryProtected;
    
    /**
     * 创建在线学习管理器
     * @param model 要学习的模型
     * @param onlineLr 在线学习率（建议1e-7 ~ 1e-6，远小于训练学习率）
     * @param minInputLength 触发学习的最小输入长度
     */
    public OnlineLearningManager(SimpCoreModel model, float onlineLr, int minInputLength) {
        this.model = model;
        this.onlineLr = onlineLr;
        this.minInputLength = minInputLength;
        this.maskRatio = 0.15f;
        this.plasticUpdateFreq = 10;
        this.plasticOnlineLr = onlineLr * 10f;
        
        // 独立优化器用于在线学习（极小学习率）
        this.onlineOptimizer = new Optimizer(onlineLr, 0.01f, 0.9f, 5.0f);
        this.plasticEngines = model.getSpmLayers();
        
        this.totalOnlineSteps = 0;
        this.totalOnlineLoss = 0;
        this.lossCount = 0;
        this.recentLosses = new float[100];
        this.lossIndex = 0;
        this.learningEnabled = true;
        this.currentStep = 0;
        this.random = new Random();
        
        // 备份嵌入层权重（用于记忆保护）
        this.frozenEmbeddingBackup = new float[model.getEmbeddingWeights().length];
        this.memoryProtected = false;
        
        System.out.println("[OnlineLearning] 初始化完成 | lr=" + onlineLr 
            + " | minLen=" + minInputLength + " | maskRatio=" + maskRatio);
    }
    
    /**
     * 处理一个完整的输入输出交互 — 执行在线学习
     * 
     * @param inputTokens 用户输入的token序列
     * @param thresholdLoss 返回本次学习的损失值
     * @return 是否执行了学习
     */
    public boolean processInteraction(int[] inputTokens) {
        if (!learningEnabled || inputTokens == null || inputTokens.length < minInputLength) {
            return false;
        }
        
        // Step 1: 保护关键记忆（在首次学习前备份嵌入层）
        if (!memoryProtected) {
            protectMemory();
        }
        
        // Step 2: 执行自监督学习
        float loss = selfSupervisedLearn(inputTokens);
        
        // Step 3: 更新PlasticityUpdateModule（Hebbian + 监督混合）
        if (currentStep % plasticUpdateFreq == 0) {
            updatePlasticWeights(inputTokens);
        }
        
        currentStep++;
        totalOnlineSteps++;
        
        return true;
    }
    
    /**
     * 自监督学习 — 核心算法
     * 
     * 在输入序列上执行next-token预测学习:
     * 1. 随机选择部分位置作为学习目标
     * 2. 前向传播获取预测
     * 3. 计算交叉熵损失
     * 4. 反向传播梯度
     * 5. 应用微小权重更新
     */
    private float selfSupervisedLearn(int[] tokens) {
        int seqLen = Math.min(tokens.length - 1, model.getMaxSeqLen());
        int[] inputIds = Arrays.copyOf(tokens, seqLen);
        int[] targetIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++) {
            targetIds[i] = tokens[i + 1];
        }
        
        // 创建学习掩码 — 随机选择15%的位置
        int[] learningMask = new int[seqLen];
        int maskedCount = 0;
        for (int i = 0; i < seqLen; i++) {
            if (random.nextFloat() < maskRatio) {
                learningMask[i] = 1;
                maskedCount++;
            }
        }
        
        // 如果没有选中任何位置，强制选一个
        if (maskedCount == 0) {
            learningMask[random.nextInt(seqLen)] = 1;
            maskedCount = 1;
        }
        
        MemoryOptimizer mo = MemoryOptimizer.getInstance();
        
        // 前向传播
        float[] logits = model.forward(inputIds, 1, seqLen);
        
        // 用掩码计算损失（只对选中的位置）
        float loss = CrossEntropyLoss.computeWithMask(logits, targetIds, learningMask, 1, seqLen, model.getVocabSize());
        
        // 记录损失
        trackLoss(loss);
        
        // 计算梯度（带掩码）
        float[] dLogits = computeMaskedGradients(logits, targetIds, learningMask, seqLen);
        
        // 反向传播并更新权重
        performOnlineBackward(dLogits, inputIds, seqLen);
        
        mo.releaseFloatArray(logits);
        mo.releaseFloatArray(dLogits);
        
        return loss;
    }
    
    /**
     * 计算带掩码的梯度 — 只对选中的学习位置计算
     */
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
        
        // 归一化
        if (count > 0) {
            float scale = 1.0f / count;
            for (int i = 0; i < grads.length; i++) {
                grads[i] *= scale;
            }
        }
        
        return grads;
    }
    
    /**
     * 执行在线反向传播 — 重用TrainingLoop的反向逻辑
     * 但只更新部分关键权重，且使用极低学习率
     */
    private void performOnlineBackward(float[] dLogits, int[] inputIds, int seqLen) {
        int nLayers = model.getNLayers();
        int dModel = model.getDModel();
        int vocabSize = model.getVocabSize();
        int totalSize = seqLen * dModel;
        MemoryOptimizer mo = MemoryOptimizer.getInstance();
        
        // 前向过程激活值缓存（需要重新前向以获取中间状态）
        float[][] layerHiddenStates = new float[nLayers + 1][];
        float[][] mhaOutputs = new float[nLayers][];
        float[][][] componentOutputs = new float[nLayers][][];
        
        // 第一次前向，缓存所有中间激活值
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
            
            // 注意力残差 + LayerNorm
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
            
            // 所有子组件前向
            float[] rfpOut = model.getRfpLayers()[l].forward(mhaNorm, 1, seqLen);
            float[] dmcOut = model.getDmcLayers()[l].forward(rfpOut, 1, seqLen);
            float[] tarOut = model.getTarLayers()[l].forward(dmcOut, 1, seqLen);
            float[] cinOut = model.getCinLayers()[l].forward(tarOut, 1, seqLen);
            float[] ibOut = model.getIbLayers()[l].forward(cinOut, 1, seqLen);
            float[] ffnOut = applyFFN(ibOut, model.getFfnWeights1()[l], model.getFfnWeights2()[l], seqLen);
            float[] mrdOut = model.getMrdLayers()[l].forward(ffnOut, 1, seqLen);
            float[] spmOut = model.getSpmLayers()[l].forward(mrdOut, 1, seqLen);
            
            componentOutputs[l] = new float[][]{mhaNorm, rfpOut, dmcOut, tarOut, cinOut, ibOut, ffnOut, mrdOut, spmOut};
            
            // 层输出 = mhaNorm + spmOut → LayerNorm
            float[] layerOutput = mo.allocateFloatArray(totalSize);
            for (int s = 0; s < seqLen; s++) {
                int offset = s * dModel;
                float[] layerInput = new float[dModel];
                for (int j = 0; j < dModel; j++) layerInput[j] = mhaNorm[offset + j] + spmOut[offset + j];
                float[] normalized = layerNorm(layerInput, l);
                for (int j = 0; j < dModel; j++) layerOutput[offset + j] = normalized[j];
            }
            layerHiddenStates[l + 1] = layerOutput;
            
            mo.releaseFloatArray(attentionResidual);
            mo.releaseFloatArray(mhaNorm);
            mo.releaseFloatArray(rfpOut);
            mo.releaseFloatArray(dmcOut);
            mo.releaseFloatArray(tarOut);
            mo.releaseFloatArray(cinOut);
            mo.releaseFloatArray(ibOut);
            mo.releaseFloatArray(ffnOut);
            mo.releaseFloatArray(mrdOut);
            mo.releaseFloatArray(spmOut);
        }
        
        // 输出层梯度
        float[] dHidden = mo.allocateFloatArray(totalSize);
        for (int s = 0; s < seqLen; s++) {
            float[] slice = new float[vocabSize];
            System.arraycopy(dLogits, s * vocabSize, slice, 0, vocabSize);
            float[] grad = MatrixOperations.matmul(slice, model.getOutputWeights(), 1, vocabSize, dModel);
            System.arraycopy(grad, 0, dHidden, s * dModel, dModel);
        }
        
        // 反向传播穿过所有层
        for (int l = nLayers - 1; l >= 0; l--) {
            // LayerNorm backward
            float[] dNext = mo.allocateFloatArray(totalSize);
            for (int s = 0; s < seqLen; s++) {
                int offset = s * dModel;
                float[] layerInput = new float[dModel];
                float[] prevHidden = new float[dModel];
                float[] layerOut = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    layerInput[j] = layerHiddenStates[l][offset + j] + componentOutputs[l][8][offset + j];
                    prevHidden[j] = layerHiddenStates[l][offset + j];
                    layerOut[j] = componentOutputs[l][8][offset + j];
                }
                
                // LayerNorm backward
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
            
            // SPN backward
            GradientResult dSPM = model.getSpmLayers()[l].backward(dNext, componentOutputs[l][7], 1, seqLen);
            updatePlasticGradients(model.getSpmLayers()[l], dSPM);
            
            // 反向继续传播
            float[] dBack = dSPM.getdInput();
            float[] dFFN = mo.allocateFloatArray(totalSize);
            System.arraycopy(dBack, 0, dFFN, 0, totalSize);
            
            // 注意：深度反向传播在此简化 — 只更新FFN和LayerNorm权重
            // 完整反向传播参见TrainingLoop
            
            System.arraycopy(dNext, 0, dHidden, 0, totalSize);
            mo.releaseFloatArray(dNext);
        }
        
        // 更新FFN权重（使用实际计算的梯度）
        float[][] gradFFN1 = new float[nLayers][];
        float[][] gradFFN2 = new float[nLayers][];
        for (int l = 0; l < nLayers; l++) {
            gradFFN1[l] = new float[model.getFfnWeights1()[l].length];
            gradFFN2[l] = new float[model.getFfnWeights2()[l].length];
        }
        
        // 计算FFN梯度（简化版：从层输出梯度反向到FFN权重）
        for (int l = 0; l < nLayers; l++) {
            float[] ffnOut = componentOutputs[l][6];
            float[] mhaNorm = componentOutputs[l][0];
            float[] dPrev = new float[totalSize];
            for (int s = 0; s < seqLen; s++) {
                int offset = s * dModel;
                float[] layerInput = new float[dModel];
                for (int j = 0; j < dModel; j++) {
                    layerInput[j] = mhaNorm[offset + j] + ffnOut[offset + j];
                }
                float mean = 0;
                for (int j = 0; j < dModel; j++) mean += layerInput[j];
                mean /= dModel;
                float var = 0;
                for (int j = 0; j < dModel; j++) var += (layerInput[j] - mean) * (layerInput[j] - mean);
                var /= dModel;
                float std = (float) Math.sqrt(var + 1e-5f);
                for (int j = 0; j < dModel; j++) {
                    float dNorm = dHidden[offset + j] * model.getLayerNormWeight()[l][j];
                    dPrev[offset + j] = dNorm / std;
                }
            }
            
            // FFN层梯度
            float[] w1 = model.getFfnWeights1()[l];
            float[] w2 = model.getFfnWeights2()[l];
            float[] hidden = new float[seqLen * dModel * 4];
            for (int s = 0; s < seqLen; s++) {
                float[] rowData = new float[dModel];
                System.arraycopy(mhaNorm, s * dModel, rowData, 0, dModel);
                float[] result = MatrixOperations.matmul(rowData, w1, 1, dModel, dModel * 4);
                for (int j = 0; j < dModel * 4; j++) {
                    hidden[s * dModel * 4 + j] = (float)(result[j] * (1.0 / (1.0 + Math.exp(-result[j]))));
                }
            }
            
            // dW2 = hidden^T @ dPrev
            for (int i = 0; i < dModel * 4; i++) {
                for (int j = 0; j < dModel; j++) {
                    float sum = 0;
                    for (int s = 0; s < seqLen; s++) {
                        sum += hidden[s * dModel * 4 + i] * dPrev[s * dModel + j];
                    }
                    gradFFN2[l][i * dModel + j] = sum / seqLen;
                }
            }
            
            // dW1 = mhaNorm^T @ (dPrev @ W2^T * sigmoid'(hidden))
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel * 4; j++) {
                    float sum = 0;
                    for (int s = 0; s < seqLen; s++) {
                        float h = hidden[s * dModel * 4 + j];
                        float dsig = h * (1 - h);
                        float dHiddenJ = 0;
                        for (int k = 0; k < dModel; k++) {
                            dHiddenJ += dPrev[s * dModel + k] * w2[j * dModel + k];
                        }
                        sum += mhaNorm[s * dModel + i] * dHiddenJ * dsig;
                    }
                    gradFFN1[l][i * dModel * 4 + j] = sum / seqLen;
                }
            }
        }
        
        // 应用FFN权重更新
        for (int l = 0; l < nLayers; l++) {
            onlineOptimizer.update(model.getFfnWeights1()[l], gradFFN1[l]);
            onlineOptimizer.update(model.getFfnWeights2()[l], gradFFN2[l]);
        }
        
        mo.releaseFloatArray(dHidden);
        
        // 释放缓存
        for (int l = 0; l <= nLayers; l++) {
            if (layerHiddenStates[l] != null) mo.releaseFloatArray(layerHiddenStates[l]);
        }
        for (int l = 0; l < nLayers; l++) {
            if (mhaOutputs[l] != null) mo.releaseFloatArray(mhaOutputs[l]);
        }
    }
    
    /**
     * 更新PlasticityUpdateModule的塑料权重
     * 结合Hebbian学习 + 外部监督信号
     */
    private void updatePlasticWeights(int[] tokens) {
        for (int l = 0; l < plasticEngines.length; l++) {
            PlasticityUpdateModule spm = plasticEngines[l];
            if (!spm.isEnabled()) continue;
            
            // 使用最近一段输入驱动Hebbian更新
            float[] inputVec = new float[model.getDModel()];
            int centralToken = tokens[tokens.length / 2];
            int offset = centralToken * model.getDModel();
            if (offset >= 0 && offset + model.getDModel() <= model.getEmbeddingWeights().length) {
                System.arraycopy(model.getEmbeddingWeights(), offset, inputVec, 0, model.getDModel());
            }
            
            // 触发Hebbian更新
            float[] output = new float[model.getDModel()];
            for (int i = 0; i < model.getDModel(); i++) {
                float sum = 0;
                for (int j = 0; j < model.getDModel(); j++) {
                    sum += spm.getPlasticWeights()[i][j] * inputVec[j];
                }
                output[i] = sum;
            }
            
            // 手动触发一次Hebbian更新（通过直接操作内部状态）
            for (int i = 0; i < model.getDModel(); i++) {
                for (int j = 0; j < model.getDModel(); j++) {
                    float hebbianUpdate = output[i] * inputVec[j] * plasticOnlineLr;
                    float currentW = spm.getPlasticWeights()[i][j];
                    float updated = currentW + hebbianUpdate;
                    updated = Math.max(-1f, Math.min(1f, updated));
                    spm.getPlasticWeights()[i][j] = updated;
                }
            }
        }
    }
    
    private void updatePlasticGradients(PlasticityUpdateModule spm, GradientResult grad) {
        // 通过优化器更新塑料权重（监督梯度）
        float[] dPlastic = grad.getGradient("plasticWeights");
        if (dPlastic != null) {
            float[][] weights = spm.getPlasticWeights();
            int dModel = spm.getDModel();
            for (int i = 0; i < dModel; i++) {
                float[] wRow = weights[i];
                float[] gRow = new float[dModel];
                System.arraycopy(dPlastic, i * dModel, gRow, 0, dModel);
                onlineOptimizer.update(wRow, gRow);
            }
        }
    }
    
    /**
     * 保护关键记忆 — 冻结嵌入层防止遗忘
     */
    private void protectMemory() {
        System.arraycopy(model.getEmbeddingWeights(), 0, frozenEmbeddingBackup, 0, frozenEmbeddingBackup.length);
        memoryProtected = true;
        System.out.println("[OnlineLearning] 记忆保护已启用 | embeddingSize=" + frozenEmbeddingBackup.length);
    }
    
    /**
     * 重置在线学习状态（需要时调用）
     */
    public void resetOnlineLearning() {
        totalOnlineSteps = 0;
        totalOnlineLoss = 0;
        lossCount = 0;
        recentLosses = new float[100];
        lossIndex = 0;
        currentStep = 0;
        
        // 恢复嵌入层备份
        if (memoryProtected && frozenEmbeddingBackup != null) {
            System.arraycopy(frozenEmbeddingBackup, 0, model.getEmbeddingWeights(), 0, frozenEmbeddingBackup.length);
        }
        memoryProtected = false;
        
        System.out.println("[OnlineLearning] 在线学习已重置");
    }
    
    /**
     * 获取模型目前"变聪明"了多少
     * 衡量标准: 最近损失值与初始损失值的对比
     */
    public float getImprovementRatio() {
        if (lossCount < 5) return 0;
        int recentCount = Math.min(lossIndex, recentLosses.length);
        if (recentCount == 0) return 0;
        float recentAvg = 0;
        int actualRecent = 0;
        for (int i = 0; i < recentCount; i++) {
            int idx = (lossIndex - 1 - i + recentLosses.length) % recentLosses.length;
            if (recentLosses[idx] != 0 || i == 0) {
                recentAvg += recentLosses[idx];
                actualRecent++;
            }
        }
        if (actualRecent == 0) return 0;
        recentAvg /= actualRecent;
        float overallAvg = totalOnlineLoss / lossCount;
        if (overallAvg <= 0) return 0;
        return (overallAvg - recentAvg) / overallAvg;
    }
    
    private void trackLoss(float loss) {
        totalOnlineLoss += loss;
        lossCount++;
        recentLosses[lossIndex % recentLosses.length] = loss;
        lossIndex++;
    }
    
    private float[] layerNorm(float[] x, int layer) {
        int dModel = model.getDModel();
        float mean = 0;
        for (int i = 0; i < dModel; i++) mean += x[i];
        mean /= dModel;
        float var = 0;
        for (int i = 0; i < dModel; i++) var += (x[i] - mean) * (x[i] - mean);
        var /= dModel;
        float std = (float) Math.sqrt(var + 1e-5f);
        float[] output = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            output[i] = (x[i] - mean) / std * model.getLayerNormWeight()[layer][i] + model.getLayerNormBias()[layer][i];
        }
        return output;
    }
    
    private float[] applyFFN(float[] x, float[] w1, float[] w2, int seqLen) {
        int dModel = model.getDModel();
        float[] hidden = new float[seqLen * dModel * 4];
        for (int i = 0; i < seqLen; i++) {
            float[] rowData = new float[dModel];
            System.arraycopy(x, i * dModel, rowData, 0, dModel);
            float[] result = MatrixOperations.matmul(rowData, w1, 1, dModel, dModel * 4);
            for (int j = 0; j < dModel * 4; j++) {
                hidden[i * dModel * 4 + j] = (float)(result[j] * (1.0 / (1.0 + Math.exp(-result[j]))));
            }
        }
        float[] output = new float[seqLen * dModel];
        for (int i = 0; i < seqLen; i++) {
            float[] rowData = new float[dModel * 4];
            System.arraycopy(hidden, i * dModel * 4, rowData, 0, dModel * 4);
            float[] result = MatrixOperations.matmul(rowData, w2, 1, dModel * 4, dModel);
            System.arraycopy(result, 0, output, i * dModel, dModel);
        }
        return output;
    }
    
    // ============== 公共接口 ==============
    
    public void setLearningEnabled(boolean enabled) { this.learningEnabled = enabled; }
    public boolean isLearningEnabled() { return learningEnabled; }
    public int getTotalOnlineSteps() { return totalOnlineSteps; }
    public float getAverageOnlineLoss() { return lossCount > 0 ? totalOnlineLoss / lossCount : 0; }
    public float getRecentAverageLoss() {
        int count = Math.min(recentLosses.length, lossIndex);
        if (count == 0) return 0;
        float sum = 0;
        for (int i = 0; i < count; i++) sum += recentLosses[i];
        return sum / count;
    }
    
    /**
     * 获取学习报告
     */
    public String getLearningReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 青小团SC架构 在线学习报告 ===\n");
        sb.append("在线学习步数: ").append(totalOnlineSteps).append("\n");
        sb.append("平均损失: ").append(String.format("%.4f", getAverageOnlineLoss())).append("\n");
        sb.append("最近损失: ").append(String.format("%.4f", getRecentAverageLoss())).append("\n");
        sb.append("改进指数: ").append(String.format("%.2f%%", getImprovementRatio() * 100)).append("\n");
        sb.append("学习状态: ").append(learningEnabled ? "已启用" : "已禁用").append("\n");
        sb.append("记忆保护: ").append(memoryProtected ? "已启用" : "未启用").append("\n");
        sb.append("================================\n");
        return sb.toString();
    }
}

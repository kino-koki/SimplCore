package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 认知推理器 — 整合所有认知模块的完整类人思维系统
 * 
 * 人类思维的完整流程：
 * 1. 感知输入 → 分层感知模块处理
 * 2. 工作记忆 → 临时存储和检索
 * 3. 思维树搜索 → 多路径探索和评估
 * 4. 联想推理 → 类比和因果推理
 * 5. 自我反思 → 评估和修正
 * 6. 情感决策 → 价值判断和选择
 * 7. 输出响应
 * 
 * 参考论文：
 * - Tree of Thoughts
 * - Reflexion
 * - Analogical Reasoning
 * - Affective Computing
 */
public class CognitiveReasoner {

    private final int dModel;
    private final int defaultSteps;
    
    private final float[] wQuery;
    private final float[] wReason;
    private final float[] wIntegration;
    
    private final float[] layerNormGamma;
    private final float[] layerNormBeta;
    
    private final List<float[]> reasoningPath;
    private final Random random;
    
    /** 子模块 */
    private final TreeOfThought treeOfThought;
    private final ReflexionModule reflexionModule;
    private final AssociativeReasoner associativeReasoner;
    private final EmotionalDecisionEngine emotionEngine;
    private final WorkingMemoryController workingMemory;
    private final HierarchicalPerceptionModule perceptionModule;

    public CognitiveReasoner(int dModel) {
        this(dModel, 3);
    }

    public CognitiveReasoner(int dModel, int defaultSteps) {
        this.dModel = dModel;
        this.defaultSteps = defaultSteps;
        this.reasoningPath = new ArrayList<>();
        this.random = new Random(42);
        
        this.wQuery = new float[dModel * dModel];
        this.wReason = new float[2 * dModel * dModel];
        this.wIntegration = new float[5 * dModel * dModel];
        
        this.layerNormGamma = new float[dModel];
        this.layerNormBeta = new float[dModel];
        
        initWeights();
        
        this.treeOfThought = new TreeOfThought(dModel, 5, 3, 0.5f);
        this.reflexionModule = new ReflexionModule(dModel);
        this.associativeReasoner = new AssociativeReasoner(dModel);
        this.emotionEngine = new EmotionalDecisionEngine(dModel);
        this.workingMemory = new WorkingMemoryController(dModel);
        this.perceptionModule = new HierarchicalPerceptionModule(dModel);
    }

    private void initWeights() {
        for (int i = 0; i < wQuery.length; i++) {
            wQuery[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        for (int i = 0; i < wReason.length; i++) {
            wReason[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
        for (int i = 0; i < wIntegration.length; i++) {
            wIntegration[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        Arrays.fill(layerNormGamma, 1.0f);
        Arrays.fill(layerNormBeta, 0.0f);
    }

    /**
     * 完整推理流程 — 模拟人类思维的完整过程
     */
    public ReasoningResult fullReasoning(float[] input, int batchSize, int seqLen) {
        reasoningPath.clear();
        
        float[] currentState = Arrays.copyOf(input, input.length);
        
        float[] perceptionOutput = perceptionModule.forward(currentState, batchSize, seqLen);
        currentState = perceptionOutput;
        reasoningPath.add(Arrays.copyOf(currentState, currentState.length));
        
        float[] query = computeQuery(currentState);
        
        float[] memoryCtx = workingMemory.read(query);
        workingMemory.write(memoryCtx);
        
        float[] thoughtResult = treeOfThought.search(query, 10);
        
        List<AssociativeReasoner.AssociationResult> associations = 
            associativeReasoner.associate(thoughtResult, 3);
        
        float[] associatedResult = thoughtResult;
        if (!associations.isEmpty()) {
            float[] bestConcept = associations.get(0).conceptVector;
            associatedResult = associativeReasoner.analogicalReasoning(thoughtResult, bestConcept);
        }
        
        float[] causalResult = associativeReasoner.causalReasoning(associatedResult);
        
        float[] reflectionInput = new float[dModel];
        int offset = 0;
        for (int i = 0; i < dModel && offset < causalResult.length; i++) {
            reflectionInput[i] = causalResult[offset++];
        }
        
        float[] correctedResult = reflexionModule.reflectAndCorrect(
            query, reflectionInput, 3, 0.7f);
        
        float[] emotionEnhanced = emotionEngine.emotionEnhancedReasoning(correctedResult);
        
        float[] integrated = integrateAll(
            thoughtResult, associatedResult, causalResult, correctedResult, emotionEnhanced);
        
        float confidence = reflexionModule.evaluate(query, integrated);
        
        return new ReasoningResult(
            integrated, 
            confidence, 
            reasoningPath, 
            associations,
            treeOfThought.getReasoningTrajectory()
        );
    }

    /**
     * 链式推理 — 简化版，适合快速推理
     */
    public float[] reason(float[] input, int steps, WorkingMemoryController memory) {
        reasoningPath.clear();
        int totalSize = input.length;
        
        float[] state = Arrays.copyOf(input, totalSize);
        List<float[]> stepOutputs = new ArrayList<>();
        
        for (int step = 0; step < steps; step++) {
            float[] query = new float[dModel];
            for (int i = 0; i < totalSize; i += dModel) {
                float[] slice = Arrays.copyOfRange(state, i, Math.min(i + dModel, totalSize));
                float[] q = MatrixOperations.matmul(slice, wQuery, 1, dModel, dModel);
                for (int j = 0; j < dModel; j++) {
                    query[j] += q[j];
                }
            }
            for (int j = 0; j < dModel; j++) {
                query[j] /= (totalSize / dModel);
            }
            
            float[] memoryCtx = memory != null ? memory.read(query) : new float[dModel];
            
            float[] newState = new float[totalSize];
            for (int i = 0; i < totalSize; i += dModel) {
                float[] concat = new float[2 * dModel];
                int end = Math.min(i + dModel, totalSize);
                int len = end - i;
                System.arraycopy(state, i, concat, 0, len);
                if (len < dModel) {
                    Arrays.fill(concat, len, dModel, 0);
                }
                System.arraycopy(memoryCtx, 0, concat, dModel, dModel);
                
                float[] reasoning = MatrixOperations.matmul(concat, wReason, 1, 2 * dModel, dModel);
                
                float[] combined = new float[dModel];
                for (int j = 0; j < len; j++) {
                    combined[j] = state[i + j] + reasoning[j];
                }
                Arrays.fill(combined, len, dModel, 0);
                
                float[] normalized = layerNorm(combined);
                System.arraycopy(normalized, 0, newState, i, len);
            }
            
            state = newState;
            stepOutputs.add(Arrays.copyOf(state, totalSize));
            
            if (memory != null) {
                memory.write(memoryCtx);
            }
        }
        
        reasoningPath.addAll(stepOutputs);
        
        float[] output = new float[totalSize];
        for (float[] stepOut : stepOutputs) {
            for (int i = 0; i < totalSize; i++) {
                output[i] += stepOut[i];
            }
        }
        for (int i = 0; i < totalSize; i++) {
            output[i] /= steps;
        }
        
        return output;
    }

    /**
     * 整合所有模块的输出
     */
    private float[] integrateAll(float[] thought, float[] associated, 
                                  float[] causal, float[] corrected, float[] emotional) {
        float[] concat = new float[5 * dModel];
        
        int pos = 0;
        for (int i = 0; i < dModel; i++) {
            concat[pos++] = i < thought.length ? thought[i] : 0;
        }
        for (int i = 0; i < dModel; i++) {
            concat[pos++] = i < associated.length ? associated[i] : 0;
        }
        for (int i = 0; i < dModel; i++) {
            concat[pos++] = i < causal.length ? causal[i] : 0;
        }
        for (int i = 0; i < dModel; i++) {
            concat[pos++] = i < corrected.length ? corrected[i] : 0;
        }
        for (int i = 0; i < dModel; i++) {
            concat[pos++] = i < emotional.length ? emotional[i] : 0;
        }
        
        float[] integrated = MatrixOperations.matmul(concat, wIntegration, 1, 5 * dModel, dModel);
        
        return layerNorm(integrated);
    }

    /**
     * 计算查询向量
     */
    private float[] computeQuery(float[] state) {
        float[] query = new float[dModel];
        int totalSize = state.length;
        
        for (int i = 0; i < totalSize; i += dModel) {
            float[] slice = Arrays.copyOfRange(state, i, Math.min(i + dModel, totalSize));
            float[] q = MatrixOperations.matmul(slice, wQuery, 1, dModel, dModel);
            for (int j = 0; j < dModel; j++) {
                query[j] += q[j];
            }
        }
        
        int numSlices = (totalSize + dModel - 1) / dModel;
        for (int j = 0; j < dModel; j++) {
            query[j] /= numSlices;
        }
        
        return query;
    }

    /**
     * 设置情感状态
     */
    public void setEmotion(EmotionalDecisionEngine.Emotion emotion, float intensity) {
        emotionEngine.setEmotion(emotion, intensity);
    }

    /**
     * 设置动机
     */
    public void setMotivation(EmotionalDecisionEngine.Motivation motivation) {
        emotionEngine.setMotivation(motivation);
    }

    /**
     * 添加概念到联想记忆
     */
    public void addConcept(String label, float[] vector) {
        associativeReasoner.addConcept(label, vector);
    }

    /**
     * 获取反思统计
     */
    public ReflexionModule.ReflectionStats getReflectionStats() {
        return reflexionModule.getStats();
    }

    /**
     * 获取当前情感状态
     */
    public EmotionalDecisionEngine.Emotion getCurrentEmotion() {
        return emotionEngine.getCurrentEmotionType();
    }

    private float[] layerNorm(float[] x) {
        float mean = 0;
        int len = x.length;
        for (float v : x) {
            mean += v;
        }
        mean /= len;
        
        float var = 0;
        for (float v : x) {
            float diff = v - mean;
            var += diff * diff;
        }
        var /= len;
        
        float invStd = 1.0f / (float) Math.sqrt(var + 1e-5f);
        float[] output = new float[len];
        for (int i = 0; i < len; i++) {
            output[i] = layerNormGamma[i % dModel] * (x[i] - mean) * invStd + layerNormBeta[i % dModel];
        }
        return output;
    }

    public static class ReasoningResult {
        public final float[] output;
        public final float confidence;
        public final List<float[]> reasoningPath;
        public final List<AssociativeReasoner.AssociationResult> associations;
        public final List<float[]> thoughtTrajectory;
        
        public ReasoningResult(float[] output, float confidence, 
                               List<float[]> path, List<AssociativeReasoner.AssociationResult> assocs,
                               List<float[]> trajectory) {
            this.output = output;
            this.confidence = confidence;
            this.reasoningPath = path;
            this.associations = assocs;
            this.thoughtTrajectory = trajectory;
        }
    }

    public List<float[]> getReasoningPath() {
        return reasoningPath;
    }

    public int getReasoningDepth() {
        if (reasoningPath.size() < 2) return reasoningPath.size();
        
        for (int i = reasoningPath.size() - 1; i > 0; i--) {
            float[] current = reasoningPath.get(i);
            float[] previous = reasoningPath.get(i - 1);
            
            float diff = 0;
            float norm = 0;
            int minLen = Math.min(current.length, previous.length);
            for (int j = 0; j < minLen; j++) {
                float d = current[j] - previous[j];
                diff += d * d;
                norm += current[j] * current[j];
            }
            float relativeChange = (float) Math.sqrt(diff) / (float) (Math.sqrt(norm) + 1e-8f);
            
            if (relativeChange > 0.01f) {
                return i + 1;
            }
        }
        return 1;
    }

    public int getDModel() { return dModel; }
    public int getDefaultSteps() { return defaultSteps; }
    public float[] getWQuery() { return wQuery; }
    public float[] getWReason() { return wReason; }
    public float[] getWIntegration() { return wIntegration; }
    public float[] getLayerNormGamma() { return layerNormGamma; }
    public float[] getLayerNormBeta() { return layerNormBeta; }
    
    public TreeOfThought getTreeOfThought() { return treeOfThought; }
    public ReflexionModule getReflexionModule() { return reflexionModule; }
    public AssociativeReasoner getAssociativeReasoner() { return associativeReasoner; }
    public EmotionalDecisionEngine getEmotionEngine() { return emotionEngine; }
    public WorkingMemoryController getWorkingMemory() { return workingMemory; }
    public HierarchicalPerceptionModule getPerceptionModule() { return perceptionModule; }
}
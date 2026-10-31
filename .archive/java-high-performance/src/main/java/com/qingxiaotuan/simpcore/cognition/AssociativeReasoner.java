package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import java.util.*;

/**
 * 联想推理器 — 模拟人类的类比和因果推理能力
 * 
 * 人类思维的核心特征：
 * 1. 语义联想：从一个概念联想到相关概念
 * 2. 类比推理：发现不同领域之间的相似性
 * 3. 因果推理：推断事件之间的因果关系
 * 4. 隐喻理解：理解抽象的比喻和隐喻
 * 
 * 参考论文：
 * - Analogical Reasoning with Large Language Models
 * - Causal Reasoning in Language Models
 */
public class AssociativeReasoner {

    private final int dModel;
    
    /** 联想记忆库 — 存储概念向量 */
    private final List<float[]> conceptMemory;
    private final List<String> conceptLabels;
    
    /** 联想权重 */
    private final float[] wAssociate;
    
    /** 类比映射权重 */
    private final float[] wAnalogy;
    
    /** 因果推理权重 */
    private final float[] wCausal;
    
    /** 联想强度阈值 */
    private final float associationThreshold;
    
    private final Random random;

    public AssociativeReasoner(int dModel) {
        this(dModel, 0.3f);
    }

    public AssociativeReasoner(int dModel, float associationThreshold) {
        this.dModel = dModel;
        this.associationThreshold = associationThreshold;
        this.random = new Random(42);
        this.conceptMemory = new ArrayList<>();
        this.conceptLabels = new ArrayList<>();
        
        this.wAssociate = new float[dModel * dModel];
        this.wAnalogy = new float[2 * dModel * dModel];
        this.wCausal = new float[dModel * dModel];
        
        initWeights();
        initConceptMemory();
    }

    private void initWeights() {
        for (int i = 0; i < wAssociate.length; i++) {
            wAssociate[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
        for (int i = 0; i < wAnalogy.length; i++) {
            wAnalogy[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
        for (int i = 0; i < wCausal.length; i++) {
            wCausal[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
    }

    private void initConceptMemory() {
        String[] basicConcepts = {
            "时间", "空间", "因果", "类比", "相似", "差异",
            "整体", "部分", "抽象", "具体", "逻辑", "情感"
        };
        
        for (String concept : basicConcepts) {
            float[] vector = new float[dModel];
            for (int i = 0; i < dModel; i++) {
                vector[i] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            conceptMemory.add(vector);
            conceptLabels.add(concept);
        }
    }

    /**
     * 语义联想 — 找到与输入最相关的概念
     */
    public List<AssociationResult> associate(float[] input, int topK) {
        float[] transformed = MatrixOperations.matmul(input, wAssociate, 1, dModel, dModel);
        
        List<AssociationResult> results = new ArrayList<>();
        
        for (int i = 0; i < conceptMemory.size(); i++) {
            float[] concept = conceptMemory.get(i);
            float similarity = cosineSimilarity(transformed, concept);
            
            if (similarity > associationThreshold) {
                results.add(new AssociationResult(conceptLabels.get(i), similarity, concept));
            }
        }
        
        results.sort((a, b) -> Float.compare(b.similarity, a.similarity));
        
        return results.size() > topK ? results.subList(0, topK) : results;
    }

    /**
     * 类比推理 — 发现源域和目标域之间的映射关系
     */
    public float[] analogicalReasoning(float[] source, float[] target) {
        float[] concat = new float[2 * dModel];
        System.arraycopy(source, 0, concat, 0, dModel);
        System.arraycopy(target, 0, concat, dModel, dModel);
        
        float[] analogy = MatrixOperations.matmul(concat, wAnalogy, 1, 2 * dModel, dModel);
        
        float[] result = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            result[i] = target[i] + analogy[i];
        }
        
        return layerNorm(result);
    }

    /**
     * 因果推理 — 推断事件之间的因果关系
     */
    public float[] causalReasoning(float[] cause) {
        float[] effect = MatrixOperations.matmul(cause, wCausal, 1, dModel, dModel);
        
        float[] normalized = layerNorm(effect);
        
        float[] confidence = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            confidence[i] = (float) Math.abs(Math.tanh(normalized[i]));
        }
        
        return normalized;
    }

    /**
     * 多步联想链 — 模拟人类的链式联想思维
     */
    public List<float[]> chainAssociation(float[] input, int steps) {
        List<float[]> chain = new ArrayList<>();
        float[] current = Arrays.copyOf(input, dModel);
        chain.add(Arrays.copyOf(current, dModel));
        
        for (int step = 0; step < steps; step++) {
            List<AssociationResult> associations = associate(current, 1);
            
            if (!associations.isEmpty()) {
                float[] associated = associations.get(0).conceptVector;
                
                float[] next = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    next[i] = current[i] + 0.5f * associated[i];
                }
                
                current = layerNorm(next);
                chain.add(Arrays.copyOf(current, dModel));
            } else {
                break;
            }
        }
        
        return chain;
    }

    /**
     * 添加新概念到联想记忆库
     */
    public void addConcept(String label, float[] vector) {
        conceptMemory.add(Arrays.copyOf(vector, dModel));
        conceptLabels.add(label);
    }

    /**
     * 删除概念
     */
    public void removeConcept(String label) {
        int index = conceptLabels.indexOf(label);
        if (index >= 0) {
            conceptMemory.remove(index);
            conceptLabels.remove(index);
        }
    }

    /**
     * 类比映射 — 将源域的知识映射到目标域
     */
    public float[] mapKnowledge(float[] sourceKnowledge, float[] sourceDomain, float[] targetDomain) {
        float sourceNorm = normalize(sourceDomain);
        float targetNorm = normalize(targetDomain);
        
        float[] direction = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            direction[i] = (targetDomain[i] - sourceDomain[i]) / (sourceNorm + 1e-8f);
        }
        
        float[] mapped = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            mapped[i] = sourceKnowledge[i] + direction[i] * (targetNorm / (sourceNorm + 1e-8f));
        }
        
        return layerNorm(mapped);
    }

    /**
     * 隐喻理解 — 解析隐喻表达
     */
    public float[] understandMetaphor(float[] literalMeaning, float[] context) {
        float[] analogy = analogicalReasoning(literalMeaning, context);
        return layerNorm(analogy);
    }

    private float cosineSimilarity(float[] a, float[] b) {
        float dot = 0;
        float norm1 = 0, norm2 = 0;
        
        for (int i = 0; i < dModel; i++) {
            dot += a[i] * b[i];
            norm1 += a[i] * a[i];
            norm2 += b[i] * b[i];
        }
        
        return dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
    }

    private float normalize(float[] vector) {
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private float[] layerNorm(float[] x) {
        float mean = 0;
        for (float v : x) {
            mean += v;
        }
        mean /= dModel;
        
        float var = 0;
        for (float v : x) {
            float diff = v - mean;
            var += diff * diff;
        }
        var /= dModel;
        
        float invStd = 1.0f / (float) Math.sqrt(var + 1e-5f);
        float[] output = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            output[i] = (x[i] - mean) * invStd;
        }
        return output;
    }

    public static class AssociationResult {
        public final String label;
        public final float similarity;
        public final float[] conceptVector;
        
        public AssociationResult(String label, float similarity, float[] conceptVector) {
            this.label = label;
            this.similarity = similarity;
            this.conceptVector = conceptVector;
        }
    }

    public int getDModel() { return dModel; }
    public float[] getWAssociate() { return wAssociate; }
    public float[] getWAnalogy() { return wAnalogy; }
    public float[] getWCausal() { return wCausal; }
    public float getAssociationThreshold() { return associationThreshold; }
    public List<String> getConceptLabels() { return conceptLabels; }
    public int getConceptCount() { return conceptMemory.size(); }
}
package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;

public class ConsistencyGate {
    private int dim;
    private float consistencyThreshold;
    
    public ConsistencyGate(int dim) {
        this.dim = dim;
        this.consistencyThreshold = 0.6f;
    }
    
    public ConsistencyResult checkConsistency(Map<String, Object> syntaxTree, List<String> verifiedFacts) {
        ConsistencyResult result = new ConsistencyResult();
        result.inconsistencies = new ArrayList<>();
        result.consistencyScore = 1.0f;
        
        List<String> tokens = (List<String>) syntaxTree.get("tokens");
        String inputType = (String) syntaxTree.get("type");
        
        if ("statement".equals(inputType) && verifiedFacts.size() > 1) {
            float pairwiseConsistency = checkPairwiseConsistency(verifiedFacts);
            result.consistencyScore = pairwiseConsistency;
            
            if (pairwiseConsistency < consistencyThreshold) {
                result.passed = false;
                result.rejectionReason = "事实声明之间存在不一致";
                result.inconsistencies.add("多个事实声明之间的一致性不足");
                return result;
            }
        }
        
        boolean logicConsistent = checkLogicalConsistency(tokens);
        if (!logicConsistent) {
            result.passed = false;
            result.rejectionReason = "逻辑不一致";
            result.consistencyScore = 0.3f;
            return result;
        }
        
        boolean temporalConsistent = checkTemporalConsistency(tokens);
        if (!temporalConsistent) {
            result.passed = false;
            result.rejectionReason = "时间逻辑不一致";
            result.consistencyScore = 0.4f;
            return result;
        }
        
        result.passed = true;
        return result;
    }
    
    private float checkPairwiseConsistency(List<String> facts) {
        if (facts.size() < 2) {
            return 1.0f;
        }
        
        float totalSimilarity = 0.0f;
        int count = 0;
        
        for (int i = 0; i < facts.size(); i++) {
            for (int j = i + 1; j < facts.size(); j++) {
                float[] emb1 = generateEmbedding(facts.get(i));
                float[] emb2 = generateEmbedding(facts.get(j));
                totalSimilarity += cosineSimilarity(emb1, emb2);
                count++;
            }
        }
        
        return totalSimilarity / count;
    }
    
    private boolean checkLogicalConsistency(List<String> tokens) {
        boolean hasPositive = false;
        boolean hasNegative = false;
        
        Set<String> negations = new HashSet<>(Arrays.asList("不", "没有", "不是", "非", "无"));
        Set<String> affirmations = new HashSet<>(Arrays.asList("是", "有", "存在", "包含"));
        
        for (String token : tokens) {
            if (negations.contains(token)) {
                hasNegative = true;
            }
            if (affirmations.contains(token)) {
                hasPositive = true;
            }
        }
        
        if (hasPositive && hasNegative) {
            for (int i = 0; i < tokens.size() - 1; i++) {
                if (negations.contains(tokens.get(i)) && affirmations.contains(tokens.get(i + 1))) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    private boolean checkTemporalConsistency(List<String> tokens) {
        boolean hasPast = false;
        boolean hasFuture = false;
        
        Set<String> pastTerms = new HashSet<>(Arrays.asList("曾经", "过去", "以前", "之前", "已经"));
        Set<String> futureTerms = new HashSet<>(Arrays.asList("将", "会", "要", "未来", "以后", "即将"));
        
        for (String token : tokens) {
            if (pastTerms.contains(token)) {
                hasPast = true;
            }
            if (futureTerms.contains(token)) {
                hasFuture = true;
            }
        }
        
        return !(hasPast && hasFuture);
    }
    
    private float[] generateEmbedding(String text) {
        float[] embedding = new float[dim];
        int seed = text.hashCode();
        
        for (int i = 0; i < dim; i++) {
            seed = (seed * 9301 + 49297) % 233280;
            embedding[i] = (float) (seed / 233280.0 * 2 - 1);
        }
        
        return embedding;
    }
    
    private float cosineSimilarity(float[] a, float[] b) {
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }
    
    public static class ConsistencyResult {
        public boolean passed;
        public List<String> inconsistencies;
        public float consistencyScore;
        public String rejectionReason;
    }
}

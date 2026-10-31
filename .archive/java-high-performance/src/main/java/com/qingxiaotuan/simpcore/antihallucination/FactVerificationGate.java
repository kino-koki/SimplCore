package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;

public class FactVerificationGate {
    private Map<String, float[]> factDatabase;
    private int dim;
    private float verificationThreshold;
    
    public FactVerificationGate(int dim) {
        this.dim = dim;
        this.factDatabase = new HashMap<>();
        this.verificationThreshold = 0.7f;
        initFactDatabase();
    }
    
    private void initFactDatabase() {
        String[] facts = {
            "地球是圆的", "光速约为30万公里/秒", "水的沸点是100摄氏度",
            "中国首都是北京", "太阳从东方升起", "人类需要呼吸氧气"
        };
        
        for (String fact : facts) {
            float[] embedding = generateEmbedding(fact);
            factDatabase.put(fact, embedding);
        }
    }
    
    public FactResult verify(Map<String, Object> syntaxTree, List<InputStructureGate.Entity> entities) {
        FactResult result = new FactResult();
        result.verifiedFacts = new ArrayList<>();
        result.unverifiedClaims = new ArrayList<>();
        result.confidenceScores = new HashMap<>();
        
        String inputType = (String) syntaxTree.get("type");
        List<String> tokens = (List<String>) syntaxTree.get("tokens");
        
        for (InputStructureGate.Entity entity : entities) {
            if ("number".equals(entity.type)) {
                boolean isVerified = verifyNumericClaim(entity.value);
                if (isVerified) {
                    result.verifiedFacts.add(entity.value);
                    result.confidenceScores.put(entity.value, 0.95f);
                } else {
                    result.unverifiedClaims.add(entity.value);
                    result.confidenceScores.put(entity.value, 0.3f);
                }
            }
        }
        
        String claim = extractClaim(tokens);
        if (claim != null && !claim.isEmpty()) {
            float confidence = checkFactConsistency(claim);
            result.confidenceScores.put(claim, confidence);
            
            if (confidence >= verificationThreshold) {
                result.verifiedFacts.add(claim);
            } else {
                result.unverifiedClaims.add(claim);
            }
        }
        
        if (!result.unverifiedClaims.isEmpty()) {
            result.passed = false;
            result.rejectionReason = "存在未验证的事实声明";
            result.confidenceScore = calculateOverallConfidence(result.confidenceScores);
            return result;
        }
        
        result.passed = true;
        result.confidenceScore = calculateOverallConfidence(result.confidenceScores);
        return result;
    }
    
    private boolean verifyNumericClaim(String number) {
        try {
            double num = Double.parseDouble(number);
            return num >= 0 && num <= 1e18;
        } catch (NumberFormatException e) {
            return false;
        }
    }
    
    private String extractClaim(List<String> tokens) {
        StringBuilder claim = new StringBuilder();
        boolean inClaim = false;
        
        for (String token : tokens) {
            if (token.matches("[\\u4e00-\\u9fa5a-zA-Z]+")) {
                if (claim.length() > 0) {
                    claim.append(" ");
                }
                claim.append(token);
            }
        }
        
        return claim.length() > 2 ? claim.toString() : null;
    }
    
    private float checkFactConsistency(String claim) {
        float[] claimEmbedding = generateEmbedding(claim);
        float maxSimilarity = 0.0f;
        
        for (Map.Entry<String, float[]> entry : factDatabase.entrySet()) {
            float similarity = cosineSimilarity(claimEmbedding, entry.getValue());
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
            }
        }
        
        return maxSimilarity;
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
    
    private float calculateOverallConfidence(Map<String, Float> scores) {
        if (scores.isEmpty()) {
            return 1.0f;
        }
        
        float sum = 0.0f;
        for (float score : scores.values()) {
            sum += score;
        }
        
        return sum / scores.size();
    }
    
    public void addFact(String fact) {
        float[] embedding = generateEmbedding(fact);
        factDatabase.put(fact, embedding);
    }
    
    public static class FactResult {
        public boolean passed;
        public List<String> verifiedFacts;
        public List<String> unverifiedClaims;
        public Map<String, Float> confidenceScores;
        public float confidenceScore;
        public String rejectionReason;
    }
}

package com.qingxiaotuan.simpcore.orchestration;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KnowledgeModuleSpace {
    private Map<String, KnowledgeModule> modules;
    private Map<String, float[]> documents;
    private int dim;
    
    public KnowledgeModuleSpace(int dim) {
        this.dim = dim;
        this.modules = new HashMap<>();
        this.documents = new HashMap<>();
    }
    
    public void addModule(String moduleId, float[] activation, String category, String description) {
        KnowledgeModule module = new KnowledgeModule(moduleId, activation, category, description);
        modules.put(moduleId, module);
    }
    
    public void addDocument(String docId, float[] embedding) {
        documents.put(docId, embedding);
    }
    
    public float[] getDocumentEmbedding(String docId) {
        return documents.get(docId);
    }
    
    public float[] retrieveContext(float[] query, int topK) {
        float[] result = new float[dim];
        
        List<SimilarityEntry> similarities = new ArrayList<>();
        for (KnowledgeModule module : modules.values()) {
            float sim = cosineSimilarity(query, module.getActivation());
            similarities.add(new SimilarityEntry(module, sim));
        }
        
        for (Map.Entry<String, float[]> entry : documents.entrySet()) {
            float sim = cosineSimilarity(query, entry.getValue());
            KnowledgeModule dummy = new KnowledgeModule(entry.getKey(), entry.getValue(), "document", "");
            similarities.add(new SimilarityEntry(dummy, sim));
        }
        
        similarities.sort((a, b) -> Float.compare(b.similarity, a.similarity));
        
        float totalWeight = 0.0f;
        for (int i = 0; i < Math.min(topK, similarities.size()); i++) {
            float weight = similarities.get(i).similarity;
            totalWeight += weight;
            
            float[] activation = similarities.get(i).module.getActivation();
            for (int j = 0; j < dim; j++) {
                result[j] += activation[j] * weight;
            }
        }
        
        if (totalWeight > 0) {
            for (int i = 0; i < dim; i++) {
                result[i] /= totalWeight;
            }
        }
        
        return result;
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
    
    public int getModuleCount() {
        return modules.size();
    }
    
    public int getDocumentCount() {
        return documents.size();
    }
    
    public static class KnowledgeModule {
        private String moduleId;
        private float[] activation;
        private String category;
        private String description;
        
        public KnowledgeModule(String moduleId, float[] activation, String category, String description) {
            this.moduleId = moduleId;
            this.activation = activation;
            this.category = category;
            this.description = description;
        }
        
        public String getModuleId() {
            return moduleId;
        }
        
        public float[] getActivation() {
            return activation;
        }
        
        public String getCategory() {
            return category;
        }
        
        public String getDescription() {
            return description;
        }
    }
    
    private static class SimilarityEntry {
        KnowledgeModule module;
        float similarity;
        
        SimilarityEntry(KnowledgeModule module, float similarity) {
            this.module = module;
            this.similarity = similarity;
        }
    }
}

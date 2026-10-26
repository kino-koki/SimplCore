package com.qingxiaotuan.simpcore.orchestration;

import java.util.ArrayList;
import java.util.List;

public class ModuleAddressor {
    private List<VectorActivation> activationSpace;
    private int dim;
    private float[] projectionMatrix;
    
    public ModuleAddressor(int dim, int maxModules) {
        this.dim = dim;
        this.activationSpace = new ArrayList<>(maxModules);
        this.projectionMatrix = new float[dim * dim];
        
        float scale = (float) (1.0 / Math.sqrt(dim));
        for (int i = 0; i < dim * dim; i++) {
            projectionMatrix[i] = (float) (Math.random() - 0.5) * 2 * scale;
        }
    }
    
    public void registerModule(String moduleId, float[] activation, String category) {
        VectorActivation va = new VectorActivation(moduleId, activation, category);
        va.setProjectedActivation(project(activation));
        activationSpace.add(va);
    }
    
    public List<VectorActivation> query(float[] query, int topK, float threshold) {
        float[] projectedQuery = project(query);
        
        List<ActivationScore> scores = new ArrayList<>();
        for (VectorActivation va : activationSpace) {
            float score = cosineSimilarity(projectedQuery, va.getProjectedActivation());
            if (score > threshold) {
                scores.add(new ActivationScore(va, score));
            }
        }
        
        scores.sort((a, b) -> Float.compare(b.score, a.score));
        
        List<VectorActivation> results = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, scores.size()); i++) {
            results.add(scores.get(i).activation);
        }
        
        return results;
    }
    
    public List<VectorActivation> queryByCategory(String category) {
        List<VectorActivation> results = new ArrayList<>();
        for (VectorActivation va : activationSpace) {
            if (va.getCategory().equals(category)) {
                results.add(va);
            }
        }
        return results;
    }
    
    private float[] project(float[] x) {
        float[] result = new float[dim];
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                sum += projectionMatrix[i * dim + j] * x[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    private float cosineSimilarity(float[] a, float[] b) {
        float dot = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (int i = 0; i < dim; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }
    
    public int getModuleCount() {
        return activationSpace.size();
    }
    
    public static class VectorActivation {
        private String moduleId;
        private float[] activation;
        private float[] projectedActivation;
        private String category;
        
        public VectorActivation(String moduleId, float[] activation, String category) {
            this.moduleId = moduleId;
            this.activation = activation;
            this.category = category;
        }
        
        public String getModuleId() {
            return moduleId;
        }
        
        public float[] getActivation() {
            return activation;
        }
        
        public float[] getProjectedActivation() {
            return projectedActivation;
        }
        
        public void setProjectedActivation(float[] projected) {
            this.projectedActivation = projected;
        }
        
        public String getCategory() {
            return category;
        }
    }
    
    private static class ActivationScore {
        VectorActivation activation;
        float score;
        
        ActivationScore(VectorActivation activation, float score) {
            this.activation = activation;
            this.score = score;
        }
    }
}

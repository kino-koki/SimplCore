package com.qingxiaotuan.simpcore.orchestration;

import java.util.ArrayList;
import java.util.List;

public class DynamicOrchestrator {
    private ModuleAddressor addressor;
    private KnowledgeModuleSpace knowledgeSpace;
    
    private long totalLatency;
    private long queryCount;
    private long ragLatency;
    
    public DynamicOrchestrator(int dim, int maxModules) {
        this.addressor = new ModuleAddressor(dim, maxModules);
        this.knowledgeSpace = new KnowledgeModuleSpace(dim);
        this.totalLatency = 0;
        this.queryCount = 0;
        this.ragLatency = 0;
    }
    
    public void registerCapabilityModule(String moduleId, float[] activation, 
                                       String category, String description) {
        addressor.registerModule(moduleId, activation, category);
        knowledgeSpace.addModule(moduleId, activation, category, description);
    }
    
    public float[] orchestrate(float[] queryEmbedding, int nSteps, float temperature) {
        long startTime = System.nanoTime();
        
        List<ModuleAddressor.VectorActivation> candidates = addressor.query(queryEmbedding, 5, 0.3f);
        
        float[] combinedActivation = new float[queryEmbedding.length];
        float totalWeight = 0.0f;
        
        for (ModuleAddressor.VectorActivation va : candidates) {
            float weight = (float) Math.pow(va.getActivation()[0], 1.0 / temperature);
            totalWeight += weight;
            
            for (int i = 0; i < queryEmbedding.length; i++) {
                combinedActivation[i] += va.getActivation()[i] * weight;
            }
        }
        
        if (totalWeight > 0) {
            for (int i = 0; i < queryEmbedding.length; i++) {
                combinedActivation[i] /= totalWeight;
            }
        }
        
        float[] finalResult = executeMultiStepReasoning(queryEmbedding, combinedActivation, nSteps);
        
        long endTime = System.nanoTime();
        totalLatency += (endTime - startTime);
        queryCount++;
        
        return finalResult;
    }
    
    private float[] executeMultiStepReasoning(float[] query, float[] moduleActivation, int nSteps) {
        float[] state = new float[query.length];
        System.arraycopy(query, 0, state, 0, query.length);
        
        for (int step = 0; step < nSteps; step++) {
            for (int i = 0; i < query.length; i++) {
                state[i] = state[i] * 0.9f + moduleActivation[i] * 0.1f;
            }
            
            float[] context = knowledgeSpace.retrieveContext(state, 3);
            for (int i = 0; i < query.length; i++) {
                state[i] += context[i] * 0.05f;
            }
        }
        
        return state;
    }
    
    public float[] crossDocumentRetrieval(float[] query, List<String> documentIds) {
        long startTime = System.nanoTime();
        
        float[] result = new float[query.length];
        
        for (String docId : documentIds) {
            float[] docEmbedding = knowledgeSpace.getDocumentEmbedding(docId);
            if (docEmbedding != null) {
                float similarity = cosineSimilarity(query, docEmbedding);
                
                for (int i = 0; i < query.length; i++) {
                    result[i] += docEmbedding[i] * similarity;
                }
            }
        }
        
        ragLatency += (System.nanoTime() - startTime);
        
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
    
    public LatencyReport getLatencyReport() {
        float avgLatency = queryCount > 0 ? (float) totalLatency / queryCount / 1_000_000.0f : 0;
        float avgRagLatency = queryCount > 0 ? (float) ragLatency / queryCount / 1_000_000.0f : 0;
        float improvement = avgRagLatency > 0 ? (1 - avgLatency / avgRagLatency) * 100 : 0;
        
        return new LatencyReport(avgLatency, avgRagLatency, improvement);
    }
    
    public static class LatencyReport {
        public float avgOrchestrationLatencyMs;
        public float avgRagLatencyMs;
        public float improvementPercent;
        
        public LatencyReport(float orchestration, float rag, float improvement) {
            this.avgOrchestrationLatencyMs = orchestration;
            this.avgRagLatencyMs = rag;
            this.improvementPercent = improvement;
        }
        
        public String toString() {
            return String.format(
                "Latency Report:\n  Orchestration: %.2f ms\n  Traditional RAG: %.2f ms\n  Improvement: %.1f%%",
                avgOrchestrationLatencyMs, avgRagLatencyMs, improvementPercent
            );
        }
    }
}

package com.qingxiaotuan.simpcore.interpretability;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import com.qingxiaotuan.simpcore.gpu.GPUMatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import java.util.*;

public class ModelExplainer {

    private final SimpCoreModel model;
    private final Tokenizer tokenizer;

    public ModelExplainer(SimpCoreModel model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public FeatureImportance explainPrediction(String text, int targetTokenId) {
        int[] inputIds = tokenizer.encode(text);
        String[] tokens = tokenizer.decodeToTokens(inputIds);
        
        float[] logits = model.forward(inputIds, 1, inputIds.length);
        
        int seqLen = inputIds.length;
        float[] gradients = computeGradients(inputIds, targetTokenId);
        
        double[] importances = new double[seqLen];
        for (int i = 0; i < seqLen; i++) {
            float[] hiddenGrad = new float[model.getDModel()];
            for (int j = 0; j < model.getDModel(); j++) {
                hiddenGrad[j] = gradients[i * model.getDModel() + j];
            }
            
            float[] embedding = new float[model.getDModel()];
            for (int j = 0; j < model.getDModel(); j++) {
                embedding[j] = model.getEmbeddingWeights()[inputIds[i] * model.getDModel() + j];
            }
            
            importances[i] = Math.abs(GPUMatrixOperations.dotProduct(hiddenGrad, embedding));
        }
        
        double maxImportance = Arrays.stream(importances).max().orElse(1.0);
        for (int i = 0; i < seqLen; i++) {
            importances[i] = importances[i] / maxImportance;
        }
        
        return new FeatureImportance(tokens, importances, logits[targetTokenId]);
    }

    private float[] computeGradients(int[] inputIds, int targetTokenId) {
        int seqLen = inputIds.length;
        int dModel = model.getDModel();
        int vocabSize = model.getVocabSize();
        MemoryOptimizer memoryOptimizer = MemoryOptimizer.getInstance();
        
        float[] logits = model.forward(inputIds, 1, seqLen);
        
        float[] dLogits = memoryOptimizer.allocateFloatArray(seqLen * vocabSize);
        for (int i = 0; i < seqLen; i++) {
            Arrays.fill(dLogits, i * vocabSize, (i + 1) * vocabSize, 0.0f);
            dLogits[i * vocabSize + targetTokenId] = 1.0f;
        }
        
        float[] dHidden = memoryOptimizer.allocateFloatArray(seqLen * dModel);
        for (int i = 0; i < seqLen; i++) {
            float[] slice = memoryOptimizer.allocateFloatArray(vocabSize);
            System.arraycopy(dLogits, i * vocabSize, slice, 0, vocabSize);
            float[] grad = GPUMatrixOperations.matmul(slice, model.getOutputWeights(), 1, vocabSize, dModel);
            System.arraycopy(grad, 0, dHidden, i * dModel, dModel);
            memoryOptimizer.releaseFloatArray(slice);
            memoryOptimizer.releaseFloatArray(grad);
        }
        
        memoryOptimizer.releaseFloatArray(dLogits);
        memoryOptimizer.releaseFloatArray(logits);
        
        return dHidden;
    }

    public AttentionAnalysis analyzeAttention(String text) {
        int[] inputIds = tokenizer.encode(text);
        String[] tokens = tokenizer.decodeToTokens(inputIds);
        int seqLen = inputIds.length;
        int nLayers = model.getNLayers();
        
        float[][][] attentionMaps = new float[nLayers][seqLen][seqLen];
        
        for (int l = 0; l < nLayers; l++) {
            float[] attention = new float[seqLen * seqLen];
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    attention[i * seqLen + j] = (float) (1.0 / seqLen);
                }
            }
            
            float[] rowSums = new float[seqLen];
            for (int i = 0; i < seqLen; i++) {
                float sum = 0;
                for (int j = 0; j < seqLen; j++) {
                    sum += attention[i * seqLen + j];
                }
                rowSums[i] = sum;
            }
            
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    attentionMaps[l][i][j] = attention[i * seqLen + j] / rowSums[i];
                }
            }
        }
        
        return new AttentionAnalysis(tokens, attentionMaps);
    }

    public SaliencyMap computeSaliency(String text) {
        int[] inputIds = tokenizer.encode(text);
        String[] tokens = tokenizer.decodeToTokens(inputIds);
        int seqLen = inputIds.length;
        
        float[] logits = model.forward(inputIds, 1, seqLen);
        
        int maxLogitIdx = 0;
        float maxLogit = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
                maxLogitIdx = i;
            }
        }
        
        float[] gradients = computeGradients(inputIds, maxLogitIdx);
        
        double[] saliency = new double[seqLen];
        for (int i = 0; i < seqLen; i++) {
            double norm = 0;
            for (int j = 0; j < model.getDModel(); j++) {
                norm += gradients[i * model.getDModel() + j] * gradients[i * model.getDModel() + j];
            }
            saliency[i] = Math.sqrt(norm);
        }
        
        double maxSaliency = Arrays.stream(saliency).max().orElse(1.0);
        for (int i = 0; i < seqLen; i++) {
            saliency[i] = saliency[i] / maxSaliency;
        }
        
        return new SaliencyMap(tokens, saliency, tokenizer.decode(new int[]{maxLogitIdx}));
    }

    public List<CounterfactualExplanation> generateCounterfactuals(String text, int targetTokenId, int numExamples) {
        List<CounterfactualExplanation> results = new ArrayList<>();
        int[] inputIds = tokenizer.encode(text);
        int seqLen = inputIds.length;
        int vocabSize = model.getVocabSize();
        
        float[] originalLogits = model.forward(inputIds, 1, seqLen);
        float originalProb = computeSoftmax(originalLogits)[targetTokenId];
        
        Random random = new Random(42);
        
        for (int n = 0; n < numExamples; n++) {
            int[] modifiedIds = Arrays.copyOf(inputIds, seqLen);
            int changedPos = random.nextInt(seqLen);
            int newToken = random.nextInt(vocabSize);
            
            while (newToken == inputIds[changedPos]) {
                newToken = random.nextInt(vocabSize);
            }
            
            modifiedIds[changedPos] = newToken;
            
            float[] modifiedLogits = model.forward(modifiedIds, 1, seqLen);
            float modifiedProb = computeSoftmax(modifiedLogits)[targetTokenId];
            
            String modifiedText = tokenizer.decode(modifiedIds);
            
            results.add(new CounterfactualExplanation(
                modifiedText,
                changedPos,
                tokenizer.decode(new int[]{inputIds[changedPos]}),
                tokenizer.decode(new int[]{newToken}),
                originalProb,
                modifiedProb,
                modifiedProb - originalProb
            ));
        }
        
        results.sort((a, b) -> Double.compare(Math.abs(b.getProbabilityChange()), Math.abs(a.getProbabilityChange())));
        
        return results.subList(0, Math.min(numExamples, results.size()));
    }

    private float[] computeSoftmax(float[] logits) {
        float maxLogit = Arrays.stream(logits).max().orElse(0.0f);
        float[] exp = new float[logits.length];
        float sum = 0;
        
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += exp[i];
        }
        
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = exp[i] / sum;
        }
        
        return probs;
    }

    public String explainGeneration(String prompt, int maxTokens) {
        StringBuilder explanation = new StringBuilder();
        explanation.append("=== Generation Explanation ===\n");
        explanation.append("Prompt: \"" + prompt + "\"\n\n");
        
        int[] inputIds = tokenizer.encode(prompt);
        String[] promptTokens = tokenizer.decodeToTokens(inputIds);
        
        explanation.append("Prompt Tokens:\n");
        for (int i = 0; i < promptTokens.length; i++) {
            explanation.append("  " + i + ": \"" + promptTokens[i] + "\"\n");
        }
        explanation.append("\n");
        
        int[] outputIds = model.generate(inputIds, maxTokens, tokenizer.getEosTokenId());
        String[] outputTokens = tokenizer.decodeToTokens(outputIds);
        
        explanation.append("Generated Tokens:\n");
        for (int i = promptTokens.length; i < outputTokens.length; i++) {
            int pos = i - promptTokens.length;
            explanation.append("  Step " + pos + ": \"" + outputTokens[i] + "\"");
            
            int[] contextIds = Arrays.copyOf(outputIds, i + 1);
            float[] logits = model.forward(contextIds, 1, contextIds.length);
            
            int topK = 5;
            Integer[] indices = new Integer[model.getVocabSize()];
            for (int j = 0; j < indices.length; j++) indices[j] = j;
            Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));
            
            explanation.append(" (Top " + topK + " candidates: ");
            for (int j = 0; j < topK; j++) {
                String candidate = tokenizer.decode(new int[]{indices[j]});
                float prob = computeSoftmax(logits)[indices[j]];
                explanation.append("\"" + candidate + "\"(" + String.format("%.2f", prob * 100) + "%)");
                if (j < topK - 1) explanation.append(", ");
            }
            explanation.append(")\n");
        }
        
        explanation.append("\n=== End of Explanation ===");
        
        return explanation.toString();
    }

    public void printFeatureImportance(FeatureImportance importance) {
        System.out.println("\n=== Feature Importance ===");
        System.out.println("Target Score: " + String.format("%.6f", importance.getTargetScore()));
        System.out.println("\nToken Importance:");
        
        for (int i = 0; i < importance.getTokens().length; i++) {
            String token = importance.getTokens()[i];
            double score = importance.getImportances()[i];
            
            int barLength = (int) (score * 50);
            StringBuilder bar = new StringBuilder();
            for (int j = 0; j < 50; j++) {
                bar.append(j < barLength ? "=" : " ");
            }
            
            System.out.printf("  %-15s | [%s] %.4f%n", 
                "\"" + token + "\"", bar.toString(), score);
        }
        System.out.println("=========================");
    }

    public void printAttentionAnalysis(AttentionAnalysis analysis) {
        System.out.println("\n=== Attention Analysis ===");
        
        for (int l = 0; l < analysis.getAttentionMaps().length; l++) {
            System.out.println("\nLayer " + l + ":");
            float[][] attention = analysis.getAttentionMaps()[l];
            
            System.out.print("      ");
            for (int j = 0; j < analysis.getTokens().length; j++) {
                System.out.printf("%8s", "\"" + analysis.getTokens()[j] + "\"");
            }
            System.out.println();
            
            for (int i = 0; i < analysis.getTokens().length; i++) {
                System.out.printf("%8s", "\"" + analysis.getTokens()[i] + "\"");
                for (int j = 0; j < analysis.getTokens().length; j++) {
                    System.out.printf("%8.2f", attention[i][j]);
                }
                System.out.println();
            }
        }
        System.out.println("==========================");
    }

    public void printSaliencyMap(SaliencyMap saliency) {
        System.out.println("\n=== Saliency Map ===");
        System.out.println("Predicted: \"" + saliency.getPredictedToken() + "\"");
        System.out.println("\nToken Saliency:");
        
        for (int i = 0; i < saliency.getTokens().length; i++) {
            String token = saliency.getTokens()[i];
            double score = saliency.getSaliency()[i];
            
            int barLength = (int) (score * 50);
            StringBuilder bar = new StringBuilder();
            for (int j = 0; j < 50; j++) {
                bar.append(j < barLength ? "=" : " ");
            }
            
            System.out.printf("  %-15s | [%s] %.4f%n", 
                "\"" + token + "\"", bar.toString(), score);
        }
        System.out.println("===================");
    }

    public static class FeatureImportance {
        private final String[] tokens;
        private final double[] importances;
        private final float targetScore;

        public FeatureImportance(String[] tokens, double[] importances, float targetScore) {
            this.tokens = tokens;
            this.importances = importances;
            this.targetScore = targetScore;
        }

        public String[] getTokens() { return tokens; }
        public double[] getImportances() { return importances; }
        public float getTargetScore() { return targetScore; }
    }

    public static class AttentionAnalysis {
        private final String[] tokens;
        private final float[][][] attentionMaps;

        public AttentionAnalysis(String[] tokens, float[][][] attentionMaps) {
            this.tokens = tokens;
            this.attentionMaps = attentionMaps;
        }

        public String[] getTokens() { return tokens; }
        public float[][][] getAttentionMaps() { return attentionMaps; }
    }

    public static class SaliencyMap {
        private final String[] tokens;
        private final double[] saliency;
        private final String predictedToken;

        public SaliencyMap(String[] tokens, double[] saliency, String predictedToken) {
            this.tokens = tokens;
            this.saliency = saliency;
            this.predictedToken = predictedToken;
        }

        public String[] getTokens() { return tokens; }
        public double[] getSaliency() { return saliency; }
        public String getPredictedToken() { return predictedToken; }
    }

    public static class CounterfactualExplanation {
        private final String modifiedText;
        private final int changedPosition;
        private final String originalToken;
        private final String newToken;
        private final float originalProbability;
        private final float modifiedProbability;
        private final float probabilityChange;

        public CounterfactualExplanation(String modifiedText, int changedPosition,
                                         String originalToken, String newToken,
                                         float originalProbability, float modifiedProbability,
                                         float probabilityChange) {
            this.modifiedText = modifiedText;
            this.changedPosition = changedPosition;
            this.originalToken = originalToken;
            this.newToken = newToken;
            this.originalProbability = originalProbability;
            this.modifiedProbability = modifiedProbability;
            this.probabilityChange = probabilityChange;
        }

        public String getModifiedText() { return modifiedText; }
        public int getChangedPosition() { return changedPosition; }
        public String getOriginalToken() { return originalToken; }
        public String getNewToken() { return newToken; }
        public float getOriginalProbability() { return originalProbability; }
        public float getModifiedProbability() { return modifiedProbability; }
        public float getProbabilityChange() { return probabilityChange; }
    }
}
package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;

import java.util.*;

public class ModelEvaluator {

    private final SimpCoreModel model;
    private final Tokenizer tokenizer;

    public ModelEvaluator(SimpCoreModel model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public EvaluationMetrics evaluate(String[] texts) {
        float totalLoss = 0;
        int totalTokens = 0;
        int totalCorrect = 0;
        List<Float> perplexities = new ArrayList<>();

        for (String text : texts) {
            int[] tokens = tokenizer.encode(text);
            if (tokens.length < 2) continue;

            int seqLen = Math.min(tokens.length - 1, model.getMaxSeqLen());
            int[] inputIds = new int[seqLen];
            int[] targetIds = new int[seqLen];

            for (int s = 0; s < seqLen; s++) {
                inputIds[s] = tokens[s];
                targetIds[s] = tokens[s + 1];
            }

            float[] logits = model.forward(inputIds, 1, seqLen);
            float loss = CrossEntropyLoss.compute(logits, targetIds, 1, seqLen, tokenizer.getVocabSize());
            totalLoss += loss;

            int correct = 0;
            for (int i = 0; i < seqLen; i++) {
                float[] tokenLogits = new float[tokenizer.getVocabSize()];
                System.arraycopy(logits, i * tokenizer.getVocabSize(), tokenLogits, 0, tokenizer.getVocabSize());
                
                int predicted = argmax(tokenLogits);
                if (predicted == targetIds[i]) {
                    correct++;
                }
            }

            totalCorrect += correct;
            totalTokens += seqLen;

            float perplexity = (float) Math.exp(loss);
            perplexities.add(perplexity);
        }

        float avgLoss = totalTokens > 0 ? totalLoss / texts.length : Float.MAX_VALUE;
        float accuracy = totalTokens > 0 ? (float) totalCorrect / totalTokens : 0;
        float avgPerplexity = perplexities.stream().mapToDouble(Float::doubleValue).average().orElse(Float.MAX_VALUE);
        float minPerplexity = perplexities.stream().mapToFloat(Float::floatValue).min().orElse(Float.MAX_VALUE);
        float maxPerplexity = perplexities.stream().mapToFloat(Float::floatValue).max().orElse(Float.MAX_VALUE);

        return new EvaluationMetrics(avgLoss, accuracy, avgPerplexity, minPerplexity, maxPerplexity, totalTokens, totalCorrect);
    }

    public GenerationMetrics evaluateGeneration(String[] prompts, int maxNewTokens) {
        List<String> generatedTexts = new ArrayList<>();
        List<Integer> generationLengths = new ArrayList<>();
        List<Float> generationTimes = new ArrayList<>();

        for (String prompt : prompts) {
            long startTime = System.nanoTime();
            String generated = generateText(prompt, maxNewTokens);
            long endTime = System.nanoTime();

            generatedTexts.add(generated);
            generationLengths.add(generated.length());
            generationTimes.add((endTime - startTime) / 1_000_000.0f);
        }

        float avgLength = generationLengths.stream().mapToInt(Integer::intValue).average().orElse(0);
        float avgTime = generationTimes.stream().mapToDouble(Float::doubleValue).average().orElse(0);
        float minTime = generationTimes.stream().mapToFloat(Float::floatValue).min().orElse(Float.MAX_VALUE);
        float maxTime = generationTimes.stream().mapToFloat(Float::floatValue).max().orElse(Float.MIN_VALUE);

        return new GenerationMetrics(generatedTexts, avgLength, avgTime, minTime, maxTime);
    }

    private String generateText(String prompt, int maxNewTokens) {
        int[] inputIds = tokenizer.encode(prompt);
        int[] outputIds = model.generate(inputIds, maxNewTokens, tokenizer.getEosTokenId());
        return tokenizer.decode(outputIds);
    }

    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public static class EvaluationMetrics {
        public final float averageLoss;
        public final float accuracy;
        public final float averagePerplexity;
        public final float minPerplexity;
        public final float maxPerplexity;
        public final int totalTokens;
        public final int correctTokens;

        public EvaluationMetrics(float averageLoss, float accuracy, float averagePerplexity,
                                 float minPerplexity, float maxPerplexity, int totalTokens, int correctTokens) {
            this.averageLoss = averageLoss;
            this.accuracy = accuracy;
            this.averagePerplexity = averagePerplexity;
            this.minPerplexity = minPerplexity;
            this.maxPerplexity = maxPerplexity;
            this.totalTokens = totalTokens;
            this.correctTokens = correctTokens;
        }

        @Override
        public String toString() {
            return String.format(
                "EvaluationMetrics{averageLoss=%.4f, accuracy=%.4f, averagePerplexity=%.4f, " +
                "minPerplexity=%.4f, maxPerplexity=%.4f, totalTokens=%d, correctTokens=%d}",
                averageLoss, accuracy, averagePerplexity, minPerplexity, maxPerplexity, totalTokens, correctTokens
            );
        }
    }

    public static class GenerationMetrics {
        public final List<String> generatedTexts;
        public final float averageLength;
        public final float averageTimeMs;
        public final float minTimeMs;
        public final float maxTimeMs;

        public GenerationMetrics(List<String> generatedTexts, float averageLength,
                                 float averageTimeMs, float minTimeMs, float maxTimeMs) {
            this.generatedTexts = generatedTexts;
            this.averageLength = averageLength;
            this.averageTimeMs = averageTimeMs;
            this.minTimeMs = minTimeMs;
            this.maxTimeMs = maxTimeMs;
        }

        @Override
        public String toString() {
            return String.format(
                "GenerationMetrics{averageLength=%.2f, averageTimeMs=%.2f, minTimeMs=%.2f, maxTimeMs=%.2f, " +
                "generatedCount=%d}",
                averageLength, averageTimeMs, minTimeMs, maxTimeMs, generatedTexts.size()
            );
        }
    }
}
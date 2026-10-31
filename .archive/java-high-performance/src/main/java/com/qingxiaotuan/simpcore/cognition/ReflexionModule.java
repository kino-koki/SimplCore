package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import java.util.*;

/**
 * 自我反思模块 — 模拟人类的元认知能力
 * 
 * 人类的高级认知能力包括：
 * 1. 自我监控：评估自己的思考过程和结果
 * 2. 错误检测：发现推理中的错误和不一致
 * 3. 策略调整：根据评估结果调整推理策略
 * 4. 知识修正：发现知识缺口并主动学习
 * 
 * 参考论文：Reflexion: Language Agents with Verbal Reinforcement Learning
 */
public class ReflexionModule {

    private final int dModel;
    
    /** 反思评估权重 */
    private final float[] wReflect;
    
    /** 修正权重 */
    private final float[] wCorrect;
    
    /** 知识缺口检测权重 */
    private final float[] wGapDetect;
    
    /** 反思历史 */
    private final List<ReflectionRecord> reflectionHistory;
    
    /** 错误模式库 */
    private final Map<Integer, float[]> errorPatterns;
    
    private final Random random;
    
    /** 反思记录 */
    private static class ReflectionRecord {
        float[] input;
        float[] output;
        float[] correctedOutput;
        float confidence;
        float correctionGain;
        boolean wasCorrected;
        long timestamp;
        
        ReflectionRecord(float[] input, float[] output, float confidence) {
            this.input = input;
            this.output = output;
            this.correctedOutput = null;
            this.confidence = confidence;
            this.correctionGain = 0;
            this.wasCorrected = false;
            this.timestamp = System.currentTimeMillis();
        }
    }

    public ReflexionModule(int dModel) {
        this.dModel = dModel;
        this.random = new Random(42);
        this.reflectionHistory = new ArrayList<>();
        this.errorPatterns = new HashMap<>();
        
        this.wReflect = new float[dModel * dModel];
        this.wCorrect = new float[2 * dModel * dModel];
        this.wGapDetect = new float[dModel];
        
        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < wReflect.length; i++) {
            wReflect[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
        for (int i = 0; i < wCorrect.length; i++) {
            wCorrect[i] = (random.nextFloat() - 0.5f) * 0.03f;
        }
        for (int i = 0; i < wGapDetect.length; i++) {
            wGapDetect[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
    }

    /**
     * 评估推理结果的质量
     * 1. 计算置信度
     * 2. 检测错误模式
     * 3. 识别知识缺口
     */
    public float evaluate(float[] input, float[] output) {
        float[] reflect = MatrixOperations.matmul(output, wReflect, 1, dModel, dModel);
        
        float confidence = 0;
        for (int i = 0; i < dModel; i++) {
            confidence += reflect[i] * output[i];
        }
        confidence = (float) Math.tanh(confidence);
        
        int patternId = detectErrorPattern(output);
        if (patternId != -1) {
            confidence *= 0.5f;
        }
        
        float gapScore = detectKnowledgeGap(input, output);
        if (gapScore > 0.5f) {
            confidence *= (1.0f - gapScore);
        }
        
        reflectionHistory.add(new ReflectionRecord(input, output, confidence));
        
        return confidence;
    }

    /**
     * 检测错误模式
     */
    private int detectErrorPattern(float[] output) {
        for (Map.Entry<Integer, float[]> entry : errorPatterns.entrySet()) {
            float[] pattern = entry.getValue();
            float similarity = 0;
            float norm1 = 0, norm2 = 0;
            
            for (int i = 0; i < dModel; i++) {
                similarity += output[i] * pattern[i];
                norm1 += output[i] * output[i];
                norm2 += pattern[i] * pattern[i];
            }
            
            float sim = similarity / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
            if (sim > 0.7f) {
                return entry.getKey();
            }
        }
        return -1;
    }

    /**
     * 检测知识缺口
     */
    private float detectKnowledgeGap(float[] input, float[] output) {
        float gapScore = 0;
        for (int i = 0; i < dModel; i++) {
            gapScore += Math.abs(input[i] - output[i]) * wGapDetect[i];
        }
        return (float) Math.tanh(gapScore);
    }

    /**
     * 修正推理结果
     * 如果评估发现问题，尝试修正
     */
    public float[] correct(float[] input, float[] output, float confidenceThreshold) {
        float confidence = evaluate(input, output);
        
        if (confidence >= confidenceThreshold) {
            return output;
        }
        
        float[] corrected = new float[dModel];
        
        float[] concat = new float[2 * dModel];
        System.arraycopy(input, 0, concat, 0, dModel);
        System.arraycopy(output, 0, concat, dModel, dModel);
        
        float[] correction = MatrixOperations.matmul(concat, wCorrect, 1, 2 * dModel, dModel);
        
        for (int i = 0; i < dModel; i++) {
            corrected[i] = output[i] + (1.0f - confidence) * correction[i];
        }
        
        float[] normalized = layerNorm(corrected);
        
        if (reflectionHistory.size() > 0) {
            ReflectionRecord last = reflectionHistory.get(reflectionHistory.size() - 1);
            last.correctedOutput = normalized;
            last.wasCorrected = true;
            last.correctionGain = computeCorrectionGain(output, normalized);
        }
        
        return normalized;
    }

    /**
     * 计算修正增益
     */
    private float computeCorrectionGain(float[] original, float[] corrected) {
        float diff = 0;
        for (int i = 0; i < dModel; i++) {
            float d = corrected[i] - original[i];
            diff += d * d;
        }
        return (float) Math.sqrt(diff);
    }

    /**
     * 反思循环 — 迭代修正直到达到满意质量
     */
    public float[] reflectAndCorrect(float[] input, float[] output, int maxIterations, 
                                      float confidenceThreshold) {
        float[] current = Arrays.copyOf(output, dModel);
        
        for (int iter = 0; iter < maxIterations; iter++) {
            float confidence = evaluate(input, current);
            
            if (confidence >= confidenceThreshold) {
                break;
            }
            
            current = correct(input, current, confidenceThreshold);
        }
        
        return current;
    }

    /**
     * 从历史中学习错误模式
     */
    public void learnErrorPattern(float[] pattern) {
        int id = errorPatterns.size() + 1;
        errorPatterns.put(id, Arrays.copyOf(pattern, dModel));
    }

    /**
     * 获取反思统计
     */
    public ReflectionStats getStats() {
        if (reflectionHistory.isEmpty()) {
            return new ReflectionStats(0, 0, 0, 0);
        }
        
        int total = reflectionHistory.size();
        int corrected = 0;
        float avgConfidence = 0;
        float avgCorrectionGain = 0;
        
        for (ReflectionRecord record : reflectionHistory) {
            avgConfidence += record.confidence;
            if (record.wasCorrected) {
                corrected++;
                avgCorrectionGain += record.correctionGain;
            }
        }
        
        avgConfidence /= total;
        avgCorrectionGain = corrected > 0 ? avgCorrectionGain / corrected : 0;
        
        return new ReflectionStats(total, corrected, avgConfidence, avgCorrectionGain);
    }

    public static class ReflectionStats {
        public final int totalReflections;
        public final int correctedCount;
        public final float avgConfidence;
        public final float avgCorrectionGain;
        
        public ReflectionStats(int total, int corrected, float confidence, float gain) {
            this.totalReflections = total;
            this.correctedCount = corrected;
            this.avgConfidence = confidence;
            this.avgCorrectionGain = gain;
        }
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

    public int getDModel() { return dModel; }
    public float[] getWReflect() { return wReflect; }
    public float[] getWCorrect() { return wCorrect; }
    public float[] getWGapDetect() { return wGapDetect; }
    public List<ReflectionRecord> getReflectionHistory() { return reflectionHistory; }
    public Map<Integer, float[]> getErrorPatterns() { return errorPatterns; }
}
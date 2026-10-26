package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import java.util.*;

/**
 * 情感决策引擎 — 模拟人类的情感驱动决策机制
 * 
 * 人类决策不是完全理性的，而是受情感影响：
 * 1. 动机驱动：内在需求和目标驱动行为
 * 2. 价值判断：基于价值观评估选项
 * 3. 风险偏好：不同情绪状态下的风险承受能力
 * 4. 情绪调节：根据情绪状态调整决策策略
 * 5. 社会偏好：考虑他人和社会规范
 * 
 * 参考论文：
 * - Affective Computing
 * - Emotional Intelligence in AI
 */
public class EmotionalDecisionEngine {

    private final int dModel;
    
    /** 情感状态维度 */
    public enum Emotion {
        HAPPINESS, SADNESS, ANGER, FEAR, SURPRISE, DISGUST, NEUTRAL
    }
    
    /** 动机类型 */
    public enum Motivation {
        SURVIVAL, SOCIAL, ACHIEVEMENT, EXPLORATION, CREATION, RELAXATION
    }
    
    /** 情感权重 — 每种情感对决策的影响 */
    private final float[] emotionWeights;
    
    /** 动机权重 */
    private final float[] motivationWeights;
    
    /** 价值权重 */
    private final float[] valueWeights;
    
    /** 当前情感状态 */
    private float[] currentEmotion;
    
    /** 当前动机 */
    private Motivation currentMotivation;
    
    /** 情绪历史 */
    private final List<float[]> emotionHistory;
    
    /** 情绪衰减率 */
    private final float emotionDecayRate;
    
    /** 风险偏好系数 */
    private float riskPreference;
    
    private final Random random;

    public EmotionalDecisionEngine(int dModel) {
        this.dModel = dModel;
        this.random = new Random(42);
        this.emotionHistory = new ArrayList<>();
        this.emotionDecayRate = 0.05f;
        this.riskPreference = 0.5f;
        this.currentMotivation = Motivation.ACHIEVEMENT;
        
        this.emotionWeights = new float[dModel];
        this.motivationWeights = new float[Motivation.values().length * dModel];
        this.valueWeights = new float[dModel];
        
        this.currentEmotion = new float[dModel];
        
        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < emotionWeights.length; i++) {
            emotionWeights[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        for (int i = 0; i < motivationWeights.length; i++) {
            motivationWeights[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        for (int i = 0; i < valueWeights.length; i++) {
            valueWeights[i] = (random.nextFloat() - 0.5f) * 0.02f;
        }
        
        Arrays.fill(currentEmotion, 0.1f);
    }

    /**
     * 设置情感状态
     */
    public void setEmotion(Emotion emotion, float intensity) {
        float[] emotionVector = new float[dModel];
        float baseValue = intensity * 0.5f;
        
        switch (emotion) {
            case HAPPINESS:
                for (int i = 0; i < dModel; i++) {
                    emotionVector[i] = baseValue * (1 + (random.nextFloat() - 0.5f) * 0.2f);
                }
                break;
            case SADNESS:
                for (int i = 0; i < dModel; i++) {
                    emotionVector[i] = -baseValue * (1 + (random.nextFloat() - 0.5f) * 0.2f);
                }
                break;
            case ANGER:
                for (int i = 0; i < dModel; i++) {
                    emotionVector[i] = baseValue * (random.nextFloat() > 0.5f ? 1 : -1);
                }
                break;
            case FEAR:
                for (int i = 0; i < dModel; i++) {
                    emotionVector[i] = -baseValue * Math.abs(random.nextFloat() - 0.5f);
                }
                break;
            case SURPRISE:
                for (int i = 0; i < dModel; i++) {
                    emotionVector[i] = (random.nextFloat() - 0.5f) * baseValue * 2;
                }
                break;
            case DISGUST:
                for (int i = 0; i < dModel; i++) {
                    emotionVector[i] = -baseValue * (0.5f + random.nextFloat() * 0.5f);
                }
                break;
            case NEUTRAL:
                Arrays.fill(emotionVector, 0);
                break;
        }
        
        for (int i = 0; i < dModel; i++) {
            currentEmotion[i] = (1 - emotionDecayRate) * currentEmotion[i] + emotionDecayRate * emotionVector[i];
        }
        
        emotionHistory.add(Arrays.copyOf(currentEmotion, dModel));
        if (emotionHistory.size() > 100) {
            emotionHistory.remove(0);
        }
    }

    /**
     * 设置当前动机
     */
    public void setMotivation(Motivation motivation) {
        this.currentMotivation = motivation;
    }

    /**
     * 设置风险偏好
     * 0 = 保守, 0.5 = 中性, 1 = 冒险
     */
    public void setRiskPreference(float preference) {
        this.riskPreference = Math.max(0, Math.min(1, preference));
    }

    /**
     * 情感驱动的决策 — 在多个选项中选择最优
     */
    public DecisionResult decide(float[][] options, float[][] optionRisks) {
        float[] scores = new float[options.length];
        
        for (int optIdx = 0; optIdx < options.length; optIdx++) {
            float[] option = options[optIdx];
            
            float valueScore = 0;
            for (int i = 0; i < dModel; i++) {
                valueScore += option[i] * valueWeights[i];
            }
            
            float emotionScore = 0;
            for (int i = 0; i < dModel; i++) {
                emotionScore += option[i] * currentEmotion[i] * emotionWeights[i];
            }
            
            float motivationScore = 0;
            int motIdx = currentMotivation.ordinal();
            for (int i = 0; i < dModel; i++) {
                motivationScore += option[i] * motivationWeights[motIdx * dModel + i];
            }
            
            float riskScore = 0;
            if (optionRisks != null && optIdx < optionRisks.length) {
                float avgRisk = 0;
                for (float risk : optionRisks[optIdx]) {
                    avgRisk += risk;
                }
                avgRisk /= optionRisks[optIdx].length;
                
                riskScore = (riskPreference - avgRisk) * 0.5f;
            }
            
            scores[optIdx] = valueScore + emotionScore + motivationScore + riskScore;
        }
        
        int bestIdx = 0;
        float bestScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }
        
        float[] certainty = new float[options.length];
        float maxScore = Float.NEGATIVE_INFINITY;
        float minScore = Float.POSITIVE_INFINITY;
        for (float s : scores) {
            if (s > maxScore) maxScore = s;
            if (s < minScore) minScore = s;
        }
        float range = maxScore - minScore + 1e-8f;
        
        for (int i = 0; i < scores.length; i++) {
            certainty[i] = (scores[i] - minScore) / range;
        }
        
        return new DecisionResult(bestIdx, scores, certainty);
    }

    /**
     * 情感感知 — 从输入中提取情感信息
     */
    public float[] perceiveEmotion(float[] input) {
        float[] emotionPerception = new float[dModel];
        
        for (int i = 0; i < dModel; i++) {
            emotionPerception[i] = input[i] * emotionWeights[i];
        }
        
        return layerNorm(emotionPerception);
    }

    /**
     * 情绪调节 — 根据目标情绪调整当前情绪
     */
    public void regulateEmotion(float[] targetEmotion, float strength) {
        for (int i = 0; i < dModel; i++) {
            currentEmotion[i] = (1 - strength) * currentEmotion[i] + strength * targetEmotion[i];
        }
    }

    /**
     * 获取情绪状态描述
     */
    public Emotion getCurrentEmotionType() {
        float mean = 0;
        float variance = 0;
        
        for (float v : currentEmotion) {
            mean += v;
        }
        mean /= dModel;
        
        for (float v : currentEmotion) {
            variance += (v - mean) * (v - mean);
        }
        variance /= dModel;
        
        if (mean > 0.1f && variance < 0.05f) {
            return Emotion.HAPPINESS;
        } else if (mean < -0.1f && variance < 0.05f) {
            return Emotion.SADNESS;
        } else if (variance > 0.1f && mean > 0) {
            return Emotion.ANGER;
        } else if (variance > 0.1f && mean < 0) {
            return Emotion.FEAR;
        } else if (Math.abs(mean) > 0.2f) {
            return Emotion.SURPRISE;
        } else if (mean < -0.05f && variance < 0.03f) {
            return Emotion.DISGUST;
        } else {
            return Emotion.NEUTRAL;
        }
    }

    /**
     * 获取情绪强度
     */
    public float getEmotionIntensity() {
        float norm = 0;
        for (float v : currentEmotion) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    /**
     * 情感增强推理 — 情绪影响推理过程
     */
    public float[] emotionEnhancedReasoning(float[] input) {
        float[] emotionInfluenced = new float[dModel];
        
        for (int i = 0; i < dModel; i++) {
            emotionInfluenced[i] = input[i] * (1 + currentEmotion[i] * 0.1f);
        }
        
        return layerNorm(emotionInfluenced);
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

    public static class DecisionResult {
        public final int chosenIndex;
        public final float[] scores;
        public final float[] certainty;
        
        public DecisionResult(int chosen, float[] scores, float[] certainty) {
            this.chosenIndex = chosen;
            this.scores = scores;
            this.certainty = certainty;
        }
    }

    public int getDModel() { return dModel; }
    public float[] getCurrentEmotion() { return currentEmotion; }
    public Motivation getCurrentMotivation() { return currentMotivation; }
    public float getRiskPreference() { return riskPreference; }
    public float[] getEmotionWeights() { return emotionWeights; }
    public float[] getMotivationWeights() { return motivationWeights; }
    public float[] getValueWeights() { return valueWeights; }
    public int getEmotionHistorySize() { return emotionHistory.size(); }
}
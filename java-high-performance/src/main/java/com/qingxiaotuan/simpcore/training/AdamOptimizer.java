package com.qingxiaotuan.simpcore.training;

import java.util.HashMap;
import java.util.Map;

public class AdamOptimizer {
    
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private int t;
    
    private Map<String, float[]> m;  // 一阶矩
    private Map<String, float[]> v;  // 二阶矩
    
    public AdamOptimizer(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f);
    }
    
    public AdamOptimizer(float learningRate, float beta1, float beta2, float epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
        this.m = new HashMap<>();
        this.v = new HashMap<>();
    }
    
    public void update(Map<String, float[]> parameters, Map<String, float[]> gradients, float lrScale) {
        t++;
        
        for (String key : gradients.keySet()) {
            float[] grad = gradients.get(key);
            float[] param = parameters.get(key);
            if (param == null || grad == null) continue;
            
            float[] mt = m.computeIfAbsent(key, k -> new float[grad.length]);
            float[] vt = v.computeIfAbsent(key, k -> new float[grad.length]);
            
            for (int i = 0; i < grad.length && i < param.length; i++) {
                // 更新一阶矩
                mt[i] = beta1 * mt[i] + (1 - beta1) * grad[i];
                // 更新二阶矩
                vt[i] = beta2 * vt[i] + (1 - beta2) * grad[i] * grad[i];
                
                // 偏差校正
                float mHat = mt[i] / (1 - (float) Math.pow(beta1, t));
                float vHat = vt[i] / (1 - (float) Math.pow(beta2, t));
                
                // 更新参数
                param[i] -= learningRate * lrScale * mHat / ((float) Math.sqrt(vHat) + epsilon);
            }
        }
    }
    
    public void reset() {
        t = 0;
        m.clear();
        v.clear();
    }
}

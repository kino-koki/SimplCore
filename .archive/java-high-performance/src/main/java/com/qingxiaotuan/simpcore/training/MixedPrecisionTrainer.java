package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.RDTModel;
import com.qingxiaotuan.simpcore.model.RDTConfig;

public class MixedPrecisionTrainer {
    private RDTModel model;
    private ZeRO3Optimizer optimizer;
    private float scale;
    private float growthFactor;
    private float backoffFactor;
    private float growthInterval;
    private int step;
    private int growthStep;
    
    private float[] grads;
    
    public MixedPrecisionTrainer(RDTModel model, ZeRO3Optimizer optimizer) {
        this.model = model;
        this.optimizer = optimizer;
        this.scale = 2048.0f;
        this.growthFactor = 2.0f;
        this.backoffFactor = 0.5f;
        this.growthInterval = 2000.0f;
        this.step = 0;
        this.growthStep = 0;
    }
    
    public float[] trainStep(int[] tokens, int[] labels, int nLoops) {
        step++;
        
        float[] logits = model.forward(tokens, nLoops);
        grads = computeGradients(logits, labels);
        
        for (int i = 0; i < grads.length; i++) {
            grads[i] *= scale;
        }
        
        optimizer.setGradients(grads);
        optimizer.allReduceGradients();
        
        boolean overflow = checkOverflow(grads);
        
        if (!overflow) {
            optimizer.step();
            growthStep++;
            
            if (growthStep >= growthInterval) {
                scale *= growthFactor;
                growthStep = 0;
            }
        } else {
            scale *= backoffFactor;
            growthStep = 0;
        }
        
        return logits;
    }
    
    private float[] computeGradients(float[] logits, int[] labels) {
        float[] grads = new float[logits.length];
        
        float[] probs = softmax(logits);
        
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            if (label >= 0 && label < logits.length) {
                grads[label] -= 1.0f / labels.length;
            }
        }
        
        for (int i = 0; i < logits.length; i++) {
            grads[i] += probs[i] / labels.length;
        }
        
        return grads;
    }
    
    private boolean checkOverflow(float[] grads) {
        for (float grad : grads) {
            if (Float.isNaN(grad) || Float.isInfinite(grad)) {
                return true;
            }
        }
        return false;
    }
    
    private float[] softmax(float[] x) {
        float max = x[0];
        for (float v : x) {
            if (v > max) max = v;
        }
        
        float sum = 0.0f;
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = (float) Math.exp(x[i] - max);
            sum += result[i];
        }
        
        for (int i = 0; i < x.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    public float getLoss(float[] logits, int[] labels) {
        float loss = 0.0f;
        
        for (int label : labels) {
            if (label >= 0 && label < logits.length) {
                float maxLogit = logits[0];
                for (float logit : logits) {
                    if (logit > maxLogit) maxLogit = logit;
                }
                
                float sumExp = 0.0f;
                for (float logit : logits) {
                    sumExp += (float) Math.exp(logit - maxLogit);
                }
                
                loss += -logits[label] + maxLogit + (float) Math.log(sumExp);
            }
        }
        
        return loss / labels.length;
    }
    
    public float getScale() {
        return scale;
    }
    
    public int getStep() {
        return step;
    }
}

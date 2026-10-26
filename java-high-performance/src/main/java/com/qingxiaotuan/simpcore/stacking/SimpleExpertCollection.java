package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.model.SequenceModel;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.*;

/**
 * 简单专家模型集合 - 使用SequenceModel
 */
public class SimpleExpertCollection {
    
    private final int nExperts;
    private final int dModel;
    private final int vocabSize;
    private final SequenceModel[] experts;
    
    public SimpleExpertCollection(int nExperts, int dModel, int vocabSize) {
        this.nExperts = nExperts;
        this.dModel = dModel;
        this.vocabSize = vocabSize;
        
        this.experts = new SequenceModel[nExperts];
        initializeExperts();
    }
    
    private void initializeExperts() {
        System.out.println("[SimpleExpertCollection] 初始化 " + nExperts + " 个专家模型...");
        
        for (int e = 0; e < nExperts; e++) {
            experts[e] = new SequenceModel(vocabSize, dModel, 2, 4);
            System.out.println("[SimpleExpertCollection] 专家 " + e + " 初始化完成");
        }
        
        System.out.println("[SimpleExpertCollection] 初始化完成");
    }
    
    /**
     * 专家推理
     */
    public float[][] infer(int[] tokens) {
        float[][] results = new float[nExperts][];
        for (int e = 0; e < nExperts; e++) {
            results[e] = experts[e].forward(tokens);
        }
        return results;
    }
    
    /**
     * 获取专家模型
     */
    public SequenceModel getExpert(int index) {
        if (index < 0 || index >= nExperts) {
            throw new IllegalArgumentException("专家索引越界: " + index);
        }
        return experts[index];
    }
    
    /**
     * 更新专家参数
     */
    public void updateExpert(int expertIdx, GradientResult grad, float learningRate) {
        experts[expertIdx].updateParametersWithGradient(grad.getParameterGradients(), learningRate);
    }
    
    /**
     * 获取总参数量
     */
    public long getTotalParameters() {
        long total = 0;
        for (SequenceModel expert : experts) {
            float[][] weights = expert.getAllWeightArrays();
            for (float[] w : weights) {
                total += w.length;
            }
        }
        return total;
    }
    
    public int getNExperts() { return nExperts; }
    public int getDModel() { return dModel; }
    public int getVocabSize() { return vocabSize; }
}

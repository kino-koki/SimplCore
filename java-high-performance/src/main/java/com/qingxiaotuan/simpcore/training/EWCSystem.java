package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.math.MatrixOperations;

import java.util.*;
import java.io.*;

public class EWCSystem {

    private final SimpCoreModel model;
    private final Map<String, float[]> fisherMatrix = new HashMap<>();
    private final Map<String, float[]> thetaStar = new HashMap<>();
    private final Map<String, Integer> samplesPerGroup = new HashMap<>();
    private final float lambda;
    private final float importanceThreshold;
    private boolean initialized;
    
    private static final String[] PARAM_GROUPS = {"embed", "output", "ffn1", "ffn2", "ln"};

    public EWCSystem(SimpCoreModel model) {
        this(model, 1000.0f);
    }

    public EWCSystem(SimpCoreModel model, float lambda) {
        this(model, lambda, 0.01f);
    }

    public EWCSystem(SimpCoreModel model, float lambda, float importanceThreshold) {
        this.model = model;
        this.lambda = lambda;
        this.importanceThreshold = importanceThreshold;
        this.initialized = false;
    }

    public void initialize(List<float[]> validationData) {
        fisherMatrix.clear();
        thetaStar.clear();
        samplesPerGroup.clear();
        
        saveInitialWeights();
        
        if (validationData == null || validationData.isEmpty()) {
            estimateFisherPerGroup();
        } else {
            computeFisherPerGroup(validationData);
        }
        
        initialized = true;
        System.out.println("[EWC] Initialized: Fisher matrix computed for " + fisherMatrix.size() + " parameter groups");
    }

    private void saveInitialWeights() {
        thetaStar.put("embed", Arrays.copyOf(model.getEmbeddingWeights(), model.getEmbeddingWeights().length));
        thetaStar.put("output", Arrays.copyOf(model.getOutputWeights(), model.getOutputWeights().length));
        thetaStar.put("ffn1", flatten(model.getFfnWeights1()));
        thetaStar.put("ffn2", flatten(model.getFfnWeights2()));
        thetaStar.put("ln", flatten(model.getLayerNormWeight()));
        
        for (String group : PARAM_GROUPS) {
            samplesPerGroup.put(group, 0);
        }
    }

    private void computeFisherPerGroup(List<float[]> validationData) {
        int nSamples = validationData.size();
        
        Map<String, float[]> gradSquaredSum = new HashMap<>();
        for (String group : PARAM_GROUPS) {
            gradSquaredSum.put(group, new float[getGroupSize(group)]);
        }
        
        for (float[] sample : validationData) {
            Map<String, float[]> grads = computeGradientsPerGroup(sample);
            
            for (String group : PARAM_GROUPS) {
                float[] groupGrads = grads.get(group);
                float[] sum = gradSquaredSum.get(group);
                if (groupGrads != null && sum != null) {
                    for (int i = 0; i < Math.min(groupGrads.length, sum.length); i++) {
                        sum[i] += groupGrads[i] * groupGrads[i];
                    }
                }
            }
        }
        
        for (String group : PARAM_GROUPS) {
            float[] sum = gradSquaredSum.get(group);
            if (sum != null && nSamples > 0) {
                float[] fisher = new float[sum.length];
                for (int i = 0; i < sum.length; i++) {
                    fisher[i] = sum[i] / nSamples;
                }
                fisherMatrix.put(group, fisher);
                samplesPerGroup.put(group, nSamples);
            }
        }
    }

    private void estimateFisherPerGroup() {
        for (String group : PARAM_GROUPS) {
            float[] weights = getGroupWeights(group);
            float[] fisher = new float[weights.length];
            
            for (int i = 0; i < weights.length; i++) {
                float importance = Math.abs(weights[i]) * (float)(0.1 + Math.random() * 0.2);
                fisher[i] = importance;
            }
            
            fisherMatrix.put(group, fisher);
            samplesPerGroup.put(group, 1);
        }
    }

    public void onlineUpdateFisher(Map<String, float[]> newGradients) {
        if (!initialized) return;
        
        for (String group : PARAM_GROUPS) {
            float[] currentFisher = fisherMatrix.get(group);
            float[] groupGradients = newGradients.get(group);
            
            if (currentFisher == null || groupGradients == null) continue;
            
            int updateLength = Math.min(currentFisher.length, groupGradients.length);
            if (updateLength == 0) continue;
            
            int currentSamples = samplesPerGroup.getOrDefault(group, 1);
            int totalSamples = currentSamples + 1;
            
            for (int i = 0; i < updateLength; i++) {
                currentFisher[i] = (currentFisher[i] * currentSamples + groupGradients[i] * groupGradients[i]) / totalSamples;
            }
            samplesPerGroup.put(group, totalSamples);
        }
    }

    public float computeEWCLoss() {
        if (!initialized) return 0.0f;
        
        float totalLoss = 0.0f;
        
        for (String group : PARAM_GROUPS) {
            float[] fisher = fisherMatrix.get(group);
            float[] star = thetaStar.get(group);
            float[] current = getGroupWeights(group);
            
            if (fisher == null || star == null || current == null) continue;
            
            int len = Math.min(Math.min(current.length, star.length), fisher.length);
            for (int i = 0; i < len; i++) {
                if (fisher[i] > importanceThreshold) {
                    float diff = current[i] - star[i];
                    totalLoss += 0.5f * lambda * fisher[i] * diff * diff;
                }
            }
        }
        
        return totalLoss;
    }

    public void applyEWCGradient(float[] weights, String paramName) {
        if (!initialized) return;
        
        float[] fisher = fisherMatrix.get(paramName);
        float[] star = thetaStar.get(paramName);
        
        if (fisher == null || star == null) return;
        
        int len = Math.min(Math.min(weights.length, star.length), fisher.length);
        for (int i = 0; i < len; i++) {
            if (fisher[i] > importanceThreshold) {
                float ewcGrad = lambda * fisher[i] * (weights[i] - star[i]);
                weights[i] -= ewcGrad;
            }
        }
    }

    private Map<String, float[]> computeGradientsPerGroup(float[] input) {
        Map<String, float[]> gradients = new HashMap<>();
        
        for (String group : PARAM_GROUPS) {
            float[] weights = getGroupWeights(group);
            float[] grads = new float[weights.length];
            
            for (int i = 0; i < grads.length; i++) {
                grads[i] = (float)(Math.random() * 2 - 1) * 0.01f;
            }
            
            gradients.put(group, grads);
        }
        
        return gradients;
    }

    private float[] getGroupWeights(String group) {
        switch (group) {
            case "embed": return model.getEmbeddingWeights();
            case "output": return model.getOutputWeights();
            case "ffn1": return flatten(model.getFfnWeights1());
            case "ffn2": return flatten(model.getFfnWeights2());
            case "ln": return flatten(model.getLayerNormWeight());
            default: return new float[0];
        }
    }

    private int getGroupSize(String group) {
        switch (group) {
            case "embed": return model.getEmbeddingWeights().length;
            case "output": return model.getOutputWeights().length;
            case "ffn1": return flatten(model.getFfnWeights1()).length;
            case "ffn2": return flatten(model.getFfnWeights2()).length;
            case "ln": return flatten(model.getLayerNormWeight()).length;
            default: return 0;
        }
    }

    private float[] flatten(float[][] matrix) {
        int size = 0;
        for (float[] row : matrix) size += row.length;
        float[] result = new float[size];
        int idx = 0;
        for (float[] row : matrix) {
            System.arraycopy(row, 0, result, idx, row.length);
            idx += row.length;
        }
        return result;
    }

    public List<Integer> getImportantParams(int topN) {
        List<int[]> indexed = new ArrayList<>();
        
        int globalIdx = 0;
        for (String group : PARAM_GROUPS) {
            float[] fisher = fisherMatrix.get(group);
            if (fisher == null) continue;
            
            for (int i = 0; i < fisher.length; i++) {
                indexed.add(new int[]{globalIdx + i, (int)(fisher[i] * 10000)});
            }
            globalIdx += fisher.length;
        }
        
        indexed.sort((a, b) -> Integer.compare(b[1], a[1]));
        
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topN, indexed.size()); i++) {
            result.add(indexed.get(i)[0]);
        }
        
        return result;
    }

    public List<String> getImportantParamGroups(int topN) {
        List<int[]> groupImportance = new ArrayList<>();
        
        for (int i = 0; i < PARAM_GROUPS.length; i++) {
            float[] fisher = fisherMatrix.get(PARAM_GROUPS[i]);
            if (fisher == null) continue;
            
            float avgImportance = 0;
            for (float f : fisher) avgImportance += f;
            avgImportance /= fisher.length;
            
            groupImportance.add(new int[]{i, (int)(avgImportance * 10000)});
        }
        
        groupImportance.sort((a, b) -> Integer.compare(b[1], a[1]));
        
        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topN, groupImportance.size()); i++) {
            result.add(PARAM_GROUPS[groupImportance.get(i)[0]]);
        }
        
        return result;
    }

    public boolean isInitialized() { return initialized; }
    public float getLambda() { return lambda; }
    public int getSamplesForFisher() { 
        int total = 0;
        for (Integer s : samplesPerGroup.values()) total += s;
        return total / Math.max(1, samplesPerGroup.size());
    }
    public Map<String, float[]> getFisherMatrix() { return fisherMatrix; }
    public Map<String, float[]> getThetaStar() { return thetaStar; }

    public void save(DataOutputStream dos) throws IOException {
        dos.writeBoolean(initialized);
        dos.writeFloat(lambda);
        dos.writeFloat(importanceThreshold);
        
        dos.writeInt(fisherMatrix.size());
        for (Map.Entry<String, float[]> e : fisherMatrix.entrySet()) {
            dos.writeUTF(e.getKey());
            dos.writeInt(e.getValue().length);
            for (float f : e.getValue()) dos.writeFloat(f);
        }
        
        dos.writeInt(thetaStar.size());
        for (Map.Entry<String, float[]> e : thetaStar.entrySet()) {
            dos.writeUTF(e.getKey());
            dos.writeInt(e.getValue().length);
            for (float f : e.getValue()) dos.writeFloat(f);
        }
        
        dos.writeInt(samplesPerGroup.size());
        for (Map.Entry<String, Integer> e : samplesPerGroup.entrySet()) {
            dos.writeUTF(e.getKey());
            dos.writeInt(e.getValue());
        }
    }

    public void load(DataInputStream dis) throws IOException {
        initialized = dis.readBoolean();
        float storedLambda = dis.readFloat();
        float storedThreshold = dis.readFloat();
        
        int fisherSize = dis.readInt();
        fisherMatrix.clear();
        for (int i = 0; i < fisherSize; i++) {
            String key = dis.readUTF();
            int len = dis.readInt();
            float[] data = new float[len];
            for (int j = 0; j < len; j++) data[j] = dis.readFloat();
            fisherMatrix.put(key, data);
        }
        
        int starSize = dis.readInt();
        thetaStar.clear();
        for (int i = 0; i < starSize; i++) {
            String key = dis.readUTF();
            int len = dis.readInt();
            float[] data = new float[len];
            for (int j = 0; j < len; j++) data[j] = dis.readFloat();
            thetaStar.put(key, data);
        }
        
        int samplesSize = dis.readInt();
        samplesPerGroup.clear();
        for (int i = 0; i < samplesSize; i++) {
            String key = dis.readUTF();
            int val = dis.readInt();
            samplesPerGroup.put(key, val);
        }
    }
}
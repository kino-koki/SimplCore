package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;

import java.util.*;
import java.io.*;

/**
 * Memory Replay Buffer - 动态记忆回放缓冲区
 * 
 * 专利级创新：
 * - 代表性样本选择：基于多样性和重要性选择存储样本
 * - 分层记忆存储：短期/中期/长期记忆分层管理
 * - 自适应回放策略：根据模型性能动态调整回放频率
 * - 遗忘机制：定期清理冗余或过时的记忆
 * - 样本增强：回放时对样本进行轻微扰动增强泛化能力
 * 
 * 工作原理：
 * 1. 在线学习时，选择代表性样本存储到缓冲区
 * 2. 定期从缓冲区回放旧样本进行巩固训练
 * 3. 与EWC配合使用，防止灾难性遗忘
 * 4. 根据新数据的分布变化调整回放策略
 */
public class MemoryReplayBuffer {

    public enum MemoryType {
        SHORT_TERM,
        MEDIUM_TERM,
        LONG_TERM
    }

    private static class MemoryEntry {
        final float[] data;
        final float[] featureVector;
        float importance;
        long timestamp;
        final MemoryType type;
        int accessCount;
        
        MemoryEntry(float[] data, float[] featureVector, float importance, MemoryType type) {
            this.data = data;
            this.featureVector = featureVector;
            this.importance = importance;
            this.type = type;
            this.timestamp = System.currentTimeMillis();
            this.accessCount = 0;
        }
    }

    private final List<MemoryEntry> shortTermBuffer = new ArrayList<>();
    private final List<MemoryEntry> mediumTermBuffer = new ArrayList<>();
    private final List<MemoryEntry> longTermBuffer = new ArrayList<>();
    
    private final int shortTermCapacity;
    private final int mediumTermCapacity;
    private final int longTermCapacity;
    
    private final float diversityThreshold;
    private final float importanceThreshold;
    
    private float[] currentFeatureMean;
    private float[] currentFeatureStd;
    private int featureUpdates;
    
    private final Random random;
    private final SimpCoreModel model;

    public MemoryReplayBuffer(SimpCoreModel model) {
        this(model, 100, 500, 2000);
    }

    public MemoryReplayBuffer(SimpCoreModel model, int shortTermCap, int mediumTermCap, int longTermCap) {
        this.model = model;
        this.shortTermCapacity = shortTermCap;
        this.mediumTermCapacity = mediumTermCap;
        this.longTermCapacity = longTermCap;
        this.diversityThreshold = 0.7f;
        this.importanceThreshold = 0.5f;
        this.random = new Random(42);
        this.featureUpdates = 0;
    }

    /**
     * 添加新样本到缓冲区
     */
    public void add(float[] data, float[] featureVector) {
        float importance = computeImportance(featureVector);
        MemoryType type = classifyMemory(importance);
        
        MemoryEntry entry = new MemoryEntry(data, featureVector, importance, type);
        
        synchronized (this) {
            switch (type) {
                case SHORT_TERM:
                    addToBuffer(shortTermBuffer, entry, shortTermCapacity);
                    break;
                case MEDIUM_TERM:
                    addToBuffer(mediumTermBuffer, entry, mediumTermCapacity);
                    break;
                case LONG_TERM:
                    addToBuffer(longTermBuffer, entry, longTermCapacity);
                    break;
            }
            
            updateFeatureStatistics(featureVector);
        }
    }

    /**
     * 添加到缓冲区（带容量限制和多样性检查）
     */
    private void addToBuffer(List<MemoryEntry> buffer, MemoryEntry entry, int capacity) {
        if (buffer.size() >= capacity) {
            buffer.remove(selectVictim(buffer));
        }
        
        if (isDiverse(buffer, entry)) {
            buffer.add(entry);
        }
    }

    /**
     * 判断样本是否具有足够多样性
     */
    private boolean isDiverse(List<MemoryEntry> buffer, MemoryEntry entry) {
        if (buffer.isEmpty()) return true;
        
        int similarCount = 0;
        for (MemoryEntry existing : buffer) {
            float similarity = cosineSimilarity(entry.featureVector, existing.featureVector);
            if (similarity > diversityThreshold) {
                similarCount++;
            }
        }
        
        return similarCount < buffer.size() * 0.3;
    }

    /**
     * 选择要移除的样本（淘汰策略）
     */
    private int selectVictim(List<MemoryEntry> buffer) {
        int victimIdx = 0;
        float minScore = Float.MAX_VALUE;
        
        for (int i = 0; i < buffer.size(); i++) {
            MemoryEntry e = buffer.get(i);
            float age = (System.currentTimeMillis() - e.timestamp) / 1000.0f;
            float score = age / (e.importance * (1 + e.accessCount));
            if (score < minScore) {
                minScore = score;
                victimIdx = i;
            }
        }
        
        return victimIdx;
    }

    /**
     * 计算样本重要性
     */
    private float computeImportance(float[] featureVector) {
        if (featureVector == null || featureVector.length == 0) {
            return 0.5f;
        }
        
        float norm = 0;
        for (float f : featureVector) norm += f * f;
        norm = (float) Math.sqrt(norm);
        
        if (currentFeatureMean != null) {
            float diff = 0;
            for (int i = 0; i < Math.min(featureVector.length, currentFeatureMean.length); i++) {
                diff += Math.abs(featureVector[i] - currentFeatureMean[i]);
            }
            return Math.min(1.0f, 0.3f + 0.7f * (diff / (currentFeatureMean.length * 2)));
        }
        
        return Math.min(1.0f, 0.5f + norm * 0.1f);
    }

    /**
     * 分类记忆类型
     */
    private MemoryType classifyMemory(float importance) {
        if (importance > 0.8f) return MemoryType.LONG_TERM;
        if (importance > 0.5f) return MemoryType.MEDIUM_TERM;
        return MemoryType.SHORT_TERM;
    }

    /**
     * 更新特征统计信息
     */
    private void updateFeatureStatistics(float[] featureVector) {
        if (featureVector == null || featureVector.length == 0) return;
        
        if (currentFeatureMean == null) {
            currentFeatureMean = new float[featureVector.length];
            currentFeatureStd = new float[featureVector.length];
        }
        
        featureUpdates++;
        float alpha = 1.0f / featureUpdates;
        
        for (int i = 0; i < featureVector.length; i++) {
            float oldMean = currentFeatureMean[i];
            currentFeatureMean[i] = alpha * featureVector[i] + (1 - alpha) * oldMean;
            
            float diff = featureVector[i] - currentFeatureMean[i];
            currentFeatureStd[i] = (float) Math.sqrt(
                alpha * diff * diff + (1 - alpha) * currentFeatureStd[i] * currentFeatureStd[i]
            );
        }
    }

    /**
     * 获取回放样本
     */
    public List<float[]> getReplayBatch(int batchSize) {
        List<float[]> batch = new ArrayList<>();
        
        synchronized (this) {
            int shortTermCount = (int)(batchSize * 0.2);
            int mediumTermCount = (int)(batchSize * 0.3);
            int longTermCount = batchSize - shortTermCount - mediumTermCount;
            
            batch.addAll(sampleFromBuffer(shortTermBuffer, shortTermCount));
            batch.addAll(sampleFromBuffer(mediumTermBuffer, mediumTermCount));
            batch.addAll(sampleFromBuffer(longTermBuffer, longTermCount));
            
            Collections.shuffle(batch, random);
        }
        
        return batch;
    }

    /**
     * 从缓冲区采样
     */
    private List<float[]> sampleFromBuffer(List<MemoryEntry> buffer, int count) {
        List<float[]> samples = new ArrayList<>();
        if (buffer.isEmpty() || count <= 0) return samples;
        
        List<MemoryEntry> shuffled = new ArrayList<>(buffer);
        Collections.shuffle(shuffled, random);
        
        int actualCount = Math.min(count, shuffled.size());
        for (int i = 0; i < actualCount; i++) {
            MemoryEntry entry = shuffled.get(i);
            entry.accessCount++;
            samples.add(enhanceSample(entry.data));
        }
        
        return samples;
    }

    /**
     * 样本增强（轻微扰动）
     */
    private float[] enhanceSample(float[] data) {
        float[] enhanced = Arrays.copyOf(data, data.length);
        
        for (int i = 0; i < enhanced.length; i++) {
            if (random.nextDouble() < 0.1) {
                enhanced[i] += (float)(random.nextGaussian()) * 0.01f;
            }
        }
        
        return enhanced;
    }

    /**
     * 余弦相似度计算
     */
    private float cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null) return 0;
        
        float dot = 0, normA = 0, normB = 0;
        int len = Math.min(a.length, b.length);
        
        for (int i = 0; i < len; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0 || normB == 0) return 0;
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    /**
     * 清理过时记忆
     */
    public void cleanup() {
        synchronized (this) {
            long now = System.currentTimeMillis();
            
            shortTermBuffer.removeIf(e -> (now - e.timestamp) > 3600000);
            mediumTermBuffer.removeIf(e -> (now - e.timestamp) > 86400000);
            
            for (MemoryEntry e : longTermBuffer) {
                if (e.accessCount < 3 && (now - e.timestamp) > 604800000) {
                    e.importance *= 0.9f;
                }
            }
            
            longTermBuffer.removeIf(e -> e.importance < 0.1f);
        }
    }

    /**
     * 获取缓冲区统计信息
     */
    public String getStatistics() {
        synchronized (this) {
            return String.format(
                "Memory Buffer: Short=%d/%d, Medium=%d/%d, Long=%d/%d, Features=%d",
                shortTermBuffer.size(), shortTermCapacity,
                mediumTermBuffer.size(), mediumTermCapacity,
                longTermBuffer.size(), longTermCapacity,
                featureUpdates
            );
        }
    }

    public int getTotalSize() {
        synchronized (this) {
            return shortTermBuffer.size() + mediumTermBuffer.size() + longTermBuffer.size();
        }
    }

    public void save(DataOutputStream dos) throws IOException {
        synchronized (this) {
            dos.writeInt(shortTermBuffer.size());
            for (MemoryEntry e : shortTermBuffer) saveEntry(dos, e);
            
            dos.writeInt(mediumTermBuffer.size());
            for (MemoryEntry e : mediumTermBuffer) saveEntry(dos, e);
            
            dos.writeInt(longTermBuffer.size());
            for (MemoryEntry e : longTermBuffer) saveEntry(dos, e);
            
            dos.writeInt(featureUpdates);
            if (currentFeatureMean != null) {
                dos.writeInt(currentFeatureMean.length);
                for (float f : currentFeatureMean) dos.writeFloat(f);
                for (float f : currentFeatureStd) dos.writeFloat(f);
            } else {
                dos.writeInt(0);
            }
        }
    }

    private void saveEntry(DataOutputStream dos, MemoryEntry e) throws IOException {
        dos.writeInt(e.data.length);
        for (float f : e.data) dos.writeFloat(f);
        
        if (e.featureVector != null) {
            dos.writeInt(e.featureVector.length);
            for (float f : e.featureVector) dos.writeFloat(f);
        } else {
            dos.writeInt(0);
        }
        
        dos.writeFloat(e.importance);
        dos.writeLong(e.timestamp);
        dos.writeInt(e.type.ordinal());
        dos.writeInt(e.accessCount);
    }

    public void load(DataInputStream dis) throws IOException {
        synchronized (this) {
            shortTermBuffer.clear();
            int shortTermSize = dis.readInt();
            for (int i = 0; i < shortTermSize; i++) shortTermBuffer.add(loadEntry(dis));
            
            mediumTermBuffer.clear();
            int mediumTermSize = dis.readInt();
            for (int i = 0; i < mediumTermSize; i++) mediumTermBuffer.add(loadEntry(dis));
            
            longTermBuffer.clear();
            int longTermSize = dis.readInt();
            for (int i = 0; i < longTermSize; i++) longTermBuffer.add(loadEntry(dis));
            
            featureUpdates = dis.readInt();
            int meanLen = dis.readInt();
            if (meanLen > 0) {
                currentFeatureMean = new float[meanLen];
                currentFeatureStd = new float[meanLen];
                for (int i = 0; i < meanLen; i++) currentFeatureMean[i] = dis.readFloat();
                for (int i = 0; i < meanLen; i++) currentFeatureStd[i] = dis.readFloat();
            }
        }
    }

    private MemoryEntry loadEntry(DataInputStream dis) throws IOException {
        int dataLen = dis.readInt();
        float[] data = new float[dataLen];
        for (int i = 0; i < dataLen; i++) data[i] = dis.readFloat();
        
        int featLen = dis.readInt();
        float[] featureVector = featLen > 0 ? new float[featLen] : null;
        if (featLen > 0) {
            for (int i = 0; i < featLen; i++) featureVector[i] = dis.readFloat();
        }
        
        float importance = dis.readFloat();
        long timestamp = dis.readLong();
        MemoryType type = MemoryType.values()[dis.readInt()];
        int accessCount = dis.readInt();
        
        MemoryEntry entry = new MemoryEntry(data, featureVector, importance, type);
        entry.timestamp = timestamp;
        entry.accessCount = accessCount;
        return entry;
    }
}
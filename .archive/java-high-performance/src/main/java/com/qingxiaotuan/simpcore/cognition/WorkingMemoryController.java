package com.qingxiaotuan.simpcore.cognition;

import com.qingxiaotuan.simpcore.math.MatrixOperations;
import com.qingxiaotuan.simpcore.memory.MemoryOptimizer;
import java.util.Arrays;
import java.util.Random;

/**
 * 工作记忆控制器 — 模拟人类工作记忆的有限容量和动态管理
 * 
 * 人类工作记忆的关键特性：
 * - 容量限制：7±2个槽位（Miller's Law）
 * - 基于注意力的读取：通过查询注意力加权读取
 * - 时间衰减：未使用的记忆随时间衰退
 * - 巩固机制：频繁访问的记忆被保护免于衰减
 * - 年龄追踪：每个槽位有年龄计数器，用于LRU替换策略
 */
public class WorkingMemoryController {

    /** 工作记忆槽位数 (7±2 = 默认7) */
    private final int numSlots;
    
    /** 每个槽位的向量维度 */
    private final int dModel;
    
    /** 槽位键矩阵: numSlots x dModel */
    private final float[][] slotKeys;
    
    /** 槽位值矩阵: numSlots x dModel */
    private final float[][] slotValues;
    
    /** 每个槽位的年龄计数器 */
    private final int[] ages;
    
    /** 每个槽位的访问计数 */
    private final int[] accessCounts;
    
    /** 每个槽位是否受保护(不被衰减) */
    private final boolean[] protectedSlots;
    
    /** 衰减率 */
    private final float decayRate;
    
    /** 保护阈值 — 访问计数超过此值则受保护 */
    private final float consolidationThreshold;
    
    /** 随机数生成器 */
    private final Random random;
    
    /** 当前步骤数 (用于年龄累计) */
    private int stepCount;

    public WorkingMemoryController(int dModel) {
        this(dModel, 7); // 默认7个槽位
    }

    public WorkingMemoryController(int dModel, int numSlots) {
        this.dModel = dModel;
        this.numSlots = Math.max(5, Math.min(9, numSlots)); // 限制在5-9之间
        this.decayRate = 0.01f;
        this.consolidationThreshold = 5.0f;
        this.random = new Random(42);
        this.stepCount = 0;
        
        this.slotKeys = new float[this.numSlots][dModel];
        this.slotValues = new float[this.numSlots][dModel];
        this.ages = new int[this.numSlots];
        this.accessCounts = new int[this.numSlots];
        this.protectedSlots = new boolean[this.numSlots];
        
        // 初始化槽位键（随机初始化，用于注意力计算）
        for (int i = 0; i < this.numSlots; i++) {
            for (int j = 0; j < dModel; j++) {
                slotKeys[i][j] = (random.nextFloat() - 0.5f) * 0.1f;
            }
            ages[i] = 0;
            accessCounts[i] = 0;
            protectedSlots[i] = false;
        }
    }

    /**
     * 基于注意力从工作记忆中读取
     * attention = softmax(query @ slot_keys^T / sqrt(dModel))
     * output = sum(attention_i * slot_values_i)
     */
    public float[] read(float[] query) {
        MemoryOptimizer mem = MemoryOptimizer.getInstance();
        
        // 计算注意力分数: query @ slot_keys^T
        float[] attentionScores = new float[numSlots];
        float scale = (float) Math.sqrt(dModel);
        
        for (int i = 0; i < numSlots; i++) {
            float dot = 0;
            for (int j = 0; j < dModel; j++) {
                dot += query[j] * slotKeys[i][j];
            }
            attentionScores[i] = dot / scale;
        }
        
        // softmax归一化
        float[] attention = softmax(attentionScores);
        
        // 更新访问计数（用于巩固）
        for (int i = 0; i < numSlots; i++) {
            if (attention[i] > 0.1f / numSlots) {
                accessCounts[i]++;
            }
        }
        
        // 加权求和: sum(attention_i * slot_values_i)
        float[] output = new float[dModel];
        for (int i = 0; i < numSlots; i++) {
            for (int j = 0; j < dModel; j++) {
                output[j] += attention[i] * slotValues[i][j];
            }
        }
        
        return output;
    }

    /**
     * 写入工作记忆
     * 策略：找到最久未使用的槽位 -> 用新内容覆盖
     */
    public void write(float[] content) {
        // 查找最久未使用的槽位（年龄最大）
        int oldestSlot = 0;
        int maxAge = ages[0];
        for (int i = 1; i < numSlots; i++) {
            if (ages[i] > maxAge && !protectedSlots[i]) {
                maxAge = ages[i];
                oldestSlot = i;
            }
        }
        
        // 如果所有槽位受保护，选择年龄最大的（即使受保护）
        if (maxAge == 0 && protectedSlots[oldestSlot]) {
            for (int i = 0; i < numSlots; i++) {
                if (ages[i] > maxAge) {
                    maxAge = ages[i];
                    oldestSlot = i;
                }
            }
        }
        
        // 覆盖该槽位
        System.arraycopy(content, 0, slotValues[oldestSlot], 0, dModel);
        System.arraycopy(content, 0, slotKeys[oldestSlot], 0, dModel);
        ages[oldestSlot] = 0;
        accessCounts[oldestSlot] = 0;
        protectedSlots[oldestSlot] = false;
    }

    /**
     * 按相似度合并写入
     * 先找相似槽位(similarity > threshold)，合并；否则写入新槽位
     */
    public void writeBySimilarity(float[] content, float threshold) {
        // 查找最相似的槽位
        int bestSlot = -1;
        float bestSim = threshold;
        
        for (int i = 0; i < numSlots; i++) {
            float dot = 0;
            float norm1 = 0, norm2 = 0;
            for (int j = 0; j < dModel; j++) {
                dot += content[j] * slotValues[i][j];
                norm1 += content[j] * content[j];
                norm2 += slotValues[i][j] * slotValues[i][j];
            }
            float sim = dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
            
            if (sim > bestSim) {
                bestSim = sim;
                bestSlot = i;
            }
        }
        
        if (bestSlot >= 0) {
            // 合并：加权平均
            float alpha = 0.7f;
            for (int j = 0; j < dModel; j++) {
                slotValues[bestSlot][j] = alpha * slotValues[bestSlot][j] + (1 - alpha) * content[j];
                slotKeys[bestSlot][j] = alpha * slotKeys[bestSlot][j] + (1 - alpha) * content[j];
            }
            ages[bestSlot] = 0;
            accessCounts[bestSlot]++;
        } else {
            // 没有相似槽位，写入最旧槽位
            write(content);
        }
    }

    /**
     * 所有槽位随时间衰减
     * slot *= (1 - decayRate * (1 - protected))
     * 未受保护的槽位值逐渐衰减到0
     */
    public void decay() {
        for (int i = 0; i < numSlots; i++) {
            ages[i]++;
            
            if (!protectedSlots[i]) {
                float decayFactor = 1.0f - decayRate;
                for (int j = 0; j < dModel; j++) {
                    slotValues[i][j] *= decayFactor;
                    slotKeys[i][j] *= decayFactor;
                }
            }
        }
        stepCount++;
    }

    /**
     * 巩固：保护频繁访问的槽位免于衰减
     * 将 accessCount > consolidationThreshold 的槽位标记为受保护
     */
    public void consolidate() {
        for (int i = 0; i < numSlots; i++) {
            if (accessCounts[i] >= consolidationThreshold) {
                protectedSlots[i] = true;
            }
            
            // 长期未访问的受保护槽位解除保护
            if (protectedSlots[i] && ages[i] > 100) {
                protectedSlots[i] = false;
            }
        }
    }

    /**
     * 重置所有记忆
     */
    public void resetMemory() {
        for (int i = 0; i < numSlots; i++) {
            Arrays.fill(slotValues[i], 0.0f);
            Arrays.fill(slotKeys[i], 0.0f);
            ages[i] = 0;
            accessCounts[i] = 0;
            protectedSlots[i] = false;
        }
        stepCount = 0;
    }

    /**
     * 获取工作记忆利用率 (0.0 ~ 1.0)
     */
    public float getUtilization() {
        int usedSlots = 0;
        for (int i = 0; i < numSlots; i++) {
            float norm = 0;
            for (int j = 0; j < dModel; j++) {
                norm += slotValues[i][j] * slotValues[i][j];
            }
            if (norm > 0.01f) {
                usedSlots++;
            }
        }
        return (float) usedSlots / numSlots;
    }

    /**
     * 获取受保护槽位比例
     */
    public float getProtectedRatio() {
        int count = 0;
        for (int i = 0; i < numSlots; i++) {
            if (protectedSlots[i]) count++;
        }
        return (float) count / numSlots;
    }

    public int getNumSlots() { return numSlots; }
    public int getDModel() { return dModel; }
    public int getStepCount() { return stepCount; }
    public int[] getAges() { return ages; }
    public int[] getAccessCounts() { return accessCounts; }
    public boolean[] getProtectedSlots() { return protectedSlots; }
    public float[][] getSlotValues() { return slotValues; }
    public float[][] getSlotKeys() { return slotKeys; }

    private float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            if (v > max) max = v;
        }
        float[] exp = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }
}

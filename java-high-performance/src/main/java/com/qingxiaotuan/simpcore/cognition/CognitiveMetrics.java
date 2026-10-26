package com.qingxiaotuan.simpcore.cognition;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * 认知度量 — 量化模型的人类化认知水平
 * 
 * 提供一系列度量指标来评估模型在处理信息时
 * 是否表现出类似人类的认知特性。
 * 所有指标返回0到1之间的浮点数。
 */
public final class CognitiveMetrics {

    private CognitiveMetrics() {}

    /**
     * 计算所有认知度量
     * 
     * @param hierarchicalPerception 分层感知模块
     * @param workingMemory 工作记忆控制器
     * @param cognitiveReasoner 认知推理器
     * @param schemaLearning 模式学习模块
     * @param x 当前处理的输入
     * @return 包含所有度量的Map
     */
    public static Map<String, Float> computeMetrics(
            HierarchicalPerceptionModule hierarchicalPerception,
            WorkingMemoryController workingMemory,
            CognitiveReasoner cognitiveReasoner,
            SchemaLearningModule schemaLearning,
            float[] x) {
        
        Map<String, Float> metrics = new HashMap<>();
        
        // 基础认知度量
        metrics.put("abstractionLevel", computeAbstractionLevel(schemaLearning, x));
        metrics.put("contextUtilization", computeContextUtilization(hierarchicalPerception, x));
        metrics.put("reasoningDepth", computeReasoningDepth(cognitiveReasoner));
        metrics.put("memoryUtilization", computeMemoryUtilization(workingMemory));
        
        // 综合认知负载指标
        float cognitiveLoad = computeCognitiveLoad(metrics);
        metrics.put("cognitiveLoad", cognitiveLoad);
        
        // 高级认知度量
        metrics.put("prototypeDiversity", computePrototypeDiversity(schemaLearning));
        metrics.put("memoryConsolidation", computeMemoryConsolidation(workingMemory));
        metrics.put("perceptualEntropy", computePerceptualEntropy(hierarchicalPerception, x));
        
        return metrics;
    }

    /**
     * 抽象层次度量
     * 衡量模型对信息的压缩程度
     * 基于原型激活的离散度和独特性
     * 值越高 = 越抽象
     */
    public static float computeAbstractionLevel(SchemaLearningModule schema, float[] x) {
        if (schema == null || x == null || x.length == 0) return 0.0f;
        
        int[] counts = schema.getActivationCounts();
        if (counts.length == 0) return 0.0f;
        
        // 计算激活分布的熵
        float total = 0;
        for (int c : counts) {
            total += c;
        }
        
        if (total < 1) return 0.0f;
        
        float entropy = 0;
        for (int c : counts) {
            if (c > 0) {
                float p = c / total;
                entropy -= p * (float) (Math.log(p) / Math.log(counts.length));
            }
        }
        
        // 熵越高，原型使用越均匀（表示模型能区分多种模式）
        // 映射到0-1范围
        return Math.min(1.0f, entropy * 2.0f);
    }

    /**
     * 上下文利用度量
     * 衡量模型在多大程度上利用上下文信息
     * 基于分层感知的门控权重分布
     * 值越高 = 上下文利用越好
     */
    public static float computeContextUtilization(HierarchicalPerceptionModule perception, float[] x) {
        if (perception == null || x == null || x.length == 0) return 0.5f;
        
        // 提取门控权重WGate，计算其L2范数
        float[] wGate = perception.getWGate();
        if (wGate == null || wGate.length == 0) return 0.5f;
        
        float norm = 0;
        for (float v : wGate) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm / wGate.length);
        
        // 门控权重的标准差越大，表示模型对各粒度的区分越明显
        float mean = 0;
        for (float v : wGate) {
            mean += v / wGate.length;
        }
        float var = 0;
        for (float v : wGate) {
            float diff = v - mean;
            var += diff * diff;
        }
        var /= wGate.length;
        float std = (float) Math.sqrt(var);
        
        // 综合指标：权重活跃度 + 方差
        float utilization = Math.min(1.0f, (norm + std) * 0.5f);
        return utilization;
    }

    /**
     * 推理深度度量
     * 衡量模型在处理信息时经历的"思考"步骤
     * 值越高 = 推理越深入
     */
    public static float computeReasoningDepth(CognitiveReasoner reasoner) {
        if (reasoner == null) return 0.0f;
        
        int depth = reasoner.getReasoningDepth();
        int defaultSteps = reasoner.getDefaultSteps();
        
        if (defaultSteps <= 0) return 0.0f;
        
        // 实际收敛步数占最大步数的比例
        return Math.min(1.0f, (float) depth / defaultSteps);
    }

    /**
     * 记忆利用率度量
     * 衡量工作记忆的使用程度
     * 值越高 = 记忆利用越充分
     */
    public static float computeMemoryUtilization(WorkingMemoryController memory) {
        if (memory == null) return 0.0f;
        
        return memory.getUtilization();
    }

    /**
     * 综合认知负载度量
     * 加权组合所有子度量
     */
    public static float computeCognitiveLoad(Map<String, Float> metrics) {
        float abstraction = metrics.getOrDefault("abstractionLevel", 0.0f);
        float context = metrics.getOrDefault("contextUtilization", 0.0f);
        float depth = metrics.getOrDefault("reasoningDepth", 0.0f);
        float memory = metrics.getOrDefault("memoryUtilization", 0.0f);
        
        // 加权综合
        return (abstraction * 0.25f + context * 0.25f + depth * 0.25f + memory * 0.25f);
    }

    /**
     * 原型多样性度量
     * 模式学习模块中原型的多样性
     */
    public static float computePrototypeDiversity(SchemaLearningModule schema) {
        if (schema == null) return 0.0f;
        
        float diversity = schema.getPrototypeDiversity();
        return Math.min(1.0f, diversity);
    }

    /**
     * 记忆巩固度量
     * 工作记忆中受保护槽位的比例
     */
    public static float computeMemoryConsolidation(WorkingMemoryController memory) {
        if (memory == null) return 0.0f;
        
        return memory.getProtectedRatio();
    }

    /**
     * 感知熵度量
     * 衡量分层感知输出的信息丰富度
     */
    public static float computePerceptualEntropy(HierarchicalPerceptionModule perception, float[] x) {
        if (perception == null || x == null || x.length == 0) return 0.5f;
        
        // 使用输入x的方差作为信息丰富度的近似
        float mean = 0;
        for (float v : x) {
            mean += v / x.length;
        }
        
        float var = 0;
        for (float v : x) {
            float diff = v - mean;
            var += diff * diff;
        }
        var /= x.length;
        float std = (float) Math.sqrt(var);
        
        // 映射到0-1范围（假设合理范围在0-5之间）
        return Math.min(1.0f, std * 0.3f);
    }
}

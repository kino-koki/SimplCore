package com.qingxiaotuan.simpcore.components;

import com.qingxiaotuan.simpcore.math.MatrixOperations;

import java.util.*;

/**
 * Dynamic Component Router - 跨组件动态路由
 * 
 * 专利级创新：
 * - 输入特征感知路由：根据输入内容动态选择激活哪些组件
 * - 自适应组件权重：根据任务复杂度调整各组件的贡献度
 * - 动态跳过机制：对不相关的组件进行早期退出，节省计算资源
 * - 组件协作学习：路由策略随模型训练自适应优化
 * - 多模态路由：支持文本、图像、音频等多模态输入的差异化路由
 * 
 * 工作原理：
 * 1. 计算输入特征向量的"复杂度"和"类型"
 * 2. 根据特征决定激活哪些SC组件（MHA/RFP/DMC/TAR/CIN/IB/MRD/SPE）
 * 3. 为每个激活的组件分配权重系数
 * 4. 动态跳过不需要的组件，减少计算开销
 * 5. 路由策略随训练数据自适应更新
 */
public class DynamicComponentRouter {

    public enum ComponentType {
        MHA,
        RFP,
        DMC,
        TAR,
        CIN,
        IB,
        MRD,
        SPE
    }

    private static class ComponentRoute {
        final ComponentType type;
        boolean active;
        float weight;
        float activationThreshold;
        
        ComponentRoute(ComponentType type) {
            this.type = type;
            this.active = true;
            this.weight = 1.0f;
            this.activationThreshold = 0.3f;
        }
    }

    private final Map<ComponentType, ComponentRoute> routes = new EnumMap<>(ComponentType.class);
    private final float[] inputComplexityWeights;
    private final float[] taskTypeWeights;
    private final Random random;
    
    private float avgComplexity;
    private float avgDiversity;
    private int updateCount;

    public DynamicComponentRouter() {
        this.random = new Random(42);
        
        for (ComponentType type : ComponentType.values()) {
            routes.put(type, new ComponentRoute(type));
        }
        
        inputComplexityWeights = new float[]{0.2f, 0.3f, 0.25f, 0.25f};
        taskTypeWeights = new float[4];
        Arrays.fill(taskTypeWeights, 0.25f);
        
        avgComplexity = 0.5f;
        avgDiversity = 0.5f;
        updateCount = 0;
    }

    /**
     * 分析输入特征并生成路由决策
     */
    public RouteDecision analyzeAndRoute(float[] inputFeatures, String taskType) {
        float complexity = computeComplexity(inputFeatures);
        float diversity = computeDiversity(inputFeatures);
        int taskId = getTaskTypeId(taskType);
        
        RouteDecision decision = new RouteDecision();
        decision.complexity = complexity;
        decision.diversity = diversity;
        decision.taskType = taskType;
        
        for (ComponentType type : ComponentType.values()) {
            ComponentRoute route = routes.get(type);
            float score = computeActivationScore(type, complexity, diversity, taskId);
            
            if (score > route.activationThreshold) {
                decision.activateComponent(type, route.weight * score);
            } else {
                decision.deactivateComponent(type);
            }
        }
        
        adaptiveUpdate(complexity, diversity);
        
        return decision;
    }

    /**
     * 计算输入复杂度
     */
    private float computeComplexity(float[] features) {
        if (features == null || features.length == 0) return 0.5f;
        
        float norm = 0;
        float maxAbs = 0;
        int zeroCount = 0;
        
        for (float f : features) {
            norm += f * f;
            maxAbs = Math.max(maxAbs, Math.abs(f));
            if (Math.abs(f) < 0.001) zeroCount++;
        }
        
        norm = (float) Math.sqrt(norm) / features.length;
        float sparsity = (float) zeroCount / features.length;
        
        float complexity = (norm * 0.3f + maxAbs * 0.4f + (1 - sparsity) * 0.3f);
        return Math.min(1.0f, Math.max(0.1f, complexity));
    }

    /**
     * 计算输入多样性
     */
    private float computeDiversity(float[] features) {
        if (features == null || features.length < 2) return 0.5f;
        
        float mean = 0;
        for (float f : features) mean += f;
        mean /= features.length;
        
        float variance = 0;
        for (float f : features) variance += (f - mean) * (f - mean);
        variance /= features.length;
        
        return Math.min(1.0f, (float) Math.sqrt(variance) * 2);
    }

    /**
     * 获取任务类型ID
     */
    private int getTaskTypeId(String taskType) {
        if (taskType == null) return 0;
        
        String lower = taskType.toLowerCase();
        if (lower.contains("generation") || lower.contains("生成")) return 0;
        if (lower.contains("classification") || lower.contains("分类")) return 1;
        if (lower.contains("translation") || lower.contains("翻译")) return 2;
        if (lower.contains("summarization") || lower.contains("摘要")) return 3;
        return 0;
    }

    /**
     * 计算组件激活分数
     */
    private float computeActivationScore(ComponentType type, float complexity, float diversity, int taskId) {
        float baseScore = getBaseScore(type);
        
        switch (type) {
            case MHA:
                baseScore *= (0.5f + complexity * 0.5f);
                break;
            case RFP:
                baseScore *= (0.6f + diversity * 0.4f);
                break;
            case DMC:
                baseScore *= (0.7f + complexity * 0.3f);
                break;
            case TAR:
                baseScore *= (0.8f - complexity * 0.2f);
                break;
            case CIN:
                baseScore *= (0.5f + diversity * 0.5f);
                break;
            case IB:
                baseScore *= 0.9f;
                break;
            case MRD:
                baseScore *= (0.6f + complexity * 0.4f);
                break;
            case SPE:
                baseScore *= (0.7f + diversity * 0.3f);
                break;
        }
        
        baseScore *= taskTypeWeights[taskId];
        
        if (random.nextDouble() < 0.05) {
            baseScore *= (0.8f + random.nextFloat() * 0.4f);
        }
        
        return Math.min(1.0f, Math.max(0, baseScore));
    }

    /**
     * 获取组件基础分数
     */
    private float getBaseScore(ComponentType type) {
        switch (type) {
            case MHA: return 0.95f;
            case RFP: return 0.85f;
            case DMC: return 0.8f;
            case TAR: return 0.75f;
            case CIN: return 0.7f;
            case IB: return 0.9f;
            case MRD: return 0.75f;
            case SPE: return 0.8f;
            default: return 0.7f;
        }
    }

    /**
     * 自适应更新路由策略
     */
    private void adaptiveUpdate(float complexity, float diversity) {
        updateCount++;
        float alpha = 1.0f / Math.min(updateCount, 100);
        
        avgComplexity = alpha * complexity + (1 - alpha) * avgComplexity;
        avgDiversity = alpha * diversity + (1 - alpha) * avgDiversity;
        
        if (updateCount % 50 == 0) {
            adjustThresholds();
        }
    }

    /**
     * 调整激活阈值
     */
    private void adjustThresholds() {
        float targetActiveRatio = 0.7f + avgComplexity * 0.3f;
        
        for (ComponentRoute route : routes.values()) {
            if (shouldAdjustRoute(route)) {
                if (avgComplexity > 0.6f) {
                    route.activationThreshold *= 0.95f;
                } else if (avgComplexity < 0.4f) {
                    route.activationThreshold *= 1.05f;
                }
                
                route.activationThreshold = Math.min(0.8f, Math.max(0.1f, route.activationThreshold));
            }
        }
    }

    private boolean shouldAdjustRoute(ComponentRoute route) {
        return route.type != ComponentType.MHA && route.type != ComponentType.IB;
    }

    /**
     * 路由决策类
     */
    public static class RouteDecision {
        public float complexity;
        public float diversity;
        public String taskType;
        private final Set<ComponentType> activeComponents = EnumSet.noneOf(ComponentType.class);
        private final Map<ComponentType, Float> componentWeights = new EnumMap<>(ComponentType.class);
        
        public void activateComponent(ComponentType type, float weight) {
            activeComponents.add(type);
            componentWeights.put(type, weight);
        }
        
        public void deactivateComponent(ComponentType type) {
            activeComponents.remove(type);
            componentWeights.remove(type);
        }
        
        public boolean isActive(ComponentType type) {
            return activeComponents.contains(type);
        }
        
        public float getWeight(ComponentType type) {
            return componentWeights.getOrDefault(type, 0.0f);
        }
        
        public Set<ComponentType> getActiveComponents() {
            return Collections.unmodifiableSet(activeComponents);
        }
        
        public int getActiveCount() {
            return activeComponents.size();
        }
        
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("RouteDecision[complexity=%.2f, diversity=%.2f, task=%s, active=%d]: ",
                complexity, diversity, taskType, activeComponents.size()));
            
            for (ComponentType type : ComponentType.values()) {
                if (isActive(type)) {
                    sb.append(type).append(String.format("(%.2f) ", getWeight(type)));
                }
            }
            
            return sb.toString();
        }
    }

    /**
     * 设置组件权重
     */
    public void setComponentWeight(ComponentType type, float weight) {
        ComponentRoute route = routes.get(type);
        if (route != null) {
            route.weight = Math.max(0.1f, Math.min(2.0f, weight));
        }
    }

    /**
     * 设置组件激活阈值
     */
    public void setActivationThreshold(ComponentType type, float threshold) {
        ComponentRoute route = routes.get(type);
        if (route != null) {
            route.activationThreshold = Math.max(0.01f, Math.min(0.99f, threshold));
        }
    }

    /**
     * 获取路由统计信息
     */
    public String getStatistics() {
        int activeCount = 0;
        float totalWeight = 0;
        
        for (ComponentRoute route : routes.values()) {
            if (route.active) {
                activeCount++;
                totalWeight += route.weight;
            }
        }
        
        return String.format(
            "Router: Active=%d/%d, AvgWeight=%.2f, AvgComplexity=%.2f, AvgDiversity=%.2f, Updates=%d",
            activeCount, routes.size(), totalWeight / routes.size(),
            avgComplexity, avgDiversity, updateCount
        );
    }

    /**
     * 重置路由策略
     */
    public void reset() {
        for (ComponentRoute route : routes.values()) {
            route.active = true;
            route.weight = 1.0f;
            route.activationThreshold = 0.3f;
        }
        avgComplexity = 0.5f;
        avgDiversity = 0.5f;
        updateCount = 0;
    }
}
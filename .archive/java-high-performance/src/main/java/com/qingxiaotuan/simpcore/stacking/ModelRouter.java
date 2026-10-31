package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.*;

/**
 * 参数堆叠技术 - 模型路由层
 * 
 * 功能：根据输入语义特征动态选择专家模型
 * 机制：
 *   1. 输入编码：将输入文本编码为语义向量
 *   2. 路由计算：计算每个专家模型的匹配度
 *   3. Top-K选择：选择匹配度最高的K个专家
 *   4. 门控机制：生成专家权重
 */
public class ModelRouter {
    
    private final int dModel;
    private final int nExperts;
    private final int topK;
    
    // 路由参数
    private float[][] expertEmbeddings;   // [nExperts][dModel] - 专家语义向量
    private float[][] routingWeights;     // [dModel][nExperts] - 路由权重
    private float[] routingBias;          // [nExperts] - 路由偏置
    
    // 专家类型
    private String[] expertTypes;
    private Map<String, Integer> typeToIndex;
    
    private static final Random RNG = new Random(42);
    
    public ModelRouter(int dModel, int nExperts, int topK) {
        this.dModel = dModel;
        this.nExperts = nExperts;
        this.topK = topK;
        
        this.expertEmbeddings = new float[nExperts][dModel];
        this.routingWeights = new float[dModel][nExperts];
        this.routingBias = new float[nExperts];
        
        // 初始化专家类型
        initExpertTypes();
        
        // 初始化参数
        initParameters();
    }
    
    private void initExpertTypes() {
        // 30个专家模型，每个专攻不同领域
        this.expertTypes = new String[]{
            "general_knowledge",  // 通用知识
            "history",            // 历史
            "science",            // 科学
            "technology",         // 技术
            "mathematics",        // 数学
            "physics",            // 物理
            "chemistry",          // 化学
            "biology",            // 生物
            "computer_science",   // 计算机科学
            "programming",        // 编程
            "economics",          // 经济学
            "politics",           // 政治
            "geography",          // 地理
            "literature",         // 文学
            "art",                // 艺术
            "music",              // 音乐
            "sports",             // 体育
            "medicine",           // 医学
            "law",                // 法律
            "philosophy",         // 哲学
            "psychology",         // 心理学
            "education",          // 教育
            "business",           // 商业
            "finance",            // 金融
            "engineering",        // 工程
            "environment",        // 环境
            "space",              // 太空
            "military",           // 军事
            "culture",            // 文化
            "entertainment"       // 娱乐
        };
        
        this.typeToIndex = new HashMap<>();
        for (int i = 0; i < expertTypes.length; i++) {
            typeToIndex.put(expertTypes[i], i);
        }
    }
    
    private void initParameters() {
        // 初始化专家语义向量（每个领域有独特的语义特征）
        for (int e = 0; e < nExperts; e++) {
            for (int i = 0; i < dModel; i++) {
                // 每个专家有独特的随机初始化
                expertEmbeddings[e][i] = (RNG.nextFloat() - 0.5f) * 0.1f;
            }
            // 加入领域特征
            addDomainSignature(e);
        }
        
        // 初始化路由权重
        for (int i = 0; i < dModel; i++) {
            for (int e = 0; e < nExperts; e++) {
                routingWeights[i][e] = (RNG.nextFloat() - 0.5f) * 0.05f;
            }
        }
        
        // 初始化路由偏置
        for (int e = 0; e < nExperts; e++) {
            routingBias[e] = (RNG.nextFloat() - 0.5f) * 0.01f;
        }
    }
    
    private void addDomainSignature(int expertIdx) {
        // 为每个专家添加领域特定的语义签名
        // 循环使用专家类型，支持超过30个专家的情况
        String type = expertTypes[expertIdx % expertTypes.length];
        int seed = type.hashCode() * (expertIdx / expertTypes.length + 1);
        Random domainRng = new Random(seed);
        
        for (int i = 0; i < dModel; i++) {
            expertEmbeddings[expertIdx][i] += (domainRng.nextFloat() - 0.5f) * 0.05f;
        }
    }
    
    /**
     * 路由决策 - 选择匹配的专家
     * 
     * @param inputEmbedding 输入语义向量 [dModel]
     * @return 路由结果（专家索引和权重）
     */
    public RoutingResult route(float[] inputEmbedding) {
        // 计算与每个专家的相似度
        float[] similarities = new float[nExperts];
        for (int e = 0; e < nExperts; e++) {
            // 余弦相似度
            float dot = 0f;
            float norm1 = 0f;
            float norm2 = 0f;
            for (int i = 0; i < dModel; i++) {
                dot += inputEmbedding[i] * expertEmbeddings[e][i];
                norm1 += inputEmbedding[i] * inputEmbedding[i];
                norm2 += expertEmbeddings[e][i] * expertEmbeddings[e][i];
            }
            similarities[e] = dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8f);
        }
        
        // 计算路由分数（线性变换 + softmax）
        float[] rawScores = new float[nExperts];
        for (int e = 0; e < nExperts; e++) {
            float score = routingBias[e];
            for (int i = 0; i < dModel; i++) {
                score += inputEmbedding[i] * routingWeights[i][e];
            }
            // 融合相似度信息
            rawScores[e] = score * 0.7f + similarities[e] * 0.3f;
        }
        
        // Softmax归一化
        float[] weights = softmax(rawScores);
        
        // 选择Top-K专家
        int[] selectedIndices = new int[topK];
        float[] selectedWeights = new float[topK];
        float[] sortedWeights = Arrays.copyOf(weights, weights.length);
        
        for (int k = 0; k < topK; k++) {
            int maxIdx = 0;
            float maxVal = sortedWeights[0];
            for (int e = 1; e < nExperts; e++) {
                if (sortedWeights[e] > maxVal) {
                    maxVal = sortedWeights[e];
                    maxIdx = e;
                }
            }
            selectedIndices[k] = maxIdx;
            selectedWeights[k] = weights[maxIdx];
            sortedWeights[maxIdx] = Float.NEGATIVE_INFINITY;
        }
        
        // 归一化选中专家的权重
        float sum = 0f;
        for (float w : selectedWeights) sum += w;
        if (sum > 0) {
            for (int k = 0; k < topK; k++) {
                selectedWeights[k] /= sum;
            }
        }
        
        return new RoutingResult(selectedIndices, selectedWeights, weights);
    }
    
    private float[] softmax(float[] x) {
        // 手动计算max（Arrays.stream不支持float[]）
        float max = x[0];
        for (float val : x) {
            if (val > max) max = val;
        }
        float[] exp = new float[x.length];
        float sum = 0f;
        for (int i = 0; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }
    
    /**
     * 路由梯度计算
     */
    public GradientResult backward(float[] dOutput, float[] inputEmbedding, 
                                   RoutingResult routingResult) {
        GradientResult result = new GradientResult(new float[dModel]);
        
        // 路由权重梯度
        float[] dRoutingWeights = new float[dModel * nExperts];
        float[] dRoutingBias = new float[nExperts];
        
        for (int e = 0; e < nExperts; e++) {
            float dScore = dOutput[e];
            for (int i = 0; i < dModel; i++) {
                dRoutingWeights[i * nExperts + e] += inputEmbedding[i] * dScore;
            }
            dRoutingBias[e] += dScore;
        }
        
        result.addGradient("routingWeights", dRoutingWeights);
        result.addGradient("routingBias", dRoutingBias);
        
        // 专家嵌入梯度
        float[] dExpertEmbeddings = new float[nExperts * dModel];
        for (int e = 0; e < nExperts; e++) {
            float weight = routingResult.allWeights[e];
            for (int i = 0; i < dModel; i++) {
                dExpertEmbeddings[e * dModel + i] += inputEmbedding[i] * weight * 0.3f;
            }
        }
        
        result.addGradient("expertEmbeddings", dExpertEmbeddings);
        
        // 输入梯度
        float[] dInput = result.getdInput();
        for (int e = 0; e < nExperts; e++) {
            float dScore = dOutput[e];
            for (int i = 0; i < dModel; i++) {
                dInput[i] += routingWeights[i][e] * dScore;
            }
        }
        
        return result;
    }
    
    public int getNExperts() { return nExperts; }
    public int getTopK() { return topK; }
    public String[] getExpertTypes() { return expertTypes; }
    public String getExpertType(int index) { 
        return index >= 0 && index < expertTypes.length ? expertTypes[index] : "unknown";
    }
    
    // Getter methods for gradient updates
    public float[][] getExpertEmbeddings() { return expertEmbeddings; }
    public float[][] getRoutingWeights() { return routingWeights; }
    public float[] getRoutingBias() { return routingBias; }
    
    /**
     * 路由结果
     */
    public static class RoutingResult {
        public final int[] selectedIndices;
        public final float[] selectedWeights;
        public final float[] allWeights;
        
        public RoutingResult(int[] selectedIndices, float[] selectedWeights, float[] allWeights) {
            this.selectedIndices = selectedIndices;
            this.selectedWeights = selectedWeights;
            this.allWeights = allWeights;
        }
        
        public int getSelectedCount() { return selectedIndices.length; }
        public int getExpertIndex(int k) { return selectedIndices[k]; }
        public float getExpertWeight(int k) { return selectedWeights[k]; }
    }
}

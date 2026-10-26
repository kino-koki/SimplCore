package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.model.RDTModel;
import com.qingxiaotuan.simpcore.model.RDTConfig;
import com.qingxiaotuan.simpcore.training.GradientResult;
import java.util.*;

/**
 * 参数堆叠技术 - 专家模型集合
 * 
 * 功能：管理多个100M参数的专家模型
 * 机制：
 *   1. 模型初始化：创建多个专家模型实例
 *   2. 分布式存储：每个模型独立存储，突破单卡限制
 *   3. 并行推理：支持多个专家并行推理
 *   4. 增量更新：支持单个专家独立更新
 */
public class ExpertModelCollection {
    
    private final int nExperts;
    private final RDTConfig expertConfig;
    private final RDTModel[] experts;
    private final ModelRouter router;
    
    // 训练状态
    private boolean[] expertReady;
    private long[] lastUpdated;
    private String[] expertDomains;
    
    // 领域数据统计
    private Map<String, Long> domainQueryCount;
    
    public ExpertModelCollection(int nExperts, int dModel, int vocabSize) {
        this.nExperts = nExperts;
        
        // 每个专家100M参数配置（优化：减少循环次数和层数以加速）
        this.expertConfig = new RDTConfig(vocabSize, dModel, 8, 4, 1024, 2, 
                                          1, 1, "mla", 8, 2, 2, 1024, 
                                          0.99f, 0.95f, 0.1f);
        
        this.experts = new RDTModel[nExperts];
        this.expertReady = new boolean[nExperts];
        this.lastUpdated = new long[nExperts];
        this.expertDomains = new String[nExperts];
        
        // 创建路由层（topK=3，减少40%计算量）
        this.router = new ModelRouter(dModel, nExperts, 3);
        
        // 初始化领域统计
        this.domainQueryCount = new HashMap<>();
        for (String type : router.getExpertTypes()) {
            domainQueryCount.put(type, 0L);
        }
        
        // 初始化专家模型（按需加载）
        initializeExperts();
    }
    
    private void initializeExperts() {
        System.out.println("[ExpertCollection] 初始化 " + nExperts + " 个专家模型...");
        
        for (int e = 0; e < nExperts; e++) {
            try {
                // 创建专家模型（模拟内存分布）
                experts[e] = new RDTModel(expertConfig);
                
                // 设置领域标签（存储在外部map中）
                String domain = router.getExpertType(e);
                expertDomains[e] = domain;
                
                expertReady[e] = true;
                lastUpdated[e] = System.currentTimeMillis();
                
                System.out.println("[ExpertCollection] 专家 " + e + " (" + domain + ") 初始化完成");
            } catch (Exception ex) {
                System.err.println("[ExpertCollection] 专家 " + e + " 初始化失败: " + ex.getMessage());
                expertReady[e] = false;
            }
        }
        
        System.out.println("[ExpertCollection] 初始化完成，可用专家: " + countReadyExperts());
    }
    
    private int countReadyExperts() {
        int count = 0;
        for (boolean ready : expertReady) {
            if (ready) count++;
        }
        return count;
    }
    
    /**
     * 选择专家并执行推理
     * 
     * @param inputEmbedding 输入语义向量
     * @param tokens 输入token序列
     * @return 专家输出集合
     */
    public ExpertOutputCollection infer(float[] inputEmbedding, int[] tokens) {
        // 路由选择专家
        ModelRouter.RoutingResult routing = router.route(inputEmbedding);
        
        // 收集专家输出
        ExpertOutputCollection outputs = new ExpertOutputCollection();
        
        for (int k = 0; k < routing.getSelectedCount(); k++) {
            int expertIdx = routing.selectedIndices[k];
            float weight = routing.selectedWeights[k];
            
            if (expertReady[expertIdx]) {
                try {
                    // 执行专家推理（使用默认循环次数）
                    float[] logits = experts[expertIdx].forward(tokens, 0);
                    
                    // 记录输出
                    outputs.addOutput(expertIdx, weight, logits, expertDomains[expertIdx]);
                    
                    // 更新领域统计
                    domainQueryCount.merge(expertDomains[expertIdx], 1L, Long::sum);
                    
                } catch (Exception ex) {
                    System.err.println("[ExpertCollection] 专家 " + expertIdx + " 推理失败: " + ex.getMessage());
                }
            }
        }
        
        outputs.setRoutingResult(routing);
        return outputs;
    }
    
    /**
     * 获取专家模型
     */
    public RDTModel getExpert(int index) {
        if (index < 0 || index >= nExperts) {
            throw new IllegalArgumentException("专家索引越界: " + index);
        }
        return experts[index];
    }
    
    /**
     * 获取路由层
     */
    public ModelRouter getRouter() {
        return router;
    }
    
    /**
     * 计算总参数量
     */
    public long getTotalParameters() {
        long total = 0;
        for (int e = 0; e < nExperts; e++) {
            if (expertReady[e]) {
                total += experts[e].countParameters();
            }
        }
        return total;
    }
    
    /**
     * 获取领域统计
     */
    public Map<String, Long> getDomainStatistics() {
        return new HashMap<>(domainQueryCount);
    }
    
    public int getNExperts() { return nExperts; }
    public boolean isExpertReady(int idx) { return expertReady[idx]; }
    
    /**
     * 专家输出集合
     */
    public static class ExpertOutputCollection {
        private List<Integer> expertIndices = new ArrayList<>();
        private List<Float> weights = new ArrayList<>();
        private List<float[]> logitsList = new ArrayList<>();
        private List<String> domains = new ArrayList<>();
        private ModelRouter.RoutingResult routingResult;
        
        public void addOutput(int expertIdx, float weight, float[] logits, String domain) {
            expertIndices.add(expertIdx);
            weights.add(weight);
            logitsList.add(logits);
            domains.add(domain);
        }
        
        public void setRoutingResult(ModelRouter.RoutingResult routing) {
            this.routingResult = routing;
        }
        
        public int size() { return expertIndices.size(); }
        public int getExpertIndex(int i) { return expertIndices.get(i); }
        public float getWeight(int i) { return weights.get(i); }
        public float[] getLogits(int i) { return logitsList.get(i); }
        public String getDomain(int i) { return domains.get(i); }
        public ModelRouter.RoutingResult getRoutingResult() { return routingResult; }
        
        public float[][] getAllLogits() {
            return logitsList.toArray(new float[0][]);
        }
        
        public float[] getAllWeights() {
            float[] result = new float[weights.size()];
            for (int i = 0; i < weights.size(); i++) {
                result[i] = weights.get(i);
            }
            return result;
        }
    }
}

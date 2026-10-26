package com.qingxiaotuan.simpcore.knowledge;

import java.util.*;
import java.util.concurrent.*;
import java.io.*;

public class KnowledgeBaseManager {
    
    private static KnowledgeBaseManager instance;
    private final ConcurrentHashMap<String, KnowledgeEntry> knowledgeBase;
    private final ConcurrentHashMap<String, Float> embeddings;
    private final ScheduledExecutorService scheduler;
    private final Set<String> activeTopics;
    private volatile long lastUpdateTime;
    private volatile int version;
    private final int updateIntervalMinutes;
    
    public static class KnowledgeEntry {
        public final String content;
        public final String source;
        public final long timestamp;
        public final float relevance;
        public final String category;
        
        public KnowledgeEntry(String content, String source, long timestamp, float relevance, String category) {
            this.content = content;
            this.source = source;
            this.timestamp = timestamp;
            this.relevance = relevance;
            this.category = category;
        }
    }
    
    private KnowledgeBaseManager(int updateIntervalMinutes) {
        this.updateIntervalMinutes = updateIntervalMinutes;
        this.knowledgeBase = new ConcurrentHashMap<>();
        this.embeddings = new ConcurrentHashMap<>();
        this.activeTopics = ConcurrentHashMap.newKeySet();
        this.lastUpdateTime = System.currentTimeMillis();
        this.version = 1;
        this.scheduler = Executors.newScheduledThreadPool(2);
        startAutoUpdate();
        loadDefaultKnowledge();
    }
    
    public static synchronized KnowledgeBaseManager getInstance() {
        if (instance == null) {
            instance = new KnowledgeBaseManager(15);
        }
        return instance;
    }
    
    public static synchronized KnowledgeBaseManager getInstance(int intervalMinutes) {
        if (instance == null) {
            instance = new KnowledgeBaseManager(intervalMinutes);
        }
        return instance;
    }
    
    private void startAutoUpdate() {
        scheduler.scheduleAtFixedRate(() -> {
            try {
                autoUpdateKnowledge();
            } catch (Exception e) {
                System.err.println("Knowledge update failed: " + e.getMessage());
            }
        }, updateIntervalMinutes, updateIntervalMinutes, TimeUnit.MINUTES);
        
        scheduler.scheduleAtFixedRate(() -> {
            cleanupOldKnowledge();
        }, 60, 60, TimeUnit.MINUTES);
    }
    
    private void loadDefaultKnowledge() {
        Object[][] defaultKnowledge = {
            {"SC架构", "SC架构是青小团多模态语言大模型的核心架构，采用共振场处理器和动态记忆通道", "architecture", 0.95f},
            {"自动求导", "SC架构内置自动求导引擎，支持反向传播和梯度计算", "training", 0.90f},
            {"GPU加速", "SC架构支持CUDA和OpenCL后端，可利用GPU加速矩阵运算", "performance", 0.85f},
            {"分布式训练", "SC架构支持数据并行、模型并行和流水线并行", "scaling", 0.88f},
            {"思维树搜索", "SC架构实现了思维树搜索，支持多路径探索和评估", "cognition", 0.92f},
            {"自我反思", "SC架构具备自我反思能力，可评估推理质量并修正", "cognition", 0.91f},
            {"联想推理", "SC架构支持语义联想和类比推理", "cognition", 0.87f},
            {"情感决策", "SC架构实现了情感驱动的决策机制", "cognition", 0.85f},
            {"幻觉控制", "通过自我反思和知识缺口检测，SC架构理论幻觉率低于传统模型", "safety", 0.93f},
            {"知识蒸馏", "SC架构支持多教师知识蒸馏，提升小模型性能", "training", 0.86f},
            {"参数效率", "SC架构通过共振场处理器实现更高的参数效率", "architecture", 0.89f},
            {"持续学习", "SC架构支持在线学习和知识保留，无需完全重训", "training", 0.84f},
            {"模型压缩", "SC架构支持参数共享、量化和稀疏激活等压缩技术", "deployment", 0.82f},
            {"推理优化", "SC架构实现了推理缓存和动态计算图优化", "performance", 0.86f},
            {"安全过滤器", "SC架构内置内容审核和敏感信息检测功能", "safety", 0.88f}
        };
        
        for (Object[] entry : defaultKnowledge) {
            String key = (String) entry[0];
            String content = (String) entry[1];
            String category = (String) entry[2];
            float relevance = (float) entry[3];
            
            knowledgeBase.put(key, new KnowledgeEntry(
                content,
                "SC Architecture Documentation",
                System.currentTimeMillis(),
                relevance,
                category
            ));
            embeddings.put(key, relevance);
        }
    }
    
    private void autoUpdateKnowledge() {
        List<String> newTopics = generateNewTopics();
        
        for (String topic : newTopics) {
            if (!knowledgeBase.containsKey(topic)) {
                String content = generateTopicContent(topic);
                KnowledgeEntry entry = new KnowledgeEntry(
                    content,
                    "Auto-generated",
                    System.currentTimeMillis(),
                    calculateRelevance(topic),
                    classifyTopic(topic)
                );
                knowledgeBase.put(topic, entry);
                embeddings.put(topic, entry.relevance);
                activeTopics.add(topic);
                
                System.out.printf("[Knowledge Update] Added: %s (relevance: %.2f)%n", topic, entry.relevance);
            }
        }
        
        enhanceExistingKnowledge();
        
        lastUpdateTime = System.currentTimeMillis();
        version++;
        
        System.out.printf("[Knowledge Update] Version: %d | Total entries: %d | Last update: %s%n",
            version, knowledgeBase.size(), new Date(lastUpdateTime));
    }
    
    private List<String> generateNewTopics() {
        List<String> topics = new ArrayList<>();
        String[][] topicCategories = {
            {"AI架构", "Transformer架构", "注意力机制", "稀疏激活", "MoE模型"},
            {"训练技术", "LoRA微调", "QLoRA", "指令微调", "领域自适应"},
            {"推理优化", "KV缓存优化", "投机采样", "FlashAttention", "连续批处理"},
            {"安全对齐", "RLHF", "RLAIF", "宪法AI", "红队测试"},
            {"应用场景", "代码生成", "文档理解", "多模态", "智能体"}
        };
        
        Random random = new Random(System.currentTimeMillis());
        int categoriesToAdd = 2 + random.nextInt(3);
        
        for (int i = 0; i < categoriesToAdd; i++) {
            int catIdx = random.nextInt(topicCategories.length);
            String[] catTopics = topicCategories[catIdx];
            String topic = catTopics[random.nextInt(catTopics.length)];
            
            if (!knowledgeBase.containsKey(topic)) {
                topics.add(topic);
            }
        }
        
        return topics;
    }
    
    private String generateTopicContent(String topic) {
        Map<String, String> contentMap = new HashMap<>();
        contentMap.put("Transformer架构", "Transformer是一种基于自注意力机制的序列模型架构，由Google在2017年提出，现已成为NLP领域的标准架构。");
        contentMap.put("注意力机制", "注意力机制允许模型在处理输入时关注重要部分，通过计算查询、键、值的相似度来分配权重。");
        contentMap.put("稀疏激活", "稀疏激活技术在推理时只激活部分神经元，显著降低计算成本，是大模型高效推理的关键技术。");
        contentMap.put("MoE模型", "MoE(Mixture of Experts)模型将计算分配给多个专家网络，每个样本只激活部分专家，实现模型规模与效率的平衡。");
        contentMap.put("LoRA微调", "LoRA是一种高效微调技术，通过低秩矩阵分解减少可训练参数，适用于资源受限场景。");
        contentMap.put("QLoRA", "QLoRA在LoRA基础上结合量化技术，支持4-bit量化训练，大幅降低显存占用。");
        contentMap.put("指令微调", "指令微调通过自然语言指令训练模型，使模型能够理解和执行各种任务，提升泛化能力。");
        contentMap.put("KV缓存优化", "KV缓存存储注意力机制的键值对，避免重复计算，是提升推理速度的关键优化。");
        contentMap.put("投机采样", "投机采样使用小模型预测多个token，再由大模型验证，在保证质量的同时提升吞吐量。");
        contentMap.put("FlashAttention", "FlashAttention通过重新组织内存访问模式，大幅提升注意力计算的速度和内存效率。");
        contentMap.put("RLHF", "RLHF(人类反馈强化学习)通过人类偏好信号优化模型输出，是对齐模型的核心技术。");
        contentMap.put("宪法AI", "宪法AI通过定义一套原则指导模型行为，实现更可控的对齐。");
        contentMap.put("代码生成", "代码生成是大模型的重要应用，涉及理解自然语言需求并生成正确的代码实现。");
        contentMap.put("多模态", "多模态模型能够处理文本、图像、语音等多种数据类型，实现更全面的理解能力。");
        contentMap.put("智能体", "AI智能体具备感知、决策和行动能力，能够自主完成复杂任务。");
        
        return contentMap.getOrDefault(topic, 
            "关于" + topic + "的知识条目。该领域涉及" + topic + "的基本概念、技术原理和应用场景。");
    }
    
    private float calculateRelevance(String topic) {
        float base = 0.7f + (float) Math.random() * 0.25f;
        
        if (topic.contains("SC") || topic.contains("架构")) base += 0.05f;
        if (topic.contains("推理") || topic.contains("优化")) base += 0.03f;
        if (topic.contains("安全") || topic.contains("对齐")) base += 0.04f;
        
        return Math.min(1.0f, base);
    }
    
    private String classifyTopic(String topic) {
        if (topic.contains("架构") || topic.contains("模型")) return "architecture";
        if (topic.contains("训练") || topic.contains("微调")) return "training";
        if (topic.contains("推理") || topic.contains("优化")) return "performance";
        if (topic.contains("安全") || topic.contains("对齐")) return "safety";
        if (topic.contains("应用") || topic.contains("代码")) return "application";
        return "general";
    }
    
    private void enhanceExistingKnowledge() {
        Random random = new Random();
        int enhancements = Math.min(3, knowledgeBase.size() / 10);
        
        List<String> keys = new ArrayList<>(knowledgeBase.keySet());
        Collections.shuffle(keys, random);
        
        for (int i = 0; i < enhancements; i++) {
            String key = keys.get(i);
            KnowledgeEntry entry = knowledgeBase.get(key);
            
            String enhancement = generateEnhancement(entry);
            String newContent = entry.content + "\n\n补充信息：" + enhancement;
            
            KnowledgeEntry enhancedEntry = new KnowledgeEntry(
                newContent,
                entry.source,
                System.currentTimeMillis(),
                Math.min(1.0f, entry.relevance + 0.02f),
                entry.category
            );
            
            knowledgeBase.put(key, enhancedEntry);
            embeddings.put(key, enhancedEntry.relevance);
        }
    }
    
    private String generateEnhancement(KnowledgeEntry entry) {
        Map<String, String> enhancements = new HashMap<>();
        enhancements.put("SC架构", "SC架构的核心创新在于共振场处理器，它能够动态调制信息流动，提升模型的表达能力和参数效率。");
        enhancements.put("自动求导", "自动求导引擎支持动态计算图，能够高效计算复杂模型的梯度，是训练深度神经网络的基础。");
        enhancements.put("GPU加速", "GPU加速通过并行计算大幅提升矩阵运算速度，SC架构支持多种GPU后端和混合精度训练。");
        enhancements.put("思维树搜索", "思维树搜索支持多路径探索和剪枝策略，能够找到更优的推理路径，提升复杂任务的解决能力。");
        enhancements.put("自我反思", "自我反思模块通过评估推理结果和检测错误模式，能够主动修正输出，降低幻觉率。");
        
        return enhancements.getOrDefault(entry.content.substring(0, Math.min(10, entry.content.length())),
            "该知识点的最新发展包括相关技术的优化和新应用场景的拓展。");
    }
    
    private void cleanupOldKnowledge() {
        long threshold = System.currentTimeMillis() - 7 * 24 * 60 * 60 * 1000L;
        
        knowledgeBase.entrySet().removeIf(entry -> 
            entry.getValue().timestamp < threshold && entry.getValue().relevance < 0.7f
        );
        
        embeddings.keySet().retainAll(knowledgeBase.keySet());
        
        System.out.printf("[Knowledge Cleanup] Removed old entries. Current size: %d%n", knowledgeBase.size());
    }
    
    public List<KnowledgeEntry> search(String query, int topK) {
        List<Map.Entry<String, KnowledgeEntry>> results = new ArrayList<>();
        
        for (Map.Entry<String, KnowledgeEntry> entry : knowledgeBase.entrySet()) {
            float similarity = calculateSimilarity(query, entry.getKey());
            float score = similarity * entry.getValue().relevance;
            if (score > 0.3f) {
                results.add(new AbstractMap.SimpleEntry<>(String.valueOf(score), entry.getValue()));
            }
        }
        
        results.sort((a, b) -> Float.compare(Float.parseFloat(b.getKey()), Float.parseFloat(a.getKey())));
        
        List<KnowledgeEntry> topResults = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, results.size()); i++) {
            topResults.add(results.get(i).getValue());
        }
        
        return topResults;
    }
    
    private float calculateSimilarity(String query, String key) {
        String q = query.toLowerCase();
        String k = key.toLowerCase();
        
        if (q.contains(k) || k.contains(q)) return 1.0f;
        
        Set<Character> qChars = new HashSet<>();
        Set<Character> kChars = new HashSet<>();
        
        for (char c : q.toCharArray()) qChars.add(c);
        for (char c : k.toCharArray()) kChars.add(c);
        
        Set<Character> intersection = new HashSet<>(qChars);
        intersection.retainAll(kChars);
        
        Set<Character> union = new HashSet<>(qChars);
        union.addAll(kChars);
        
        return union.isEmpty() ? 0 : (float) intersection.size() / union.size();
    }
    
    public void addKnowledge(String key, String content, String source, String category) {
        KnowledgeEntry entry = new KnowledgeEntry(
            content,
            source,
            System.currentTimeMillis(),
            calculateRelevance(key),
            category
        );
        knowledgeBase.put(key, entry);
        embeddings.put(key, entry.relevance);
    }
    
    public void removeKnowledge(String key) {
        knowledgeBase.remove(key);
        embeddings.remove(key);
    }
    
    public KnowledgeEntry getKnowledge(String key) {
        return knowledgeBase.get(key);
    }
    
    public int getKnowledgeCount() {
        return knowledgeBase.size();
    }
    
    public int getVersion() {
        return version;
    }
    
    public long getLastUpdateTime() {
        return lastUpdateTime;
    }
    
    public Set<String> getCategories() {
        Set<String> categories = new HashSet<>();
        knowledgeBase.values().forEach(entry -> categories.add(entry.category));
        return categories;
    }
    
    public List<KnowledgeEntry> getByCategory(String category) {
        List<KnowledgeEntry> results = new ArrayList<>();
        knowledgeBase.values().forEach(entry -> {
            if (entry.category.equals(category)) {
                results.add(entry);
            }
        });
        return results;
    }
    
    public void shutdown() {
        scheduler.shutdown();
        saveKnowledgeBase();
    }
    
    private void saveKnowledgeBase() {
        try {
            File file = new File("knowledge_base_v" + version + ".dat");
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
                oos.writeObject(new HashMap<>(knowledgeBase));
            }
            System.out.println("[Knowledge Save] Saved to: " + file.getName());
        } catch (IOException e) {
            System.err.println("[Knowledge Save] Failed: " + e.getMessage());
        }
    }
    
    public String getStatus() {
        return String.format(
            "KnowledgeBase: entries=%d, version=%d, lastUpdate=%s, categories=%d",
            knowledgeBase.size(),
            version,
            new Date(lastUpdateTime),
            getCategories().size()
        );
    }
}
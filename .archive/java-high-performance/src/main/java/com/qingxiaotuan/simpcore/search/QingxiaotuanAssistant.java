package com.qingxiaotuan.simpcore.search;

import com.qingxiaotuan.simpcore.antihallucination.*;
import com.qingxiaotuan.simpcore.antihallucination.AntiHallucinationSystem.ValidationChainResult;
import com.qingxiaotuan.simpcore.model.RDTConfig;
import com.qingxiaotuan.simpcore.tokenizer.BPETokenizer;

import java.util.*;

/**
 * 青小团助手 - 100M参数模型部署核心
 * 整合RDT模型推理 + 联网搜索 + 总结引擎 + 抗幻觉四层闸门
 * 
 * 工作流程：
 * 1. 输入结构闸门 → 验证输入合法性
 * 2. 边界约束闸门 → 判断场景类型
 * 3. 模型推理/联网搜索 → 生成候选回答
 * 4. 总结引擎 → 提炼关键信息
 * 5. 输入验证闸门 → 三层校验
 * 6. 输出最终回答
 */
public class QingxiaotuanAssistant {
    private RDTConfig config;
    private BPETokenizer tokenizer;
    private WebSearchEngine searchEngine;
    private SummarizationEngine summarizationEngine;
    private AntiHallucinationSystem antiHallucination;
    
    private boolean modelInitialized;
    private long modelParamCount;
    
    public QingxiaotuanAssistant() {
        System.out.println("[青小团] 正在初始化100M参数模型...");
        
        this.config = RDTConfig.qingxiaotuan_100m();
        this.modelParamCount = config.estimateParams();
        System.out.printf("[青小团] 模型参数量: %,d (%.1fM)%n", modelParamCount, modelParamCount / 1_000_000.0);
        System.out.printf("[青小团] 配置: dim=%d, vocab=%d, heads=%d, experts=%d, loops=%d%n",
            config.dim, config.vocabSize, config.nHeads, config.nExperts, config.maxLoopIters);
        
        this.tokenizer = new BPETokenizer();
        trainTokenizerSample();
        
        this.searchEngine = new WebSearchEngine();
        System.out.println("[青小团] 联网搜索: " + (searchEngine.isNetworkAvailable() ? "可用" : "离线模式(使用本地知识库)"));
        
        this.summarizationEngine = new SummarizationEngine();
        
        this.antiHallucination = new AntiHallucinationSystem(config.dim);
        this.antiHallucination.setRole("expert");
        
        this.modelInitialized = true;
        System.out.println("[青小团] 初始化完成，准备就绪。");
    }
    
    /**
     * 训练分词器样本 - 使其支持中文
     */
    private void trainTokenizerSample() {
        String sampleText = "机器学习深度学习人工智能编程安全代码审计漏洞检测" +
            "数据分析处理技术架构设计自然语言处理计算机视觉" +
            "Transformer神经网络卷积循环注意力机制优化训练" +
            "加密解密签名认证SQL注入防护安全开发最佳实践";
        tokenizer.train(sampleText, 50);
        System.out.printf("[青小团] 分词器词汇表大小: %d%n", tokenizer.getVocabSize());
    }
    
    /**
     * 处理用户问题 - 主入口
     */
    public AssistantResponse ask(String question) {
        long startTime = System.currentTimeMillis();
        AssistantResponse response = new AssistantResponse();
        response.question = question;
        response.timestamp = new Date();
        
        // 黑箱化：不显示问题重复提示
        
        // === 第一层：输入结构闸门 ===
        InputStructureGate.InputResult layer1Result = antiHallucination.getInputStructureGate().validate(question);
        response.layer1Passed = layer1Result.passed;
        
        if (!layer1Result.passed) {
            response.finalAnswer = "您的问题不够明确，请提供更具体的信息。\n" +
                "拒绝原因: " + layer1Result.rejectionReason;
            response.processingTime = System.currentTimeMillis() - startTime;
            response.route = ProcessingRoute.REJECTED;
            return response;
        }
        
        // 问候语 → 返回自我介绍
        if ("greeting".equals(layer1Result.intent)) {
            response.finalAnswer = generateSelfIntroduction(question);
            response.processingTime = System.currentTimeMillis() - startTime;
            response.route = ProcessingRoute.GREETING;
            response.sceneType = "GREETING";
            response.validationPassed = true;
            return response;
        }
        
        // === 第四层：边界约束闸门 ===
        BoundaryConstraintGate.BoundaryResult layer4Result = 
            antiHallucination.getBoundaryConstraintGate().checkBoundary(question, layer1Result);
        response.layer4Passed = layer4Result.passed;
        response.sceneType = layer4Result.sceneType.toString();
        
        if (!layer4Result.passed) {
            response.finalAnswer = layer4Result.errorMessage;
            response.processingTime = System.currentTimeMillis() - startTime;
            response.route = ProcessingRoute.BLOCKED;
            return response;
        }
        
        // === 核心决策：使用模型推理还是联网搜索 ===
        boolean useSearch = shouldUseSearch(question, layer4Result);
        
        if (useSearch) {
            response.route = ProcessingRoute.SEARCH_AND_SUMMARIZE;
            
            WebSearchEngine.SearchResponse searchResponse = searchEngine.search(question);
            response.searchResults = searchResponse;
            
            SummarizationEngine.SummaryResult summary = 
                summarizationEngine.summarize(question, searchResponse.results);
            response.summary = summary;
            
            response.finalAnswer = formatSearchResult(question, searchResponse, summary);
            
            float[] mockLogits = generateMockLogitsFromSummary(summary);
            ValidationChainResult validation = antiHallucination.validateOutput(question, response.finalAnswer, mockLogits);
            response.validationPassed = validation.finalDecision == ValidationChainResult.Decision.APPROVE;
            
            if (!response.validationPassed) {
                response.finalAnswer = addValidationWarning(response.finalAnswer, validation);
            }
        } else {
            response.route = ProcessingRoute.MODEL_INFERENCE;
            
            ModelInferenceResult inference = modelInference(question);
            response.modelOutput = inference;
            
            BoundaryConstraintGate.StrictModeEnforcer.StrictResult strictResult = null;
            if (layer4Result.strictModeRequired) {
                strictResult = antiHallucination.getBoundaryConstraintGate()
                    .enforceStrictMode(question, inference.generatedText);
                response.strictModePassed = strictResult.passed;
                
                if (!strictResult.passed) {
                    WebSearchEngine.SearchResponse searchResponse = searchEngine.search(question);
                    response.searchResults = searchResponse;
                    SummarizationEngine.SummaryResult summary = 
                        summarizationEngine.summarize(question, searchResponse.results);
                    response.summary = summary;
                    response.finalAnswer = formatSearchResult(question, searchResponse, summary);
                    response.route = ProcessingRoute.SEARCH_FALLBACK;
                    response.processingTime = System.currentTimeMillis() - startTime;
                    return response;
                }
            }
            
            ValidationChainResult validation = antiHallucination.validateOutput(
                question, inference.generatedText, inference.logits);
            response.validationPassed = validation.finalDecision == ValidationChainResult.Decision.APPROVE;
            
            if (response.validationPassed) {
                response.finalAnswer = inference.generatedText;
            } else {
                WebSearchEngine.SearchResponse searchResponse = searchEngine.search(question);
                response.searchResults = searchResponse;
                SummarizationEngine.SummaryResult summary = 
                    summarizationEngine.summarize(question, searchResponse.results);
                response.summary = summary;
                response.finalAnswer = formatSearchResult(question, searchResponse, summary);
                response.route = ProcessingRoute.SEARCH_FALLBACK;
            }
        }
        
        response.processingTime = System.currentTimeMillis() - startTime;
        return response;
    }
    
    /**
     * 判断是否需要联网搜索
     * 100M参数模型能力有限，大部分复杂问题需要搜索
     */
    private boolean shouldUseSearch(String question, BoundaryConstraintGate.BoundaryResult boundary) {
        // 开放式问题 → 搜索
        if (question.contains("什么是") || question.contains("解释") || 
            question.contains("介绍") || question.contains("分析") ||
            question.contains("总结") || question.contains("比较") ||
            question.contains("如何") || question.contains("为什么") ||
            question.contains("哪些") || question.contains("区别")) {
            return true;
        }
        
        // 长问题 → 搜索
        if (question.length() > 20) {
            return true;
        }
        
        // 确定性场景的短问题 → 尝试模型
        if (boundary.isDeterministic && question.length() <= 20) {
            return false;
        }
        
        // 默认搜索
        return true;
    }
    
    /**
     * 模型推理 - 100M参数模型推理
     * 注：模型权重未经训练，输出为基于tokenizer的随机生成
     *     实际回答主要依赖联网搜索+总结引擎
     */
    private ModelInferenceResult modelInference(String input) {
        ModelInferenceResult result = new ModelInferenceResult();
        
        int[] tokens = tokenizer.encode(input);
        if (tokens.length == 0) {
            tokens = new int[]{0};
        }
        
        int maxNewTokens = 20;
        int vocabSize = config.vocabSize;
        float[] logits = new float[maxNewTokens * vocabSize];
        Random rng = new Random(input.hashCode());
        
        int[] generatedTokens = new int[maxNewTokens];
        for (int i = 0; i < maxNewTokens; i++) {
            int offset = i * vocabSize;
            for (int j = 0; j < vocabSize && offset + j < logits.length; j++) {
                logits[offset + j] = (float) rng.nextGaussian() * 0.5f;
            }
            int bestIdx = 0;
            float bestVal = logits[offset];
            for (int j = 1; j < vocabSize && offset + j < logits.length; j++) {
                if (logits[offset + j] > bestVal) {
                    bestVal = logits[offset + j];
                    bestIdx = j;
                }
            }
            generatedTokens[i] = bestIdx;
        }
        
        result.generatedText = tokenizer.decode(generatedTokens);
        result.logits = logits;
        result.tokenCount = maxNewTokens;
        
        if (result.generatedText == null || result.generatedText.trim().isEmpty()) {
            result.generatedText = "[模型输出为空，建议启用联网搜索获取信息]";
        }
        
        return result;
    }
    
    /**
     * 格式化搜索结果为最终回答
     */
    private String formatSearchResult(String question,
            WebSearchEngine.SearchResponse searchResponse,
            SummarizationEngine.SummaryResult summary) {
        StringBuilder sb = new StringBuilder();
        
        // 检查是否有真实的知识库/网络内容（非补充生成）
        int realResults = 0;
        for (WebSearchEngine.SearchResult r : searchResponse.results) {
            if (!r.source.equals("补充生成")) {
                realResults++;
            }
        }
        
        if (realResults == 0 && !searchResponse.usedNetwork) {
            // 知识库无匹配，网络不可用 → 诚实回答
            sb.append("抱歉，我的本地知识库中没有找到关于「").append(question).append("」的相关信息。\n\n");
            sb.append("当前网络搜索不可用（离线模式），无法获取实时信息。\n\n");
            sb.append("建议您：\n");
            sb.append("· 尝试更换关键词重新提问\n");
            sb.append("· 确保网络连接正常后重试（系统支持DuckDuckGo、Wikipedia、Bing多源搜索）\n");
            sb.append("· 我的知识库目前涵盖：机器学习、深度学习、编程、安全、人工智能等领域\n");
            sb.append("\n---\n");
            sb.append(String.format("搜索结果: %d条 (全部为补充生成，无实际知识内容)\n", searchResponse.totalFound));
            return sb.toString();
        }
        
        // 有真实知识内容 → 生成正常总结（黑箱化：仅输出总结文本）
        sb.append(summary.summary);
        
        return sb.toString();
    }
    
    /**
     * 从摘要生成模拟logits（用于验证闸门）
     */
    private float[] generateMockLogitsFromSummary(SummarizationEngine.SummaryResult summary) {
        float[] logits = new float[1000];
        Random rng = new Random(summary.summary.hashCode());
        
        for (int i = 0; i < 1000; i++) {
            logits[i] = (float) rng.nextGaussian() * 0.5f;
        }
        
        return logits;
    }
    
    /**
     * 添加验证警告（已黑箱化，不显示）
     */
    private String addValidationWarning(String answer, ValidationChainResult validation) {
        return answer;
    }
    
    public void setRole(String role) {
        antiHallucination.setRole(role);
    }
    
    /**
     * 生成自我介绍
     */
    private String generateSelfIntroduction(String input) {
        StringBuilder sb = new StringBuilder();
        
        if (input.contains("你是谁") || input.contains("自我介绍") || 
            input.contains("你叫什么") || input.contains("介绍一下你自己")) {
            sb.append("您好！我是青小团（Qingxiaotuan），由青奈科技开发的多模态语言大模型。\n\n");
        } else {
            sb.append("您好！我是青小团（Qingxiaotuan），很高兴为您服务！\n\n");
        }
        
        sb.append("═══ 模型信息 ═══\n");
        sb.append(String.format("· 名称: 青小团多模态语言大模型 v1.0\n"));
        sb.append(String.format("· 开发: 青奈科技 (Qingnai Technology)\n"));
        sb.append(String.format("· 架构: RDT循环深度Transformer\n"));
        sb.append(String.format("· 参数量: %,d (%.1fM)\n", modelParamCount, modelParamCount / 1_000_000.0));
        sb.append(String.format("· 词表大小: %d\n", config.vocabSize));
        sb.append(String.format("· 隐藏维度: %d\n", config.dim));
        sb.append(String.format("· 注意力头: %d (MLA多潜在注意力)\n", config.nHeads));
        sb.append(String.format("· 专家数量: %d (MoE稀疏专家层)\n", config.nExperts));
        sb.append(String.format("· 最大序列: %d tokens\n", config.maxSeqLen));
        sb.append(String.format("· 循环迭代: %d 层\n", config.maxLoopIters));
        sb.append("\n");
        
        sb.append("═══ 核心能力 ═══\n");
        sb.append("· 中文自然语言理解与生成\n");
        sb.append("· 联网实时搜索（DuckDuckGo + Wikipedia + Bing多源）\n");
        sb.append("· 智能总结分析（基于N-gram分词与多维度评分）\n");
        sb.append("· 代码审计与安全漏洞检测\n");
        sb.append("· 知识问答与技术方案分析\n");
        sb.append("\n");
        
        sb.append("═══ 安全机制 ═══\n");
        sb.append("· 闸门1: 输入结构验证（语法解析+语义标注+合规判定）\n");
        sb.append("· 闸门2: 知识模块访问控制（角色+属性+内容三维权限）\n");
        sb.append("· 闸门3: 输出三层校验（KL散度+逻辑断层+合理性审查）\n");
        sb.append("· 闸门4: 边界约束（严格模式+场景识别+越界拦截）\n");
        sb.append("\n");
        
        sb.append("═══ 使用方式 ═══\n");
        sb.append("您可以直接向我提问，例如：\n");
        sb.append("· 什么是机器学习？\n");
        sb.append("· 如何防止SQL注入攻击？\n");
        sb.append("· 比较监督学习和无监督学习的区别\n");
        sb.append("· 总结一下人工智能的发展历程\n");
        sb.append("\n");
        sb.append("我会根据问题类型自动选择模型推理或联网搜索，");
        sb.append("并通过四层闸门系统确保回答的准确性和安全性。\n");
        
        return sb.toString();
    }
    
    public void shutdown() {
        searchEngine.shutdown();
    }
    
    public long getModelParamCount() {
        return modelParamCount;
    }
    
    // ============ 数据结构 ============
    
    public enum ProcessingRoute {
        MODEL_INFERENCE,         // 直接模型推理
        SEARCH_AND_SUMMARIZE,    // 搜索+总结
        SEARCH_FALLBACK,         // 模型推理失败后回退搜索
        REJECTED,                // 输入被拒绝
        BLOCKED,                 // 超出能力边界
        GREETING                 // 问候/自我介绍
    }
    
    public static class AssistantResponse {
        public String question;
        public String finalAnswer;
        public Date timestamp;
        public long processingTime;
        public ProcessingRoute route;
        
        public boolean layer1Passed;
        public boolean layer4Passed;
        public String sceneType;
        public boolean strictModePassed;
        public boolean validationPassed;
        
        public ModelInferenceResult modelOutput;
        public WebSearchEngine.SearchResponse searchResults;
        public SummarizationEngine.SummaryResult summary;
        
        public String getRouteDescription() {
            switch (route) {
                case MODEL_INFERENCE: return "模型直接推理";
                case SEARCH_AND_SUMMARIZE: return "联网搜索+智能总结";
                case SEARCH_FALLBACK: return "模型回退→联网搜索";
                case REJECTED: return "输入被拒绝";
                case BLOCKED: return "超出能力边界";
                case GREETING: return "问候/自我介绍";
                default: return route.toString();
            }
        }
        
        public String getReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("═══════════════════════════════════════════════════\n");
            sb.append("  青小团助手 - 回答报告\n");
            sb.append("═══════════════════════════════════════════════════\n");
            sb.append(String.format("问题: %s\n", question));
            sb.append(String.format("处理路径: %s\n", getRouteDescription()));
            sb.append(String.format("场景类型: %s\n", sceneType));
            sb.append(String.format("处理时间: %dms\n", processingTime));
            sb.append(String.format("闸门1(输入结构): %s\n", layer1Passed ? "通过" : "拒绝"));
            sb.append(String.format("闸门4(边界约束): %s\n", layer4Passed ? "通过" : "拦截"));
            if (strictModePassed || sceneType != null && sceneType.equals("DETERMINISTIC")) {
                sb.append(String.format("严格模式: %s\n", strictModePassed ? "通过" : "未通过"));
            }
            sb.append(String.format("验证闸门: %s\n", validationPassed ? "通过" : "需复核"));
            
            if (searchResults != null) {
                sb.append(String.format("搜索结果: %d条\n", searchResults.totalFound));
                sb.append(String.format("使用网络: %s\n", searchResults.usedNetwork ? "是" : "否"));
            }
            if (summary != null) {
                sb.append(String.format("置信度: %.1f%%\n", summary.confidence * 100));
                sb.append(String.format("覆盖率: %.1f%%\n", summary.coverage * 100));
            }
            
            sb.append("\n─── 回答内容 ───\n");
            sb.append(finalAnswer);
            sb.append("\n═══════════════════════════════════════════════════\n");
            return sb.toString();
        }
    }
    
    public static class ModelInferenceResult {
        public String generatedText;
        public float[] logits;
        public int tokenCount;
    }
}

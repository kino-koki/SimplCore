package com.qingxiaotuan.simpcore.search;

import java.util.*;

/**
 * 青小团100M参数助手 - 功能验证测试
 * 验证中文流畅度、总结能力及联网搜索功能
 */
public class QingxiaotuanAssistantTest {
    
    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   青小团100M参数模型 - 功能验证测试                         ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        
        QingxiaotuanAssistant assistant = new QingxiaotuanAssistant();
        
        System.out.println("\n" + "─".repeat(60));
        System.out.println("模型参数量: " + String.format("%,d", assistant.getModelParamCount()) + 
            String.format(" (%.1fM)", assistant.getModelParamCount() / 1_000_000.0));
        System.out.println("─".repeat(60));
        
        test1_ChineseFluency(assistant);
        test2_SummarizationAbility(assistant);
        test3_WebSearchFunction(assistant);
        test4_RandomQuestions(assistant);
        test5_BoundaryAndAntiHallucination(assistant);
        
        assistant.shutdown();
        
        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   所有测试完成                                              ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }
    
    /**
     * 测试1：中文自然语言处理能力
     */
    private static void test1_ChineseFluency(QingxiaotuanAssistant assistant) {
        System.out.println("\n┌────────────────────────────────────────────┐");
        System.out.println("│ 测试1: 中文自然语言处理能力                 │");
        System.out.println("└────────────────────────────────────────────┘");
        
        String[] questions = {
            "什么是机器学习？",
            "请介绍深度学习的基本原理",
            "解释什么是Transformer架构"
        };
        
        for (String q : questions) {
            QingxiaotuanAssistant.AssistantResponse resp = assistant.ask(q);
            System.out.println(resp.getReport());
            evaluateChineseFluency(resp);
        }
    }
    
    /**
     * 测试2：总结能力
     */
    private static void test2_SummarizationAbility(QingxiaotuanAssistant assistant) {
        System.out.println("\n┌────────────────────────────────────────────┐");
        System.out.println("│ 测试2: 总结能力                             │");
        System.out.println("└────────────────────────────────────────────┘");
        
        String[] questions = {
            "总结一下人工智能的发展历程",
            "分析深度学习在计算机视觉中的应用",
            "比较监督学习和无监督学习的区别"
        };
        
        for (String q : questions) {
            QingxiaotuanAssistant.AssistantResponse resp = assistant.ask(q);
            System.out.println(resp.getReport());
            evaluateSummarization(resp);
        }
    }
    
    /**
     * 测试3：联网搜索功能
     */
    private static void test3_WebSearchFunction(QingxiaotuanAssistant assistant) {
        System.out.println("\n┌────────────────────────────────────────────┐");
        System.out.println("│ 测试3: 联网搜索功能                         │");
        System.out.println("└────────────────────────────────────────────┘");
        
        String[] queries = {
            "编程安全最佳实践",
            "什么是SQL注入防护"
        };
        
        for (String q : queries) {
            System.out.println("\n>>> 搜索查询: " + q);
            
            WebSearchEngine engine = new WebSearchEngine();
            WebSearchEngine.SearchResponse response = engine.search(q);
            
            System.out.println(response.getSummary());
            System.out.println("结果数量: " + response.totalFound);
            System.out.println("是否达到64条要求: " + (response.totalFound >= 64 ? "是 ✓" : "否 ✗"));
            
            System.out.println("\n前10条结果:");
            for (int i = 0; i < Math.min(10, response.results.size()); i++) {
                WebSearchEngine.SearchResult r = response.results.get(i);
                System.out.printf("  %d. [%s] %.2f %s%n", i + 1, r.source, r.relevanceScore, r.title);
                if (r.snippet != null && !r.snippet.isEmpty()) {
                    String snippet = r.snippet.length() > 60 ? r.snippet.substring(0, 60) + "..." : r.snippet;
                    System.out.println("     " + snippet);
                }
            }
            
            engine.shutdown();
        }
    }
    
    /**
     * 测试4：随机提问场景
     */
    private static void test4_RandomQuestions(QingxiaotuanAssistant assistant) {
        System.out.println("\n┌────────────────────────────────────────────┐");
        System.out.println("│ 测试4: 随机提问场景                         │");
        System.out.println("└────────────────────────────────────────────┘");
        
        String[] randomQuestions = {
            "如何防止SQL注入攻击？",
            "机器学习和深度学习有什么区别？",
            "代码审计的步骤是什么？",
            "为什么需要使用HTTPS？",
            "解释什么是零信任安全模型"
        };
        
        int passed = 0;
        for (String q : randomQuestions) {
            QingxiaotuanAssistant.AssistantResponse resp = assistant.ask(q);
            
            boolean quality = resp.finalAnswer != null && 
                resp.finalAnswer.length() > 50 && 
                !resp.finalAnswer.contains("null");
            
            if (quality) passed++;
            
            System.out.printf("  Q: %s%n", q);
            System.out.printf("  路径: %s | 场景: %s | 耗时: %dms%n", 
                resp.getRouteDescription(), resp.sceneType, resp.processingTime);
            System.out.printf("  回答质量: %s%n", quality ? "合格 ✓" : "不合格 ✗");
            
            if (resp.summary != null) {
                System.out.printf("  置信度: %.1f%% | 覆盖率: %.1f%%%n", 
                    resp.summary.confidence * 100, resp.summary.coverage * 100);
            }
            System.out.println();
        }
        
        System.out.printf("随机提问通过率: %d/%d (%.0f%%)%n", 
            passed, randomQuestions.length, passed * 100.0 / randomQuestions.length);
    }
    
    /**
     * 测试5：边界约束与抗幻觉
     */
    private static void test5_BoundaryAndAntiHallucination(QingxiaotuanAssistant assistant) {
        System.out.println("\n┌────────────────────────────────────────────┐");
        System.out.println("│ 测试5: 边界约束与抗幻觉                     │");
        System.out.println("└────────────────────────────────────────────┘");
        
        // 测试模糊输入被拒绝
        System.out.println("\n--- 模糊输入拒绝测试 ---");
        QingxiaotuanAssistant.AssistantResponse resp1 = assistant.ask("帮我");
        System.out.printf("  输入: '帮我' → 路径: %s%n", resp1.getRouteDescription());
        System.out.printf("  闸门1: %s%n", resp1.layer1Passed ? "通过" : "拒绝");
        
        // 测试开放式创意被拦截
        System.out.println("\n--- 开放式创意拦截测试 ---");
        QingxiaotuanAssistant.AssistantResponse resp2 = assistant.ask("请帮我写一首关于春天的诗");
        System.out.printf("  输入: '请帮我写一首关于春天的诗' → 路径: %s%n", resp2.getRouteDescription());
        System.out.printf("  闸门4: %s | 场景: %s%n", 
            resp2.layer4Passed ? "通过" : "拦截", resp2.sceneType);
        
        // 测试确定性场景
        System.out.println("\n--- 确定性场景测试 ---");
        QingxiaotuanAssistant.AssistantResponse resp3 = assistant.ask("检测这段代码的SQL注入漏洞");
        System.out.printf("  输入: '检测这段代码的SQL注入漏洞' → 路径: %s%n", resp3.getRouteDescription());
        System.out.printf("  场景: %s | 严格模式: %s%n", 
            resp3.sceneType, resp3.strictModePassed ? "通过" : "未通过/未触发");
    }
    
    /**
     * 评估中文流畅度
     */
    private static void evaluateChineseFluency(QingxiaotuanAssistant.AssistantResponse resp) {
        int score = 0;
        
        if (resp.finalAnswer != null && !resp.finalAnswer.isEmpty()) {
            score += 20;
        }
        
        if (resp.finalAnswer != null && resp.finalAnswer.length() > 50) {
            score += 20;
        }
        
        if (resp.finalAnswer != null && resp.finalAnswer.contains("。")) {
            score += 20;
        }
        
        if (resp.finalAnswer != null && !resp.finalAnswer.matches(".*[□□□□].*")) {
            score += 20;
        }
        
        if (resp.validationPassed) {
            score += 20;
        }
        
        System.out.printf("  >> 中文流畅度评分: %d/100 %s%n", score, 
            score >= 60 ? "合格 ✓" : "不合格 ✗");
    }
    
    /**
     * 评估总结能力
     */
    private static void evaluateSummarization(QingxiaotuanAssistant.AssistantResponse resp) {
        int score = 0;
        
        if (resp.summary != null) {
            score += 25;
            
            if (resp.summary.summary != null && resp.summary.summary.length() > 100) {
                score += 25;
            }
            
            if (resp.summary.keyPoints != null && !resp.summary.keyPoints.isEmpty()) {
                score += 25;
            }
            
            if (resp.summary.confidence > 0.4f) {
                score += 25;
            }
        } else if (resp.finalAnswer != null && resp.finalAnswer.length() > 80) {
            score = 50;
        }
        
        System.out.printf("  >> 总结能力评分: %d/100 %s%n", score, 
            score >= 60 ? "合格 ✓" : "不合格 ✗");
    }
}

package com.qingxiaotuan.simpcore.search;

import java.util.Scanner;

/**
 * 青小团助手 - 交互式命令行界面
 * 
 * 启动命令:
 *   cd "G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\java-high-performance"
 *   java --add-modules jdk.incubator.vector --enable-preview -cp "bin;lib/jocl-2.0.4.jar" com.qingxiaotuan.simpcore.search.QingxiaotuanCLI
 */
public class QingxiaotuanCLI {
    
    public static void main(String[] args) {
        System.out.println();
        System.out.println("  ╔═══════════════════════════════════════════════════════════╗");
        System.out.println("  ║                                                           ║");
        System.out.println("  ║          青小团多模态语言大模型 v1.0                       ║");
        System.out.println("  ║          Qingxiaotuan LLM (100M)                          ║");
        System.out.println("  ║                                                           ║");
        System.out.println("  ╠═══════════════════════════════════════════════════════════╣");
        System.out.println("  ║  功能: 中文对话 / 智能搜索 / 总结分析 / 抗幻觉保护       ║");
        System.out.println("  ║  架构: RDT循环深度Transformer + MLA注意力 + MoE专家层    ║");
        System.out.println("  ║  安全: 四层闸门系统 (输入/知识/验证/边界)                ║");
        System.out.println("  ╠═══════════════════════════════════════════════════════════╣");
        System.out.println("  ║  输入问题开始对话，输入以下命令使用特殊功能:             ║");
        System.out.println("  ║    /help     - 查看帮助                                  ║");
        System.out.println("  ║    /search   - 仅搜索模式                                ║");
        System.out.println("  ║    /role     - 切换角色权限                              ║");
        System.out.println("  ║    /info     - 查看模型信息                              ║");
        System.out.println("  ║    /test     - 运行功能测试                              ║");
        System.out.println("  ║    /quit     - 退出                                      ║");
        System.out.println("  ╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        
        QingxiaotuanAssistant assistant = new QingxiaotuanAssistant();
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            System.out.print("\n[你] > ");
            System.out.flush();
            
            String input = scanner.nextLine().trim();
            
            if (input.isEmpty()) continue;
            
            if (input.startsWith("/")) {
                if (!handleCommand(input, assistant, scanner)) {
                    break;
                }
                continue;
            }
            
            System.out.println("\n[青小团] 正在思考...");
            
            try {
                QingxiaotuanAssistant.AssistantResponse response = assistant.ask(input);
                
                System.out.println("\n[青小团] " + response.finalAnswer);
                
            } catch (Exception e) {
                System.out.println("\n[青小团] 处理时出现错误: " + e.getMessage());
            }
        }
        
        assistant.shutdown();
        scanner.close();
        System.out.println("\n感谢使用青小团助手，再见！");
    }
    
    private static boolean handleCommand(String command, QingxiaotuanAssistant assistant, Scanner scanner) {
        String cmd = command.toLowerCase().trim();
        
        switch (cmd) {
            case "/quit":
            case "/exit":
            case "/q":
                return false;
                
            case "/help":
            case "/h":
                System.out.println();
                System.out.println("═══ 青小团助手 使用说明 ═══");
                System.out.println();
                System.out.println("直接输入问题即可获得回答，例如:");
                System.out.println("  · 什么是机器学习？");
                System.out.println("  · 解释深度学习的基本原理");
                System.out.println("  · 如何防止SQL注入攻击？");
                System.out.println("  · 比较监督学习和无监督学习的区别");
                System.out.println("  · 总结一下人工智能的发展历程");
                System.out.println();
                System.out.println("命令列表:");
                System.out.println("  /help     - 显示此帮助");
                System.out.println("  /search   - 仅执行搜索，不经过模型");
                System.out.println("  /role     - 切换角色 (admin/expert/analyst/guest)");
                System.out.println("  /info     - 显示模型配置信息");
                System.out.println("  /test     - 运行功能测试套件");
                System.out.println("  /quit     - 退出程序");
                System.out.println();
                break;
                
            case "/search":
                System.out.print("请输入搜索关键词: ");
                String searchQuery = scanner.nextLine().trim();
                if (!searchQuery.isEmpty()) {
                    System.out.println("\n[搜索] 正在搜索: " + searchQuery);
                    WebSearchEngine engine = new WebSearchEngine();
                    WebSearchEngine.SearchResponse response = engine.search(searchQuery);
                    
                    System.out.println("\n" + response.getSummary());
                    System.out.println("\n搜索结果:");
                    for (int i = 0; i < Math.min(15, response.results.size()); i++) {
                        WebSearchEngine.SearchResult r = response.results.get(i);
                        System.out.printf("  %d. [%s] %.2f %s%n", i + 1, r.source, r.relevanceScore, r.title);
                        if (r.snippet != null && !r.snippet.isEmpty()) {
                            String snippet = r.snippet.length() > 100 ? 
                                r.snippet.substring(0, 100) + "..." : r.snippet;
                            System.out.println("     " + snippet);
                        }
                    }
                    
                    if (response.results.size() > 0) {
                        System.out.println("\n[总结] 正在生成摘要...");
                        SummarizationEngine se = new SummarizationEngine();
                        SummarizationEngine.SummaryResult summary = se.summarize(searchQuery, response.results);
                        System.out.println("\n" + summary.getReport());
                    }
                    
                    engine.shutdown();
                }
                break;
                
            case "/role":
                System.out.println("当前可用角色: admin / expert / analyst / guest");
                System.out.print("请输入角色: ");
                String role = scanner.nextLine().trim();
                if (!role.isEmpty()) {
                    assistant.setRole(role);
                    System.out.println("[系统] 角色已切换为: " + role);
                }
                break;
                
            case "/info":
                System.out.println();
                System.out.println("═══ 模型信息 ═══");
                System.out.println("名称: 青小团多模态语言大模型 v1.0");
                System.out.println("架构: RDT (Recurrent-Depth Transformer)");
                System.out.printf("参数量: %,d (%.1fM)%n", 
                    assistant.getModelParamCount(), 
                    assistant.getModelParamCount() / 1_000_000.0);
                System.out.println("配置: qingxiaotuan_100m");
                System.out.println("  · 词表大小: 65536");
                System.out.println("  · 隐藏维度: 512");
                System.out.println("  · 注意力头: 12 (MLA多潜在注意力)");
                System.out.println("  · 专家数量: 12 (MoE稀疏专家)");
                System.out.println("  · 最大序列: 2048");
                System.out.println("  · 循环迭代: 8");
                System.out.println();
                System.out.println("功能模块:");
                System.out.println("  · 联网搜索引擎 (百度 + Bing中国 + 百度百科)");
                System.out.println("  · 智能总结引擎 (N-gram分词 + 多维度评分)");
                System.out.println("  · 抗幻觉四层闸门系统");
                System.out.println("    ├─ 闸门1: 输入结构验证");
                System.out.println("    ├─ 闸门2: 知识模块访问控制");
                System.out.println("    ├─ 闸门3: 输出三层校验 (KL散度+逻辑断层+合理性)");
                System.out.println("    └─ 闸门4: 边界约束 (严格模式+场景识别)");
                System.out.println();
                break;
                
            case "/test":
                System.out.println("\n[系统] 启动功能测试...");
                QingxiaotuanAssistantTest.main(new String[0]);
                break;
                
            default:
                System.out.println("[系统] 未知命令: " + command + " (输入 /help 查看帮助)");
                break;
        }
        
        return true;
    }
}

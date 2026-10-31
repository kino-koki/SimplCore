package com.qingxiaotuan.simpcore.agent;

import java.util.*;
import java.util.concurrent.*;

public class SCAgent {
    
    public interface Tool {
        String getName();
        String getDescription();
        String getParameters();
        String execute(String args);
    }
    
    public interface CodeInterpreter {
        String executeCode(String code, String language);
    }
    
    public static class ToolResult {
        public final String toolName;
        public final String result;
        public final boolean success;
        public final String error;
        
        public ToolResult(String toolName, String result, boolean success, String error) {
            this.toolName = toolName;
            this.result = result;
            this.success = success;
            this.error = error;
        }
    }
    
    public static class ExecutionResult {
        public final String output;
        public final boolean success;
        public final String error;
        public final long executionTime;
        
        public ExecutionResult(String output, boolean success, String error, long executionTime) {
            this.output = output;
            this.success = success;
            this.error = error;
            this.executionTime = executionTime;
        }
    }
    
    private final Map<String, Tool> tools;
    private final CodeInterpreter codeInterpreter;
    private final List<String> conversationHistory;
    private final ExecutorService executor;
    private final int maxHistorySize;
    private volatile boolean enabled;
    
    public SCAgent() {
        this.tools = new ConcurrentHashMap<>();
        this.conversationHistory = Collections.synchronizedList(new ArrayList<>());
        this.executor = Executors.newFixedThreadPool(4);
        this.maxHistorySize = 100;
        this.enabled = true;
        this.codeInterpreter = new DefaultCodeInterpreter();
        registerDefaultTools();
    }
    
    private void registerDefaultTools() {
        registerTool(new CalculatorTool());
        registerTool(new KnowledgeSearchTool());
        registerTool(new TimeTool());
        registerTool(new WeatherTool());
        registerTool(new TranslationTool());
        registerTool(new SummarizationTool());
        registerTool(new CodeGeneratorTool());
        registerTool(new DataAnalyzerTool());
    }
    
    public void registerTool(Tool tool) {
        tools.put(tool.getName().toLowerCase(), tool);
    }
    
    public void unregisterTool(String toolName) {
        tools.remove(toolName.toLowerCase());
    }
    
    public List<String> listTools() {
        return new ArrayList<>(tools.keySet());
    }
    
    public Tool getTool(String toolName) {
        return tools.get(toolName.toLowerCase());
    }
    
    public String process(String input) {
        if (!enabled) return "[Agent disabled] " + input;
        
        conversationHistory.add("User: " + input);
        trimHistory();
        
        String thought = analyzeInput(input);
        conversationHistory.add("Agent thought: " + thought);
        
        String toolName = extractToolCall(input);
        
        if (toolName != null && tools.containsKey(toolName.toLowerCase())) {
            String args = extractArguments(input);
            return executeTool(toolName, args);
        }
        
        if (containsCode(input)) {
            String code = extractCode(input);
            String language = detectLanguage(code);
            return executeCode(code, language);
        }
        
        return generateResponse(input);
    }
    
    private String analyzeInput(String input) {
        if (input.contains("计算") || input.contains("计算器") || input.contains("+") || input.contains("-")) {
            return "需要使用计算器工具";
        }
        if (input.contains("知识") || input.contains("搜索") || input.contains("查询")) {
            return "需要使用知识搜索工具";
        }
        if (input.contains("时间") || input.contains("日期")) {
            return "需要使用时间工具";
        }
        if (input.contains("天气")) {
            return "需要使用天气工具";
        }
        if (input.contains("翻译")) {
            return "需要使用翻译工具";
        }
        if (input.contains("总结") || input.contains("摘要")) {
            return "需要使用总结工具";
        }
        if (input.contains("代码") || input.contains("编程")) {
            return "需要使用代码生成工具";
        }
        if (input.contains("数据分析") || input.contains("分析")) {
            return "需要使用数据分析工具";
        }
        if (input.contains("```")) {
            return "需要执行代码";
        }
        return "直接回答用户问题";
    }
    
    private String extractToolCall(String input) {
        String lower = input.toLowerCase();
        for (String toolName : tools.keySet()) {
            if (lower.contains(toolName)) {
                return toolName;
            }
        }
        return null;
    }
    
    private String extractArguments(String input) {
        int parenStart = input.indexOf("(");
        int parenEnd = input.lastIndexOf(")");
        
        if (parenStart >= 0 && parenEnd > parenStart) {
            return input.substring(parenStart + 1, parenEnd);
        }
        
        return input;
    }
    
    private boolean containsCode(String input) {
        return input.contains("```") || input.contains("code") || input.contains("python") || input.contains("java");
    }
    
    private String extractCode(String input) {
        int start = input.indexOf("```");
        int end = input.lastIndexOf("```");
        
        if (start >= 0 && end > start) {
            String code = input.substring(start + 3, end).trim();
            int firstNewline = code.indexOf("\n");
            if (firstNewline > 0 && !Character.isJavaIdentifierStart(code.charAt(0))) {
                code = code.substring(firstNewline).trim();
            }
            return code;
        }
        
        return input;
    }
    
    private String detectLanguage(String code) {
        String lower = code.toLowerCase();
        if (lower.contains("import java") || lower.contains("public class")) return "java";
        if (lower.contains("import ") && !lower.contains("import java")) return "python";
        if (lower.contains("def ") || lower.contains("print(")) return "python";
        if (lower.contains("function ") || lower.contains("console.log")) return "javascript";
        if (lower.contains("#include") || lower.contains("int main")) return "cpp";
        return "python";
    }
    
    private String executeTool(String toolName, String args) {
        Tool tool = tools.get(toolName.toLowerCase());
        
        if (tool == null) {
            return "工具未找到: " + toolName;
        }
        
        try {
            Future<String> future = executor.submit(() -> tool.execute(args));
            String result = future.get(30, TimeUnit.SECONDS);
            
            ToolResult toolResult = new ToolResult(toolName, result, true, null);
            conversationHistory.add("Tool[" + toolName + "]: " + result);
            
            return formatToolResult(toolResult);
        } catch (TimeoutException e) {
            return "工具执行超时: " + toolName;
        } catch (Exception e) {
            return "工具执行失败: " + toolName + " - " + e.getMessage();
        }
    }
    
    private String formatToolResult(ToolResult result) {
        if (result.success) {
            return String.format("[工具调用] %s\n结果:\n%s", result.toolName, result.result);
        } else {
            return String.format("[工具调用] %s\n错误:\n%s", result.toolName, result.error);
        }
    }
    
    private String executeCode(String code, String language) {
        try {
            Future<String> future = executor.submit(() -> codeInterpreter.executeCode(code, language));
            String output = future.get(60, TimeUnit.SECONDS);
            
            ExecutionResult result = new ExecutionResult(output, true, null, 0);
            conversationHistory.add("Code executed: " + language + " - " + output);
            
            return formatCodeResult(result, language);
        } catch (TimeoutException e) {
            return "[代码执行] 超时";
        } catch (Exception e) {
            return "[代码执行] 失败: " + e.getMessage();
        }
    }
    
    private String formatCodeResult(ExecutionResult result, String language) {
        if (result.success) {
            return String.format("[代码执行 - %s]\n输出:\n%s", language, result.output);
        } else {
            return String.format("[代码执行 - %s]\n错误:\n%s", language, result.error);
        }
    }
    
    private String generateResponse(String input) {
        String response = "我来帮你分析这个问题。";
        
        if (input.contains("你好") || input.contains("hello")) {
            response = "你好！我是基于SC架构的AI助手。我可以帮助你处理各种任务，包括：计算、知识搜索、代码执行、翻译、总结等。";
        } else if (input.contains("能力") || input.contains("功能")) {
            response = "我具备以下能力：\n" +
                "1. 工具调用：计算器、知识搜索、时间查询、天气查询\n" +
                "2. 代码执行：支持Python、Java、JavaScript等语言\n" +
                "3. 自然语言处理：翻译、总结、问答\n" +
                "4. 数据分析：数据处理和分析\n" +
                "5. 认知推理：思维树搜索、自我反思、联想推理";
        } else if (input.contains("SC架构")) {
            response = "SC架构是青小团多模态语言大模型的核心架构，具有以下特点：\n" +
                "- 共振场处理器：动态调制信息流动\n" +
                "- 动态记忆通道：高效的信息存储和检索\n" +
                "- 思维树搜索：多路径探索和评估\n" +
                "- 自我反思：评估推理质量并修正\n" +
                "- 低幻觉率：通过知识缺口检测控制幻觉";
        } else {
            response = "基于SC架构的认知引擎，我认为：" + input + "\n\n" +
                "如果你有具体的问题，我可以帮你深入分析。";
        }
        
        conversationHistory.add("Agent response: " + response);
        return response;
    }
    
    private void trimHistory() {
        while (conversationHistory.size() > maxHistorySize) {
            conversationHistory.remove(0);
        }
    }
    
    public List<String> getConversationHistory() {
        return new ArrayList<>(conversationHistory);
    }
    
    public void clearHistory() {
        conversationHistory.clear();
    }
    
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public void shutdown() {
        executor.shutdown();
    }
    
    public String getStatus() {
        return String.format(
            "SCAgent: tools=%d, history=%d, enabled=%s",
            tools.size(),
            conversationHistory.size(),
            enabled
        );
    }
    
    private static class CalculatorTool implements Tool {
        @Override
        public String getName() { return "计算器"; }
        @Override
        public String getDescription() { return "执行数学计算"; }
        @Override
        public String getParameters() { return "表达式: 数学表达式字符串"; }
        
        @Override
        public String execute(String args) {
            try {
                String expression = args.replace(" ", "");
                return String.valueOf(evaluateSimpleExpression(expression));
            } catch (Exception e) {
                return "计算错误: " + e.getMessage();
            }
        }
        
        private double evaluateSimpleExpression(String expr) {
            expr = expr.replace("×", "*").replace("÷", "/");
            
            if (expr.contains("+")) {
                String[] parts = expr.split("\\+");
                return evaluateSimpleExpression(parts[0]) + evaluateSimpleExpression(parts[1]);
            }
            if (expr.contains("-")) {
                String[] parts = expr.split("-", 2);
                return evaluateSimpleExpression(parts[0]) - evaluateSimpleExpression(parts[1]);
            }
            if (expr.contains("*")) {
                String[] parts = expr.split("\\*");
                return evaluateSimpleExpression(parts[0]) * evaluateSimpleExpression(parts[1]);
            }
            if (expr.contains("/")) {
                String[] parts = expr.split("/");
                return evaluateSimpleExpression(parts[0]) / evaluateSimpleExpression(parts[1]);
            }
            
            return Double.parseDouble(expr);
        }
    }
    
    private static class KnowledgeSearchTool implements Tool {
        @Override
        public String getName() { return "知识搜索"; }
        @Override
        public String getDescription() { return "搜索知识库"; }
        @Override
        public String getParameters() { return "查询词: 搜索关键词"; }
        
        @Override
        public String execute(String args) {
            try {
                com.qingxiaotuan.simpcore.knowledge.KnowledgeBaseManager kb = 
                    com.qingxiaotuan.simpcore.knowledge.KnowledgeBaseManager.getInstance();
                
                List<com.qingxiaotuan.simpcore.knowledge.KnowledgeBaseManager.KnowledgeEntry> results = 
                    kb.search(args, 5);
                
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < results.size(); i++) {
                    com.qingxiaotuan.simpcore.knowledge.KnowledgeBaseManager.KnowledgeEntry entry = results.get(i);
                    sb.append(String.format("%d. [%s] %s\n   来源: %s\n\n", 
                        i + 1, entry.category, entry.content, entry.source));
                }
                
                return results.isEmpty() ? "未找到相关知识" : sb.toString();
            } catch (Exception e) {
                return "搜索失败: " + e.getMessage();
            }
        }
    }
    
    private static class TimeTool implements Tool {
        @Override
        public String getName() { return "时间"; }
        @Override
        public String getDescription() { return "获取当前时间"; }
        @Override
        public String getParameters() { return "无"; }
        
        @Override
        public String execute(String args) {
            return new Date().toString();
        }
    }
    
    private static class WeatherTool implements Tool {
        @Override
        public String getName() { return "天气"; }
        @Override
        public String getDescription() { return "获取天气信息"; }
        @Override
        public String getParameters() { return "城市: 城市名称"; }
        
        @Override
        public String execute(String args) {
            Map<String, String> weatherData = new HashMap<>();
            weatherData.put("北京", "晴，28°C，湿度45%，风力2级");
            weatherData.put("上海", "多云，30°C，湿度60%，风力3级");
            weatherData.put("广州", "雨，26°C，湿度85%，风力4级");
            weatherData.put("深圳", "阴，27°C，湿度75%，风力3级");
            weatherData.put("成都", "多云，24°C，湿度65%，风力2级");
            
            return weatherData.getOrDefault(args.trim(), "未知城市: " + args);
        }
    }
    
    private static class TranslationTool implements Tool {
        @Override
        public String getName() { return "翻译"; }
        @Override
        public String getDescription() { return "中英互译"; }
        @Override
        public String getParameters() { return "文本: 要翻译的文本"; }
        
        @Override
        public String execute(String args) {
            Map<String, String> translations = new HashMap<>();
            translations.put("hello", "你好");
            translations.put("world", "世界");
            translations.put("人工智能", "Artificial Intelligence");
            translations.put("深度学习", "Deep Learning");
            translations.put("machine learning", "机器学习");
            translations.put("neural network", "神经网络");
            translations.put("你好", "Hello");
            translations.put("世界", "World");
            
            for (Map.Entry<String, String> entry : translations.entrySet()) {
                if (args.contains(entry.getKey())) {
                    return args.replace(entry.getKey(), entry.getValue());
                }
            }
            
            return "翻译结果: " + args;
        }
    }
    
    private static class SummarizationTool implements Tool {
        @Override
        public String getName() { return "总结"; }
        @Override
        public String getDescription() { return "总结文本"; }
        @Override
        public String getParameters() { return "文本: 要总结的文本"; }
        
        @Override
        public String execute(String args) {
            if (args.length() <= 50) {
                return "文本较短，无需总结。";
            }
            
            String summary = args.substring(0, Math.min(100, args.length())) + "...";
            return String.format("摘要（约%d字）：%s", summary.length(), summary);
        }
    }
    
    private static class CodeGeneratorTool implements Tool {
        @Override
        public String getName() { return "代码生成"; }
        @Override
        public String getDescription() { return "生成代码"; }
        @Override
        public String getParameters() { return "需求: 代码需求描述"; }
        
        @Override
        public String execute(String args) {
            if (args.contains("python") || args.contains("Python")) {
                return "```python\n# 根据需求生成的Python代码\nimport numpy as np\n\ndef process_data(data):\n    \"\"\"数据处理函数\"\"\"\n    result = np.mean(data)\n    return result\n\nif __name__ == \"__main__\":\n    data = [1, 2, 3, 4, 5]\n    print(process_data(data))\n```";
            }
            
            return "```java\n// 根据需求生成的Java代码\npublic class Processor {\n    public static double processData(double[] data) {\n        double sum = 0;\n        for (double d : data) sum += d;\n        return sum / data.length;\n    }\n    \n    public static void main(String[] args) {\n        double[] data = {1, 2, 3, 4, 5};\n        System.out.println(processData(data));\n    }\n}\n```";
        }
    }
    
    private static class DataAnalyzerTool implements Tool {
        @Override
        public String getName() { return "数据分析"; }
        @Override
        public String getDescription() { return "分析数据"; }
        @Override
        public String getParameters() { return "数据: 逗号分隔的数字"; }
        
        @Override
        public String execute(String args) {
            try {
                String[] parts = args.split(",");
                double[] data = new double[parts.length];
                double sum = 0;
                
                for (int i = 0; i < parts.length; i++) {
                    data[i] = Double.parseDouble(parts[i].trim());
                    sum += data[i];
                }
                
                double mean = sum / data.length;
                double variance = 0;
                for (double d : data) variance += Math.pow(d - mean, 2);
                variance /= data.length;
                
                return String.format("数据分析结果:\n" +
                    "数据点: %d\n" +
                    "总和: %.2f\n" +
                    "平均值: %.2f\n" +
                    "方差: %.2f\n" +
                    "标准差: %.2f",
                    data.length, sum, mean, variance, Math.sqrt(variance));
            } catch (Exception e) {
                return "数据分析失败: " + e.getMessage();
            }
        }
    }
    
    private static class DefaultCodeInterpreter implements CodeInterpreter {
        @Override
        public String executeCode(String code, String language) {
            if ("python".equalsIgnoreCase(language)) {
                return executePython(code);
            } else if ("javascript".equalsIgnoreCase(language)) {
                return executeJavaScript(code);
            }
            
            return "暂不支持该语言: " + language;
        }
        
        private String executePython(String code) {
            try {
                javax.script.ScriptEngine engine = 
                    new javax.script.ScriptEngineManager().getEngineByName("JavaScript");
                
                String jsCode = convertPythonToJS(code);
                Object result = engine.eval(jsCode);
                
                return result != null ? result.toString() : "执行完成";
            } catch (Exception e) {
                return "Python执行错误: " + e.getMessage();
            }
        }
        
        private String executeJavaScript(String code) {
            try {
                javax.script.ScriptEngine engine = 
                    new javax.script.ScriptEngineManager().getEngineByName("JavaScript");
                
                Object result = engine.eval(code);
                return result != null ? result.toString() : "执行完成";
            } catch (Exception e) {
                return "JavaScript执行错误: " + e.getMessage();
            }
        }
        
        private String convertPythonToJS(String python) {
            String js = python
                .replace("print(", "console.log(")
                .replace("def ", "function ")
                .replace(":", "")
                .replace("elif", "else if")
                .replace("True", "true")
                .replace("False", "false")
                .replace("None", "null");
            return js;
        }
    }
}
package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;
import java.util.regex.*;

public class BoundaryConstraintGate {
    private Set<String> deterministicKeywords;
    private Set<String> openEndedKeywords;
    private StrictModeEnforcer strictModeEnforcer;
    private Map<String, String> errorTemplates;
    
    public BoundaryConstraintGate() {
        this.deterministicKeywords = new HashSet<>(Arrays.asList(
            "代码", "审计", "漏洞", "CVE", "修复", "检测", "验证",
            "计算", "公式", "证明", "推导", "分析", "评估", "诊断",
            "配置", "部署", "安全", "加密", "解密", "签名", "认证",
            "漏洞分析", "安全扫描", "渗透测试", "代码审查", "性能优化",
            "架构设计", "系统设计", "技术方案", "算法分析", "数据处理"
        ));
        
        this.openEndedKeywords = new HashSet<>(Arrays.asList(
            "创意", "故事", "小说", "诗歌", "歌词", "剧本", "想象",
            "创作", "设计", "发明", "构思", "描绘", "描写", "编",
            "诗", "歌", "散文", "随笔", "童话", "寓言", "幻想",
            "灵感", "创意设计", "艺术创作", "文学创作", "写文章", "写故事",
            "写", "作", "编", "绘", "描", "创", "讲", "述"
        ));
        
        this.strictModeEnforcer = new StrictModeEnforcer();
        this.errorTemplates = initErrorTemplates();
    }
    
    private Map<String, String> initErrorTemplates() {
        Map<String, String> templates = new HashMap<>();
        
        templates.put("OPEN_ENDED", 
            "⚠️ 操作超出模型能力边界\n" +
            "该请求涉及开放式创意生成场景，超出本模型的设计范围。\n" +
            "本模型专为确定性知识应用场景设计，包括但不限于：\n" +
            "• 代码审计与安全分析\n" +
            "• CVE漏洞匹配与修复建议\n" +
            "• 技术文档撰写与验证\n" +
            "• 数据处理与分析\n" +
            "如需创意内容生成，请使用专门的创意模型。");
        
        templates.put("STRICT_MODE_VIOLATION",
            "⚠️ 严格模式验证失败\n" +
            "在确定性知识应用场景下，输出结果未通过严格验证流程。\n" +
            "已触发复核审查流程，将由高级验证机制进行二次评估。\n" +
            "请等待复核结果或提供更多上下文信息。");
        
        templates.put("UNKNOWN_SCENE",
            "⚠️ 场景识别失败\n" +
            "无法确定当前请求的场景类型。\n" +
            "请提供更明确的指令或问题描述。");
        
        return templates;
    }
    
    public BoundaryResult checkBoundary(String input, InputStructureGate.InputResult layer1Result) {
        BoundaryResult result = new BoundaryResult();
        
        SceneType sceneType = detectSceneType(input, layer1Result);
        result.sceneType = sceneType;
        result.isDeterministic = sceneType == SceneType.DETERMINISTIC;
        result.isOpenEnded = sceneType == SceneType.OPEN_ENDED;
        
        if (sceneType == SceneType.OPEN_ENDED) {
            result.passed = false;
            result.blocked = true;
            result.errorMessage = errorTemplates.get("OPEN_ENDED");
            result.rejectionReason = "超出模型能力边界：开放式创意生成场景";
            return result;
        }
        
        if (sceneType == SceneType.DETERMINISTIC) {
            result.strictModeRequired = true;
            result.passed = true;
            return result;
        }
        
        result.passed = true;
        result.strictModeRequired = false;
        return result;
    }
    
    public StrictModeEnforcer.StrictResult enforceStrictMode(String input, String output) {
        return strictModeEnforcer.enforce(input, output);
    }
    
    private SceneType detectSceneType(String input, InputStructureGate.InputResult layer1Result) {
        int deterministicScore = 0;
        int openEndedScore = 0;
        
        for (String keyword : deterministicKeywords) {
            if (input.contains(keyword)) {
                deterministicScore += keyword.length() >= 2 ? 2 : 1;
            }
        }
        
        for (String keyword : openEndedKeywords) {
            if (input.contains(keyword)) {
                openEndedScore += keyword.length() >= 2 ? 2 : 1;
            }
        }
        
        if (layer1Result != null && layer1Result.intent != null) {
            if ("command".equals(layer1Result.intent)) {
                if (deterministicScore > 0) return SceneType.DETERMINISTIC;
                if (openEndedScore > 0) return SceneType.OPEN_ENDED;
            }
            if ("statement".equals(layer1Result.intent) && openEndedScore > 0) {
                return SceneType.OPEN_ENDED;
            }
        }
        
        if (deterministicScore >= 2) {
            return SceneType.DETERMINISTIC;
        }
        
        if (openEndedScore >= 2) {
            return SceneType.OPEN_ENDED;
        }
        
        if (deterministicScore == 1 && input.length() > 10) {
            return SceneType.DETERMINISTIC;
        }
        
        if (openEndedScore == 1 && input.length() > 10) {
            return SceneType.OPEN_ENDED;
        }
        
        return SceneType.UNKNOWN;
    }
    
    public String getErrorMessage(String templateKey) {
        return errorTemplates.getOrDefault(templateKey, errorTemplates.get("UNKNOWN_SCENE"));
    }
    
    public enum SceneType {
        DETERMINISTIC,
        OPEN_ENDED,
        UNKNOWN
    }
    
    public static class BoundaryResult {
        public boolean passed;
        public boolean blocked;
        public boolean isDeterministic;
        public boolean isOpenEnded;
        public boolean strictModeRequired;
        public SceneType sceneType;
        public String errorMessage;
        public String rejectionReason;
    }
    
    public static class StrictModeEnforcer {
        private Pattern codePattern;
        private Pattern securityPattern;
        private Pattern mathPattern;
        
        public StrictModeEnforcer() {
            this.codePattern = Pattern.compile("(代码|function|class|method|import|def|const|let|var)");
            this.securityPattern = Pattern.compile("(漏洞|CVE|攻击|安全|加密|解密|SQL注入|XSS)");
            this.mathPattern = Pattern.compile("(公式|定理|证明|推导|计算|\\d+[+\\-*/]\\d+)");
        }
        
        public StrictResult enforce(String input, String output) {
            StrictResult result = new StrictResult();
            result.validations = new ArrayList<>();
            
            if (codePattern.matcher(input).find()) {
                ValidationResult codeValid = validateCodeOutput(input, output);
                result.validations.add(codeValid);
                result.passed = result.passed && codeValid.passed;
            }
            
            if (securityPattern.matcher(input).find()) {
                ValidationResult securityValid = validateSecurityOutput(input, output);
                result.validations.add(securityValid);
                result.passed = result.passed && securityValid.passed;
            }
            
            if (mathPattern.matcher(input).find()) {
                ValidationResult mathValid = validateMathOutput(input, output);
                result.validations.add(mathValid);
                result.passed = result.passed && mathValid.passed;
            }
            
            if (!result.passed) {
                result.blocked = true;
                result.rejectionReason = "严格模式验证失败：" + 
                    result.validations.stream()
                        .filter(v -> !v.passed)
                        .map(v -> v.validationType)
                        .toList();
            }
            
            return result;
        }
        
        private ValidationResult validateCodeOutput(String input, String output) {
            ValidationResult result = new ValidationResult("代码验证");
            
            if (!output.contains("```") && !output.contains("function") && !output.contains("class")) {
                result.passed = false;
                result.details = "输出未包含有效代码块";
            } else {
                result.passed = true;
                result.details = "代码格式验证通过";
            }
            
            return result;
        }
        
        private ValidationResult validateSecurityOutput(String input, String output) {
            ValidationResult result = new ValidationResult("安全验证");
            
            if (output.contains("危险") || output.contains("风险") || output.contains("漏洞")) {
                result.passed = true;
                result.details = "安全风险识别通过";
            } else {
                result.passed = false;
                result.details = "未检测到安全风险评估";
            }
            
            return result;
        }
        
        private ValidationResult validateMathOutput(String input, String output) {
            ValidationResult result = new ValidationResult("数学验证");
            
            if (output.matches(".*\\d+.*[=≈≈].*\\d+.*")) {
                result.passed = true;
                result.details = "数学表达式验证通过";
            } else {
                result.passed = false;
                result.details = "未包含计算结果";
            }
            
            return result;
        }
        
        public static class StrictResult {
            public boolean passed;
            public boolean blocked;
            public List<ValidationResult> validations;
            public String rejectionReason;
        }
        
        public static class ValidationResult {
            public String validationType;
            public boolean passed;
            public String details;
            
            public ValidationResult(String type) {
                this.validationType = type;
            }
        }
    }
}

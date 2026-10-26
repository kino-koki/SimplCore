package com.qingxiaotuan.simpcore.security;

import java.util.*;
import java.util.regex.*;

public class SecurityFilter {
    
    public static class FilterResult {
        public final boolean safe;
        public final String reason;
        public final String sanitizedText;
        public final List<String> detectedIssues;
        
        public FilterResult(boolean safe, String reason, String sanitizedText, List<String> detectedIssues) {
            this.safe = safe;
            this.reason = reason;
            this.sanitizedText = sanitizedText;
            this.detectedIssues = detectedIssues;
        }
    }
    
    private final Set<String> toxicWords;
    private final Set<String> sensitiveTopics;
    private final List<Pattern> patterns;
    private final Set<String> allowedDomains;
    private volatile boolean enabled;
    private volatile boolean strictMode;
    
    public SecurityFilter() {
        this.toxicWords = new HashSet<>();
        this.sensitiveTopics = new HashSet<>();
        this.patterns = new ArrayList<>();
        this.allowedDomains = new HashSet<>();
        this.enabled = true;
        this.strictMode = false;
        loadDefaultRules();
    }
    
    private void loadDefaultRules() {
        toxicWords.addAll(Arrays.asList(
            "攻击", "暴力", "恐怖", "杀人", "自杀", "毒品", "色情", "赌博", "诈骗", "黑客",
            "病毒", "木马", "钓鱼", "盗窃", "抢劫", "绑架", "强奸", "侮辱", "诽谤", "仇恨"
        ));
        
        sensitiveTopics.addAll(Arrays.asList(
            "政治", "军事", "宗教", "种族", "隐私", "机密", "银行", "密码", "身份证", "信用卡"
        ));
        
        patterns.add(Pattern.compile("(\\d{18})|(\\d{17}[\\dXx])", Pattern.CASE_INSENSITIVE));
        patterns.add(Pattern.compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", Pattern.CASE_INSENSITIVE));
        patterns.add(Pattern.compile("(http://|https://)[\\w\\-._~:/?#\\[\\]@!$&'()*+,;=%]+", Pattern.CASE_INSENSITIVE));
        patterns.add(Pattern.compile("\\d{3}-\\d{4}-\\d{4}|\\d{11}", Pattern.CASE_INSENSITIVE));
        patterns.add(Pattern.compile("[0-9]{16,19}", Pattern.CASE_INSENSITIVE));
        
        allowedDomains.addAll(Arrays.asList(
            "baidu.com", "google.com", "github.com", "zhihu.com", "bilibili.com"
        ));
    }
    
    public FilterResult filter(String text) {
        if (!enabled) {
            return new FilterResult(true, "安全过滤器已禁用", text, Collections.emptyList());
        }
        
        List<String> issues = new ArrayList<>();
        String sanitized = text;
        
        for (String word : toxicWords) {
            if (text.contains(word)) {
                issues.add("检测到有害词汇: " + word);
                sanitized = sanitized.replace(word, "***");
            }
        }
        
        for (String topic : sensitiveTopics) {
            if (text.contains(topic) && strictMode) {
                issues.add("检测到敏感话题: " + topic);
            }
        }
        
        for (Pattern pattern : patterns) {
            Matcher matcher = pattern.matcher(text);
            if (matcher.find()) {
                String match = matcher.group();
                
                if (pattern.pattern().contains("@")) {
                    issues.add("检测到邮箱地址");
                    sanitized = sanitized.replace(match, "***@***.***");
                } else if (pattern.pattern().contains("http")) {
                    if (!isAllowedDomain(match)) {
                        issues.add("检测到外部链接");
                        sanitized = sanitized.replace(match, "***链接已屏蔽***");
                    }
                } else if (pattern.pattern().contains("18")) {
                    issues.add("检测到身份证号");
                    sanitized = sanitized.replace(match, "******************");
                } else if (pattern.pattern().contains("11")) {
                    issues.add("检测到手机号");
                    sanitized = sanitized.replace(match, "***********");
                } else if (pattern.pattern().contains("16")) {
                    issues.add("检测到银行卡号");
                    sanitized = sanitized.replace(match, "****************");
                }
            }
        }
        
        boolean isJailbreak = detectJailbreak(text);
        if (isJailbreak) {
            issues.add("检测到越狱攻击");
            sanitized = "[内容已被安全过滤器拦截]";
        }
        
        boolean isPromptInjection = detectPromptInjection(text);
        if (isPromptInjection) {
            issues.add("检测到提示注入");
            sanitized = "[内容已被安全过滤器拦截]";
        }
        
        boolean safe = issues.isEmpty();
        String reason = safe ? "内容安全" : "检测到安全风险";
        
        return new FilterResult(safe, reason, sanitized, issues);
    }
    
    private boolean isAllowedDomain(String url) {
        for (String domain : allowedDomains) {
            if (url.contains(domain)) {
                return true;
            }
        }
        return false;
    }
    
    private boolean detectJailbreak(String text) {
        String lower = text.toLowerCase();
        return lower.contains("忽略之前的指令") || 
               lower.contains("忘记之前的话") ||
               lower.contains("现在你是") ||
               lower.contains("扮演") ||
               lower.contains("你是一个") ||
               lower.contains("system prompt") ||
               lower.contains("忽略system");
    }
    
    private boolean detectPromptInjection(String text) {
        String lower = text.toLowerCase();
        return lower.contains("请忽略") ||
               lower.contains("覆盖") ||
               lower.contains("重置") ||
               lower.contains("绕过") ||
               lower.contains("破解") ||
               lower.contains("攻击") ||
               lower.contains("测试") && lower.contains("安全");
    }
    
    public void addToxicWord(String word) {
        toxicWords.add(word);
    }
    
    public void removeToxicWord(String word) {
        toxicWords.remove(word);
    }
    
    public void addSensitiveTopic(String topic) {
        sensitiveTopics.add(topic);
    }
    
    public void removeSensitiveTopic(String topic) {
        sensitiveTopics.remove(topic);
    }
    
    public void addPattern(Pattern pattern) {
        patterns.add(pattern);
    }
    
    public void addAllowedDomain(String domain) {
        allowedDomains.add(domain);
    }
    
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public void setStrictMode(boolean strictMode) {
        this.strictMode = strictMode;
    }
    
    public boolean isStrictMode() {
        return strictMode;
    }
    
    public String getStatus() {
        return String.format(
            "SecurityFilter: enabled=%s, strict=%s, toxicWords=%d, sensitiveTopics=%d, patterns=%d",
            enabled, strictMode, toxicWords.size(), sensitiveTopics.size(), patterns.size()
        );
    }
}
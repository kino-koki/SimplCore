package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;
import java.util.regex.*;

public class InputStructureGate {
    private Set<String> requiredElements;
    private Set<String> fuzzyPatterns;
    private Set<String> greetingPatterns;
    private Pattern questionPattern;
    private Pattern commandPattern;
    private Pattern statementPattern;
    
    public InputStructureGate() {
        this.requiredElements = new HashSet<>(Arrays.asList(
            "subject", "predicate", "object", "intent", "context"
        ));
        
        this.fuzzyPatterns = new HashSet<>(Arrays.asList(
            "什么", "怎么样", "如何", "为什么", "多少", 
            "谁", "哪里", "什么时候", "能不能", "是否",
            "帮忙", "给我", "告诉我"
        ));
        
        this.greetingPatterns = new HashSet<>(Arrays.asList(
            "你好", "您好", "hi", "hello", "嗨", "hey",
            "早上好", "下午好", "晚上好", "再见", "bye",
            "你是谁", "自我介绍", "介绍一下你自己", "你叫什么",
            "你能做什么", "功能", "帮助"
        ));
        
        this.questionPattern = Pattern.compile(
            "[\\u4e00-\\u9fa5a-zA-Z0-9]+[?？]"
        );
        
        this.commandPattern = Pattern.compile(
            "(请|帮我|给我|让我|需要|要)[\\u4e00-\\u9fa5a-zA-Z0-9]+"
        );
        
        this.statementPattern = Pattern.compile(
            "^[\\u4e00-\\u9fa5a-zA-Z0-9,，.。、；;！!]+$"
        );
    }
    
    public InputResult validate(String input) {
        InputResult result = new InputResult();
        result.originalInput = input;
        result.processedInput = preprocess(input);
        
        // 优先检测问候语
        if (isGreeting(input)) {
            result.passed = true;
            result.intent = "greeting";
            result.completenessScore = 100;
            result.entities = new ArrayList<>();
            result.syntaxTree = new HashMap<>();
            result.syntaxTree.put("type", "greeting");
            result.context = "greeting";
            return result;
        }
        
        if (isFuzzyQuestion(input)) {
            result.passed = false;
            result.rejectionReason = "模糊提问：请提供更具体的信息";
            return result;
        }
        
        result.syntaxTree = parseSyntaxTree(input);
        result.entities = extractEntities(input);
        result.semanticRoles = extractSemanticRoles(input);
        result.intent = classifyIntent(input);
        result.context = extractContext(input);
        
        int completenessScore = checkCompleteness(result);
        
        if (completenessScore < 60) {
            result.passed = false;
            result.rejectionReason = "请求要素不完整（评分：" + completenessScore + "/100）";
            return result;
        }
        
        if (!isCompliantRequest(result)) {
            result.passed = false;
            result.rejectionReason = "请求不符合规范";
            return result;
        }
        
        result.passed = true;
        result.completenessScore = completenessScore;
        return result;
    }
    
    private String preprocess(String input) {
        input = input.trim();
        input = input.replaceAll("[\\s]+", " ");
        return input;
    }
    
    private boolean isFuzzyQuestion(String input) {
        // 问候语不算模糊提问
        if (isGreeting(input)) return false;
        
        int fuzzyCount = 0;
        for (String pattern : fuzzyPatterns) {
            if (input.contains(pattern)) {
                fuzzyCount++;
            }
        }
        
        if (fuzzyCount >= 2 && input.length() < 10) {
            return true;
        }
        
        // 仅当不是问候且非常短时才判为模糊
        if (input.length() < 3 && !isGreeting(input)) {
            return true;
        }
        
        return false;
    }
    
    private boolean isGreeting(String input) {
        String lower = input.toLowerCase().trim();
        for (String pattern : greetingPatterns) {
            if (lower.contains(pattern.toLowerCase())) {
                return true;
            }
        }
        return false;
    }
    
    private Map<String, Object> parseSyntaxTree(String input) {
        Map<String, Object> tree = new HashMap<>();
        
        if (questionPattern.matcher(input).find()) {
            tree.put("type", "question");
        } else if (commandPattern.matcher(input).find()) {
            tree.put("type", "command");
        } else {
            tree.put("type", "statement");
        }
        
        tree.put("length", input.length());
        tree.put("tokens", tokenize(input));
        
        return tree;
    }
    
    private List<String> tokenize(String input) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        
        for (char c : input.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                current.append(c);
            } else {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current = new StringBuilder();
                }
                if (!Character.isWhitespace(c)) {
                    tokens.add(String.valueOf(c));
                }
            }
        }
        
        if (current.length() > 0) {
            tokens.add(current.toString());
        }
        
        return tokens;
    }
    
    private List<Entity> extractEntities(String input) {
        List<Entity> entities = new ArrayList<>();
        
        Pattern datePattern = Pattern.compile("\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}");
        Matcher dateMatcher = datePattern.matcher(input);
        while (dateMatcher.find()) {
            entities.add(new Entity(dateMatcher.group(), "date"));
        }
        
        Pattern numberPattern = Pattern.compile("\\d+[\\d.]*");
        Matcher numberMatcher = numberPattern.matcher(input);
        while (numberMatcher.find()) {
            entities.add(new Entity(numberMatcher.group(), "number"));
        }
        
        Pattern personPattern = Pattern.compile("[\\u4e00-\\u9fa5]{2,4}(先生|女士|博士|教授|老师)?");
        Matcher personMatcher = personPattern.matcher(input);
        while (personMatcher.find()) {
            entities.add(new Entity(personMatcher.group(), "person"));
        }
        
        return entities;
    }
    
    private Map<String, String> extractSemanticRoles(String input) {
        Map<String, String> roles = new HashMap<>();
        
        if (input.contains("因为")) {
            roles.put("cause", input.substring(input.indexOf("因为") + 2));
        }
        
        if (input.contains("所以") || input.contains("因此")) {
            roles.put("effect", input.substring(input.indexOf("所以") > 0 ? 
                input.indexOf("所以") + 2 : input.indexOf("因此") + 2));
        }
        
        if (input.contains("如果") || input.contains("假如")) {
            roles.put("condition", input.substring(input.indexOf("如果") > 0 ? 
                input.indexOf("如果") + 2 : input.indexOf("假如") + 2));
        }
        
        return roles;
    }
    
    private String classifyIntent(String input) {
        if (questionPattern.matcher(input).find()) {
            return "question";
        }
        if (commandPattern.matcher(input).find()) {
            return "command";
        }
        return "statement";
    }
    
    private String extractContext(String input) {
        if (input.length() < 20) {
            return "short";
        }
        if (input.length() < 50) {
            return "medium";
        }
        return "long";
    }
    
    private int checkCompleteness(InputResult result) {
        int score = 0;
        
        if (result.intent != null && !result.intent.isEmpty()) {
            score += 20;
        }
        
        if (result.entities != null && result.entities.size() > 0) {
            score += 30;
        }
        
        if (result.semanticRoles != null && result.semanticRoles.size() > 0) {
            score += 20;
        }
        
        if (result.context != null) {
            score += 15;
        }
        
        if (result.processedInput.length() >= 5) {
            score += 15;
        }
        
        return score;
    }
    
    private boolean isCompliantRequest(InputResult result) {
        if (result.intent.equals("question") && result.entities.isEmpty()) {
            return false;
        }
        
        return true;
    }
    
    public static class InputResult {
        public boolean passed;
        public String originalInput;
        public String processedInput;
        public Map<String, Object> syntaxTree;
        public List<Entity> entities;
        public Map<String, String> semanticRoles;
        public String intent;
        public String context;
        public int completenessScore;
        public String rejectionReason;
    }
    
    public static class Entity {
        public String value;
        public String type;
        
        public Entity(String value, String type) {
            this.value = value;
            this.type = type;
        }
    }
}

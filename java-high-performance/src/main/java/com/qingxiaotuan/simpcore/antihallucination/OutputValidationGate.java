package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;
import java.util.regex.*;

public class OutputValidationGate {
    private Set<String> hallucinationPatterns;
    private Pattern uncertainPattern;
    private Pattern contradictionPattern;
    private Pattern excessiveClaimPattern;
    
    public OutputValidationGate() {
        this.hallucinationPatterns = new HashSet<>(Arrays.asList(
            "可能", "也许", "大概", "据说", "据称", "有人说",
            "应该是", "似乎", "好像", "仿佛", "感觉", "认为"
        ));
        
        this.uncertainPattern = Pattern.compile(
            "(可能|也许|大概|据说|据称|有人说|应该是|似乎|好像|仿佛)"
        );
        
        this.contradictionPattern = Pattern.compile(
            "(但是|然而|却|可是)[^。；;！!？?]*([不|没有|非|无])"
        );
        
        this.excessiveClaimPattern = Pattern.compile(
            "(所有|全部|完全|绝对|必须|一定|肯定)[^。；;！!？?]{0,10}(是|有|会|能)"
        );
    }
    
    public OutputResult validate(String input, String output) {
        OutputResult result = new OutputResult();
        result.hallucinationDetected = false;
        result.hallucinationScore = 0.0f;
        result.issues = new ArrayList<>();
        
        int hallucinationCount = 0;
        int totalWords = countWords(output);
        
        for (String pattern : hallucinationPatterns) {
            if (output.contains(pattern)) {
                hallucinationCount++;
            }
        }
        
        if (uncertainPattern.matcher(output).find()) {
            hallucinationCount += 2;
            result.issues.add("包含不确定表述");
        }
        
        if (contradictionPattern.matcher(output).find()) {
            hallucinationCount += 3;
            result.issues.add("存在矛盾表述");
        }
        
        if (excessiveClaimPattern.matcher(output).find()) {
            hallucinationCount += 2;
            result.issues.add("包含过度绝对化表述");
        }
        
        boolean isRelevant = checkRelevance(input, output);
        if (!isRelevant) {
            result.hallucinationDetected = true;
            result.hallucinationScore = 0.8f;
            result.issues.add("输出与输入不相关");
            return result;
        }
        
        boolean hasFacts = checkFactuality(output);
        if (!hasFacts && totalWords > 20) {
            result.hallucinationScore = 0.4f;
            result.issues.add("缺少事实依据");
        }
        
        float score = (float) hallucinationCount / Math.max(totalWords, 1);
        result.hallucinationScore = Math.min(score, 1.0f);
        
        if (result.hallucinationScore > 0.3f) {
            result.hallucinationDetected = true;
        }
        
        result.valid = !result.hallucinationDetected;
        return result;
    }
    
    private int countWords(String text) {
        int count = 0;
        boolean inWord = false;
        
        for (char c : text.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                if (!inWord) {
                    count++;
                    inWord = true;
                }
            } else {
                inWord = false;
            }
        }
        
        return count;
    }
    
    private boolean checkRelevance(String input, String output) {
        Set<String> inputWords = extractKeyWords(input);
        Set<String> outputWords = extractKeyWords(output);
        
        if (inputWords.isEmpty()) {
            return true;
        }
        
        int intersection = 0;
        for (String word : inputWords) {
            if (outputWords.contains(word)) {
                intersection++;
            }
        }
        
        return (float) intersection / inputWords.size() >= 0.3f;
    }
    
    private Set<String> extractKeyWords(String text) {
        Set<String> words = new HashSet<>();
        Pattern wordPattern = Pattern.compile("[\\u4e00-\\u9fa5]{2,}|[a-zA-Z]+");
        Matcher matcher = wordPattern.matcher(text);
        
        while (matcher.find()) {
            String word = matcher.group().toLowerCase();
            if (word.length() >= 2) {
                words.add(word);
            }
        }
        
        return words;
    }
    
    private boolean checkFactuality(String text) {
        Pattern datePattern = Pattern.compile("\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}");
        Pattern numberPattern = Pattern.compile("\\d+[\\d.]*[万|亿|%|倍|年|月|日|米|公里|元]");
        Pattern sourcePattern = Pattern.compile("(根据|依据|参考|来自)[^。；;！!？?]{0,20}(报告|研究|数据|文献)");
        
        return datePattern.matcher(text).find() || 
               numberPattern.matcher(text).find() ||
               sourcePattern.matcher(text).find();
    }
    
    public static class OutputResult {
        public boolean valid;
        public boolean hallucinationDetected;
        public float hallucinationScore;
        public List<String> issues;
    }
}

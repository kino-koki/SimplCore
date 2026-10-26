package com.qingxiaotuan.simpcore.search;

import java.util.*;
import java.util.regex.*;

/**
 * 总结引擎 - 从搜索结果中提取关键信息，生成流畅的中文摘要
 * 实现基于关键词频率、语义相关性和信息覆盖度的多维度摘要算法
 */
public class SummarizationEngine {
    private static final int MAX_SUMMARY_LENGTH = 500;
    private static final int MIN_SENTENCE_LENGTH = 10;
    private static final int TOP_KEYWORDS = 15;
    private static final int TOP_SENTENCES = 8;
    
    private Set<String> stopWords;
    
    public SummarizationEngine() {
        initStopWords();
    }
    
    private void initStopWords() {
        stopWords = new HashSet<>(Arrays.asList(
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "自己", "这", "那", "它", "他", "她", "们", "这个", "那个", "什么", "怎么",
            "可以", "能够", "应该", "需要", "可能", "或者", "但是", "因为", "所以", "如果",
            "虽然", "然而", "此外", "另外", "同时", "以及", "对于", "关于", "通过", "利用",
            "进行", "实现", "使用", "采用", "基于", "根据", "按照", "随着", "由于",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "in", "on", "at", "to", "for", "of", "and", "or", "but", "not"
        ));
    }
    
    /**
     * 从搜索结果生成总结性回答
     */
    public SummaryResult summarize(String query, List<WebSearchEngine.SearchResult> results) {
        SummaryResult summary = new SummaryResult();
        summary.query = query;
        summary.sourceCount = results.size();
        
        List<TextWithSource> allTexts = new ArrayList<>();
        for (WebSearchEngine.SearchResult r : results) {
            if (r.snippet != null && !r.snippet.isEmpty()) {
                allTexts.add(new TextWithSource(r.snippet, r.source, r.relevanceScore));
            }
        }
        
        List<String> rawTexts = new ArrayList<>();
        for (TextWithSource t : allTexts) {
            rawTexts.add(t.text);
        }
        
        Map<String, Integer> keywordFreq = extractKeywords(rawTexts, query);
        summary.keywords = new ArrayList<>(keywordFreq.keySet());
        
        List<SentenceInfo> sentences = extractAndScoreSentences(allTexts, query, keywordFreq);
        
        List<SentenceInfo> topSentences = selectTopSentences(sentences, TOP_SENTENCES);
        
        summary.summary = generateSummary(topSentences, query, keywordFreq);
        summary.keyPoints = extractKeyPoints(topSentences, keywordFreq);
        summary.confidence = calculateConfidence(results, sentences);
        summary.coverage = calculateCoverage(keywordFreq, rawTexts);
        
        return summary;
    }
    
    /**
     * 提取关键词及其频率
     */
    private Map<String, Integer> extractKeywords(List<String> texts, String query) {
        Map<String, Integer> freq = new HashMap<>();
        
        for (String text : texts) {
            List<String> words = segmentChinese(text);
            for (String word : words) {
                if (word.length() >= 2 && !stopWords.contains(word)) {
                    freq.put(word, freq.getOrDefault(word, 0) + 1);
                }
            }
        }
        
        List<String> queryWords = segmentChinese(query);
        for (String word : queryWords) {
            if (word.length() >= 2) {
                freq.put(word, freq.getOrDefault(word, 0) + 5);
            }
        }
        
        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(freq.entrySet());
        sorted.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        Map<String, Integer> topKeywords = new LinkedHashMap<>();
        int count = 0;
        for (Map.Entry<String, Integer> entry : sorted) {
            topKeywords.put(entry.getKey(), entry.getValue());
            if (++count >= TOP_KEYWORDS) break;
        }
        
        return topKeywords;
    }
    
    /**
     * 中文分词 - 基于N-gram的简化分词，过滤无意义片段
     */
    private List<String> segmentChinese(String text) {
        List<String> words = new ArrayList<>();
        
        Pattern chinesePattern = Pattern.compile("[\\u4e00-\\u9fa5]+");
        Matcher matcher = chinesePattern.matcher(text);
        
        Set<String> meaninglessNgrams = new HashSet<>(Arrays.asList(
            "什么是", "解释什", "么是", "释什么", "介绍", "总结", "比较", "分析",
            "如何", "为什么", "哪些", "区别", "请介", "请解", "请总", "请分",
            "帮", "帮我", "的吗", "是什么", "样的", "怎么", "怎样"
        ));
        
        while (matcher.find()) {
            String segment = matcher.group();
            
            for (int len = 2; len <= Math.min(4, segment.length()); len++) {
                for (int i = 0; i <= segment.length() - len; i++) {
                    String ngram = segment.substring(i, i + len);
                    if (!stopWords.contains(ngram) && !meaninglessNgrams.contains(ngram) && 
                        !isFragmentOf(ngram, meaninglessNgrams)) {
                        words.add(ngram);
                    }
                }
            }
            
            if (segment.length() == 1 && !stopWords.contains(segment)) {
                words.add(segment);
            }
        }
        
        Pattern englishPattern = Pattern.compile("[a-zA-Z]{2,}");
        Matcher engMatcher = englishPattern.matcher(text);
        while (engMatcher.find()) {
            words.add(engMatcher.group().toLowerCase());
        }
        
        return words;
    }
    
    private boolean isFragmentOf(String ngram, Set<String> meaningless) {
        for (String m : meaningless) {
            if (m.contains(ngram) && ngram.length() < m.length()) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 信息密度评分：包含数字、年份、定义性动词的句子加分
     */
    private float calculateDensityBonus(String sentence) {
        float bonus = 0.5f; // 基础分
        
        // 包含年份（4位数字，1800-2100范围）
        if (sentence.matches(".*1[89]\\d{2}.*") || sentence.matches(".*2[01]\\d{2}.*")) {
            bonus += 0.4f;
        }
        
        // 包含数字（非年份）
        if (sentence.matches(".*\\d+.*")) {
            bonus += 0.1f;
        }
        
        // 包含定义性动词（是、成立、位于、属于、称为、包括、由、来自）
        String[] definitionVerbs = {"是", "成立", "位于", "属于", "称为", "包括", "由", "来自", "发布", "推出"};
        for (String verb : definitionVerbs) {
            if (sentence.contains(verb)) {
                bonus += 0.1f;
                break;
            }
        }
        
        // 包含日期（月/日）
        if (sentence.matches(".*\\d+月\\d+日.*")) {
            bonus += 0.2f;
        }
        
        // 包含专业术语（科技、技术、系统、网络、公司、企业等）
        String[] technicalTerms = {"科技", "技术", "系统", "网络", "公司", "企业", "产品", "服务", "研发", "创新"};
        int termCount = 0;
        for (String term : technicalTerms) {
            if (sentence.contains(term)) {
                termCount++;
            }
        }
        bonus += Math.min(termCount * 0.1f, 0.3f);
        
        // 限制最大加分
        return Math.min(bonus, 1.0f);
    }
    
    /**
     * 提取并评分句子 - 考虑来源权重
     */
    private List<SentenceInfo> extractAndScoreSentences(List<TextWithSource> texts, String query,
            Map<String, Integer> keywords) {
        List<SentenceInfo> sentences = new ArrayList<>();
        int textIdx = 0;
        
        for (TextWithSource tws : texts) {
            String[] splitSentences = tws.text.split("[。！？.!?\n;；]");
            
            for (String sent : splitSentences) {
                sent = sent.trim();
                if (sent.length() < MIN_SENTENCE_LENGTH) continue;
                
                SentenceInfo info = new SentenceInfo();
                info.text = sent;
                info.sourceIdx = textIdx;
                
                info.keywordScore = calculateKeywordScore(sent, keywords);
                info.queryScore = calculateQueryScore(sent, query);
                info.lengthScore = calculateLengthScore(sent);
                info.positionScore = 1.0f - (float) textIdx / Math.max(texts.size(), 1);
                
                float sourceWeight = 1.0f;
                if (tws.source.equals("补充生成")) {
                    sourceWeight = 0.3f;
                } else if (tws.source.contains("百度百科")) {
                    sourceWeight = 1.4f;
                } else if (tws.source.contains("本地知识库")) {
                    sourceWeight = 1.2f;
                } else if (tws.source.contains("Wikipedia")) {
                    sourceWeight = 1.3f;
                }
                
                // 信息密度评分：包含数字、年份、定义性动词的句子加分
                float densityBonus = calculateDensityBonus(sent);
                
                info.totalScore = (info.keywordScore * 0.35f + 
                                  info.queryScore * 0.3f + 
                                  info.lengthScore * 0.15f + 
                                  info.positionScore * 0.1f +
                                  densityBonus * 0.1f) * sourceWeight;
                
                sentences.add(info);
            }
            textIdx++;
        }
        
        return sentences;
    }
    
    private float calculateKeywordScore(String sentence, Map<String, Integer> keywords) {
        float score = 0;
        for (Map.Entry<String, Integer> entry : keywords.entrySet()) {
            if (sentence.contains(entry.getKey())) {
                score += entry.getValue();
            }
        }
        return score / Math.max(sentence.length() / 10, 1);
    }
    
    private float calculateQueryScore(String sentence, String query) {
        List<String> queryWords = segmentChinese(query);
        int matchCount = 0;
        for (String word : queryWords) {
            if (word.length() >= 2 && sentence.contains(word)) {
                matchCount++;
            }
        }
        return (float) matchCount / Math.max(queryWords.size(), 1);
    }
    
    private float calculateLengthScore(String sentence) {
        int len = sentence.length();
        if (len < 15) return 0.3f;
        if (len < 30) return 0.6f;
        if (len < 60) return 1.0f;
        if (len < 100) return 0.8f;
        return 0.5f;
    }
    
    /**
     * 选择得分最高的句子，保证多样性
     */
    private List<SentenceInfo> selectTopSentences(List<SentenceInfo> sentences, int topN) {
        sentences.sort((a, b) -> Float.compare(b.totalScore, a.totalScore));
        
        List<SentenceInfo> selected = new ArrayList<>();
        Set<String> usedNgrams = new HashSet<>();
        
        for (SentenceInfo sent : sentences) {
            if (selected.size() >= topN) break;
            
            boolean duplicate = false;
            List<String> ngrams = segmentChinese(sent.text);
            int overlap = 0;
            for (String ngram : ngrams) {
                if (ngram.length() >= 3 && usedNgrams.contains(ngram)) {
                    overlap++;
                }
            }
            
            if (overlap > ngrams.size() * 0.5) {
                duplicate = true;
            }
            
            if (!duplicate) {
                selected.add(sent);
                for (String ngram : ngrams) {
                    if (ngram.length() >= 3) {
                        usedNgrams.add(ngram);
                    }
                }
            }
        }
        
        selected.sort(Comparator.comparingInt(s -> s.sourceIdx));
        return selected;
    }
    
    /**
     * 生成流畅的中文摘要
     */
    private String generateSummary(List<SentenceInfo> sentences, String query, 
            Map<String, Integer> keywords) {
        StringBuilder sb = new StringBuilder();
        
        if (sentences.isEmpty()) {
            return "未能从搜索结果中提取到有效的总结信息。";
        }
        
        sb.append("\n");
        
        for (int i = 0; i < sentences.size(); i++) {
            String sent = sentences.get(i).text;
            
            if (i == 0) {
                sb.append(sent);
                if (!sent.endsWith("。") && !sent.endsWith("!") && !sent.endsWith("?")) {
                    sb.append("。");
                }
                sb.append("\n\n");
            } else {
                sb.append(sent);
                if (!sent.endsWith("。") && !sent.endsWith("!") && !sent.endsWith("?")) {
                    sb.append("。");
                }
                sb.append(" ");
            }
        }
        
        if (sb.length() > MAX_SUMMARY_LENGTH) {
            sb.setLength(MAX_SUMMARY_LENGTH);
            int lastPeriod = sb.lastIndexOf("。");
            if (lastPeriod > MAX_SUMMARY_LENGTH * 0.7) {
                sb.setLength(lastPeriod + 1);
            }
        }
        
        // 去掉模板化结论句，使回答更自然
        return sb.toString();
    }
    
    private String generateConclusion(String query, Map<String, Integer> keywords) {
        StringBuilder sb = new StringBuilder();
        
        List<String> topKeywords = new ArrayList<>(keywords.keySet());
        int keywordCount = Math.min(3, topKeywords.size());
        
        if (keywordCount > 0) {
            sb.append("该主题主要涉及");
            for (int i = 0; i < keywordCount; i++) {
                if (i > 0) {
                    sb.append(i == keywordCount - 1 ? "以及" : "、");
                }
                sb.append(topKeywords.get(i));
            }
            sb.append("等方面。");
        }
        
        sb.append("建议结合具体应用场景，参考多方资料进行深入分析。");
        
        return sb.toString();
    }
    
    /**
     * 提取关键要点
     */
    private List<String> extractKeyPoints(List<SentenceInfo> sentences, 
            Map<String, Integer> keywords) {
        List<String> points = new ArrayList<>();
        
        for (SentenceInfo sent : sentences) {
            String text = sent.text;
            if (text.length() > 20) {
                String point = text;
                if (text.length() > 80) {
                    int cutPos = text.indexOf("，", 60);
                    if (cutPos > 0) {
                        point = text.substring(0, cutPos) + "...";
                    } else {
                        point = text.substring(0, 80) + "...";
                    }
                }
                points.add(point);
            }
        }
        
        return points.size() > 5 ? points.subList(0, 5) : points;
    }
    
    /**
     * 计算置信度
     */
    private float calculateConfidence(List<WebSearchEngine.SearchResult> results, 
            List<SentenceInfo> sentences) {
        if (results.isEmpty()) return 0.0f;
        
        float sourceDiversity = 0;
        Set<String> sources = new HashSet<>();
        for (WebSearchEngine.SearchResult r : results) {
            sources.add(r.source);
        }
        sourceDiversity = Math.min(1.0f, sources.size() / 4.0f);
        
        float avgRelevance = 0;
        for (WebSearchEngine.SearchResult r : results) {
            avgRelevance += r.relevanceScore;
        }
        avgRelevance /= results.size();
        
        float sentenceQuality = 0;
        if (!sentences.isEmpty()) {
            for (SentenceInfo s : sentences) {
                sentenceQuality += s.totalScore;
            }
            sentenceQuality /= sentences.size();
        }
        
        return sourceDiversity * 0.3f + avgRelevance * 0.4f + Math.min(1.0f, sentenceQuality) * 0.3f;
    }
    
    /**
     * 计算信息覆盖率
     */
    private float calculateCoverage(Map<String, Integer> keywords, List<String> texts) {
        if (keywords.isEmpty() || texts.isEmpty()) return 0.0f;
        
        int coveredTexts = 0;
        List<String> topKeywords = new ArrayList<>(keywords.keySet());
        
        for (String text : texts) {
            int matchCount = 0;
            for (String keyword : topKeywords) {
                if (text.contains(keyword)) {
                    matchCount++;
                }
            }
            if (matchCount >= topKeywords.size() * 0.3) {
                coveredTexts++;
            }
        }
        
        return (float) coveredTexts / texts.size();
    }
    
    // ============ 数据结构 ============
    
    private static class SentenceInfo {
        String text;
        int sourceIdx;
        float keywordScore;
        float queryScore;
        float lengthScore;
        float positionScore;
        float totalScore;
    }
    
    private static class TextWithSource {
        String text;
        String source;
        float relevance;
        
        TextWithSource(String text, String source, float relevance) {
            this.text = text;
            this.source = source;
            this.relevance = relevance;
        }
    }
    
    public static class SummaryResult {
        public String query;
        public String summary;
        public List<String> keyPoints;
        public List<String> keywords;
        public int sourceCount;
        public float confidence;
        public float coverage;
        
        public String getReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("═══════════════════════════════════════════════════\n");
            sb.append("  总结报告\n");
            sb.append("═══════════════════════════════════════════════════\n");
            sb.append(String.format("查询: %s\n", query));
            sb.append(String.format("信息来源数: %d\n", sourceCount));
            sb.append(String.format("置信度: %.1f%%\n", confidence * 100));
            sb.append(String.format("覆盖率: %.1f%%\n", coverage * 100));
            sb.append(String.format("关键词: %s\n\n", String.join("、", keywords)));
            sb.append("─── 总结内容 ───\n");
            sb.append(summary).append("\n\n");
            sb.append("─── 关键要点 ───\n");
            for (int i = 0; i < keyPoints.size(); i++) {
                sb.append(String.format("%d. %s\n", i + 1, keyPoints.get(i)));
            }
            sb.append("═══════════════════════════════════════════════════\n");
            return sb.toString();
        }
    }
}

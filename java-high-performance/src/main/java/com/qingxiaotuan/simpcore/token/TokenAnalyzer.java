package com.qingxiaotuan.simpcore.token;

import com.qingxiaotuan.simpcore.model.Tokenizer;
import java.util.*;

public class TokenAnalyzer {

    public static class TokenHeatmap {
        public Map<String, Integer> tokenFrequency = new HashMap<>();
        public Map<String, Double> tokenInformationContent = new HashMap<>();
        public Map<String, String> tokenCategory = new HashMap<>();
        public int totalTokens;
        public int uniqueTokens;
        public double averageTokenLength;
        public double entropy;

        public void print() {
            System.out.println("=== Token Heatmap ===");
            System.out.println("Total tokens: " + totalTokens);
            System.out.println("Unique tokens: " + uniqueTokens);
            System.out.println("Average token length: " + String.format("%.2f", averageTokenLength));
            System.out.println("Entropy: " + String.format("%.2f", entropy));
            System.out.println("Top 20 frequent tokens:");
            tokenFrequency.entrySet().stream()
                    .sorted((a, b) -> b.getValue().compareTo(a.getValue()))
                    .limit(20)
                    .forEach(e -> System.out.println("  " + e.getKey() + ": " + e.getValue() + " (" + tokenCategory.get(e.getKey()) + ")"));
        }
    }

    public static class BottleneckReport {
        public List<String> bottlenecks = new ArrayList<>();
        public Map<String, Double> consumptionRatio = new HashMap<>();
        public Map<String, String> severity = new HashMap<>();

        public void addBottleneck(String stage, double ratio, String level) {
            bottlenecks.add(stage);
            consumptionRatio.put(stage, ratio);
            severity.put(stage, level);
        }

        public void print() {
            System.out.println("=== Token Consumption Bottleneck Report ===");
            bottlenecks.stream()
                    .sorted((a, b) -> Double.compare(consumptionRatio.get(b), consumptionRatio.get(a)))
                    .forEach(stage -> System.out.println("  [" + severity.get(stage) + "] " + stage + ": " + String.format("%.1f%%", consumptionRatio.get(stage) * 100)));
        }
    }

    private static final Set<String> FUNCTIONAL_TOKENS = Set.of(" ", ".", ",", "!", "?", ":", ";", "-", "_", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", "\"", "'", "\\", "/", "<", ">", "=", "+", "~", "`", "|");
    private static final Set<String> LOW_INFO_TOKENS = Set.of("the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very");

    public static TokenHeatmap analyzeTokens(int[] tokenIds, Tokenizer tokenizer) {
        TokenHeatmap heatmap = new TokenHeatmap();
        Map<String, Integer> freq = new HashMap<>();
        int totalLength = 0;

        for (int id : tokenIds) {
            String token = tokenizer.idToToken.get(id);
            if (token == null) continue;
            freq.put(token, freq.getOrDefault(token, 0) + 1);
            totalLength += token.length();
        }

        heatmap.tokenFrequency = freq;
        heatmap.totalTokens = tokenIds.length;
        heatmap.uniqueTokens = freq.size();
        heatmap.averageTokenLength = (double) totalLength / tokenIds.length;

        double entropy = 0;
        for (int count : freq.values()) {
            double prob = (double) count / tokenIds.length;
            entropy -= prob * Math.log(prob) / Math.log(2);
        }
        heatmap.entropy = entropy;

        for (String token : freq.keySet()) {
            if (token.equals("<bos>") || token.equals("<eos>") || token.equals("<pad>") || token.equals("<unk>")) {
                heatmap.tokenCategory.put(token, "SPECIAL");
            } else if (FUNCTIONAL_TOKENS.contains(token)) {
                heatmap.tokenCategory.put(token, "FUNCTIONAL");
            } else if (LOW_INFO_TOKENS.contains(token.toLowerCase())) {
                heatmap.tokenCategory.put(token, "LOW_INFO");
            } else if (token.matches("\\d+")) {
                heatmap.tokenCategory.put(token, "NUMERIC");
            } else if (token.length() == 1 && Character.isLetter(token.charAt(0))) {
                heatmap.tokenCategory.put(token, "SINGLE_CHAR");
            } else {
                heatmap.tokenCategory.put(token, "CONTENT");
            }
        }

        for (String token : freq.keySet()) {
            int freqCount = freq.get(token);
            double prob = (double) freqCount / tokenIds.length;
            double infoContent = -Math.log(prob) / Math.log(2);
            heatmap.tokenInformationContent.put(token, infoContent);
        }

        return heatmap;
    }

    public static BottleneckReport analyzeTokenFlow(String text, Tokenizer tokenizer, int maxSequenceLength) {
        BottleneckReport report = new BottleneckReport();

        int[] tokens = tokenizer.encode(text);
        int totalTokens = tokens.length;

        int preprocessingOverhead = 2;
        double preprocessingRatio = (double) preprocessingOverhead / totalTokens;
        report.addBottleneck("Preprocessing (BOS/EOS tokens)", preprocessingRatio, "LOW");

        int paddingNeeded = Math.max(0, maxSequenceLength - totalTokens);
        double paddingRatio = (double) paddingNeeded / Math.max(maxSequenceLength, totalTokens);
        String paddingSeverity = paddingRatio > 0.5 ? "HIGH" : paddingRatio > 0.2 ? "MEDIUM" : "LOW";
        report.addBottleneck("Padding", paddingRatio, paddingSeverity);

        int truncationLoss = Math.max(0, totalTokens - maxSequenceLength);
        double truncationRatio = (double) truncationLoss / totalTokens;
        String truncationSeverity = truncationRatio > 0.3 ? "HIGH" : truncationRatio > 0.1 ? "MEDIUM" : "LOW";
        report.addBottleneck("Truncation loss", truncationRatio, truncationSeverity);

        TokenHeatmap heatmap = analyzeTokens(tokens, tokenizer);
        int lowInfoCount = 0;
        for (String token : heatmap.tokenFrequency.keySet()) {
            if ("LOW_INFO".equals(heatmap.tokenCategory.get(token))) {
                lowInfoCount += heatmap.tokenFrequency.get(token);
            }
        }
        double lowInfoRatio = (double) lowInfoCount / totalTokens;
        String lowInfoSeverity = lowInfoRatio > 0.4 ? "HIGH" : lowInfoRatio > 0.2 ? "MEDIUM" : "LOW";
        report.addBottleneck("Low-information tokens", lowInfoRatio, lowInfoSeverity);

        int functionalCount = 0;
        for (String token : heatmap.tokenFrequency.keySet()) {
            if ("FUNCTIONAL".equals(heatmap.tokenCategory.get(token))) {
                functionalCount += heatmap.tokenFrequency.get(token);
            }
        }
        double functionalRatio = (double) functionalCount / totalTokens;
        String functionalSeverity = functionalRatio > 0.3 ? "HIGH" : functionalRatio > 0.15 ? "MEDIUM" : "LOW";
        report.addBottleneck("Functional tokens", functionalRatio, functionalSeverity);

        int singleCharCount = 0;
        for (String token : heatmap.tokenFrequency.keySet()) {
            if ("SINGLE_CHAR".equals(heatmap.tokenCategory.get(token))) {
                singleCharCount += heatmap.tokenFrequency.get(token);
            }
        }
        double singleCharRatio = (double) singleCharCount / totalTokens;
        String singleCharSeverity = singleCharRatio > 0.3 ? "HIGH" : singleCharRatio > 0.15 ? "MEDIUM" : "LOW";
        report.addBottleneck("Single-character tokens", singleCharRatio, singleCharSeverity);

        return report;
    }

    public static double calculateCompressionPotential(String text, Tokenizer tokenizer) {
        TokenHeatmap heatmap = analyzeTokens(tokenizer.encode(text), tokenizer);

        double lowInfoRatio = 0;
        double functionalRatio = 0;
        double singleCharRatio = 0;

        int total = heatmap.totalTokens;
        for (String token : heatmap.tokenFrequency.keySet()) {
            int count = heatmap.tokenFrequency.get(token);
            String category = heatmap.tokenCategory.get(token);
            if ("LOW_INFO".equals(category)) {
                lowInfoRatio += (double) count / total;
            } else if ("FUNCTIONAL".equals(category)) {
                functionalRatio += (double) count / total;
            } else if ("SINGLE_CHAR".equals(category)) {
                singleCharRatio += (double) count / total;
            }
        }

        double potential = lowInfoRatio * 0.8 + functionalRatio * 0.5 + singleCharRatio * 0.6;
        return Math.min(0.7, potential);
    }
}
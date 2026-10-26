package com.qingxiaotuan.simpcore.inference;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

public class InferenceEngine {
    
    public interface StreamCallback {
        void onToken(String token);
        void onComplete();
        void onError(String error);
    }
    
    public static class GenerationResult {
        public final String text;
        public final int tokenCount;
        public final long latencyMs;
        public final float perplexity;
        public final boolean fromCache;
        
        public GenerationResult(String text, int tokenCount, long latencyMs, float perplexity, boolean fromCache) {
            this.text = text;
            this.tokenCount = tokenCount;
            this.latencyMs = latencyMs;
            this.perplexity = perplexity;
            this.fromCache = fromCache;
        }
    }
    
    public static class BatchResult {
        public final List<GenerationResult> results;
        public final long totalLatencyMs;
        public final int totalTokens;
        
        public BatchResult(List<GenerationResult> results, long totalLatencyMs, int totalTokens) {
            this.results = results;
            this.totalLatencyMs = totalLatencyMs;
            this.totalTokens = totalTokens;
        }
    }
    
    private final ConcurrentHashMap<String, GenerationResult> cache;
    private final int cacheSize;
    private final ExecutorService executor;
    private volatile int maxBatchSize;
    private volatile boolean cacheEnabled;
    private volatile int maxSeqLen;
    
    public InferenceEngine() {
        this.cache = new ConcurrentHashMap<>();
        this.cacheSize = 10000;
        this.executor = Executors.newFixedThreadPool(8);
        this.maxBatchSize = 32;
        this.cacheEnabled = true;
        this.maxSeqLen = 2048;
    }
    
    public GenerationResult generate(String prompt, int maxTokens) {
        long start = System.currentTimeMillis();
        
        if (cacheEnabled) {
            GenerationResult cached = cache.get(prompt);
            if (cached != null && cached.tokenCount >= maxTokens) {
                return new GenerationResult(
                    cached.text.substring(0, Math.min(cached.text.length(), maxTokens * 2)),
                    maxTokens,
                    System.currentTimeMillis() - start,
                    cached.perplexity,
                    true
                );
            }
        }
        
        String text = generateText(prompt, maxTokens);
        float perplexity = calculatePerplexity(prompt, text);
        
        GenerationResult result = new GenerationResult(
            text,
            text.length() / 2,
            System.currentTimeMillis() - start,
            perplexity,
            false
        );
        
        if (cacheEnabled) {
            addToCache(prompt, result);
        }
        
        return result;
    }
    
    public void generateStream(String prompt, int maxTokens, StreamCallback callback) {
        executor.submit(() -> {
            try {
                String text = generateText(prompt, maxTokens);
                
                for (int i = 0; i < text.length(); i++) {
                    char c = text.charAt(i);
                    callback.onToken(String.valueOf(c));
                    Thread.sleep(10);
                }
                
                callback.onComplete();
            } catch (Exception e) {
                callback.onError(e.getMessage());
            }
        });
    }
    
    public BatchResult generateBatch(List<String> prompts, int maxTokens) {
        long start = System.currentTimeMillis();
        List<GenerationResult> results = new ArrayList<>();
        int totalTokens = 0;
        
        List<List<String>> batches = partition(prompts, maxBatchSize);
        
        for (List<String> batch : batches) {
            List<Future<GenerationResult>> futures = new ArrayList<>();
            
            for (String prompt : batch) {
                futures.add(executor.submit(() -> generate(prompt, maxTokens)));
            }
            
            for (Future<GenerationResult> future : futures) {
                try {
                    GenerationResult result = future.get(60, TimeUnit.SECONDS);
                    results.add(result);
                    totalTokens += result.tokenCount;
                } catch (Exception e) {
                    results.add(new GenerationResult("", 0, 0, Float.MAX_VALUE, false));
                }
            }
        }
        
        return new BatchResult(results, System.currentTimeMillis() - start, totalTokens);
    }
    
    private String generateText(String prompt, int maxTokens) {
        StringBuilder sb = new StringBuilder(prompt);
        
        for (int i = 0; i < maxTokens; i++) {
            char nextChar = (char) ('a' + (int) (Math.random() * 26));
            if (Math.random() < 0.1) nextChar = ' ';
            if (Math.random() < 0.05) nextChar = '.';
            sb.append(nextChar);
        }
        
        return sb.toString();
    }
    
    private float calculatePerplexity(String prompt, String text) {
        return 1.0f + (float) Math.random() * 2.0f;
    }
    
    private void addToCache(String prompt, GenerationResult result) {
        if (cache.size() >= cacheSize) {
            String oldestKey = cache.keySet().iterator().next();
            cache.remove(oldestKey);
        }
        cache.put(prompt, result);
    }
    
    public void clearCache() {
        cache.clear();
    }
    
    public int getCacheSize() {
        return cache.size();
    }
    
    public void setCacheEnabled(boolean enabled) {
        this.cacheEnabled = enabled;
    }
    
    public boolean isCacheEnabled() {
        return cacheEnabled;
    }
    
    public void setMaxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
    }
    
    public int getMaxSeqLen() {
        return maxSeqLen;
    }
    
    public void setMaxBatchSize(int size) {
        this.maxBatchSize = size;
    }
    
    public int getMaxBatchSize() {
        return maxBatchSize;
    }
    
    public void shutdown() {
        executor.shutdown();
    }
    
    private <T> List<List<T>> partition(List<T> list, int size) {
        List<List<T>> partitions = new ArrayList<>();
        for (int i = 0; i < list.size(); i += size) {
            partitions.add(list.subList(i, Math.min(i + size, list.size())));
        }
        return partitions;
    }
    
    public String getStatus() {
        return String.format(
            "InferenceEngine: cache=%d/%d, cacheEnabled=%s, maxBatch=%d, maxSeq=%d",
            cache.size(), cacheSize, cacheEnabled, maxBatchSize, maxSeqLen
        );
    }
}
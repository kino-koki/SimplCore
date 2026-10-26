package com.qingxiaotuan.simpcore.context;

import java.util.*;
import java.util.concurrent.*;

public class ContextWindowManager {
    
    public static class ContextEntry {
        public final String content;
        public final long timestamp;
        public final float importance;
        public final String type;
        
        public ContextEntry(String content, float importance, String type) {
            this.content = content;
            this.timestamp = System.currentTimeMillis();
            this.importance = importance;
            this.type = type;
        }
    }
    
    public static class ContextSnapshot {
        public final List<ContextEntry> entries;
        public final int totalTokens;
        public final long timestamp;
        
        public ContextSnapshot(List<ContextEntry> entries, int totalTokens) {
            this.entries = entries;
            this.totalTokens = totalTokens;
            this.timestamp = System.currentTimeMillis();
        }
    }
    
    private final Deque<ContextEntry> contextHistory;
    private final int maxWindowSize;
    private final int maxTokens;
    private volatile int currentTokens;
    private final ScheduledExecutorService scheduler;
    private volatile boolean enabled;
    
    public ContextWindowManager(int maxWindowSize, int maxTokens) {
        this.maxWindowSize = maxWindowSize;
        this.maxTokens = maxTokens;
        this.contextHistory = new LinkedList<>();
        this.currentTokens = 0;
        this.scheduler = Executors.newScheduledThreadPool(1);
        this.enabled = true;
        startCleanupTask();
    }
    
    public ContextWindowManager() {
        this(100, 8192);
    }
    
    private void startCleanupTask() {
        scheduler.scheduleAtFixedRate(() -> {
            if (enabled) {
                cleanupOldContext();
            }
        }, 60, 60, TimeUnit.SECONDS);
    }
    
    public void addContext(String content, String type) {
        addContext(content, 0.5f, type);
    }
    
    public void addContext(String content, float importance, String type) {
        if (!enabled) return;
        
        ContextEntry entry = new ContextEntry(content, importance, type);
        
        synchronized (contextHistory) {
            contextHistory.addLast(entry);
            currentTokens += content.length() / 2;
            
            while (contextHistory.size() > maxWindowSize || currentTokens > maxTokens) {
                ContextEntry removed = contextHistory.pollFirst();
                if (removed != null) {
                    currentTokens -= removed.content.length() / 2;
                }
            }
        }
    }
    
    public ContextSnapshot getContext(int maxEntries) {
        synchronized (contextHistory) {
            List<ContextEntry> entries = new ArrayList<>();
            int tokens = 0;
            
            for (ContextEntry entry : contextHistory) {
                if (entries.size() >= maxEntries) break;
                entries.add(entry);
                tokens += entry.content.length() / 2;
            }
            
            return new ContextSnapshot(entries, tokens);
        }
    }
    
    public ContextSnapshot getRelevantContext(String query, int maxEntries) {
        synchronized (contextHistory) {
            List<Map.Entry<ContextEntry, Float>> scored = new ArrayList<>();
            
            for (ContextEntry entry : contextHistory) {
                float similarity = calculateSimilarity(query, entry.content);
                float score = similarity * entry.importance;
                if (score > 0.1f) {
                    scored.add(new AbstractMap.SimpleEntry<>(entry, score));
                }
            }
            
            scored.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
            
            List<ContextEntry> entries = new ArrayList<>();
            int tokens = 0;
            
            for (int i = 0; i < Math.min(maxEntries, scored.size()); i++) {
                ContextEntry entry = scored.get(i).getKey();
                entries.add(entry);
                tokens += entry.content.length() / 2;
            }
            
            return new ContextSnapshot(entries, tokens);
        }
    }
    
    private float calculateSimilarity(String query, String content) {
        String q = query.toLowerCase();
        String c = content.toLowerCase();
        
        if (q.isEmpty() || c.isEmpty()) return 0;
        
        Set<String> qWords = new HashSet<>(Arrays.asList(q.split("\\s+")));
        Set<String> cWords = new HashSet<>(Arrays.asList(c.split("\\s+")));
        
        if (qWords.isEmpty()) return 0;
        
        Set<String> intersection = new HashSet<>(qWords);
        intersection.retainAll(cWords);
        
        return (float) intersection.size() / qWords.size();
    }
    
    private void cleanupOldContext() {
        long threshold = System.currentTimeMillis() - 24 * 60 * 60 * 1000L;
        
        synchronized (contextHistory) {
            contextHistory.removeIf(entry -> entry.timestamp < threshold && entry.importance < 0.7f);
        }
    }
    
    public void clearContext() {
        synchronized (contextHistory) {
            contextHistory.clear();
            currentTokens = 0;
        }
    }
    
    public void setWindowSize(int size) {
        synchronized (contextHistory) {
            while (contextHistory.size() > size) {
                ContextEntry removed = contextHistory.pollFirst();
                if (removed != null) {
                    currentTokens -= removed.content.length() / 2;
                }
            }
        }
    }
    
    public int getContextSize() {
        synchronized (contextHistory) {
            return contextHistory.size();
        }
    }
    
    public int getCurrentTokens() {
        return currentTokens;
    }
    
    public int getMaxTokens() {
        return maxTokens;
    }
    
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    public boolean isEnabled() {
        return enabled;
    }
    
    public void shutdown() {
        scheduler.shutdown();
    }
    
    public String getStatus() {
        return String.format(
            "ContextWindow: entries=%d, tokens=%d/%d, enabled=%s",
            getContextSize(), currentTokens, maxTokens, enabled
        );
    }
}
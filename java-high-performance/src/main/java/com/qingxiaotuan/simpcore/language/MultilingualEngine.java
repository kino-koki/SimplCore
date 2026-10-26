package com.qingxiaotuan.simpcore.language;

import java.util.*;
import java.util.concurrent.*;

public class MultilingualEngine {
    
    public static class TranslationResult {
        public final String translatedText;
        public final String sourceLang;
        public final String targetLang;
        public final float confidence;
        public final long processingTimeMs;
        
        public TranslationResult(String translatedText, String sourceLang, String targetLang, 
                                float confidence, long processingTimeMs) {
            this.translatedText = translatedText;
            this.sourceLang = sourceLang;
            this.targetLang = targetLang;
            this.confidence = confidence;
            this.processingTimeMs = processingTimeMs;
        }
    }
    
    public static class LanguageDetectionResult {
        public final String language;
        public final float confidence;
        public final List<String> alternatives;
        
        public LanguageDetectionResult(String language, float confidence, List<String> alternatives) {
            this.language = language;
            this.confidence = confidence;
            this.alternatives = alternatives;
        }
    }
    
    private final Map<String, Set<String>> languagePairs;
    private final Map<String, String> langCodes;
    private final ExecutorService executor;
    private volatile boolean enabled;
    
    public MultilingualEngine() {
        this.languagePairs = new ConcurrentHashMap<>();
        this.langCodes = new HashMap<>();
        this.executor = Executors.newFixedThreadPool(4);
        this.enabled = true;
        initializeLanguages();
    }
    
    private void initializeLanguages() {
        langCodes.put("zh", "中文");
        langCodes.put("en", "英文");
        langCodes.put("ja", "日文");
        langCodes.put("ko", "韩文");
        langCodes.put("fr", "法文");
        langCodes.put("de", "德文");
        langCodes.put("es", "西班牙文");
        langCodes.put("ru", "俄文");
        langCodes.put("ar", "阿拉伯文");
        langCodes.put("pt", "葡萄牙文");
        
        Set<String> allLangs = new HashSet<>(langCodes.keySet());
        for (String lang : langCodes.keySet()) {
            languagePairs.put(lang, allLangs);
        }
    }
    
    public LanguageDetectionResult detectLanguage(String text) {
        long start = System.currentTimeMillis();
        
        int[] charCounts = new int[langCodes.size()];
        int idx = 0;
        for (String lang : langCodes.keySet()) {
            for (char c : text.toCharArray()) {
                if (isLangChar(c, lang)) {
                    charCounts[idx]++;
                }
            }
            idx++;
        }
        
        int maxIdx = 0;
        int maxCount = 0;
        for (int i = 0; i < charCounts.length; i++) {
            if (charCounts[i] > maxCount) {
                maxCount = charCounts[i];
                maxIdx = i;
            }
        }
        
        String detectedLang = new ArrayList<>(langCodes.keySet()).get(maxIdx);
        float confidence = maxCount > 0 ? Math.min(1.0f, (float) maxCount / text.length() + 0.2f) : 0.5f;
        
        List<String> alternatives = new ArrayList<>();
        for (int i = 0; i < charCounts.length; i++) {
            if (i != maxIdx && charCounts[i] > 0) {
                alternatives.add(new ArrayList<>(langCodes.keySet()).get(i));
            }
        }
        
        return new LanguageDetectionResult(detectedLang, confidence, alternatives);
    }
    
    private boolean isLangChar(char c, String lang) {
        switch (lang) {
            case "zh": return c >= '\u4e00' && c <= '\u9fff' && !(c >= '\u3040' && c <= '\u30ff');
            case "ja": return c >= '\u3040' && c <= '\u30ff';
            case "ko": return c >= '\uac00' && c <= '\ud7af';
            case "en": return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
            case "fr": return c == 'é' || c == 'è' || c == 'ê' || c == 'à' || c == 'ç';
            case "de": return c == 'ä' || c == 'ö' || c == 'ü' || c == 'ß';
            case "es": return c == 'ñ' || c == 'á' || c == 'é' || c == 'í' || c == 'ó' || c == 'ú';
            case "ru": return c >= '\u0400' && c <= '\u04ff';
            case "ar": return c >= '\u0600' && c <= '\u06ff';
            case "pt": return c == 'ç' || c == 'ã' || c == 'õ';
            default: return Character.isLetter(c);
        }
    }
    
    public TranslationResult translate(String text, String targetLang) {
        return translate(text, null, targetLang);
    }
    
    public TranslationResult translate(String text, String sourceLang, String targetLang) {
        long start = System.currentTimeMillis();
        
        if (sourceLang == null) {
            sourceLang = detectLanguage(text).language;
        }
        
        String translated = performTranslation(text, sourceLang, targetLang);
        float confidence = 0.85f + (float) Math.random() * 0.15f;
        
        return new TranslationResult(translated, sourceLang, targetLang, confidence, System.currentTimeMillis() - start);
    }
    
    public Future<TranslationResult> translateAsync(String text, String targetLang) {
        return executor.submit(() -> translate(text, targetLang));
    }
    
    private String performTranslation(String text, String sourceLang, String targetLang) {
        if (sourceLang.equals(targetLang)) {
            return text;
        }
        
        Map<String, Map<String, String>> translations = new HashMap<>();
        translations.put("zh", new HashMap<>());
        translations.get("zh").put("hello", "你好");
        translations.get("zh").put("world", "世界");
        translations.get("zh").put("AI", "人工智能");
        translations.get("zh").put("learning", "学习");
        
        translations.put("en", new HashMap<>());
        translations.get("en").put("你好", "Hello");
        translations.get("en").put("世界", "World");
        translations.get("en").put("人工智能", "AI");
        translations.get("en").put("学习", "Learning");
        
        translations.put("ja", new HashMap<>());
        translations.get("ja").put("hello", "こんにちは");
        translations.get("ja").put("world", "世界");
        
        translations.put("ko", new HashMap<>());
        translations.get("ko").put("hello", "안녕하세요");
        translations.get("ko").put("world", "세계");
        
        Map<String, String> langMap = translations.get(targetLang);
        if (langMap != null) {
            for (Map.Entry<String, String> entry : langMap.entrySet()) {
                text = text.replace(entry.getKey(), entry.getValue());
            }
        }
        
        return "[" + langCodes.get(targetLang) + "] " + text;
    }
    
    public List<String> getSupportedLanguages() {
        return new ArrayList<>(langCodes.values());
    }
    
    public boolean isLanguageSupported(String lang) {
        return langCodes.containsKey(lang) || langCodes.containsValue(lang);
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
            "MultilingualEngine: enabled=%s, languages=%d",
            enabled, langCodes.size()
        );
    }
}
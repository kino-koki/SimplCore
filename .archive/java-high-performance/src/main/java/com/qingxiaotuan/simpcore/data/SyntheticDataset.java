package com.qingxiaotuan.simpcore.data;

import java.util.*;

public class SyntheticDataset {
    
    private static final String[] COMMON_WORDS = {
        "the", "and", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "even", "still", "also", "almost", "enough", "already",
        "hardly", "nearly", "quite", "rather", "really", "soon", "yet",
        "about", "across", "against", "along", "among", "around", "behind",
        "beside", "beyond", "but", "down", "except", "inside", "like",
        "near", "off", "out", "over", "past", "since", "toward", "under",
        "until", "up", "upon", "while", "within", "without", "a", "an",
        "this", "that", "these", "those", "it", "its", "he", "him", "his",
        "she", "her", "they", "them", "we", "us", "you", "your", "my",
        "me", "I", "what", "which", "who", "whom", "whose", "whatever",
        "whichever", "whoever", "whomever", "something", "anything", "nothing",
        "everything", "someone", "anyone", "no one", "everyone", "somebody",
        "anybody", "nobody", "everybody"
    };
    
    private static final String[] NOUNS = {
        "cat", "dog", "bird", "fish", "tree", "flower", "house", "car",
        "computer", "book", "table", "chair", "phone", "water", "food",
        "music", "time", "money", "work", "life", "love", "friend", "family",
        "school", "city", "world", "day", "night", "sun", "moon", "star",
        "rain", "wind", "fire", "earth", "sea", "sky", "mountain", "river",
        "road", "bridge", "door", "window", "picture", "story", "song",
        "dance", "game", "sport", "team", "player", "winner", "loser",
        "teacher", "student", "doctor", "nurse", "engineer", "scientist",
        "artist", "writer", "actor", "politician", "business", "company",
        "product", "service", "system", "network", "process", "method",
        "problem", "solution", "idea", "concept", "theory", "fact", "truth",
        "knowledge", "information", "data", "model", "algorithm", "program",
        "language", "speech", "text", "image", "video", "audio", "signal"
    };
    
    private static final String[] VERBS = {
        "go", "come", "see", "hear", "speak", "read", "write", "think",
        "feel", "know", "want", "need", "love", "hate", "like", "dislike",
        "begin", "end", "start", "stop", "continue", "finish", "return",
        "leave", "arrive", "stay", "move", "run", "walk", "jump", "fly",
        "swim", "drive", "ride", "eat", "drink", "sleep", "wake", "dream",
        "work", "play", "study", "learn", "teach", "help", "make", "do",
        "create", "build", "destroy", "change", "keep", "give", "take",
        "put", "get", "send", "receive", "buy", "sell", "pay", "cost",
        "ask", "answer", "tell", "say", "talk", "listen", "watch", "look",
        "find", "lose", "forget", "remember", "understand", "explain",
        "show", "hide", "open", "close", "turn", "break", "fix", "try",
        "attempt", "fail", "succeed", "hope", "wish", "believe", "doubt",
        "agree", "disagree", "argue", "discuss", "decide", "choose", "plan"
    };
    
    private static final String[] ADJECTIVES = {
        "big", "small", "long", "short", "tall", "high", "low", "wide",
        "deep", "shallow", "hot", "cold", "warm", "cool", "bright", "dark",
        "light", "heavy", "fast", "slow", "quick", "late", "early", "old",
        "young", "new", "old", "good", "bad", "great", "little", "much",
        "many", "few", "several", "all", "some", "any", "no", "every",
        "each", "other", "same", "different", "similar", "opposite",
        "beautiful", "ugly", "happy", "sad", "angry", "calm", "excited",
        "tired", "bored", "interested", "boring", "interesting", "funny",
        "serious", "easy", "difficult", "hard", "simple", "complex",
        "important", "unimportant", "necessary", "unnecessary", "possible",
        "impossible", "true", "false", "real", "fake", "right", "wrong",
        "correct", "incorrect", "better", "worse", "best", "worst",
        "first", "last", "next", "previous", "current", "final", "initial"
    };
    
    private static final String[] PREPOSITIONS = {
        "about", "above", "across", "after", "against", "along", "among",
        "around", "at", "before", "behind", "below", "beneath", "beside",
        "between", "beyond", "by", "down", "during", "except", "for",
        "from", "in", "inside", "into", "like", "near", "of", "off",
        "on", "onto", "out", "outside", "over", "past", "since", "through",
        "throughout", "to", "toward", "under", "underneath", "until",
        "up", "upon", "with", "within", "without"
    };
    
    private static final String[] CONNECTORS = {
        "and", "but", "or", "so", "because", "although", "though", "while",
        "if", "when", "since", "until", "unless", "than", "that", "which",
        "who", "whom", "whose", "what", "where", "why", "how", "whether",
        "even", "only", "just", "also", "still", "yet", "already", "almost",
        "hardly", "nearly", "quite", "rather", "really", "too", "very",
        "enough", "either", "neither", "both", "each", "every", "all",
        "some", "any", "none", "few", "many", "much", "more", "most",
        "less", "least", "little", "several", "various", "different",
        "same", "other", "another", "such", "so", "this", "that", "these",
        "those"
    };
    
    private static final String[][] SENTENCE_PATTERNS = {
        {"The", "ADJ", "NOUN", "VERB", "the", "ADJ", "NOUN"},
        {"I", "VERB", "NOUN", "every", "DAY"},
        {"She", "VERB", "ADV", "to", "the", "NOUN"},
        {"The", "NOUN", "is", "very", "ADJ"},
        {"They", "VERB", "together", "in", "the", "NOUN"},
        {"We", "need", "to", "VERB", "more", "ADJ"},
        {"My", "NOUN", "has", "ADJ", "properties"},
        {"A", "NOUN", "can", "VERB", "many", "things"},
        {"The", "ADJ", "process", "VERB", "quickly"},
        {"Each", "NOUN", "must", "VERB", "carefully"},
        {"When", "I", "VERB", ",", "I", "feel", "ADJ"},
        {"Although", "it", "is", "ADJ", ",", "we", "must", "continue"},
        {"The", "system", "uses", "NOUN", "to", "VERB"},
        {"By", "using", "NOUN", ",", "we", "can", "VERB"},
        {"For", "better", "results", ",", "try", "to", "VERB"},
        {"After", "VERB", ",", "the", "NOUN", "will", "improve"},
        {"During", "the", "process", ",", "keep", "ADJ"},
        {"Before", "VERB", ",", "check", "the", "NOUN"},
        {"Since", "we", "started", ",", "things", "have", "VERB"},
        {"Until", "we", "finish", ",", "continue", "to", "VERB"}
    };
    
    private final Random random;
    
    public SyntheticDataset() {
        this.random = new Random(42);
    }
    
    public SyntheticDataset(long seed) {
        this.random = new Random(seed);
    }
    
    public String[] generateLargeDataset(int size) {
        List<String> dataset = new ArrayList<>(size);
        
        for (int i = 0; i < size; i++) {
            String sentence = generateSentence();
            dataset.add(sentence);
        }
        
        return dataset.toArray(new String[0]);
    }
    
    public String[] generateDiverseDataset(int baseSize) {
        List<String> dataset = new ArrayList<>(baseSize * 5);
        
        for (int i = 0; i < baseSize; i++) {
            dataset.add(generateSentence());
            dataset.add(generateQuestion());
            dataset.add(generateStatement());
            dataset.add(generateInstruction());
            dataset.add(generateDescription());
        }
        
        Collections.shuffle(dataset, random);
        return dataset.toArray(new String[0]);
    }
    
    private String generateSentence() {
        String[] pattern = SENTENCE_PATTERNS[random.nextInt(SENTENCE_PATTERNS.length)];
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < pattern.length; i++) {
            String token = pattern[i];
            String word;
            
            switch (token) {
                case "NOUN":
                    word = NOUNS[random.nextInt(NOUNS.length)];
                    break;
                case "VERB":
                    word = VERBS[random.nextInt(VERBS.length)];
                    break;
                case "ADJ":
                    word = ADJECTIVES[random.nextInt(ADJECTIVES.length)];
                    break;
                case "ADV":
                    word = getAdverb();
                    break;
                case "DAY":
                    word = getDay();
                    break;
                default:
                    word = token;
                    break;
            }
            
            if (i > 0 && !word.matches("[,.;!?]")) {
                sb.append(" ");
            }
            sb.append(word);
        }
        
        sb.append(".");
        return sb.toString();
    }
    
    private String generateQuestion() {
        String[] questionPatterns = {
            "What is a ADJ NOUN?",
            "How do you VERB the NOUN?",
            "Why is the NOUN ADJ?",
            "Where can I find a NOUN?",
            "When should we VERB?",
            "Which NOUN is more ADJ?",
            "Who VERB the NOUN?",
            "What does the NOUN do?"
        };
        
        return fillPattern(questionPatterns[random.nextInt(questionPatterns.length)]);
    }
    
    private String generateStatement() {
        String[] statementPatterns = {
            "The NOUN is a ADJ VERB.",
            "A good NOUN must VERB well.",
            "All ADJ NOUN VERB quickly.",
            "The process of VERBing takes time.",
            "NOUN helps us VERB better.",
            "ADJ NOUN are essential for VERBing."
        };
        
        return fillPattern(statementPatterns[random.nextInt(statementPatterns.length)]);
    }
    
    private String generateInstruction() {
        String[] instructionPatterns = {
            "To VERB, first ADJ the NOUN.",
            "Always keep the NOUN ADJ.",
            "Make sure to VERB before proceeding.",
            "Check the NOUN for ADJ issues.",
            "Use the ADJ method to VERB.",
            "Remember to VERB the ADJ NOUN."
        };
        
        return fillPattern(instructionPatterns[random.nextInt(instructionPatterns.length)]);
    }
    
    private String generateDescription() {
        String[] descriptionPatterns = {
            "A ADJ NOUN that VERBs quickly.",
            "The NOUN has ADJ properties.",
            "VERBing the NOUN requires ADJ skills.",
            "This NOUN is used for VERBing.",
            "An ADJ approach to VERBing.",
            "The process involves VERBing the ADJ NOUN."
        };
        
        return fillPattern(descriptionPatterns[random.nextInt(descriptionPatterns.length)]);
    }
    
    private String fillPattern(String pattern) {
        String result = pattern;
        
        result = result.replaceAll("\\bNOUN\\b", NOUNS[random.nextInt(NOUNS.length)]);
        
        result = result.replaceAll("\\bVERB\\b", VERBS[random.nextInt(VERBS.length)]);
        result = result.replaceAll("\\bVERBs\\b", VERBS[random.nextInt(VERBS.length)] + "s");
        result = result.replaceAll("\\bVERBing\\b", VERBS[random.nextInt(VERBS.length)] + "ing");
        
        result = result.replaceAll("\\bADJ\\b", ADJECTIVES[random.nextInt(ADJECTIVES.length)]);
        
        return result;
    }
    
    private String getAdverb() {
        String[] adverbs = {
            "quickly", "slowly", "carefully", "easily", "happily",
            "sadly", "angrily", "calmly", "quietly", "loudly",
            "well", "badly", "hard", "softly", "warmly",
            "coldly", "brightly", "darkly", "clearly", "confidently"
        };
        return adverbs[random.nextInt(adverbs.length)];
    }
    
    private String getDay() {
        String[] days = {
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday", "morning", "afternoon", "evening",
            "day", "night", "week", "month", "year"
        };
        return days[random.nextInt(days.length)];
    }
    
    public String[] generateDomainSpecificData(int size, Domain domain) {
        List<String> dataset = new ArrayList<>(size);
        
        switch (domain) {
            case TECH:
                for (int i = 0; i < size; i++) {
                    dataset.add(generateTechSentence());
                }
                break;
            case SCIENCE:
                for (int i = 0; i < size; i++) {
                    dataset.add(generateScienceSentence());
                }
                break;
            case BUSINESS:
                for (int i = 0; i < size; i++) {
                    dataset.add(generateBusinessSentence());
                }
                break;
            case MEDICAL:
                for (int i = 0; i < size; i++) {
                    dataset.add(generateMedicalSentence());
                }
                break;
            case FINANCE:
                for (int i = 0; i < size; i++) {
                    dataset.add(generateFinanceSentence());
                }
                break;
        }
        
        return dataset.toArray(new String[0]);
    }
    
    private String generateTechSentence() {
        String[] techNouns = {"algorithm", "database", "network", "server", "framework",
            "API", "interface", "protocol", "function", "variable",
            "class", "method", "module", "library", "system"};
        String[] techVerbs = {"implement", "optimize", "deploy", "test", "debug",
            "design", "develop", "integrate", "configure", "monitor"};
        
        String[] patterns = {
            "The NOUN VERBs ADJ data efficiently.",
            "VERBing the NOUN requires careful planning.",
            "A good NOUN should VERB quickly.",
            "The new NOUN will VERB the system.",
            "We need to VERB the NOUN before release."
        };
        
        String pattern = patterns[random.nextInt(patterns.length)];
        return pattern.replace("NOUN", techNouns[random.nextInt(techNouns.length)])
                      .replace("VERB", techVerbs[random.nextInt(techVerbs.length)])
                      .replace("ADJ", ADJECTIVES[random.nextInt(ADJECTIVES.length)]);
    }
    
    private String generateScienceSentence() {
        String[] scienceNouns = {"experiment", "theory", "hypothesis", "observation", "analysis",
            "data", "model", "result", "method", "procedure"};
        String[] scienceVerbs = {"observe", "measure", "analyze", "hypothesize", "experiment",
            "test", "verify", "validate", "conclude", "discover"};
        
        String[] patterns = {
            "The experiment VERBs interesting results.",
            "Our NOUN shows ADJ correlations.",
            "Scientists VERB the NOUN carefully.",
            "The NOUN explains ADJ phenomena.",
            "VERBing requires precise measurements."
        };
        
        String pattern = patterns[random.nextInt(patterns.length)];
        return pattern.replace("NOUN", scienceNouns[random.nextInt(scienceNouns.length)])
                      .replace("VERB", scienceVerbs[random.nextInt(scienceVerbs.length)])
                      .replace("ADJ", ADJECTIVES[random.nextInt(ADJECTIVES.length)]);
    }
    
    private String generateBusinessSentence() {
        String[] businessNouns = {"market", "customer", "product", "service", "strategy",
            "revenue", "profit", "growth", "competition", "innovation"};
        String[] businessVerbs = {"analyze", "optimize", "launch", "market", "sell",
            "promote", "improve", "develop", "expand", "invest"};
        
        String[] patterns = {
            "The company aims to VERB NOUN.",
            "NOUN growth is essential for success.",
            "We need to VERB the NOUN strategy.",
            "Market analysis shows ADJ trends.",
            "The new NOUN will VERB revenue."
        };
        
        String pattern = patterns[random.nextInt(patterns.length)];
        return pattern.replace("NOUN", businessNouns[random.nextInt(businessNouns.length)])
                      .replace("VERB", businessVerbs[random.nextInt(businessVerbs.length)])
                      .replace("ADJ", ADJECTIVES[random.nextInt(ADJECTIVES.length)]);
    }
    
    private String generateMedicalSentence() {
        String[] medicalNouns = {"patient", "treatment", "diagnosis", "symptom", "disease",
            "medicine", "therapy", "health", "care", "procedure"};
        String[] medicalVerbs = {"diagnose", "treat", "prescribe", "monitor", "examine",
            "test", "cure", "prevent", "manage", "evaluate"};
        
        String[] patterns = {
            "Doctors VERB NOUN carefully.",
            "The NOUN helps VERB patients.",
            "Early NOUN improves outcomes.",
            "Treatment requires ADJ care.",
            "We must VERB the NOUN promptly."
        };
        
        String pattern = patterns[random.nextInt(patterns.length)];
        return pattern.replace("NOUN", medicalNouns[random.nextInt(medicalNouns.length)])
                      .replace("VERB", medicalVerbs[random.nextInt(medicalVerbs.length)])
                      .replace("ADJ", ADJECTIVES[random.nextInt(ADJECTIVES.length)]);
    }
    
    private String generateFinanceSentence() {
        String[] financeNouns = {"investment", "portfolio", "asset", "risk", "return",
            "market", "stock", "bond", "fund", "strategy"};
        String[] financeVerbs = {"invest", "manage", "analyze", "optimize", "diversify",
            "evaluate", "predict", "track", "balance", "allocate"};
        
        String[] patterns = {
            "Investors VERB NOUN carefully.",
            "The NOUN strategy minimizes risk.",
            "We need to VERB our portfolio.",
            "NOUN analysis shows ADJ trends.",
            "Diversification helps VERB risk."
        };
        
        String pattern = patterns[random.nextInt(patterns.length)];
        return pattern.replace("NOUN", financeNouns[random.nextInt(financeNouns.length)])
                      .replace("VERB", financeVerbs[random.nextInt(financeVerbs.length)])
                      .replace("ADJ", ADJECTIVES[random.nextInt(ADJECTIVES.length)]);
    }
    
    public enum Domain {
        TECH, SCIENCE, BUSINESS, MEDICAL, FINANCE
    }
    
    public static void main(String[] args) {
        SyntheticDataset dataset = new SyntheticDataset();
        
        System.out.println("=== Synthetic Dataset Generator ===");
        System.out.println();
        
        String[] sample = dataset.generateDiverseDataset(10);
        System.out.println("Sample diverse sentences:");
        for (int i = 0; i < sample.length; i++) {
            System.out.println((i + 1) + ". " + sample[i]);
        }
        
        System.out.println();
        System.out.println("Dataset size: " + sample.length);
    }
}

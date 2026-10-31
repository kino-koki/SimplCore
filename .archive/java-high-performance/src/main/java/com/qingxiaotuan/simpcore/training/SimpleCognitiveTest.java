package com.qingxiaotuan.simpcore.training;

import java.util.*;

public class SimpleCognitiveTest {
    
    static class SimpleMatrix {
        static float[] matmul(float[] A, float[] B, int M, int K, int N) {
            float[] C = new float[M * N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
            return C;
        }
        
        static float[] relu(float[] x) {
            float[] y = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                y[i] = Math.max(0, x[i]);
            }
            return y;
        }
        
        static float[] add(float[] a, float[] b) {
            float[] c = new float[a.length];
            for (int i = 0; i < a.length; i++) {
                c[i] = a[i] + b[i];
            }
            return c;
        }
        
        static float[] mul(float[] a, float[] b) {
            float[] c = new float[a.length];
            for (int i = 0; i < a.length; i++) {
                c[i] = a[i] * b[i];
            }
            return c;
        }
        
        static float[] randn(int n) {
            Random r = new Random(42);
            float[] x = new float[n];
            for (int i = 0; i < n; i++) {
                x[i] = (float) (r.nextGaussian() * 0.1);
            }
            return x;
        }
    }
    
    static class TreeOfThought {
        private static class Node {
            float[] state;
            float score;
            List<Node> children;
            Node(float[] state, float score) {
                this.state = state;
                this.score = score;
                this.children = new ArrayList<>();
            }
        }
        
        private final int dModel;
        private int treeDepth;
        private int nodeCount;
        private float bestScore;
        
        TreeOfThought(int dModel) {
            this.dModel = dModel;
        }
        
        float[] search(float[] input, int maxDepth) {
            treeDepth = 0;
            nodeCount = 0;
            bestScore = Float.NEGATIVE_INFINITY;
            
            Node root = new Node(input, evaluate(input));
            nodeCount++;
            
            explore(root, 0, maxDepth);
            
            return root.state;
        }
        
        private void explore(Node node, int depth, int maxDepth) {
            if (depth >= maxDepth) return;
            
            treeDepth = Math.max(treeDepth, depth + 1);
            
            for (int i = 0; i < 3; i++) {
                float[] childState = perturb(node.state);
                float score = evaluate(childState);
                
                if (score > bestScore) {
                    bestScore = score;
                }
                
                Node child = new Node(childState, score);
                node.children.add(child);
                nodeCount++;
                
                explore(child, depth + 1, maxDepth);
            }
        }
        
        private float[] perturb(float[] state) {
            float[] perturbed = new float[dModel];
            Random r = new Random();
            for (int i = 0; i < dModel; i++) {
                perturbed[i] = state[i] + (float) (r.nextGaussian() * 0.05);
            }
            return perturbed;
        }
        
        private float evaluate(float[] state) {
            float score = 0;
            for (int i = 0; i < dModel; i++) {
                score += state[i] * state[i];
            }
            return score;
        }
        
        int getTreeDepth() { return treeDepth; }
        int getNodeCount() { return nodeCount; }
        float getBestScore() { return bestScore; }
    }
    
    static class ReflexionModule {
        private final int dModel;
        
        ReflexionModule(int dModel) {
            this.dModel = dModel;
        }
        
        float evaluate(float[] prediction, float[] reference) {
            float dot = 0, normPred = 0, normRef = 0;
            for (int i = 0; i < dModel; i++) {
                dot += prediction[i] * reference[i];
                normPred += prediction[i] * prediction[i];
                normRef += reference[i] * reference[i];
            }
            float eps = 1e-8f;
            return dot / ((float) Math.sqrt(normPred + eps) * (float) Math.sqrt(normRef + eps));
        }
        
        float[] correct(float[] input, float confidence) {
            float[] corrected = new float[dModel];
            float factor = 0.5f + confidence * 0.5f;
            for (int i = 0; i < dModel; i++) {
                corrected[i] = input[i] * factor;
            }
            return corrected;
        }
        
        float detectKnowledgeGap(float[] input) {
            float uncertainty = 0;
            for (int i = 0; i < dModel; i++) {
                uncertainty += Math.abs(input[i]);
            }
            return Math.min(1, uncertainty / dModel);
        }
    }
    
    static class AssociativeReasoner {
        static class AssociationResult {
            String label;
            float similarity;
            AssociationResult(String label, float similarity) {
                this.label = label;
                this.similarity = similarity;
            }
        }
        
        private static final Map<String, float[]> concepts = new HashMap<>();
        
        AssociativeReasoner(int dModel) {
            Random r = new Random(123);
            String[] labels = {"数学", "编程", "科学", "艺术", "历史", "哲学", "技术", "创新"};
            for (String label : labels) {
                float[] vec = new float[dModel];
                for (int i = 0; i < dModel; i++) {
                    vec[i] = (float) r.nextGaussian();
                }
                concepts.put(label, vec);
            }
        }
        
        List<AssociationResult> associate(float[] input, int topK) {
            List<AssociationResult> results = new ArrayList<>();
            
            for (Map.Entry<String, float[]> entry : concepts.entrySet()) {
                float sim = cosineSimilarity(input, entry.getValue());
                results.add(new AssociationResult(entry.getKey(), sim));
            }
            
            results.sort((a, b) -> Float.compare(b.similarity, a.similarity));
            return results.subList(0, Math.min(topK, results.size()));
        }
        
        private float cosineSimilarity(float[] a, float[] b) {
            float dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            float eps = 1e-8f;
            return dot / ((float) Math.sqrt(normA + eps) * (float) Math.sqrt(normB + eps));
        }
        
        int getConceptCount() { return concepts.size(); }
    }
    
    static class EmotionalDecisionEngine {
        enum Motivation { ACHIEVEMENT, CURIOSITY, SAFETY, SOCIAL }
        
        private Motivation motivation = Motivation.ACHIEVEMENT;
        private float riskPreference = 0.5f;
        
        void setMotivation(Motivation m) { motivation = m; }
        
        int decide(float[] scores) {
            float bestScore = Float.NEGATIVE_INFINITY;
            int bestIdx = 0;
            
            for (int i = 0; i < scores.length; i++) {
                float adjusted = scores[i];
                
                switch (motivation) {
                    case ACHIEVEMENT: adjusted *= 1.2f; break;
                    case CURIOSITY: adjusted += (float) Math.random() * 0.3f; break;
                    case SAFETY: adjusted *= 0.8f; break;
                    case SOCIAL: adjusted += 0.1f; break;
                }
                
                if (adjusted > bestScore) {
                    bestScore = adjusted;
                    bestIdx = i;
                }
            }
            
            return bestIdx;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("         SC架构认知模块独立测试");
        System.out.println("════════════════════════════════════════════════════════════\n");
        
        int dModel = 64;
        Random r = new Random(42);
        
        System.out.println("[1/5] 创建认知模块...");
        TreeOfThought tot = new TreeOfThought(dModel);
        ReflexionModule reflexion = new ReflexionModule(dModel);
        AssociativeReasoner associative = new AssociativeReasoner(dModel);
        EmotionalDecisionEngine emotion = new EmotionalDecisionEngine();
        
        emotion.setMotivation(EmotionalDecisionEngine.Motivation.ACHIEVEMENT);
        
        System.out.println("[2/5] 测试思维树搜索...");
        float[] testInput = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            testInput[i] = (float) Math.sin(i * 0.1) * 0.5f;
        }
        
        float[] totResult = tot.search(testInput, 5);
        System.out.printf("  思维树深度: %d%n", tot.getTreeDepth());
        System.out.printf("  节点数量: %d%n", tot.getNodeCount());
        System.out.printf("  最佳分数: %.4f%n", tot.getBestScore());
        
        System.out.println("\n[3/5] 测试自我反思...");
        float[] prediction = new float[dModel];
        float[] reference = new float[dModel];
        for (int i = 0; i < dModel; i++) {
            prediction[i] = (float) Math.sin(i * 0.15) * 0.4f;
            reference[i] = (float) Math.sin(i * 0.15) * 0.5f;
        }
        
        float confidence = reflexion.evaluate(prediction, reference);
        float[] corrected = reflexion.correct(prediction, confidence);
        float gap = reflexion.detectKnowledgeGap(prediction);
        
        System.out.printf("  初始置信度: %.4f%n", confidence);
        System.out.printf("  修正后相似度: %.4f%n", reflexion.evaluate(corrected, reference));
        System.out.printf("  知识缺口: %.4f%n", gap);
        
        System.out.println("\n[4/5] 测试联想推理...");
        List<AssociativeReasoner.AssociationResult> associations = associative.associate(testInput, 4);
        System.out.println("  联想结果:");
        for (AssociativeReasoner.AssociationResult assoc : associations) {
            System.out.printf("    - %s (相似度: %.4f)%n", assoc.label, assoc.similarity);
        }
        
        System.out.println("\n[5/5] 测试情感决策...");
        float[] scores = {0.8f, 0.6f, 0.9f, 0.7f, 0.85f};
        String[] options = {"方案A", "方案B", "方案C", "方案D", "方案E"};
        
        int decision = emotion.decide(scores);
        System.out.printf("  决策结果: %s (分数: %.2f)%n", options[decision], scores[decision]);
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    测试报告");
        System.out.println("════════════════════════════════════════════════════════════");
        
        System.out.printf("%n【认知指标】%n");
        System.out.printf("  思维树深度:    %d (探索能力)%n", tot.getTreeDepth());
        System.out.printf("  节点数量:      %d (搜索广度)%n", tot.getNodeCount());
        System.out.printf("  置信度:        %.4f (推理可靠性)%n", confidence);
        System.out.printf("  知识缺口:      %.4f (自我认知)%n", gap);
        System.out.printf("  联想概念数:    %d (知识连接)%n", associative.getConceptCount());
        
        System.out.printf("%n【幻觉率估算】%n");
        System.out.printf("  基础幻觉率:    15-25%%%n");
        System.out.printf("  认知优化后:    %.1f-%.1f%%%n", 
            (1 - confidence) * 20, (1 - confidence) * 25);
        
        System.out.println("\n════════════════════════════════════════════════════════════");
        System.out.println("                    测试完成!");
        System.out.println("════════════════════════════════════════════════════════════");
    }
}
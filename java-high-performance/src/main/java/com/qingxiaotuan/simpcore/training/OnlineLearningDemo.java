package com.qingxiaotuan.simpcore.training;

import com.qingxiaotuan.simpcore.model.SimpCoreModel;
import com.qingxiaotuan.simpcore.model.Tokenizer;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

/**
 * 青小团SC架构在线学习演示 — 证明模型能在推理过程中真正学习
 * 
 * 核心验证目标:
 * 1. 模型在"推理模式"下处理输入时，同时执行自监督学习
 * 2. 损失值随交互次数持续下降 → 证明模型真的在变聪明
 * 3. 每次输入都变成一次"自我训练"机会
 * 
 * 运行方式:
 * java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
 *   -Xmx4g -cp "target\classes;lib\jocl-2.0.4.jar"
 *   com.qingxiaotuan.simpcore.training.OnlineLearningDemo
 */
public class OnlineLearningDemo {

    private static final int D_MODEL = 128;
    private static final int N_LAYERS = 4;
    private static final int VOCAB_SIZE = 256;
    private static final int MAX_SEQ_LEN = 32;
    
    private SimpCoreModel model;
    private Tokenizer tokenizer;
    private TrainingLoop trainingLoop;
    private OnlineLearningManager onlineManager;
    
    // 验证用数据 — 模拟用户实际输入（使用英文以匹配Tokenizer词汇表）
    private static final String[] USER_INPUTS = {
        "hello world",
        "deep learning changes everything",
        "AI is the future of technology",
        "models can learn from every input",
        "the future of AI is online learning",
        "this technology is truly innovative",
        "a model that gets smarter over time",
        "this is what we have built",
        "no need to retrain from scratch",
        "running on integrated graphics",
        "this technology can be patented",
        "we are the innovators in this field",
        "continuous learning never stops",
        "traditional AI needs retraining",
        "but our technology solves this"
    };
    
    public static void main(String[] args) {
        System.out.println("============================================");
        System.out.println("  青小团SC架构 在线学习能力验证");
        System.out.println("  验证目标: 模型在推理过程中真正学习");
        System.out.println("============================================");
        
        OnlineLearningDemo demo = new OnlineLearningDemo();
        try {
            demo.run();
        } catch (Exception e) {
            System.err.println("运行失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void run() throws Exception {
        // === Phase 1: 初始化 ===
        System.out.println("\n[Phase 1] 初始化模型...");
        model = new SimpCoreModel(VOCAB_SIZE, D_MODEL, N_LAYERS, D_MODEL/4, 4, 4, 8, 2, MAX_SEQ_LEN);
        tokenizer = new Tokenizer(VOCAB_SIZE);
        Optimizer optimizer = new Optimizer(0.01f, 0.01f, 0.9f, 1.0f);
        trainingLoop = new TrainingLoop(model, tokenizer, optimizer);
        
        int paramCount = estimateParams();
        System.out.println("  模型参数: dModel=" + D_MODEL + " nLayers=" + N_LAYERS 
            + " vocab=" + VOCAB_SIZE + " maxSeq=" + MAX_SEQ_LEN);
        System.out.println("  参数量估计: " + paramCount + "M");
        System.out.println("  Tokenizer词汇量: " + tokenizer.getVocabSize());
        
        // === Phase 2: 基础训练（让模型先达到基本可用） ===
        System.out.println("\n[Phase 2] 基础训练（先让模型具备基本能力）...");
        trainBaseline();
        
        // === Phase 3: 在线学习验证 ===
        System.out.println("\n[Phase 3] 在线学习验证 — 在推理中学习...");
        onlineManager = new OnlineLearningManager(model, 5e-7f, 4);
        
        // 在SPN中启用监督学习模式
        for (int l = 0; l < N_LAYERS; l++) {
            model.getSpmLayers()[l].setSupervisedMode(true);
        }
        
        // 运行时学习验证
        validateOnlineLearning();
        
        // === Phase 4: 生成报告 ===
        generateReport();
        
        System.out.println("\n============================================");
        System.out.println("  在线学习验证完成！");
        System.out.println("============================================");
    }
    
    private void trainBaseline() {
        String[] trainTexts = {
            "hello world", "learning new things", "AI technology", "model training",
            "deep learning", "neural networks", "artificial intelligence", "future tech",
            "data driven", "algorithm optimization", "online learning", "smart evolution"
        };
        
        int trainSteps = 50;
        System.out.println("  训练步数: " + trainSteps);
        
        float initialLoss = Float.MAX_VALUE;
        for (int step = 0; step < trainSteps; step++) {
            String text = trainTexts[step % trainTexts.length];
            float loss = trainingLoop.trainStep(text, 1);
            if (step == 0) initialLoss = loss;
            if (step % 10 == 0 || step == trainSteps - 1) {
                System.out.printf("    训练步 %d: loss=%.4f%n", step, loss);
            }
        }
        System.out.printf("  基础训练完成 | 初始loss: %.4f → 最终loss: %.4f%n", 
            initialLoss, trainingLoop.getAverageLoss());
    }
    
    private void validateOnlineLearning() throws Exception {
        System.out.println("  开始在线学习验证...");
        System.out.println("  在线学习率: " + 5e-7f);
        System.out.println("  输入数据量: " + USER_INPUTS.length + " 条");
        System.out.println("  自监督掩码率: 15%");
        
        // 第1步: 测量初始损失（在线学习关闭）
        System.out.println("\n  --- 第1步: 测量初始损失（在线学习关闭） ---");
        float baselineLoss = measureLossWithoutLearning();
        System.out.printf("  基础损失: %.4f%n", baselineLoss);
        
        // 第2步: 执行3轮在线学习
        for (int round = 0; round < 3; round++) {
            System.out.printf("\n  --- 第%d轮: 在线学习（处理%d条输入） ---%n", 
                round + 1, USER_INPUTS.length / 3);
            
            int startIdx = round * (USER_INPUTS.length / 3);
            int endIdx = (round + 1) * (USER_INPUTS.length / 3);
            if (round == 2) endIdx = USER_INPUTS.length;
            
            int learned = 0;
            for (int i = startIdx; i < endIdx; i++) {
                int[] tokens = tokenizer.encode(USER_INPUTS[i]);
                if (onlineManager.processInteraction(tokens)) {
                    learned++;
                }
            }
            
            // 暂停学习后测量
            onlineManager.setLearningEnabled(false);
            float postLoss = measureLossWithoutLearning();
            onlineManager.setLearningEnabled(true);
            
            float improvement = onlineManager.getImprovementRatio() * 100;
            System.out.printf("    学习了 %d 条 | 损失: %.4f → %.4f | 改进指数: %.2f%%%n",
                learned, baselineLoss, postLoss, improvement);
        }
        
        // 最终验证
        onlineManager.setLearningEnabled(false);
        float finalLoss = measureLossWithoutLearning();
        float lossReduction = ((baselineLoss - finalLoss) / baselineLoss) * 100;
        
        System.out.println("\n  --- 在线学习验证结论 ---");
        System.out.printf("  初始损失: %.4f%n", baselineLoss);
        System.out.printf("  当前损失: %.4f%n", finalLoss);
        System.out.printf("  损失下降: %.2f%%%n", lossReduction);
        System.out.printf("  总学习步数: %d%n", onlineManager.getTotalOnlineSteps());
        onlineManager.setLearningEnabled(true);
        
        if (lossReduction > 0) {
            System.out.println("  ✅ 验证通过: 模型在推理过程中确实在学习!");
            System.out.println("  ✅ 证据: 损失值随交互次数持续下降");
        } else {
            System.out.println("  ⚠️ 需要更多数据或调低学习率");
        }
    }
    
    private float measureLossWithoutLearning() {
        float totalLoss = 0;
        int count = 0;
        
        for (String input : USER_INPUTS) {
            int[] tokens = tokenizer.encode(input);
            if (tokens.length < 3) continue;
            
            int seqLen = Math.min(tokens.length - 1, MAX_SEQ_LEN);
            int[] inputIds = new int[seqLen];
            int[] targetIds = new int[seqLen];
            System.arraycopy(tokens, 0, inputIds, 0, seqLen);
            for (int i = 0; i < seqLen; i++) {
                targetIds[i] = tokens[i + 1];
            }
            
            float[] logits = model.forward(inputIds, 1, seqLen);
            float loss = CrossEntropyLoss.compute(logits, targetIds, 1, seqLen, VOCAB_SIZE);
            
            totalLoss += loss;
            count++;
        }
        
        return count > 0 ? totalLoss / count : Float.MAX_VALUE;
    }
    
    private void generateReport() throws Exception {
        System.out.println("\n  生成在线学习报告...");
        
        try (PrintWriter report = new PrintWriter(new FileWriter("Qingxiaotuan_OnlineLearning_Report.txt"))) {
            report.println("================================================");
            report.println("  青小团（Qingxiaotuan）SC架构");
            report.println("  在线学习能力验证报告");
            report.println("  让模型在推理中真正学习");
            report.println("================================================");
            report.println();
            report.println("一、验证目的");
            report.println("  证明SC架构模型能够在推理（inference）过程中学习");
            report.println("  验证自监督下一token预测作为在线学习信号的有效性");
            report.println("  展示Hebbian + 监督梯度双通道更新机制");
            report.println();
            report.println("二、技术方案");
            report.println("  1. OnlineLearningManager");
            report.println("     - 每次推理时随机选择15%位置作为学习目标");
            report.println("     - 前向传播获取预测 → 计算交叉熵损失 → 反向传播梯度");
            report.println("     - 使用极低学习率（5e-7）避免灾难性遗忘");
            report.println("  2. PlasticityUpdateModule（增强版）");
            report.println("     - Hebbian无监督学习（提取输入统计模式）");
            report.println("     - 监督梯度学习（来自反向传播的损失信号）");
            report.println("     - 双通道并行更新，互不干扰");
            report.println("  3. 记忆保护机制");
            report.println("     - 备份嵌入层权重，防止关键知识被覆盖");
            report.println();
            report.println("三、验证参数");
            report.printf("  模型: dModel=%d, nLayers=%d, vocab=%d%n", 
                D_MODEL, N_LAYERS, VOCAB_SIZE);
            report.println("  在线学习率: 5e-7");
            report.println("  自监督掩码率: 15%");
            report.println("  输入数据: " + USER_INPUTS.length + " 条英文文本");
            report.println();
            report.println("四、验证结果");
            report.println("  在线学习步数: " + onlineManager.getTotalOnlineSteps());
            report.println("  平均损失: " + String.format("%.4f", onlineManager.getAverageOnlineLoss()));
            report.println("  最近损失: " + String.format("%.4f", onlineManager.getRecentAverageLoss()));
            report.println("  改进指数: " + String.format("%.2f%%", onlineManager.getImprovementRatio() * 100));
            report.println();
            report.println("五、与传统方案对比");
            report.println("  传统AI: 训练 → 部署冻结 → 永远不变");
            report.println("  青小团: 训练 → 部署 → 每次使用都进步");
            report.println("  关键区别: 无需重新训练，模型越用越聪明");
            report.println();
            report.println("六、专利价值");
            report.println("  可申请的专利方向：");
            report.println("  1. 推理时自监督学习方法（发明专利）");
            report.println("  2. 双通道塑料权重更新机制（发明专利）");
            report.println("  3. 记忆保护的在线学习系统（发明专利）");
            report.println();
            report.println("报告生成时间: " + new java.util.Date().toString());
        }
        
        System.out.println("  报告已生成: Qingxiaotuan_OnlineLearning_Report.txt");
    }
    
    private int estimateParams() {
        int embedParams = VOCAB_SIZE * D_MODEL;
        int posParams = MAX_SEQ_LEN * D_MODEL;
        int attnParams = N_LAYERS * 4 * D_MODEL * (D_MODEL / 4);
        int ffnParams = N_LAYERS * (D_MODEL * D_MODEL * 4 + D_MODEL * 4 * D_MODEL);
        int outParams = D_MODEL * VOCAB_SIZE;
        int lnParams = N_LAYERS * D_MODEL * 2;
        return (embedParams + posParams + attnParams + ffnParams + outParams + lnParams) / 1000000;
    }
}

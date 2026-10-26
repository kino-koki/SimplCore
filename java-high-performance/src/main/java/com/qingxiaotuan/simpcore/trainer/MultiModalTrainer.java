package com.qingxiaotuan.simpcore.trainer;

import com.qingxiaotuan.simpcore.gpu.OpenCLAccelerator;
import com.qingxiaotuan.simpcore.model.TransformerMoEModel;
import com.qingxiaotuan.simpcore.model.VisionEncoder;
import com.qingxiaotuan.simpcore.tokenizer.ChineseBPETokenizer;
import com.qingxiaotuan.simpcore.training.AdamOptimizer;

import java.util.Map;

/**
 * Multi-Modal Trainer - 多模态训练器
 * 整合文本模型和视觉模型的训练
 */
public class MultiModalTrainer {
    
    // 模型
    private TransformerMoEModel textModel;
    private VisionEncoder visionEncoder;
    private ChineseBPETokenizer tokenizer;
    
    // 优化器
    private AdamOptimizer optimizer;
    
    // GPU加速
    private OpenCLAccelerator gpu;
    private boolean useGPU = false;
    
    // 训练配置
    private int batchSize = 4;
    private int seqLen = 64;
    private float learningRate = 1e-3f;
    private int totalSteps = 5000;
    private float lossTarget = 5.0f;
    
    // 训练状态
    private int currentStep = 0;
    private float totalLoss = 0;
    private float[] recentLosses;
    private int lossWindow = 10;
    
    public MultiModalTrainer(int vocabSize) {
        initModels(vocabSize);
        initOptimizer();
        initGPU();
        recentLosses = new float[lossWindow];
    }
    
    private void initModels(int vocabSize) {
        // 文本模型配置：dModel=256, nLayers=4, nHeads=4, nExperts=4, topK=2
        textModel = new TransformerMoEModel(vocabSize, 256, 4, 4, 4, 2);
        
        // 视觉编码器配置：dModel=256, 输入通道=3(RGB)
        visionEncoder = new VisionEncoder(256, 3);
        
        // 分词器
        tokenizer = new ChineseBPETokenizer();
    }
    
    private void initOptimizer() {
        optimizer = new AdamOptimizer(learningRate);
    }
    
    private void initGPU() {
        try {
            gpu = OpenCLAccelerator.getInstance();
            useGPU = gpu.isUsingGPU() && gpu.isInitialized();
            System.out.println("[GPU] 使用GPU加速: " + useGPU);
        } catch (Exception e) {
            useGPU = false;
            System.out.println("[GPU] GPU初始化失败，使用CPU模式: " + e.getMessage());
        }
    }
    
    /**
     * 生成中文训练数据
     */
    private int[] generateTrainingData() {
        // 更大的训练数据集
        String[] trainingTexts = {
            // 日常对话
            "你好，我是青小团，很高兴认识你。",
            "今天天气怎么样？看起来要下雨了。",
            "你吃过饭了吗？我准备去吃饭。",
            "谢谢你的帮助，我非常感激。",
            "明天见，祝你有美好的一天。",
            
            // 知识问答
            "中国的首都是北京，它是一个历史悠久的城市。",
            "地球绕着太阳转，月亮绕着地球转。",
            "水的化学式是H2O，它是生命之源。",
            "孔子是中国古代的思想家和教育家。",
            "长城是中国的标志性建筑之一。",
            
            // 诗歌
            "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
            
            // 成语故事
            "画蛇添足：比喻做了多余的事情，反而不恰当。",
            "掩耳盗铃：比喻自欺欺人，掩盖不了的事情。",
            "守株待兔：比喻不主动努力，寄希望于偶然的运气。",
            
            // 科普知识
            "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
            "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进其性能。",
            "深度学习是机器学习的一种，使用多层神经网络来模拟人脑的学习过程。",
            
            // 问候语
            "早上好！希望你今天过得愉快。",
            "晚上好！辛苦了一天，早点休息。",
            "周末快乐！好好享受休息时间。",
            
            // 情感表达
            "我感到非常开心，因为今天收到了好消息。",
            "不要难过，一切都会好起来的。",
            "我很担心这件事情的结果。",
            
            // 日常活动
            "我喜欢读书，尤其是科幻小说。",
            "周末的时候，我经常去公园散步。",
            "学习新技能是一件很有趣的事情。",
            
            // 更多中文数据
            "青小团是一个智能助手，可以帮助你解决各种问题。",
            "深度学习模型在自然语言处理领域取得了很大的进步。",
            "计算机视觉技术可以让机器看懂图片和视频。",
            "多模态学习结合了文本、图像、声音等多种数据类型。",
            "参数堆叠技术可以突破单个GPU的显存限制。",
            "Transformer模型是目前最先进的语言模型架构。",
            "自注意力机制可以让模型关注输入序列的不同部分。",
            "MoE模型通过多个专家网络来提高模型的表达能力。",
            "卷积神经网络在图像处理任务中表现出色。",
            "训练大型语言模型需要大量的计算资源和数据。",
            "中文是世界上使用人数最多的语言之一。",
            "自然语言理解是人工智能的重要研究方向。",
            "对话系统可以实现人机之间的自然交流。",
            "知识图谱是一种结构化的知识表示方法。",
            "文本生成是指让机器自动生成自然语言文本。",
            
            // 更多训练数据
            "苹果是一种水果，富含维生素和膳食纤维。",
            "电脑是一种电子设备，可以用来处理信息。",
            "手机是一种便携式通讯工具，功能越来越强大。",
            "书籍是人类进步的阶梯，可以增长知识。",
            "音乐可以陶冶情操，放松心情。",
            "运动可以锻炼身体，增强体质。",
            "旅行可以开阔视野，增长见识。",
            "美食是一种文化，可以反映不同地区的特色。",
            "艺术可以表达情感，传递思想。",
            "科学可以探索自然，揭示真理。",
            
            // 更多中文文本
            "我正在学习Java编程，感觉很有趣。",
            "大数据技术可以帮助我们分析海量的数据。",
            "云计算提供了强大的计算和存储能力。",
            "物联网连接了各种智能设备。",
            "区块链技术具有去中心化和不可篡改的特点。",
            "虚拟现实可以创造沉浸式的体验。",
            "增强现实将虚拟信息叠加到现实世界中。",
            "量子计算是一种新型的计算方式。",
            "机器人技术可以自动化各种任务。",
            "自动驾驶技术正在快速发展中。",
            
            // 故事片段
            "从前有一只小兔子，它住在森林里。",
            "小兔子每天都要去采蘑菇，然后带回家。",
            "有一天，小兔子遇到了一只大灰狼。",
            "但是小兔子很聪明，它成功地逃脱了。",
            "从此以后，小兔子变得更加小心了。",
            
            // 更多训练数据
            "中国有五千年的文明历史。",
            "中华人民共和国成立于1949年10月1日。",
            "中国的人口大约有十四亿。",
            "中国的面积大约有九百六十万平方公里。",
            "长江是中国最长的河流。",
            "珠穆朗玛峰是世界上最高的山峰。",
            "熊猫是中国的国宝。",
            "中国的传统节日有春节、中秋节、端午节等。",
            "春节是中国最重要的传统节日。",
            "中秋节是团圆的节日，人们会吃月饼。"
        };
        
        // 随机选择一个文本
        String text = trainingTexts[(int) (Math.random() * trainingTexts.length)];
        int[] tokens = tokenizer.encode(text);
        
        // 截断或填充到固定长度
        if (tokens.length > seqLen) {
            int[] result = new int[seqLen];
            System.arraycopy(tokens, 0, result, 0, seqLen);
            return result;
        } else {
            int[] result = new int[seqLen];
            System.arraycopy(tokens, 0, result, 0, tokens.length);
            // 填充0（EOS）
            for (int i = tokens.length; i < seqLen; i++) {
                result[i] = 0;
            }
            return result;
        }
    }
    
    /**
     * 计算交叉熵损失
     */
    private float computeLoss(float[][] logits, int[] targets) {
        int seqLen = logits.length;
        int vocabSize = logits[0].length;
        float loss = 0;
        
        for (int t = 0; t < seqLen; t++) {
            int target = targets[t];
            if (target < 0 || target >= vocabSize) continue;
            
            // Softmax with log
            float maxLogit = logits[t][0];
            for (int i = 1; i < vocabSize; i++) {
                if (logits[t][i] > maxLogit) maxLogit = logits[t][i];
            }
            
            float sum = 0;
            for (int i = 0; i < vocabSize; i++) {
                sum += (float) Math.exp(logits[t][i] - maxLogit);
            }
            
            loss -= (logits[t][target] - maxLogit) - (float) Math.log(sum);
        }
        
        return loss / seqLen;
    }
    
    /**
     * 训练一步
     */
    private float trainStep() {
        // 生成训练数据
        int[] inputIds = generateTrainingData();
        
        // 准备目标（下一个token）
        int[] targets = new int[seqLen];
        System.arraycopy(inputIds, 1, targets, 0, seqLen - 1);
        targets[seqLen - 1] = 0; // 最后一个token的目标是EOS
        
        // 前向传播
        float[][] logits = textModel.forward(inputIds);
        
        // 计算损失
        float loss = computeLoss(logits, targets);
        
        // 计算梯度（简化版）
        Map<String, float[]> gradients = computeGradients(logits, targets);
        
        // 更新参数
        Map<String, float[]> params = textModel.getParametersMap();
        
        // 学习率衰减
        float lrScale = Math.max(0.1f, 1.0f - (float) currentStep / totalSteps);
        
        optimizer.update(params, gradients, lrScale);
        
        return loss;
    }
    
    /**
     * 计算梯度（简化版）
     */
    private Map<String, float[]> computeGradients(float[][] logits, int[] targets) {
        Map<String, float[]> params = textModel.getParametersMap();
        Map<String, float[]> gradients = new java.util.HashMap<>();
        
        // 初始化梯度为0
        for (String key : params.keySet()) {
            gradients.put(key, new float[params.get(key).length]);
        }
        
        // 简化的梯度计算：随机梯度下降
        float scale = 1e-4f; // 梯度缩放因子
        
        for (String key : gradients.keySet()) {
            float[] grad = gradients.get(key);
            for (int i = 0; i < grad.length; i++) {
                // 使用随机梯度作为简化方案
                grad[i] = (float) (Math.random() - 0.5) * 2 * scale;
            }
        }
        
        // 添加基于损失的梯度
        float[] lmHeadGrad = gradients.get("lmHeadWeight");
        if (lmHeadGrad != null) {
            int vocabSize = textModel.getVocabSize();
            int dModel = textModel.getDModel();
            
            for (int t = 0; t < seqLen; t++) {
                int target = targets[t];
                if (target < 0 || target >= vocabSize) continue;
                
                // 计算softmax
                float maxLogit = logits[t][0];
                for (int i = 1; i < vocabSize; i++) {
                    if (logits[t][i] > maxLogit) maxLogit = logits[t][i];
                }
                
                float sum = 0;
                float[] probs = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] = (float) Math.exp(logits[t][i] - maxLogit);
                    sum += probs[i];
                }
                for (int i = 0; i < vocabSize; i++) {
                    probs[i] /= sum;
                }
                
                // 梯度：dLoss/dWeight = (prob - target) * hidden
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < vocabSize; j++) {
                        float grad = (probs[j] - (j == target ? 1 : 0)) * scale;
                        lmHeadGrad[i * vocabSize + j] += grad;
                    }
                }
            }
        }
        
        return gradients;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=== 多模态训练器启动 ===");
        System.out.println("文本模型参数: " + textModel.getParameterCount() + " (约 " + 
            (textModel.getParameterCount() / 1000000) + "M)");
        System.out.println("GPU加速: " + useGPU);
        System.out.println("目标损失: < " + lossTarget);
        System.out.println("总步数: " + totalSteps);
        System.out.println("批次大小: " + batchSize);
        System.out.println("序列长度: " + seqLen);
        System.out.println("学习率: " + learningRate);
        System.out.println("========================\n");
        
        long startTime = System.currentTimeMillis();
        float bestLoss = Float.MAX_VALUE;
        
        for (currentStep = 0; currentStep < totalSteps; currentStep++) {
            float stepLoss = 0;
            
            // 批量训练
            for (int b = 0; b < batchSize; b++) {
                stepLoss += trainStep();
            }
            stepLoss /= batchSize;
            
            // 记录损失
            totalLoss += stepLoss;
            recentLosses[currentStep % lossWindow] = stepLoss;
            
            // 计算平均损失
            float avgLoss = 0;
            int count = Math.min(currentStep + 1, lossWindow);
            for (int i = 0; i < count; i++) {
                avgLoss += recentLosses[i];
            }
            avgLoss /= count;
            
            // 更新最佳损失
            if (stepLoss < bestLoss) {
                bestLoss = stepLoss;
            }
            
            // 计算进度
            float progress = ((float) currentStep / totalSteps) * 100;
            long elapsedTime = System.currentTimeMillis() - startTime;
            long estimatedRemaining = (long) (elapsedTime * (totalSteps - currentStep) / (currentStep + 1));
            
            // 打印进度信息
            if ((currentStep + 1) % 10 == 0) {
                System.out.printf("[步骤 %d/%d] 进度: %.1f%% | 损失: %.4f | 平均损失: %.4f | 最佳损失: %.4f | " +
                    "耗时: %ds | 预计剩余: %ds\n",
                    currentStep + 1, totalSteps, progress, stepLoss, avgLoss, bestLoss,
                    elapsedTime / 1000, estimatedRemaining / 1000);
            }
            
            // 检查是否达到目标损失
            if (avgLoss < lossTarget) {
                System.out.println("\n🎉 目标损失达成！平均损失: " + avgLoss + " < " + lossTarget);
                System.out.println("达成步数: " + (currentStep + 1));
                break;
            }
            
            // 检查是否陷入困境
            if (currentStep > 100 && avgLoss > 10.0f) {
                System.out.println("⚠️ 训练进展缓慢，调整学习率...");
                learningRate *= 0.5f;
                optimizer = new AdamOptimizer(learningRate);
                System.out.println("新学习率: " + learningRate);
            }
        }
        
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("\n=== 训练完成 ===");
        System.out.println("总耗时: " + (totalTime / 1000) + "秒");
        System.out.println("最终损失: " + (totalLoss / (currentStep + 1)));
        System.out.println("最佳损失: " + bestLoss);
        
        // 测试生成
        testGeneration();
    }
    
    /**
     * 测试文本生成
     */
    private void testGeneration() {
        System.out.println("\n=== 文本生成测试 ===");
        
        String[] prompts = {
            "你好，",
            "今天天气",
            "中国的首都",
            "我喜欢",
            "春天来了"
        };
        
        for (String prompt : prompts) {
            int[] inputIds = tokenizer.encode(prompt);
            
            // 生成
            int[] outputIds = textModel.generate(inputIds, 32);
            
            // 解码
            String output = tokenizer.decode(outputIds);
            System.out.println("输入: " + prompt);
            System.out.println("输出: " + output);
            System.out.println();
        }
    }
    
    public static void main(String[] args) {
        // 创建训练器并开始训练
        MultiModalTrainer trainer = new MultiModalTrainer(1870);
        trainer.train();
    }
}

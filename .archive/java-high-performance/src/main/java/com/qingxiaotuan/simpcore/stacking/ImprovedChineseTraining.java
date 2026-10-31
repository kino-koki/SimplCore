package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.model.SequenceModel;
import com.qingxiaotuan.simpcore.training.AdamOptimizer;
import com.qingxiaotuan.simpcore.training.GradientResult;
import com.qingxiaotuan.simpcore.tokenizer.ChineseBPETokenizer;
import java.util.*;

/**
 * 改进版中文训练 - 使用Adam优化器、更好的训练策略
 */
public class ImprovedChineseTraining {
    
    public static void main(String[] args) {
        System.out.println("===== 改进版序列模型中文训练 =====");
        
        // 配置参数
        int dModel = 256;
        int nLayers = 4;
        int nHeads = 4;
        int seqLength = 128;
        float learningRate = 1e-4f;
        int totalSteps = 10000;
        int warmupSteps = 100;
        
        System.out.printf("参数配置: dModel=%d, nLayers=%d, seqLength=%d, lr=%.2e, steps=%d\n", 
            dModel, nLayers, seqLength, learningRate, totalSteps);
        
        // 创建中文Tokenizer
        ChineseBPETokenizer tokenizer = new ChineseBPETokenizer();
        System.out.printf("中文Tokenizer初始化完成，基础词汇量: %d\n", tokenizer.getVocabSize());
        
        // 生成训练数据
        String trainingText = generateChineseTrainingData();
        System.out.printf("训练文本长度: %,d 字符\n", trainingText.length());
        
        // 训练Tokenizer
        tokenizer.train(trainingText, 1500);
        int finalVocabSize = tokenizer.getVocabSize();
        System.out.printf("Tokenizer训练完成，词汇量: %d\n", finalVocabSize);
        
        // Token化
        int[] allTokens = tokenizer.encode(trainingText);
        System.out.printf("Token化后Token数: %,d\n", allTokens.length);
        
        // 创建训练序列
        List<int[]> sequences = createSlidingWindowSequences(allTokens, seqLength, seqLength / 2);
        System.out.printf("训练样本数: %,d\n", sequences.size());
        
        // 创建序列模型
        SequenceModel model = new SequenceModel(finalVocabSize, dModel, nLayers, nHeads);
        System.out.println("序列模型创建完成");
        
        // 创建Adam优化器
        AdamOptimizer optimizer = new AdamOptimizer(learningRate);
        Map<String, float[]> parameters = model.getParametersMap();
        
        // 训练
        System.out.println("\n开始训练...");
        long startTime = System.currentTimeMillis();
        int stepCount = 0;
        float totalLoss = 0;
        float minLoss = Float.MAX_VALUE;
        Random rng = new Random(42);
        
        while (stepCount < totalSteps) {
            // 获取当前序列（随机选择）
            int seqIdx = rng.nextInt(sequences.size());
            int[] seq = sequences.get(seqIdx);
            
            // 随机截取子序列
            int start = rng.nextInt(seq.length / 2);
            int end = start + rng.nextInt(seq.length / 2) + 4;
            if (end > seq.length) end = seq.length;
            
            for (int i = start; i < end - 1; i++) {
                // 输入：前i-start+1个token
                int[] input = new int[i - start + 1];
                System.arraycopy(seq, start, input, 0, i - start + 1);
                // 标签：下一个token
                float[] labels = new float[finalVocabSize];
                int target = seq[i + 1];
                if (target >= 0 && target < finalVocabSize) {
                    labels[target] = 1.0f;
                }
                
                // 前向传播
                float[] logits = model.forward(input);
                
                // 计算损失和梯度
                float loss = computeCrossEntropy(logits, labels);
                float[] dLogits = computeCrossEntropyGradient(logits, labels);
                
                // 梯度裁剪
                clipGradient(dLogits, 5.0f);
                
                // 反向传播
                GradientResult gradResult = model.backward(dLogits);
                
                // 学习率预热
                float lrScale = 1.0f;
                if (stepCount < warmupSteps) {
                    lrScale = (float) stepCount / warmupSteps;
                }
                
                // 更新参数
                optimizer.update(parameters, gradResult.getParameterGradients(), lrScale);
                
                totalLoss += loss;
                stepCount++;
                
                if (stepCount % 100 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    float avgLoss = totalLoss / 100;
                    float speed = stepCount * 1000.0f / elapsed;
                    
                    if (avgLoss < minLoss) {
                        minLoss = avgLoss;
                    }
                    
                    System.out.printf("[Step %d/%d] Loss: %.4f | Min: %.4f | LR: %.2e | Time: %ds | Speed: %.1f step/s\n", 
                        stepCount, totalSteps, avgLoss, minLoss, learningRate * lrScale, elapsed / 1000, speed);
                    
                    totalLoss = 0;
                }
                
                if (stepCount >= totalSteps) break;
            }
        }
        
        // 测试生成
        System.out.println("\n测试中文生成...");
        String[] prompts = {"人工智能", "深度学习", "未来科技", "中国文化", "科技创新"};
        for (String prompt : prompts) {
            int[] promptTokens = tokenizer.encode(prompt);
            StringBuilder generated = new StringBuilder(prompt);
            
            System.out.println("输入: " + prompt);
            System.out.println("输入Token: " + Arrays.toString(promptTokens));
            
            for (int i = 0; i < 30; i++) {
                float[] logits = model.forward(promptTokens);
                
                // 使用温度采样
                int nextToken = sampleWithTemperature(logits, 0.7f);
                
                float maxLogit = logits[0];
                    for (float v : logits) {
                        if (v > maxLogit) maxLogit = v;
                    }
                    System.out.printf("  第%d步: nextToken=%d, maxLogit=%.2f\n", 
                        i, nextToken, maxLogit);
                
                // 确保token ID在有效范围内
                if (nextToken < 0 || nextToken >= finalVocabSize) {
                    nextToken = 0;
                }
                
                String tokenStr = tokenizer.decode(new int[]{nextToken});
                generated.append(tokenStr);
                
                // 限制序列长度
                if (promptTokens.length >= seqLength) {
                    int[] newPrompt = new int[seqLength];
                    System.arraycopy(promptTokens, 1, newPrompt, 0, seqLength - 1);
                    newPrompt[seqLength - 1] = nextToken;
                    promptTokens = newPrompt;
                } else {
                    int[] newPrompt = new int[promptTokens.length + 1];
                    System.arraycopy(promptTokens, 0, newPrompt, 0, promptTokens.length);
                    newPrompt[promptTokens.length] = nextToken;
                    promptTokens = newPrompt;
                }
            }
            
            System.out.println("生成: " + generated.toString());
            System.out.println();
        }
        
        System.out.println("训练完成！");
    }
    
    private static String generateChineseTrainingData() {
        StringBuilder sb = new StringBuilder();
        
        // 添加真实的中文文本段落
        sb.append("人工智能是一门研究如何使计算机模拟人类智能行为的技术科学。");
        sb.append("机器学习是人工智能的一个分支，它使用算法让计算机从数据中学习。");
        sb.append("深度学习是机器学习的一种方法，使用多层神经网络来模拟人脑的学习过程。");
        sb.append("神经网络是由许多神经元组成的网络结构，能够处理复杂的模式识别任务。");
        sb.append("大数据是指无法在一定时间内用常规软件工具进行捕捉、管理和处理的数据集合。");
        sb.append("云计算是一种基于互联网的计算方式，通过网络提供动态可扩展的虚拟化资源。");
        sb.append("物联网是指通过信息传感设备将物品与互联网连接起来，实现智能识别和管理。");
        sb.append("区块链是一种分布式数据库技术，具有去中心化、不可篡改的特点。");
        sb.append("量子计算是利用量子力学现象进行计算的新型计算模式，具有强大的并行计算能力。");
        sb.append("自然语言处理是让计算机理解、处理和生成人类语言的技术。");
        sb.append("计算机视觉是让计算机能够识别和理解图像和视频内容的技术。");
        sb.append("语音识别是将人类语音转换为文字的技术，广泛应用于智能助手和语音输入。");
        sb.append("推荐系统是根据用户的兴趣和行为，为用户推荐相关内容的算法。");
        sb.append("知识图谱是一种结构化的语义知识库，用于存储和管理知识。");
        sb.append("强化学习是一种通过与环境互动来学习最优行为策略的机器学习方法。");
        sb.append("迁移学习是将从一个任务中学到的知识迁移到另一个相关任务的技术。");
        sb.append("对抗训练是一种通过生成对抗网络来提高模型鲁棒性的训练方法。");
        sb.append("自监督学习是一种不需要人工标注数据的学习方法，利用数据本身的结构进行学习。");
        sb.append("多模态学习是指同时处理多种类型数据的学习方法，如图像、文本和语音。");
        sb.append("Transformer是一种基于自注意力机制的深度学习模型，在自然语言处理领域取得了巨大成功。");
        sb.append("注意力机制是一种让模型能够聚焦于输入中重要部分的机制。");
        sb.append("预训练模型是在大规模数据上预先训练好的模型，可以通过微调应用于各种下游任务。");
        sb.append("微调整体预训练模型以适应特定任务的过程。");
        sb.append("推理是指使用训练好的模型进行预测的过程。");
        sb.append("训练是指通过数据调整模型参数的过程。");
        sb.append("参数是模型中可学习的变量。");
        sb.append("算力是指计算机处理数据的能力。");
        sb.append("数据是机器学习的燃料，高质量的数据对于训练好的模型至关重要。");
        sb.append("算法是解决问题的步骤和方法。");
        sb.append("模型是对现实世界的抽象表示。");
        sb.append("架构是指模型的整体结构和设计。");
        sb.append("优化是指改进模型性能的过程。");
        sb.append("收敛是指训练过程中损失函数逐渐下降并趋于稳定的状态。");
        sb.append("梯度是损失函数相对于参数的导数，用于指导参数更新。");
        sb.append("损失函数是衡量模型预测与真实值之间差异的函数。");
        sb.append("激活函数是神经网络中用于引入非线性的函数。");
        sb.append("正则化是一种防止模型过拟合的技术。");
        
        // 添加更多领域的文本
        String[][] domains = {
            {"物理学", "化学", "生物学", "数学", "天文学", "地理学", "医学", "工程学", "计算机科学"},
            {"中国", "历史", "文化", "文明", "传统", "古代", "现代", "近代", "当代"},
            {"经济", "市场", "贸易", "金融", "投资", "消费", "生产", "供给", "需求"},
            {"教育", "学习", "知识", "技能", "学校", "大学", "课程", "研究", "学术"},
            {"健康", "身体", "心理", "运动", "营养", "饮食", "睡眠", "医疗", "康复"},
            {"艺术", "文学", "音乐", "绘画", "建筑", "戏剧", "电影", "诗歌", "小说"},
            {"哲学", "思想", "智慧", "存在", "知识", "真理", "道德", "价值", "伦理"}
        };
        
        Random rng = new Random(42);
        for (int i = 0; i < 30000; i++) {
            int domainIdx = rng.nextInt(domains.length);
            String[] terms = domains[domainIdx];
            
            int sentenceType = rng.nextInt(10);
            
            switch (sentenceType) {
                case 0:
                    String term1 = terms[rng.nextInt(terms.length)];
                    String term2 = terms[rng.nextInt(terms.length)];
                    sb.append(term1).append("是").append(term2).append("的一个重要分支。");
                    break;
                case 1:
                    String term3 = terms[rng.nextInt(terms.length)];
                    String[] adjectives = {"重要的", "核心的", "基础的", "关键的", "现代的", "先进的"};
                    sb.append(term3).append("是").append(adjectives[rng.nextInt(adjectives.length)]).append("领域。");
                    break;
                case 2:
                    String term4 = terms[rng.nextInt(terms.length)];
                    String[] applications = {"广泛应用于", "被用于", "可以用来", "能够实现"};
                    sb.append(term4).append(applications[rng.nextInt(applications.length)]).append("各个行业。");
                    break;
                case 3:
                    String term5 = terms[rng.nextInt(terms.length)];
                    sb.append(term5).append("正在快速发展，推动着社会进步。");
                    break;
                case 4:
                    String term6 = terms[rng.nextInt(terms.length)];
                    String term7 = terms[rng.nextInt(terms.length)];
                    sb.append(term6).append("和").append(term7).append("密切相关，共同发展。");
                    break;
                case 5:
                    String term8 = terms[rng.nextInt(terms.length)];
                    sb.append("学习").append(term8).append("需要不断积累知识和实践。");
                    break;
                case 6:
                    String term9 = terms[rng.nextInt(terms.length)];
                    sb.append("未来，").append(term9).append("将成为重要力量。");
                    break;
                case 7:
                    String term10 = terms[rng.nextInt(terms.length)];
                    sb.append(term10).append("在现代社会中具有重要意义。");
                    break;
                case 8:
                    String term11 = terms[rng.nextInt(terms.length)];
                    sb.append("研究").append(term11).append("有助于推动科学进步。");
                    break;
                case 9:
                    String term12 = terms[rng.nextInt(terms.length)];
                    sb.append(term12).append("的发展离不开创新和实践。");
                    break;
            }
        }
        
        return sb.toString();
    }
    
    private static List<int[]> createSlidingWindowSequences(int[] tokens, int seqLength, int stride) {
        List<int[]> sequences = new ArrayList<>();
        for (int i = 0; i < tokens.length - seqLength; i += stride) {
            int[] seq = new int[seqLength];
            System.arraycopy(tokens, i, seq, 0, seqLength);
            sequences.add(seq);
        }
        return sequences;
    }
    
    private static float computeCrossEntropy(float[] logits, float[] labels) {
        float loss = 0f;
        int vocabSize = logits.length;
        
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            sum += (float) Math.exp(logits[i] - max);
        }
        
        for (int i = 0; i < vocabSize; i++) {
            float prob = (float) Math.exp(logits[i] - max) / sum;
            if (labels[i] > 0) {
                loss -= labels[i] * (float) Math.log(prob + 1e-8f);
            }
        }
        
        return loss;
    }
    
    private static float[] computeCrossEntropyGradient(float[] logits, float[] labels) {
        int vocabSize = logits.length;
        float[] grad = new float[vocabSize];
        
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        
        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            sum += (float) Math.exp(logits[i] - max);
        }
        
        for (int i = 0; i < vocabSize; i++) {
            float prob = (float) Math.exp(logits[i] - max) / sum;
            grad[i] = prob - labels[i];
        }
        
        return grad;
    }
    
    private static void clipGradient(float[] grad, float maxNorm) {
        float norm = 0f;
        for (float g : grad) {
            norm += g * g;
        }
        norm = (float) Math.sqrt(norm);
        
        if (norm > maxNorm) {
            float scale = maxNorm / norm;
            for (int i = 0; i < grad.length; i++) {
                grad[i] *= scale;
            }
        }
    }
    
    private static int sampleWithTemperature(float[] logits, float temperature) {
        // 应用温度
        float[] adjusted = new float[logits.length];
        float max = logits[0];
        for (float val : logits) {
            if (val > max) max = val;
        }
        
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            adjusted[i] = (float) Math.exp((logits[i] - max) / temperature);
            sum += adjusted[i];
        }
        
        // 归一化
        for (int i = 0; i < adjusted.length; i++) {
            adjusted[i] /= sum;
        }
        
        // 采样
        float r = (float) Math.random();
        float cdf = 0f;
        for (int i = 0; i < adjusted.length; i++) {
            cdf += adjusted[i];
            if (r < cdf) {
                return i;
            }
        }
        
        // fallback to argmax
        int idx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                idx = i;
            }
        }
        return idx;
    }
}

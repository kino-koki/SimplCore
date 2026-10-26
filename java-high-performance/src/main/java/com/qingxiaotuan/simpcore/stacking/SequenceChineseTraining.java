package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.model.SequenceModel;
import com.qingxiaotuan.simpcore.training.GradientResult;
import com.qingxiaotuan.simpcore.tokenizer.ChineseBPETokenizer;
import java.util.*;

/**
 * 序列模型中文训练 - 使用SequenceModel进行中文训练
 */
public class SequenceChineseTraining {
    
    public static void main(String[] args) {
        System.out.println("===== 序列模型中文训练 =====");
        
        // 配置参数
        int nExperts = 1;
        int dModel = 128;
        int vocabSize = 4096;
        int seqLength = 64;
        float learningRate = 1e-3f;
        int totalSteps = 5000;
        
        System.out.printf("参数配置: experts=%d, dModel=%d, vocab=%d, lr=%.2e\n", 
            nExperts, dModel, vocabSize, learningRate);
        
        // 创建中文Tokenizer
        ChineseBPETokenizer tokenizer = new ChineseBPETokenizer();
        System.out.printf("中文Tokenizer初始化完成，基础词汇量: %d\n", tokenizer.getVocabSize());
        
        // 生成训练数据
        String trainingText = generateChineseTrainingData();
        System.out.printf("训练文本长度: %,d 字符\n", trainingText.length());
        
        // 训练Tokenizer
        tokenizer.train(trainingText, 1000);
        int finalVocabSize = tokenizer.getVocabSize();
        System.out.printf("Tokenizer训练完成，词汇量: %d\n", finalVocabSize);
        
        // Token化
        int[] allTokens = tokenizer.encode(trainingText);
        System.out.printf("Token化后Token数: %,d\n", allTokens.length);
        
        // 创建训练序列
        List<int[]> sequences = createSlidingWindowSequences(allTokens, seqLength, seqLength / 2);
        System.out.printf("训练样本数: %,d\n", sequences.size());
        
        // 创建序列模型
        SequenceModel model = new SequenceModel(finalVocabSize, dModel, 2, 4);
        System.out.println("序列模型创建完成");
        
        // 训练
        System.out.println("\n开始训练...");
        long startTime = System.currentTimeMillis();
        int stepCount = 0;
        float totalLoss = 0;
        float prevLoss = Float.MAX_VALUE;
        
        while (stepCount < totalSteps) {
            // 获取当前序列
            int[] seq = sequences.get(stepCount % sequences.size());
            
            for (int i = 0; i < seq.length - 1; i++) {
                // 输入：前i+1个token
                int[] input = new int[i + 1];
                System.arraycopy(seq, 0, input, 0, i + 1);
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
                
                // 反向传播
                GradientResult gradResult = model.backward(dLogits);
                
                // 更新参数
                model.updateParametersWithGradient(gradResult.getParameterGradients(), learningRate);
                
                totalLoss += loss;
                stepCount++;
                
                if (stepCount % 50 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    float avgLoss = totalLoss / 50;
                    System.out.printf("[Step %d/%d] Loss: %.4f | LR: %.2e | Time: %ds | Tokens: %dK\n", 
                        stepCount, totalSteps, avgLoss, learningRate, elapsed / 1000, stepCount * seqLength / 1000);
                    
                    if (avgLoss < prevLoss * 0.99) {
                        System.out.printf("         ↓ 损失下降: %.2f%%\n", (1 - avgLoss / prevLoss) * 100);
                    }
                    prevLoss = avgLoss;
                    totalLoss = 0;
                }
                
                if (stepCount >= totalSteps) break;
            }
        }
        
        // 测试生成
        System.out.println("\n测试中文生成...");
        String[] prompts = {"人工智能", "深度学习", "未来科技"};
        for (String prompt : prompts) {
            int[] promptTokens = tokenizer.encode(prompt);
            StringBuilder generated = new StringBuilder(prompt);
            
            System.out.println("输入: " + prompt);
            System.out.println("输入Token: " + java.util.Arrays.toString(promptTokens));
            
            for (int i = 0; i < 20; i++) {
                float[] logits = model.forward(promptTokens);
                int nextToken = argmax(logits);
                
                System.out.printf("  第%d步: nextToken=%d, logits长度=%d, vocabSize=%d\n", 
                    i, nextToken, logits.length, finalVocabSize);
                
                // 确保token ID在有效范围内
                if (nextToken < 0 || nextToken >= finalVocabSize) {
                    System.out.println("  WARNING: token ID超出范围，回退到0");
                    nextToken = 0;
                }
                
                String tokenStr = tokenizer.decode(new int[]{nextToken});
                generated.append(tokenStr);
                
                int[] newPrompt = new int[promptTokens.length + 1];
                System.arraycopy(promptTokens, 0, newPrompt, 0, promptTokens.length);
                newPrompt[promptTokens.length] = nextToken;
                promptTokens = newPrompt;
            }
            
            System.out.println("生成: " + generated.toString());
            System.out.println();
        }
        
        System.out.println("训练完成！");
    }
    
    private static String generateChineseTrainingData() {
        StringBuilder sb = new StringBuilder();
        
        String[][] domains = {
            {"人工智能", "机器学习", "深度学习", "神经网络", "大数据", "云计算", "物联网", "区块链", "量子计算", "自然语言处理", "计算机视觉", "语音识别", "推荐系统", "知识图谱", "强化学习", "迁移学习", "对抗训练", "自监督学习", "多模态学习", "Transformer", "注意力机制", "预训练模型", "微调", "推理", "训练", "参数", "算力", "数据", "算法", "模型", "架构", "优化", "收敛", "梯度", "损失", "激活函数", "正则化"},
            {"物理学", "化学", "生物学", "数学", "天文学", "地理学", "医学", "工程学", "计算机科学", "统计学", "概率论", "微积分", "线性代数", "相对论", "量子力学", "热力学", "电磁学", "光学", "声学", "力学", "材料科学", "环境科学", "生态学", "遗传学", "进化论", "细胞生物学", "分子生物学"},
            {"中国", "历史", "文化", "文明", "传统", "古代", "现代", "近代", "当代", "朝代", "汉朝", "唐朝", "宋朝", "明朝", "清朝", "丝绸之路", "四大发明", "儒家", "道家", "佛家", "诗词", "书法", "绘画", "音乐", "戏曲", "建筑", "服饰", "饮食", "节日", "习俗", "礼仪", "哲学"},
            {"经济", "市场", "贸易", "金融", "投资", "消费", "生产", "供给", "需求", "GDP", "通货膨胀", "失业率", "利率", "汇率", "股票", "债券", "基金", "银行", "保险", "企业", "公司", "创业", "创新", "管理", "营销", "品牌", "战略", "规划", "目标", "实现", "成功", "挑战", "机遇"},
            {"教育", "学习", "知识", "技能", "学校", "大学", "学院", "课程", "考试", "研究", "学术", "论文", "学位", "导师", "学生", "教师", "培训", "辅导", "阅读", "写作", "思考", "实践", "创新", "创造", "探索", "发现", "理解"},
            {"健康", "身体", "心理", "运动", "体育", "锻炼", "跑步", "游泳", "骑行", "瑜伽", "健身", "营养", "饮食", "睡眠", "休息", "医疗", "疾病", "预防", "治疗", "康复", "体检", "疫苗", "药物", "手术", "中医", "西医"},
            {"艺术", "文学", "音乐", "绘画", "雕塑", "建筑", "戏剧", "电影", "舞蹈", "诗歌", "小说", "散文", "故事", "传说", "神话", "寓言", "童话", "传记", "历史", "纪实", "虚构", "想象", "创作", "灵感", "表达", "风格", "流派"},
            {"哲学", "思想", "智慧", "存在", "知识", "真理", "道德", "价值", "伦理", "形而上学", "认识论", "本体论", "辩证法", "逻辑", "理性", "经验", "意识", "自由", "平等", "正义", "幸福", "意义", "目的", "因果", "必然", "偶然"}
        };
        
        Random rng = new Random(42);
        for (int i = 0; i < 50000; i++) {
            int domainIdx = rng.nextInt(domains.length);
            String[] terms = domains[domainIdx];
            
            int sentenceType = rng.nextInt(8);
            
            switch (sentenceType) {
                case 0:
                    String term1 = terms[rng.nextInt(terms.length)];
                    String term2 = terms[rng.nextInt(terms.length)];
                    sb.append(term1).append("是").append(term2).append("的一个重要分支。");
                    break;
                case 1:
                    String term3 = terms[rng.nextInt(terms.length)];
                    String[] adjectives = {"重要的", "核心的", "基础的", "关键的", "现代的", "先进的", "创新的", "前沿的"};
                    sb.append(term3).append("是").append(adjectives[rng.nextInt(adjectives.length)]).append("技术领域。");
                    break;
                case 2:
                    String term4 = terms[rng.nextInt(terms.length)];
                    String[] applications = {"广泛应用于", "被用于", "可以用来", "能够实现", "有助于", "促进了"};
                    sb.append(term4).append(applications[rng.nextInt(applications.length)]).append("各个行业和领域。");
                    break;
                case 3:
                    String term5 = terms[rng.nextInt(terms.length)];
                    sb.append(term5).append("正在快速发展，推动着科技进步和社会变革。");
                    break;
                case 4:
                    String term6 = terms[rng.nextInt(terms.length)];
                    String term7 = terms[rng.nextInt(terms.length)];
                    sb.append(term6).append("和").append(term7).append("密切相关，相辅相成，共同发展。");
                    break;
                case 5:
                    String term8 = terms[rng.nextInt(terms.length)];
                    sb.append("学习").append(term8).append("需要不断积累知识和实践经验。");
                    break;
                case 6:
                    String term9 = terms[rng.nextInt(terms.length)];
                    sb.append("未来，").append(term9).append("将成为推动社会发展的重要力量。");
                    break;
                case 7:
                    String term10 = terms[rng.nextInt(terms.length)];
                    sb.append(term10).append("在现代社会中具有重要意义和广泛影响。");
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
    
    private static int argmax(float[] x) {
        int idx = 0;
        float max = x[0];
        for (int i = 1; i < x.length; i++) {
            if (x[i] > max) {
                max = x[i];
                idx = i;
            }
        }
        return idx;
    }
}

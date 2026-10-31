package com.qingxiaotuan.simpcore.stacking;

import com.qingxiaotuan.simpcore.data.SCDataLoader;
import java.util.*;

public class StackedTrainingTest {
    
    public static void main(String[] args) {
        System.out.println("═══════════════════════════════════════════════════════════════════");
        System.out.println("           参数堆叠技术 - 大规模训练测试");
        System.out.println("═══════════════════════════════════════════════════════════════════\n");
        
        // 大规模配置
        int nExperts = 8;
        int dModel = 256;
        int vocabSize = 8192;
        int batchSize = 8;
        float learningRate = 3e-4f;
        int warmupSteps = 500;
        int totalSteps = 10000;
        
        System.out.println("[1/5] 初始化堆叠训练器...");
        System.out.printf("      专家数量: %d | ", nExperts);
        System.out.printf("dModel: %d | ", dModel);
        System.out.printf("词汇量: %d | ", vocabSize);
        System.out.printf("总参数: ~%dM\n", nExperts * 10);
        
        StackedTrainer trainer = new StackedTrainer(nExperts, dModel, vocabSize, 
                                                    batchSize, learningRate,
                                                    warmupSteps, totalSteps);
        
        System.out.println("\n[2/5] 准备大规模训练数据...");
        
        // 生成大量训练数据
        StringBuilder trainingData = new StringBuilder();
        
        // 添加基础词汇（确保Tokenizer能学习到常用中文）
        String basicChars = "一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾佰仟" +
                           "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥" +
                           "东南西北中上下左右前后内外大小长短高低宽窄" +
                           "天地人日月星辰山川河海风云雨雪雷电雾露霜雹" +
                           "金木水火土风雨雷电霜雪云雾春夏秋冬" +
                           "红黄蓝绿黑白灰紫粉橙棕青" +
                           "爱恨情愁喜怒哀惧忧思悲恐惊" +
                           "仁义礼智信忠孝廉耻勇" +
                           "家国天下人生世界宇宙自然";
        
        // 添加大量中文句子
        String[][] domains = {
            // AI/科技
            {"人工智能", "机器学习", "深度学习", "神经网络", "大数据", "云计算", "物联网", 
             "区块链", "量子计算", "自然语言处理", "计算机视觉", "语音识别", "推荐系统", 
             "知识图谱", "强化学习", "迁移学习", "对抗训练", "自监督学习", "多模态学习", 
             "Transformer", "GPT", "BERT", "注意力机制", "预训练模型", "微调", "推理", 
             "训练", "参数", "算力", "数据", "算法", "模型", "架构", "优化", "收敛"},
            
            // 科学
            {"物理学", "化学", "生物学", "数学", "天文学", "地理学", "地质学", "医学", 
             "工程学", "计算机科学", "统计学", "概率论", "微积分", "线性代数", "拓扑学", 
             "相对论", "量子力学", "热力学", "电磁学", "光学", "声学", "力学", "流体力学", 
             "材料科学", "环境科学", "生态学", "遗传学", "进化论", "细胞生物学"},
            
            // 历史/文化
            {"中国", "历史", "文化", "文明", "传统", "古代", "现代", "近代", "当代", 
             "朝代", "汉朝", "唐朝", "宋朝", "明朝", "清朝", "秦朝", "三国", "南北朝", 
             "丝绸之路", "四大发明", "儒家", "道家", "佛家", "诗词", "书法", "绘画", 
             "音乐", "戏曲", "建筑", "服饰", "饮食", "节日", "习俗"},
            
            // 经济/商业
            {"经济", "市场", "贸易", "金融", "投资", "消费", "生产", "供给", "需求", 
             "GDP", "通货膨胀", "失业率", "利率", "汇率", "股票", "债券", "基金", 
             "银行", "保险", "企业", "公司", "创业", "创新", "管理", "营销", "品牌"},
            
            // 哲学/思想
            {"哲学", "思想", "智慧", "存在", "知识", "真理", "道德", "价值", "伦理", 
             "形而上学", "认识论", "本体论", "辩证法", "逻辑", "理性", "经验", "意识", 
             "自由", "平等", "正义", "幸福", "意义", "目的", "因果", "必然", "偶然"},
            
            // 教育/学习
            {"教育", "学习", "知识", "技能", "学校", "大学", "学院", "课程", "考试", 
             "研究", "学术", "论文", "学位", "导师", "学生", "教师", "培训", "辅导", 
             "阅读", "写作", "思考", "实践", "创新", "创造", "探索", "发现", "理解"},
            
            // 健康/体育
            {"健康", "身体", "心理", "运动", "体育", "锻炼", "跑步", "游泳", "骑行", 
             "瑜伽", "健身", "营养", "饮食", "睡眠", "休息", "医疗", "疾病", "预防", 
             "治疗", "康复", "体检", "疫苗", "药物", "手术", "中医", "西医"},
            
            // 艺术/文学
            {"艺术", "文学", "音乐", "绘画", "雕塑", "建筑", "戏剧", "电影", "舞蹈", 
             "诗歌", "小说", "散文", "故事", "传说", "神话", "寓言", "童话", "传记", 
             "历史", "纪实", "虚构", "想象", "创作", "灵感", "表达", "风格", "流派"}
        };
        
        // 生成大量训练句子
        Random rng = new Random(42);
        for (int i = 0; i < 5000; i++) {
            int domainIdx = rng.nextInt(domains.length);
            String[] terms = domains[domainIdx];
            
            // 生成不同类型的句子
            int sentenceType = rng.nextInt(6);
            
            switch (sentenceType) {
                case 0: // 定义句
                    String term1 = terms[rng.nextInt(terms.length)];
                    String term2 = terms[rng.nextInt(terms.length)];
                    trainingData.append(term1).append("是").append(term2).append("的一个分支。");
                    break;
                case 1: // 描述句
                    String term3 = terms[rng.nextInt(terms.length)];
                    String[] adjectives = {"重要的", "核心的", "基础的", "关键的", "现代的", "先进的", "创新的"};
                    trainingData.append(term3).append("是").append(adjectives[rng.nextInt(adjectives.length)]).append("技术。");
                    break;
                case 2: // 应用句
                    String term4 = terms[rng.nextInt(terms.length)];
                    String[] applications = {"广泛应用于", "被用于", "可以用来", "能够实现", "有助于"};
                    trainingData.append(term4).append(applications[rng.nextInt(applications.length)]).append("各个领域。");
                    break;
                case 3: // 发展句
                    String term5 = terms[rng.nextInt(terms.length)];
                    trainingData.append(term5).append("正在快速发展，推动着科技进步。");
                    break;
                case 4: // 关系句
                    String term6 = terms[rng.nextInt(terms.length)];
                    String term7 = terms[rng.nextInt(terms.length)];
                    trainingData.append(term6).append("和").append(term7).append("密切相关，相辅相成。");
                    break;
                case 5: // 知识句
                    String term8 = terms[rng.nextInt(terms.length)];
                    trainingData.append("学习").append(term8).append("需要不断积累和实践。");
                    break;
            }
            
            trainingData.append(" ");
        }
        
        // 添加一些完整的段落
        String[] paragraphs = {
            "人工智能是计算机科学的一个分支，致力于研究如何使计算机系统能够模拟人类的智能行为。" +
            "机器学习是人工智能的核心技术，通过从数据中学习模式来自动改进算法性能。" +
            "深度学习是机器学习的一种方法，使用多层神经网络来学习复杂的特征表示。",
            
            "Transformer架构是近年来自然语言处理领域最重要的突破之一，" +
            "它引入了自注意力机制，使模型能够在处理序列时关注不同位置的信息。" +
            "预训练模型通过在大规模文本数据上进行训练，学习到通用的语言表示，" +
            "然后可以通过微调适应特定的下游任务。",
            
            "参数堆叠技术是一种创新的模型训练方法，通过训练多个专家模型并将它们的输出集成，" +
            "可以在有限的计算资源下实现更大规模的模型效果。路由层根据输入的语义特征动态选择专家，" +
            "集成层将多个专家的输出融合为统一的结果。",
            
            "物理学是研究物质运动规律的科学，探索宇宙的基本法则。" +
            "相对论揭示了时间和空间的相对性，量子力学描述了微观粒子的行为。" +
            "热力学研究热量和能量的转换，电磁学研究电和磁的相互作用。",
            
            "数学是科学的语言，研究数量、结构、变化和空间。" +
            "代数研究数和符号的运算，几何研究空间和形状。" +
            "微积分研究变化和运动，概率论研究随机事件的可能性。",
            
            "中国是世界上历史最悠久的文明之一，拥有五千年灿烂文化。" +
            "古代中国发明了造纸术、印刷术、火药和指南针，对世界文明做出了重要贡献。" +
            "丝绸之路连接了东西方文明，促进了贸易和文化交流。",
            
            "教育是培养人的过程，传授知识、技能和价值观。" +
            "基础教育是国民教育的基础，高等教育培养专业人才。" +
            "终身学习是现代社会的要求，知识更新速度不断加快。",
            
            "健康是幸福生活的基础，包括身体健康和心理健康。" +
            "体育锻炼促进身体健康，增强体质和免疫力。" +
            "合理饮食和充足睡眠是维持健康的重要因素。"
        };
        
        for (String p : paragraphs) {
            for (int j = 0; j < 10; j++) {
                trainingData.append(p).append(" ");
            }
        }
        
        String trainingText = trainingData.toString();
        System.out.printf("      训练文本长度: %,d 字符\n", trainingText.length());
        
        // 训练Tokenizer（增加合并次数）
        trainer.trainTokenizer(trainingText, 1000);
        
        int[] allTokens = trainer.getTokenizer().encode(trainingText);
        System.out.printf("      Token化后Token数: %,d\n", allTokens.length);
        
        // 创建滑动窗口序列
        List<int[]> sequences = SCDataLoader.createSlidingWindowSequences(allTokens, 128, 64);
        System.out.printf("      训练样本数: %,d\n", sequences.size());
        
        SCDataLoader dataLoader = new SCDataLoader(sequences, batchSize, 128);
        
        System.out.println("\n[3/5] 开始大规模训练...");
        System.out.println("      预计训练步数: 2000\n");
        
        int stepsToRun = 2000;
        StackedTrainer.TrainingStats stats = trainer.train(dataLoader, stepsToRun);
        
        System.out.println("\n[4/5] 训练完成！最终统计:");
        System.out.printf("      总步数: %,d\n", stats.totalSteps);
        System.out.printf("      平均损失: %.4f\n", stats.averageLoss);
        System.out.printf("      处理Token数: %,d\n", stats.totalTokens);
        System.out.printf("      耗时: %.1fs\n", stats.elapsedSeconds);
        
        System.out.println("\n[5/5] 测试文本生成能力...");
        
        String[] prompts = {
            "人工智能是",
            "深度学习的核心技术包括",
            "未来科技发展趋势",
            "量子计算与经典计算的区别",
            "自然语言处理的应用场景",
            "参数堆叠技术的优势",
            "中国古代四大发明",
            "健康生活的重要性",
            "数学在科学中的作用",
            "教育的意义"
        };
        
        for (String prompt : prompts) {
            String generated = trainer.generate(prompt, 80);
            System.out.printf("\n      Prompt: %s\n", prompt);
            System.out.printf("      生成: %s\n", generated);
        }
        
        System.out.println("\n═══════════════════════════════════════════════════════════════════");
        System.out.println("                    参数堆叠大规模训练测试完成！");
        System.out.println("═══════════════════════════════════════════════════════════════════");
    }
}

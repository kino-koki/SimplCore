package com.qingxiaotuan.simpcore.tokenizer;

import java.util.*;

public class ChineseBPETokenizer extends BPETokenizer {
    
    private static final String CHINESE_BASIC_CHARS = 
        "一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾佰仟" +
        "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥" +
        "东南西北中上下左右前后内外大小长短高低宽窄" +
        "天地人日月星辰山川河海风云雨雪雷电雾露霜雹" +
        "金木水火土风雨雷电霜雪云雾春夏秋冬" +
        "红黄蓝绿黑白灰紫粉橙棕青" +
        "爱恨情愁喜怒哀惧忧思悲恐惊" +
        "仁义礼智信忠孝廉耻勇" +
        "家国天下人生世界宇宙自然";
    
    private static final String[] CHINESE_COMMON_WORDS = getCommonWords();
    
    private static String[] getCommonWords() {
        return new String[] {
            "人工智能", "机器学习", "深度学习", "神经网络", "大数据", "云计算",
            "物联网", "区块链", "量子计算", "自然语言", "计算机", "互联网",
            "科技", "技术", "算法", "数据", "模型", "系统", "应用", "开发",
            "研究", "创新", "智能", "自动化", "信息化", "数字化", "智能化",
            "中国", "国家", "人民", "社会", "经济", "文化", "历史", "传统",
            "教育", "学习", "知识", "科学", "工程", "数学", "物理", "化学",
            "生物", "医学", "健康", "生活", "工作", "发展", "进步", "创新",
            "未来", "现在", "过去", "时间", "空间", "世界", "宇宙", "地球",
            "人类", "生命", "思维", "意识", "语言", "文字", "图像", "声音",
            "视频", "音频", "信息", "通信", "网络", "安全", "隐私", "保护",
            "产品", "服务", "用户", "市场", "商业", "金融", "投资", "消费",
            "生产", "供给", "需求", "管理", "领导", "团队", "合作", "竞争",
            "战略", "规划", "目标", "实现", "成功", "失败", "挑战", "机遇",
            "问题", "解决方案", "方法", "理论", "实践", "经验", "教训",
            "分析", "设计", "实现", "测试", "优化", "改进", "升级", "更新",
            "发布", "部署", "运行", "维护", "监控", "评估", "报告", "总结",
            "思考", "讨论", "交流", "分享", "学习", "培训", "教育", "辅导",
            "阅读", "写作", "编辑", "翻译", "解释", "说明", "介绍", "描述",
            "理解", "掌握", "应用", "实践", "创新", "创造", "发现", "探索",
            "支持", "帮助", "服务", "贡献", "参与", "合作", "协调", "配合",
            "领导", "管理", "组织", "规划", "决策", "执行", "监督", "评估",
            "分析", "预测", "判断", "决策", "行动", "实施", "完成", "达成",
            "目标", "任务", "项目", "计划", "进度", "质量", "效率", "效果",
            "成本", "收益", "价值", "重要性", "必要性", "可行性", "有效性",
            "影响", "作用", "意义", "目的", "原因", "结果", "过程", "方法",
            "技术", "工具", "平台", "框架", "系统", "架构", "模式", "标准",
            "规范", "流程", "步骤", "阶段", "环节", "因素", "条件", "环境",
            "资源", "能力", "水平", "程度", "范围", "领域", "行业", "市场",
            "趋势", "方向", "前景", "机遇", "挑战", "风险", "问题", "对策",
            "建议", "方案", "措施", "办法", "途径", "策略", "技巧", "经验",
            "知识", "信息", "数据", "内容", "材料", "文档", "报告", "论文",
            "书籍", "文章", "新闻", "资讯", "消息", "公告", "通知", "声明",
            "评论", "观点", "意见", "想法", "思考", "分析", "研究", "探讨",
            "讨论", "交流", "沟通", "对话", "会议", "论坛", "研讨会", "讲座",
            "培训", "课程", "学习", "教育", "教学", "指导", "辅导", "帮助",
            "支持", "服务", "咨询", "解答", "回应", "反馈", "建议", "意见",
            "感谢", "祝贺", "欢迎", "问候", "告别", "祝福", "鼓励", "表扬",
            "批评", "建议", "提醒", "警告", "通知", "告知", "说明", "解释",
            "介绍", "推荐", "分享", "展示", "演示", "讲解", "培训", "指导",
            "使用", "操作", "设置", "配置", "管理", "维护", "升级", "更新",
            "安装", "卸载", "启动", "停止", "运行", "调试", "测试", "验证",
            "开发", "设计", "实现", "部署", "发布", "上线", "运维", "监控",
            "分析", "优化", "改进", "修复", "解决", "处理", "应对", "防范",
            "安全", "防护", "加密", "解密", "认证", "授权", "访问", "权限",
            "网络", "连接", "通信", "传输", "存储", "备份", "恢复", "同步",
            "数据", "信息", "内容", "媒体", "文件", "文档", "图片", "视频",
            "音频", "文本", "代码", "程序", "软件", "硬件", "设备", "系统",
            "平台", "工具", "应用", "服务", "网站", "应用", "功能", "特性",
            "界面", "交互", "体验", "性能", "效率", "速度", "质量", "稳定",
            "可靠", "安全", "兼容", "扩展", "灵活", "易用", "便捷", "高效",
            "智能", "自动", "快速", "准确", "全面", "深入", "细致", "专业",
            "创新", "领先", "优秀", "卓越", "杰出", "重要", "关键", "核心",
            "基础", "根本", "本质", "原理", "机制", "规律", "方法", "技术",
            "理论", "实践", "经验", "教训", "总结", "反思", "展望", "规划",
            "战略", "战术", "执行", "落实", "推进", "实施", "完成", "达成",
            "实现", "取得", "获得", "达到", "超过", "领先", "保持", "巩固",
            "提升", "增强", "改善", "优化", "改进", "完善", "创新", "突破",
            "发展", "进步", "成长", "壮大", "扩展", "扩张", "深化", "细化",
            "整合", "融合", "协同", "联动", "配合", "支持", "保障", "促进",
            "推动", "带动", "引领", "指导", "帮助", "服务", "贡献", "参与",
            "合作", "沟通", "交流", "协调", "协商", "谈判", "达成", "签署",
            "执行", "履行", "遵守", "维护", "保障", "保护", "维护", "捍卫",
            "建设", "构建", "打造", "创造", "创新", "开发", "利用", "使用",
            "发挥", "体现", "展示", "表达", "传播", "扩散", "影响", "作用",
            "产生", "引起", "导致", "造成", "形成", "建立", "完善", "健全",
            "规范", "标准", "制度", "机制", "体系", "系统", "结构", "框架",
            "模式", "方式", "方法", "途径", "手段", "工具", "技术", "设备",
            "资源", "条件", "环境", "背景", "现状", "趋势", "方向", "前景",
            "机遇", "挑战", "问题", "困难", "障碍", "风险", "威胁", "危机",
            "应对", "解决", "处理", "克服", "消除", "降低", "减少", "控制",
            "预防", "防范", "预警", "监测", "评估", "分析", "判断", "决策",
            "行动", "措施", "方案", "策略", "计划", "安排", "部署", "实施",
            "推进", "落实", "执行", "监督", "检查", "验收", "评估", "总结",
            "汇报", "报告", "沟通", "反馈", "调整", "优化", "改进", "完善",
            "持续", "不断", "逐步", "稳步", "快速", "高效", "精准", "精细",
            "全面", "深入", "系统", "科学", "专业", "规范", "严格", "认真",
            "负责", "积极", "主动", "热情", "耐心", "细心", "严谨", "务实",
            "创新", "开放", "包容", "合作", "共赢", "共享", "协同", "联动",
            "智慧", "智能", "绿色", "低碳", "环保", "可持续", "和谐", "稳定",
            "安全", "健康", "幸福", "美好", "繁荣", "富强", "文明", "进步",
            "民主", "自由", "平等", "公正", "法治", "诚信", "友善", "敬业",
            "爱国", "创新", "协调", "绿色", "开放", "共享", "发展", "改革",
            "稳定", "统一", "团结", "奋斗", "实干", "担当", "奉献", "服务",
            "人民", "国家", "民族", "社会", "时代", "未来", "梦想", "希望",
            "信念", "信心", "勇气", "毅力", "坚持", "努力", "奋斗", "拼搏",
            "成功", "胜利", "成就", "辉煌", "伟大"
        };
    }
    
    public ChineseBPETokenizer() {
        super(); // 调用父类构造函数初始化基本词汇
        initializeChineseVocab(); // 添加中文词汇
    }
    
    private void initializeChineseVocab() {
        for (char c : CHINESE_BASIC_CHARS.toCharArray()) {
            String token = String.valueOf(c);
            if (!vocab.containsKey(token)) {
                vocab.put(token, nextId);
                idToToken.put(nextId, token);
                nextId++;
            }
        }
        
        for (String word : CHINESE_COMMON_WORDS) {
            if (!vocab.containsKey(word)) {
                vocab.put(word, nextId);
                idToToken.put(nextId, word);
                nextId++;
            }
        }
    }
    
    public void train(String text, int iterations) {
        List<String> tokens = new ArrayList<>();
        
        int i = 0;
        while (i < text.length()) {
            boolean found = false;
            
            for (int len = Math.min(8, text.length() - i); len >= 1; len--) {
                String candidate = text.substring(i, i + len);
                if (vocab.containsKey(candidate)) {
                    tokens.add(candidate);
                    i += len;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                tokens.add(String.valueOf(text.charAt(i)));
                i++;
            }
        }
        
        for (int iter = 0; iter < iterations; iter++) {
            Map<String, Integer> pairCounts = new HashMap<>();
            
            for (int j = 0; j < tokens.size() - 1; j++) {
                String pair = tokens.get(j) + "|" + tokens.get(j + 1);
                pairCounts.put(pair, pairCounts.getOrDefault(pair, 0) + 1);
            }
            
            if (pairCounts.isEmpty()) break;
            
            String bestPair = null;
            int maxCount = 0;
            for (Map.Entry<String, Integer> entry : pairCounts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    bestPair = entry.getKey();
                }
            }
            
            if (bestPair == null || maxCount < 2) break;
            
            String[] parts = bestPair.split("\\|");
            String newToken = parts[0] + parts[1];
            
            vocab.put(newToken, nextId);
            idToToken.put(nextId, newToken);
            merges.put(bestPair, newToken);
            nextId++;
            
            List<String> newTokens = new ArrayList<>();
            int j = 0;
            while (j < tokens.size()) {
                if (j < tokens.size() - 1 && tokens.get(j).equals(parts[0]) && tokens.get(j + 1).equals(parts[1])) {
                    newTokens.add(newToken);
                    j += 2;
                } else {
                    newTokens.add(tokens.get(j));
                    j++;
                }
            }
            tokens = newTokens;
        }
        
        System.out.printf("中文BPE训练完成. 词汇量: %d, 合并次数: %d%n", vocab.size(), merges.size());
    }
    
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        
        int i = 0;
        while (i < text.length()) {
            boolean found = false;
            
            for (int len = Math.min(8, text.length() - i); len >= 1; len--) {
                String candidate = text.substring(i, i + len);
                if (vocab.containsKey(candidate)) {
                    ids.add(vocab.get(candidate));
                    i += len;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                ids.add(vocab.getOrDefault(String.valueOf(text.charAt(i)), vocab.get("<|unknown|>")));
                i++;
            }
        }
        
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }
    
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String token = idToToken.getOrDefault(id, "<|unknown|>");
            sb.append(token);
        }
        return sb.toString();
    }
    
    public int getVocabSize() {
        return vocab.size();
    }
    
    public Set<String> getVocab() {
        return vocab.keySet();
    }
}

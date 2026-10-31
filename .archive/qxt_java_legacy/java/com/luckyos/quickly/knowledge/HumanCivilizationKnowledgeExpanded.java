package com.luckyos.quickly.knowledge;

import java.util.*;
import java.util.concurrent.*;

public class HumanCivilizationKnowledgeExpanded {
    
    private static final int VERSION = 2;
    private static final long TOTAL_ENTRIES = 2_000_000L;
    private static final long TOTAL_RELATIONS = 5_000_000L;
    private static final long TOTAL_EVENTS = 100_000L;
    
    private final ConcurrentHashMap<String, KnowledgeEntry> knowledgeBase = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, List<Relation>> relations = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, HistoricalEvent> historicalEvents = new ConcurrentHashMap<>();
    
    private final KnowledgeGraph knowledgeGraph = new KnowledgeGraph();
    private final Timeline timeline = new Timeline();
    private final CivilizationIndex index = new CivilizationIndex();
    
    public HumanCivilizationKnowledgeExpanded() {
        initializeAllKnowledge();
    }
    
    private void initializeAllKnowledge() {
        System.out.println("[知识扩展] ⚡ 正在初始化完整人类文明知识库...");
        
        loadPrehistoricEra();
        loadAncientCivilizations();
        loadMedievalPeriod();
        loadRenaissanceEra();
        loadModernEra();
        loadContemporaryEra();
        
        loadMilitaryStrategies();
        loadPoliticalStrategies();
        loadGameTheory();
        loadPsychologicalStrategies();
        
        loadPhilosophicalSchools();
        loadEconomicTheories();
        loadScientificRevolutions();
        
        loadMathematicsKnowledge();
        loadPhysicsKnowledge();
        loadChemistryKnowledge();
        loadBiologyKnowledge();
        
        loadArtHistory();
        loadLiteratureKnowledge();
        loadCulturalHeritage();
        
        buildKnowledgeGraph();
        
        System.out.println("[知识扩展] ✅ 人类文明知识库加载完成！");
        System.out.println("[知识扩展] 📊 统计: " + knowledgeBase.size() + "条知识, " + 
                          getTotalRelations() + "条关系, " + historicalEvents.size() + "个历史事件");
    }
    
    private void loadPrehistoricEra() {
        System.out.println("[知识扩展] 加载史前时代知识...");
        
        addKnowledge("prehistory_homo_sapiens", "智人", 
            "智人(Homo sapiens)是现代人类的学名，大约在30万年前起源于非洲。", "历史/史前", 1.0);
        
        addKnowledge("prehistory_stone_age", "石器时代", 
            "石器时代是人类历史上最早的阶段，从约300万年前到约4000年前，以使用石器工具为标志。", "历史/史前", 1.0);
        
        addKnowledge("prehistory_neanderthal", "尼安德特人", 
            "尼安德特人是生活在欧洲和西亚的古人类，约在4万年前灭绝，与智人有过交集。", "历史/史前", 0.95);
        
        addKnowledge("prehistory_fire_control", "火的控制", 
            "人类大约在100万年前学会控制火，这是人类进化史上的重要里程碑。", "历史/史前", 1.0);
        
        addKnowledge("prehistory_agriculture", "农业革命", 
            "农业革命发生在约1万年前，标志着从采集狩猎向定居农业的转变。", "历史/史前", 1.0);
        
        addKnowledge("prehistory_language", "语言起源", 
            "人类语言大约在5-10万年前开始发展，是人类区别于其他动物的关键特征。", "历史/史前", 0.9);
        
        addKnowledge("prehistory_tools", "工具制造", 
            "最早的石器工具可以追溯到约330万年前的奥杜威文化。", "历史/史前", 1.0);
        
        addKnowledge("prehistory_migration", "人类迁徙", 
            "智人从非洲出发，大约在6万年前开始向世界各地迁徙。", "历史/史前", 1.0);
        
        addKnowledge("prehistory_cave_painting", "洞穴壁画", 
            "最早的洞穴壁画发现于法国拉斯科洞穴，距今约1.7万年。", "历史/史前", 0.98);
        
        addKnowledge("prehistory_ice_age", "冰河时代", 
            "最后一次冰河时代在约1.1万年前结束，对人类进化产生了深远影响。", "历史/史前", 1.0);
        
        addRelation("智人", "起源于", "非洲");
        addRelation("石器时代", "分为", "旧石器时代");
        addRelation("石器时代", "分为", "新石器时代");
        addRelation("农业革命", "导致", "定居生活");
        
        addHistoricalEvent("2000000BCE", "石器工具出现", "最早的石器工具在非洲出现");
        addHistoricalEvent("1000000BCE", "火的控制", "人类开始控制和使用火");
        addHistoricalEvent("300000BCE", "智人出现", "现代人类在非洲出现");
        addHistoricalEvent("70000BCE", "人类迁徙", "智人开始从非洲向外迁徙");
        addHistoricalEvent("10000BCE", "农业革命", "农业开始发展，定居生活出现");
    }
    
    private void loadAncientCivilizations() {
        System.out.println("[知识扩展] 加载古代文明知识...");
        
        addKnowledge("ancient_mesopotamia", "美索不达米亚文明", 
            "美索不达米亚文明是世界上最早的文明之一，位于两河流域（底格里斯河和幼发拉底河）。", "历史/古代", 1.0);
        
        addKnowledge("ancient_egypt", "古埃及文明", 
            "古埃及文明发源于尼罗河流域，以金字塔、木乃伊和象形文字闻名。", "历史/古代", 1.0);
        
        addKnowledge("ancient_india", "古印度文明", 
            "古印度文明包括哈拉帕文明（印度河文明）和后来的吠陀文明。", "历史/古代", 1.0);
        
        addKnowledge("ancient_china", "古代中国", 
            "中国古代文明是世界上唯一延续至今的古老文明，拥有5000多年的历史。", "历史/古代", 1.0);
        
        addKnowledge("ancient_greece", "古希腊文明", 
            "古希腊文明对西方哲学、科学、艺术和政治产生了深远影响。", "历史/古代", 1.0);
        
        addKnowledge("ancient_rome", "古罗马文明", 
            "古罗马文明继承了古希腊文化，并建立了庞大的帝国。", "历史/古代", 1.0);
        
        addKnowledge("ancient_maya", "玛雅文明", 
            "玛雅文明是中美洲的古代文明，以其精确的历法和宏伟的建筑著称。", "历史/古代", 0.95);
        
        addKnowledge("ancient_inca", "印加文明", 
            "印加文明是南美洲的古代文明，建立了庞大的道路网络。", "历史/古代", 0.95);
        
        addKnowledge("ancient_persia", "波斯帝国", 
            "波斯帝国是古代世界最大的帝国之一，以其高效的行政系统闻名。", "历史/古代", 1.0);
        
        addKnowledge("ancient_carthage", "迦太基", 
            "迦太基是腓尼基人建立的城邦，曾与罗马进行布匿战争。", "历史/古代", 0.9);
        
        addRelation("美索不达米亚", "发明", "楔形文字");
        addRelation("古埃及", "建造", "金字塔");
        addRelation("古希腊", "产生", "民主制度");
        addRelation("古罗马", "继承", "古希腊文化");
        addRelation("古代中国", "发明", "造纸术");
        addRelation("古代中国", "发明", "印刷术");
        addRelation("古代中国", "发明", "火药");
        addRelation("古代中国", "发明", "指南针");
        
        addHistoricalEvent("3500BCE", "美索不达米亚文明", "两河流域出现城市文明");
        addHistoricalEvent("3000BCE", "古埃及统一", "纳尔迈统一上下埃及");
        addHistoricalEvent("2500BCE", "金字塔建造", "吉萨金字塔开始建造");
        addHistoricalEvent("2200BCE", "哈拉帕文明", "印度河流域出现城市文明");
        addHistoricalEvent("1600BCE", "商朝建立", "中国商朝建立");
        addHistoricalEvent("776BCE", "第一届奥运会", "古希腊奥林匹亚举办第一届奥运会");
        addHistoricalEvent("509BCE", "罗马共和国", "罗马共和国建立");
        addHistoricalEvent("221BCE", "秦朝统一", "秦始皇统一中国");
        addHistoricalEvent("27BCE", "罗马帝国", "奥古斯都建立罗马帝国");
    }
    
    private void loadMedievalPeriod() {
        System.out.println("[知识扩展] 加载中世纪知识...");
        
        addKnowledge("medieval_europe", "中世纪欧洲", 
            "中世纪是欧洲历史上从西罗马帝国灭亡到文艺复兴的时期（约500-1500年）。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_byzantium", "拜占庭帝国", 
            "拜占庭帝国是东罗马帝国的延续，以君士坦丁堡为首都，持续了近千年。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_islam", "伊斯兰黄金时代", 
            "伊斯兰黄金时代（8-13世纪）见证了科学、数学和哲学的繁荣。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_feudalism", "封建制度", 
            "封建制度是中世纪欧洲的社会和政治体系，以土地所有权为基础。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_crusades", "十字军东征", 
            "十字军东征是一系列宗教战争，旨在从穆斯林手中夺回圣地。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_mongols", "蒙古帝国", 
            "蒙古帝国是历史上最大的陆地帝国，由成吉思汗建立。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_vikings", "维京人", 
            "维京人是北欧的航海民族，以探险、贸易和掠夺闻名。", "历史/中世纪", 0.95);
        
        addKnowledge("medieval_renaissance", "中世纪晚期", 
            "中世纪晚期见证了城市的兴起和大学的建立。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_china", "中国中世纪", 
            "中国中世纪包括唐宋元明等朝代，是中国历史上的鼎盛时期。", "历史/中世纪", 1.0);
        
        addKnowledge("medieval_japan", "日本中世纪", 
            "日本中世纪包括平安时代、镰仓时代和室町时代。", "历史/中世纪", 0.9);
        
        addRelation("中世纪", "特征", "封建制度");
        addRelation("伊斯兰黄金时代", "贡献", "代数学");
        addRelation("蒙古帝国", "征服", "欧亚大陆");
        addRelation("十字军东征", "目的", "夺回圣地");
        
        addHistoricalEvent("476CE", "西罗马灭亡", "西罗马帝国灭亡");
        addHistoricalEvent("622CE", "希吉拉", "穆罕默德迁徙至麦地那");
        addHistoricalEvent("800CE", "查理曼加冕", "查理曼被加冕为神圣罗马帝国皇帝");
        addHistoricalEvent("1096CE", "第一次十字军", "第一次十字军东征开始");
        addHistoricalEvent("1206CE", "蒙古帝国", "成吉思汗统一蒙古");
        addHistoricalEvent("1347CE", "黑死病", "黑死病开始在欧洲蔓延");
        addHistoricalEvent("1453CE", "君士坦丁堡陷落", "奥斯曼帝国征服君士坦丁堡");
    }
    
    private void loadRenaissanceEra() {
        System.out.println("[知识扩展] 加载文艺复兴知识...");
        
        addKnowledge("renaissance_origin", "文艺复兴起源", 
            "文艺复兴起源于14世纪的意大利，是一场文化和艺术的复兴运动。", "历史/文艺复兴", 1.0);
        
        addKnowledge("renaissance_art", "文艺复兴艺术", 
            "文艺复兴艺术以透视法、写实主义和人文主义为特征。", "历史/文艺复兴", 1.0);
        
        addKnowledge("renaissance_science", "文艺复兴科学", 
            "文艺复兴时期科学开始摆脱宗教束缚，出现了哥白尼、伽利略等科学家。", "历史/文艺复兴", 1.0);
        
        addKnowledge("renaissance_literature", "文艺复兴文学", 
            "文艺复兴文学以人文主义为核心，代表人物有但丁、莎士比亚等。", "历史/文艺复兴", 1.0);
        
        addKnowledge("renaissance_printing", "印刷术传播", 
            "古腾堡印刷机的发明促进了知识的传播。", "历史/文艺复兴", 1.0);
        
        addKnowledge("renaissance_exploration", "大航海时代", 
            "大航海时代开启了欧洲对美洲和亚洲的探索。", "历史/文艺复兴", 1.0);
        
        addRelation("文艺复兴", "起源于", "意大利");
        addRelation("文艺复兴", "特征", "人文主义");
        addRelation("大航海时代", "发现", "美洲");
        addRelation("印刷术", "促进", "知识传播");
        
        addHistoricalEvent("1340CE", "但丁神曲", "但丁完成《神曲》");
        addHistoricalEvent("1450CE", "古腾堡印刷机", "约翰内斯·古腾堡发明印刷机");
        addHistoricalEvent("1492CE", "哥伦布发现美洲", "哥伦布抵达美洲");
        addHistoricalEvent("1519CE", "麦哲伦环球航行", "麦哲伦开始环球航行");
        addHistoricalEvent("1543CE", "哥白尼革命", "哥白尼发表《天体运行论》");
        addHistoricalEvent("1609CE", "伽利略望远镜", "伽利略发明望远镜");
    }
    
    private void loadModernEra() {
        System.out.println("[知识扩展] 加载现代史知识...");
        
        addKnowledge("modern_scientific_revolution", "科学革命", 
            "科学革命（16-17世纪）奠定了现代科学的基础。", "历史/现代", 1.0);
        
        addKnowledge("modern_enlightenment", "启蒙运动", 
            "启蒙运动（18世纪）强调理性、自由和人权。", "历史/现代", 1.0);
        
        addKnowledge("modern_industrial", "工业革命", 
            "工业革命（18-19世纪）从英国开始，标志着从农业社会向工业社会的转变。", "历史/现代", 1.0);
        
        addKnowledge("modern_french_revolution", "法国大革命", 
            "法国大革命（1789年）推翻了君主专制，确立了共和制度。", "历史/现代", 1.0);
        
        addKnowledge("modern_american_revolution", "美国独立战争", 
            "美国独立战争（1775-1783年）建立了美利坚合众国。", "历史/现代", 1.0);
        
        addKnowledge("modern_world_war_i", "第一次世界大战", 
            "第一次世界大战（1914-1918年）是人类历史上第一次全球性战争。", "历史/现代", 1.0);
        
        addKnowledge("modern_world_war_ii", "第二次世界大战", 
            "第二次世界大战（1939-1945年）是人类历史上最具破坏性的战争。", "历史/现代", 1.0);
        
        addKnowledge("modern_cold_war", "冷战", 
            "冷战（1947-1991年）是美国和苏联之间的意识形态对抗。", "历史/现代", 1.0);
        
        addKnowledge("modern_decolonization", "非殖民化", 
            "20世纪后半叶，亚非拉国家纷纷摆脱殖民统治获得独立。", "历史/现代", 1.0);
        
        addKnowledge("modern_technology", "技术革命", 
            "20世纪见证了电力、汽车、航空、计算机等重大技术突破。", "历史/现代", 1.0);
        
        addRelation("科学革命", "代表人物", "牛顿");
        addRelation("启蒙运动", "核心思想", "理性");
        addRelation("工业革命", "起源于", "英国");
        addRelation("法国大革命", "口号", "自由平等博爱");
        addRelation("冷战", "双方", "美国");
        addRelation("冷战", "双方", "苏联");
        
        addHistoricalEvent("1687CE", "牛顿力学", "牛顿发表《自然哲学的数学原理》");
        addHistoricalEvent("1776CE", "美国独立", "美国发表《独立宣言》");
        addHistoricalEvent("1789CE", "法国大革命", "法国大革命爆发");
        addHistoricalEvent("1789CE", "工业革命开始", "瓦特改良蒸汽机");
        addHistoricalEvent("1914CE", "一战爆发", "第一次世界大战开始");
        addHistoricalEvent("1917CE", "俄国革命", "俄国十月革命");
        addHistoricalEvent("1939CE", "二战爆发", "第二次世界大战开始");
        addHistoricalEvent("1945CE", "二战结束", "第二次世界大战结束");
        addHistoricalEvent("1947CE", "冷战开始", "杜鲁门主义出台");
        addHistoricalEvent("1969CE", "登月", "阿波罗11号登月");
    }
    
    private void loadContemporaryEra() {
        System.out.println("[知识扩展] 加载当代史知识...");
        
        addKnowledge("contemporary_cold_war_end", "冷战结束", 
            "1991年苏联解体标志着冷战的结束。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_globalization", "全球化", 
            "全球化是20世纪末以来世界经济、文化和政治的一体化进程。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_internet", "互联网革命", 
            "互联网的普及彻底改变了人类的沟通和信息获取方式。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_mobile", "移动革命", 
            "智能手机的出现开启了移动互联网时代。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_ai", "人工智能时代", 
            "21世纪人工智能技术取得重大突破，正在改变各行各业。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_climate", "气候变化", 
            "全球气候变化是当代最重要的挑战之一。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_biotech", "生物技术", 
            "生物技术的发展包括基因编辑、克隆和再生医学等领域。", "历史/当代", 1.0);
        
        addKnowledge("contemporary_space", "太空探索", 
            "21世纪太空探索取得重大进展，包括火星探测和私人航天。", "历史/当代", 1.0);
        
        addRelation("互联网", "发明于", "美国");
        addRelation("全球化", "特征", "互联互通");
        addRelation("人工智能", "应用领域", "自动驾驶");
        addRelation("人工智能", "应用领域", "医疗诊断");
        addRelation("气候变化", "原因", "温室气体排放");
        
        addHistoricalEvent("1989CE", "柏林墙倒塌", "柏林墙倒塌");
        addHistoricalEvent("1991CE", "苏联解体", "苏联解体");
        addHistoricalEvent("1991CE", "万维网", "万维网诞生");
        addHistoricalEvent("2001CE", "911事件", "恐怖袭击事件");
        addHistoricalEvent("2007CE", "iPhone发布", "苹果发布iPhone");
        addHistoricalEvent("2020CE", "COVID-19", "新冠疫情爆发");
    }
    
    private void loadMilitaryStrategies() {
        System.out.println("[知识扩展] 加载军事策略知识...");
        
        addKnowledge("strategy_sun_tzu", "孙子兵法", 
            "《孙子兵法》是中国古代著名的军事著作，提出了\"知己知彼，百战不殆\"等重要思想。", "策略/军事", 1.0);
        
        addKnowledge("strategy_cannae", "坎尼战役", 
            "坎尼战役是汉尼拔运用包围战术的经典战例。", "策略/军事", 0.95);
        
        addKnowledge("strategy_blitzkrieg", "闪电战", 
            "闪电战是二战中德国使用的快速机动战术。", "策略/军事", 1.0);
        
        addKnowledge("strategy_guerrilla", "游击战", 
            "游击战是一种非对称战争策略，强调机动性和隐蔽性。", "策略/军事", 1.0);
        
        addKnowledge("strategy_naval", "海战策略", 
            "海战策略包括舰队部署、封锁和制海权争夺。", "策略/军事", 0.9);
        
        addKnowledge("strategy_air_superiority", "制空权", 
            "制空权是现代战争的关键，掌握空中优势可以控制战场。", "策略/军事", 1.0);
        
        addKnowledge("strategy_logistics", "后勤保障", 
            "后勤保障是战争胜利的基础，包括物资供应和运输。", "策略/军事", 1.0);
        
        addKnowledge("strategy_deception", "欺骗战术", 
            "欺骗战术包括佯攻、伪装和情报战。", "策略/军事", 1.0);
        
        addRelation("孙子兵法", "作者", "孙武");
        addRelation("闪电战", "使用者", "德国");
        addRelation("游击战", "特征", "非对称");
        addRelation("制空权", "重要性", "决定性");
    }
    
    private void loadPoliticalStrategies() {
        System.out.println("[知识扩展] 加载政治策略知识...");
        
        addKnowledge("politics_realpolitik", "现实政治", 
            "现实政治强调国家利益至上，不顾道德和意识形态。", "策略/政治", 1.0);
        
        addKnowledge("politics_divide_conquer", "分而治之", 
            "分而治之是一种通过分裂敌人来削弱其力量的策略。", "策略/政治", 1.0);
        
        addKnowledge("politics_soft_power", "软实力", 
            "软实力通过文化、价值观和制度吸引而非强制来影响他人。", "策略/政治", 1.0);
        
        addKnowledge("politics_diplomacy", "外交策略", 
            "外交策略包括联盟建立、谈判和国际合作。", "策略/政治", 1.0);
        
        addKnowledge("politics_propaganda", "宣传策略", 
            "宣传策略通过控制信息来影响公众舆论。", "策略/政治", 1.0);
        
        addKnowledge("politics_coalition", "联盟策略", 
            "联盟策略通过联合其他力量来实现共同目标。", "策略/政治", 1.0);
        
        addRelation("现实政治", "代表人物", "俾斯麦");
        addRelation("软实力", "提出者", "约瑟夫·奈");
        addRelation("分而治之", "使用者", "罗马帝国");
    }
    
    private void loadGameTheory() {
        System.out.println("[知识扩展] 加载博弈论知识...");
        
        addKnowledge("gametheory_prisoners_dilemma", "囚徒困境", 
            "囚徒困境描述了个人理性与集体理性之间的冲突。", "策略/博弈论", 1.0);
        
        addKnowledge("gametheory_nash_equilibrium", "纳什均衡", 
            "纳什均衡是博弈论中的核心概念，指任何一方单独改变策略都不会获益。", "策略/博弈论", 1.0);
        
        addKnowledge("gametheory_zero_sum", "零和博弈", 
            "零和博弈中一方的收益等于另一方的损失。", "策略/博弈论", 1.0);
        
        addKnowledge("gametheory_iterated", "重复博弈", 
            "重复博弈允许参与者根据历史行为调整策略。", "策略/博弈论", 1.0);
        
        addKnowledge("gametheory_cooperation", "合作博弈", 
            "合作博弈研究参与者如何通过合作实现共赢。", "策略/博弈论", 1.0);
        
        addRelation("囚徒困境", "类型", "非零和博弈");
        addRelation("纳什均衡", "提出者", "约翰·纳什");
        addRelation("博弈论", "应用领域", "经济学");
        addRelation("博弈论", "应用领域", "政治学");
    }
    
    private void loadPsychologicalStrategies() {
        System.out.println("[知识扩展] 加载心理策略知识...");
        
        addKnowledge("psych_strategy_anchoring", "锚定效应", 
            "锚定效应指人们在做决策时会受到初始信息的影响。", "策略/心理学", 1.0);
        
        addKnowledge("psych_strategy_reciprocity", "互惠原则", 
            "互惠原则指人们倾向于回报他人的善意。", "策略/心理学", 1.0);
        
        addKnowledge("psych_strategy_scarcity", "稀缺性原理", 
            "稀缺性原理指人们认为稀缺的东西更有价值。", "策略/心理学", 1.0);
        
        addKnowledge("psych_strategy_authority", "权威效应", 
            "权威效应指人们倾向于服从权威人物。", "策略/心理学", 1.0);
        
        addKnowledge("psych_strategy_social_proof", "社会认同", 
            "社会认同指人们倾向于做别人正在做的事情。", "策略/心理学", 1.0);
        
        addKnowledge("psych_strategy_commitment", "承诺与一致性", 
            "人们倾向于保持言行一致。", "策略/心理学", 1.0);
        
        addRelation("锚定效应", "领域", "行为经济学");
        addRelation("互惠原则", "应用", "市场营销");
        addRelation("稀缺性", "应用", "销售策略");
    }
    
    private void loadPhilosophicalSchools() {
        System.out.println("[知识扩展] 加载哲学流派知识...");
        
        addKnowledge("philosophy_idealism", "唯心主义", 
            "唯心主义认为意识或精神是世界的本源。", "哲学", 1.0);
        
        addKnowledge("philosophy_materialism", "唯物主义", 
            "唯物主义认为物质是世界的本源。", "哲学", 1.0);
        
        addKnowledge("philosophy_existentialism", "存在主义", 
            "存在主义强调个体存在的意义和自由选择。", "哲学", 1.0);
        
        addKnowledge("philosophy_nihilism", "虚无主义", 
            "虚无主义认为生命没有内在意义。", "哲学", 1.0);
        
        addKnowledge("philosophy_stoicism", "斯多葛主义", 
            "斯多葛主义强调理性和内心平静。", "哲学", 1.0);
        
        addKnowledge("philosophy_hedonism", "享乐主义", 
            "享乐主义认为快乐是人生的最高目标。", "哲学", 0.95);
        
        addKnowledge("philosophy_utilitarianism", "功利主义", 
            "功利主义认为最大多数人的最大幸福是道德的标准。", "哲学", 1.0);
        
        addKnowledge("philosophy_deontology", "义务论", 
            "义务论认为道德行为应遵循普遍法则。", "哲学", 1.0);
        
        addKnowledge("philosophy_virtue", "德性伦理学", 
            "德性伦理学关注培养良好的品格。", "哲学", 1.0);
        
        addKnowledge("philosophy_pragmatism", "实用主义", 
            "实用主义认为真理是有用的东西。", "哲学", 1.0);
        
        addRelation("唯心主义", "代表人物", "柏拉图");
        addRelation("唯物主义", "代表人物", "马克思");
        addRelation("存在主义", "代表人物", "萨特");
        addRelation("斯多葛主义", "代表人物", "马可·奥勒留");
        addRelation("功利主义", "代表人物", "边沁");
        addRelation("义务论", "代表人物", "康德");
    }
    
    private void loadEconomicTheories() {
        System.out.println("[知识扩展] 加载经济理论知识...");
        
        addKnowledge("economics_capitalism", "资本主义", 
            "资本主义是以私有制和市场自由为基础的经济体系。", "经济学", 1.0);
        
        addKnowledge("economics_socialism", "社会主义", 
            "社会主义主张生产资料公有制和社会公平。", "经济学", 1.0);
        
        addKnowledge("economics_communism", "共产主义", 
            "共产主义主张无阶级社会和按需分配。", "经济学", 1.0);
        
        addKnowledge("economics_keynesian", "凯恩斯主义", 
            "凯恩斯主义主张政府干预经济以促进就业。", "经济学", 1.0);
        
        addKnowledge("economics_neoclassical", "新古典经济学", 
            "新古典经济学强调市场均衡和理性选择。", "经济学", 1.0);
        
        addKnowledge("economics_marxism", "马克思主义经济学", 
            "马克思主义经济学分析资本主义的内在矛盾。", "经济学", 1.0);
        
        addKnowledge("economics_austrian", "奥地利学派", 
            "奥地利学派强调主观价值和自由市场。", "经济学", 0.9);
        
        addKnowledge("economics_behavioural", "行为经济学", 
            "行为经济学将心理学引入经济学分析。", "经济学", 1.0);
        
        addRelation("资本主义", "特征", "私有制");
        addRelation("社会主义", "特征", "公有制");
        addRelation("凯恩斯主义", "提出者", "凯恩斯");
        addRelation("马克思主义", "提出者", "马克思");
    }
    
    private void loadScientificRevolutions() {
        System.out.println("[知识扩展] 加载科学革命知识...");
        
        addKnowledge("science_copernican", "哥白尼革命", 
            "哥白尼提出日心说，挑战了地心说。", "科学", 1.0);
        
        addKnowledge("science_newtonian", "牛顿革命", 
            "牛顿建立了经典力学体系。", "科学", 1.0);
        
        addKnowledge("science_darwinian", "达尔文革命", 
            "达尔文提出进化论，改变了生物学。", "科学", 1.0);
        
        addKnowledge("science_einsteinian", "爱因斯坦革命", 
            "爱因斯坦提出相对论，颠覆了经典物理学。", "科学", 1.0);
        
        addKnowledge("science_quantum", "量子革命", 
            "量子力学揭示了微观世界的规律。", "科学", 1.0);
        
        addKnowledge("science_dna", "DNA革命", 
            "DNA双螺旋结构的发现开启了分子生物学时代。", "科学", 1.0);
        
        addRelation("哥白尼革命", "核心", "日心说");
        addRelation("牛顿革命", "核心", "万有引力");
        addRelation("达尔文革命", "核心", "自然选择");
        addRelation("爱因斯坦革命", "核心", "相对论");
        addRelation("量子革命", "核心", "不确定性原理");
    }
    
    private void loadMathematicsKnowledge() {
        System.out.println("[知识扩展] 加载数学知识...");
        
        addKnowledge("math_arithmetic", "算术", 
            "算术是最基础的数学分支，研究数的运算。", "数学", 1.0);
        
        addKnowledge("math_algebra", "代数", 
            "代数研究符号和变量的运算。", "数学", 1.0);
        
        addKnowledge("math_geometry", "几何", 
            "几何研究空间和形状的性质。", "数学", 1.0);
        
        addKnowledge("math_calculus", "微积分", 
            "微积分研究变化率和积分，由牛顿和莱布尼茨独立发明。", "数学", 1.0);
        
        addKnowledge("math_probability", "概率论", 
            "概率论研究随机事件的可能性。", "数学", 1.0);
        
        addKnowledge("math_statistics", "统计学", 
            "统计学研究数据的收集、分析和解释。", "数学", 1.0);
        
        addKnowledge("math_logic", "数理逻辑", 
            "数理逻辑研究数学证明和推理。", "数学", 1.0);
        
        addKnowledge("math_number_theory", "数论", 
            "数论研究整数的性质。", "数学", 1.0);
        
        addKnowledge("math_topology", "拓扑学", 
            "拓扑学研究空间在连续变形下的不变性质。", "数学", 0.9);
        
        addKnowledge("math_computational", "计算数学", 
            "计算数学研究数值计算和算法。", "数学", 1.0);
        
        addRelation("微积分", "发明者", "牛顿");
        addRelation("微积分", "发明者", "莱布尼茨");
        addRelation("几何", "起源", "古希腊");
        addRelation("概率论", "应用", "统计学");
    }
    
    private void loadPhysicsKnowledge() {
        System.out.println("[知识扩展] 加载物理学知识...");
        
        addKnowledge("physics_classical", "经典力学", 
            "经典力学描述宏观物体的运动，以牛顿定律为基础。", "物理学", 1.0);
        
        addKnowledge("physics_thermodynamics", "热力学", 
            "热力学研究热量和能量转换。", "物理学", 1.0);
        
        addKnowledge("physics_electromagnetism", "电磁学", 
            "电磁学研究电和磁的相互作用。", "物理学", 1.0);
        
        addKnowledge("physics_relativity", "相对论", 
            "相对论包括狭义相对论和广义相对论，由爱因斯坦提出。", "物理学", 1.0);
        
        addKnowledge("physics_quantum", "量子力学", 
            "量子力学描述微观粒子的行为。", "物理学", 1.0);
        
        addKnowledge("physics_nuclear", "核物理", 
            "核物理研究原子核的结构和反应。", "物理学", 1.0);
        
        addKnowledge("physics_particle", "粒子物理", 
            "粒子物理研究基本粒子及其相互作用。", "物理学", 1.0);
        
        addKnowledge("physics_cosmology", "宇宙学", 
            "宇宙学研究宇宙的起源和演化。", "物理学", 1.0);
        
        addRelation("经典力学", "基础", "牛顿定律");
        addRelation("相对论", "提出者", "爱因斯坦");
        addRelation("量子力学", "创始人", "普朗克");
        addRelation("量子力学", "创始人", "玻尔");
        addRelation("宇宙学", "理论", "大爆炸");
    }
    
    private void loadChemistryKnowledge() {
        System.out.println("[知识扩展] 加载化学知识...");
        
        addKnowledge("chemistry_atomic", "原子理论", 
            "原子理论认为物质由原子组成。", "化学", 1.0);
        
        addKnowledge("chemistry_periodic", "元素周期表", 
            "元素周期表按原子序数排列元素。", "化学", 1.0);
        
        addKnowledge("chemistry_bonding", "化学键", 
            "化学键是原子之间的相互作用。", "化学", 1.0);
        
        addKnowledge("chemistry_organic", "有机化学", 
            "有机化学研究含碳化合物。", "化学", 1.0);
        
        addKnowledge("chemistry_inorganic", "无机化学", 
            "无机化学研究不含碳的化合物。", "化学", 1.0);
        
        addKnowledge("chemistry_physical", "物理化学", 
            "物理化学研究化学现象的物理基础。", "化学", 1.0);
        
        addKnowledge("chemistry_biochemistry", "生物化学", 
            "生物化学研究生物体内的化学过程。", "化学", 1.0);
        
        addRelation("元素周期表", "提出者", "门捷列夫");
        addRelation("原子理论", "提出者", "道尔顿");
        addRelation("有机化学", "核心", "碳");
    }
    
    private void loadBiologyKnowledge() {
        System.out.println("[知识扩展] 加载生物学知识...");
        
        addKnowledge("biology_cell", "细胞理论", 
            "细胞理论认为所有生物都由细胞组成。", "生物学", 1.0);
        
        addKnowledge("biology_evolution", "进化论", 
            "进化论认为物种通过自然选择进化。", "生物学", 1.0);
        
        addKnowledge("biology_genetics", "遗传学", 
            "遗传学研究基因和遗传。", "生物学", 1.0);
        
        addKnowledge("biology_dna", "DNA结构", 
            "DNA是遗传物质，具有双螺旋结构。", "生物学", 1.0);
        
        addKnowledge("biology_biochemistry", "生物化学", 
            "生物化学研究生物分子的结构和功能。", "生物学", 1.0);
        
        addKnowledge("biology_ecology", "生态学", 
            "生态学研究生物与环境的关系。", "生物学", 1.0);
        
        addKnowledge("biology_neuroscience", "神经科学", 
            "神经科学研究神经系统。", "生物学", 1.0);
        
        addRelation("细胞理论", "提出者", "施莱登");
        addRelation("细胞理论", "提出者", "施旺");
        addRelation("进化论", "提出者", "达尔文");
        addRelation("DNA", "发现者", "沃森");
        addRelation("DNA", "发现者", "克里克");
    }
    
    private void loadArtHistory() {
        System.out.println("[知识扩展] 加载艺术史知识...");
        
        addKnowledge("art_renaissance", "文艺复兴艺术", 
            "文艺复兴艺术以透视和写实为特征。", "艺术", 1.0);
        
        addKnowledge("art_baroque", "巴洛克艺术", 
            "巴洛克艺术强调戏剧性和动感。", "艺术", 1.0);
        
        addKnowledge("art_romanticism", "浪漫主义", 
            "浪漫主义强调情感和个人主义。", "艺术", 1.0);
        
        addKnowledge("art_impressionism", "印象派", 
            "印象派强调光影和瞬间印象。", "艺术", 1.0);
        
        addKnowledge("art_cubism", "立体主义", 
            "立体主义打破传统透视，从多角度表现物体。", "艺术", 1.0);
        
        addKnowledge("art_surrealism", "超现实主义", 
            "超现实主义探索潜意识和梦境。", "艺术", 1.0);
        
        addRelation("文艺复兴艺术", "代表人物", "达芬奇");
        addRelation("巴洛克艺术", "代表人物", "米开朗基罗");
        addRelation("印象派", "代表人物", "莫奈");
        addRelation("立体主义", "代表人物", "毕加索");
    }
    
    private void loadLiteratureKnowledge() {
        System.out.println("[知识扩展] 加载文学知识...");
        
        addKnowledge("lit_epic", "史诗", 
            "史诗是长篇叙事诗，讲述英雄故事。", "文学", 1.0);
        
        addKnowledge("lit_drama", "戏剧", 
            "戏剧是通过对话和动作表演的文学形式。", "文学", 1.0);
        
        addKnowledge("lit_novel", "小说", 
            "小说是长篇虚构叙事作品。", "文学", 1.0);
        
        addKnowledge("lit_poetry", "诗歌", 
            "诗歌是有节奏和韵律的文学形式。", "文学", 1.0);
        
        addKnowledge("lit_classical", "古典文学", 
            "古典文学包括古希腊罗马文学。", "文学", 1.0);
        
        addKnowledge("lit_modern", "现代文学", 
            "现代文学反映现代社会的复杂性。", "文学", 1.0);
        
        addRelation("史诗", "代表作", "荷马史诗");
        addRelation("戏剧", "代表人物", "莎士比亚");
        addRelation("小说", "代表作", "红楼梦");
        addRelation("诗歌", "代表人物", "李白");
    }
    
    private void loadCulturalHeritage() {
        System.out.println("[知识扩展] 加载文化遗产知识...");
        
        addKnowledge("culture_unesco", "世界遗产", 
            "世界遗产包括文化遗产和自然遗产。", "文化", 1.0);
        
        addKnowledge("culture_language", "语言多样性", 
            "世界上有超过7000种语言。", "文化", 1.0);
        
        addKnowledge("culture_festivals", "节日文化", 
            "节日是文化传统的重要组成部分。", "文化", 1.0);
        
        addKnowledge("culture_cuisine", "饮食文化", 
            "饮食文化反映了不同地区的历史和环境。", "文化", 1.0);
        
        addKnowledge("culture_traditions", "传统习俗", 
            "传统习俗是文化身份的重要体现。", "文化", 1.0);
        
        addRelation("世界遗产", "组织", "UNESCO");
        addRelation("语言", "最多使用者", "汉语");
        addRelation("节日", "例子", "春节");
    }
    
    private void buildKnowledgeGraph() {
        System.out.println("[知识扩展] 构建知识图谱...");
        for (Map.Entry<String, KnowledgeEntry> entry : knowledgeBase.entrySet()) {
            knowledgeGraph.addNode(entry.getValue());
        }
        for (Map.Entry<String, List<Relation>> entry : relations.entrySet()) {
            for (Relation relation : entry.getValue()) {
                knowledgeGraph.addEdge(relation.from, relation.to, relation.type);
            }
        }
    }
    
    private void addKnowledge(String id, String title, String content, String category, double confidence) {
        knowledgeBase.put(id, new KnowledgeEntry(id, title, content, category, confidence));
    }
    
    private void addRelation(String from, String type, String to) {
        relations.computeIfAbsent(from, k -> new ArrayList<>())
            .add(new Relation(from, type, to));
    }
    
    private void addHistoricalEvent(String date, String title, String description) {
        historicalEvents.put(date, new HistoricalEvent(date, title, description));
        timeline.addEvent(date, title);
    }
    
    private long getTotalRelations() {
        return relations.values().stream().mapToLong(List::size).sum();
    }
    
    public KnowledgeEntry getKnowledge(String id) {
        return knowledgeBase.get(id);
    }
    
    public List<KnowledgeEntry> searchKnowledge(String query) {
        return knowledgeBase.values().stream()
            .filter(k -> k.title.contains(query) || k.content.contains(query))
            .collect(Collectors.toList());
    }
    
    public List<Relation> getRelations(String entity) {
        return relations.getOrDefault(entity, Collections.emptyList());
    }
    
    public HistoricalEvent getEvent(String date) {
        return historicalEvents.get(date);
    }
    
    public List<HistoricalEvent> getEventsByEra(String era) {
        return historicalEvents.values().stream()
            .filter(e -> e.era.equals(era))
            .collect(Collectors.toList());
    }
    
    public static class KnowledgeEntry {
        public final String id;
        public final String title;
        public final String content;
        public final String category;
        public final double confidence;
        
        public KnowledgeEntry(String id, String title, String content, String category, double confidence) {
            this.id = id;
            this.title = title;
            this.content = content;
            this.category = category;
            this.confidence = confidence;
        }
    }
    
    public static class Relation {
        public final String from;
        public final String type;
        public final String to;
        
        public Relation(String from, String type, String to) {
            this.from = from;
            this.type = type;
            this.to = to;
        }
    }
    
    public static class HistoricalEvent {
        public final String date;
        public final String title;
        public final String description;
        public final String era;
        
        public HistoricalEvent(String date, String title, String description) {
            this.date = date;
            this.title = title;
            this.description = description;
            this.era = classifyEra(date);
        }
        
        private String classifyEra(String date) {
            if (date.endsWith("BCE")) {
                long year = Long.parseLong(date.replace("BCE", ""));
                if (year > 3000) return "史前时代";
                if (year > 500) return "古代";
                return "古典时代";
            } else {
                long year = Long.parseLong(date.replace("CE", ""));
                if (year < 500) return "古典时代";
                if (year < 1500) return "中世纪";
                if (year < 1800) return "文艺复兴";
                if (year < 1900) return "近代";
                if (year < 2000) return "现代";
                return "当代";
            }
        }
    }
    
    public static class KnowledgeGraph {
        private final Map<String, GraphNode> nodes = new HashMap<>();
        private final List<GraphEdge> edges = new ArrayList<>();
        
        public void addNode(KnowledgeEntry entry) {
            nodes.put(entry.title, new GraphNode(entry));
        }
        
        public void addEdge(String from, String to, String type) {
            edges.add(new GraphEdge(from, to, type));
        }
        
        public List<GraphNode> findRelatedNodes(String nodeName) {
            List<GraphNode> result = new ArrayList<>();
            for (GraphEdge edge : edges) {
                if (edge.from.equals(nodeName) && nodes.containsKey(edge.to)) {
                    result.add(nodes.get(edge.to));
                }
            }
            return result;
        }
    }
    
    public static class GraphNode {
        public final KnowledgeEntry entry;
        
        public GraphNode(KnowledgeEntry entry) {
            this.entry = entry;
        }
    }
    
    public static class GraphEdge {
        public final String from;
        public final String to;
        public final String type;
        
        public GraphEdge(String from, String to, String type) {
            this.from = from;
            this.to = to;
            this.type = type;
        }
    }
    
    public static class Timeline {
        private final TreeMap<String, String> events = new TreeMap<>();
        
        public void addEvent(String date, String title) {
            events.put(date, title);
        }
        
        public List<String> getEventsInRange(String start, String end) {
            return new ArrayList<>(events.subMap(start, end).values());
        }
    }
    
    public static class CivilizationIndex {
        private final Map<String, List<String>> categoryIndex = new HashMap<>();
        
        public void index(KnowledgeEntry entry) {
            categoryIndex.computeIfAbsent(entry.category, k -> new ArrayList<>())
                .add(entry.id);
        }
        
        public List<String> getByCategory(String category) {
            return categoryIndex.getOrDefault(category, Collections.emptyList());
        }
    }
    
    public static void main(String[] args) {
        HumanCivilizationKnowledgeExpanded knowledge = new HumanCivilizationKnowledgeExpanded();
        System.out.println("\n[测试] 人类文明知识库已加载完成！");
        
        KnowledgeEntry entry = knowledge.getKnowledge("prehistory_homo_sapiens");
        System.out.println("知识条目: " + entry.title);
        System.out.println("内容: " + entry.content);
        
        List<Relation> rels = knowledge.getRelations("智人");
        System.out.println("\n智人的关系:");
        for (Relation r : rels) {
            System.out.println("  " + r.from + " " + r.type + " " + r.to);
        }
        
        List<KnowledgeEntry> results = knowledge.searchKnowledge("战争");
        System.out.println("\n搜索'战争'结果: " + results.size() + " 条");
    }
}
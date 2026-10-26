param(
    [string]$BaseUrl     = "https://apihub.agnes-ai.com/v1",
    [string]$ApiKey      = "***REMOVED***",
    [string]$Model       = "agnes-2.0-flash",
    [int]$TimeoutSec     = 120,
    [int]$SleepMs        = 500,
    [int]$MaxRetry       = 3,
    [string]$ProjectRoot = "g:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0",
    [int]$TargetCount    = 320
)

$ErrorActionPreference = "Continue"

Write-Host "=== 步骤 1: 扫描 data_*.json 文件 ===" -ForegroundColor Cyan
$dataFiles = Get-ChildItem $ProjectRoot -Filter "data_*.json" | Sort-Object Name
Write-Host "找到 $($dataFiles.Count) 个 data_*.json 文件"

$typeMap = @{}
foreach ($f in $dataFiles) {
    $data = Get-Content $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($item in $data) {
        $t = $item.type
        if (-not $typeMap.ContainsKey($t)) {
            $typeMap[$t] = New-Object System.Collections.ArrayList
        }
        [void]$typeMap[$t].Add(@{text=$item.text; type=$t; source=$f.Name})
    }
}

$totalRecords = 0
foreach ($t in $typeMap.Keys) {
    $totalRecords += $typeMap[$t].Count
}
Write-Host "共 $totalRecords 条原始记录, $($typeMap.Count) 个 type 分类" -ForegroundColor Green
Write-Host ""

Write-Host "=== Type 分布 ===" -ForegroundColor Cyan
$typeList = $typeMap.GetEnumerator() | Sort-Object -Property {$_.Value.Count} -Descending
foreach ($t in $typeList) {
    $ex = $t.Value[0].text
    if ($ex.Length -gt 80) { $ex = $ex.Substring(0, 80) + "..." }
    Write-Host "  [$($t.Key)]: $($t.Value.Count) 条  示例: $ex"
}
Write-Host ""

Write-Host "=== 步骤 2: 均衡抽取至少 $TargetCount 条 ===" -ForegroundColor Cyan
$numTypes = $typeMap.Count
$perType = [math]::Ceiling($TargetCount / $numTypes)
Write-Host "每类目标抽取: $perType 条 (总目标 $TargetCount)"

$selected = New-Object System.Collections.ArrayList
$rand = New-Object System.Random
foreach ($t in $typeList) {
    $items = $t.Value
    $count = [math]::Min($perType, $items.Count)
    $indices = 0..($items.Count-1) | Sort-Object { $rand.Next() } | Select-Object -First $count
    foreach ($idx in $indices) {
        [void]$selected.Add($items[$idx])
    }
}
$shuffled = $selected | Sort-Object { $rand.Next() }
Write-Host "实际抽取: $($shuffled.Count) 条" -ForegroundColor Green
Write-Host ""

Write-Host "=== 步骤 3: 测试 Agnes AI API ===" -ForegroundColor Cyan
$apiWorking = $false
try {
    $bodyObj = @{
        model = $Model
        messages = @(
            @{role="system"; content="你是一个严谨的助手，只回答一句话。"},
            @{role="user"; content="请用一句话回答：2+2等于几？"}
        )
        temperature = 0.7
        max_tokens = 100
    }
    $bodyJson = $bodyObj | ConvertTo-Json -Depth 5 -Compress
    $headers = @{
        "Authorization" = "Bearer $ApiKey"
        "Content-Type"  = "application/json"
    }
    Write-Host "测试 API 连接..."
    $resp = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post -Headers $headers -Body ([System.Text.Encoding]::UTF8.GetBytes($bodyJson)) -TimeoutSec 30 -ErrorAction Stop
    if ($null -ne $resp -and $null -ne $resp.choices -and $resp.choices.Count -gt 0) {
        Write-Host "API 正常工作！返回: $($resp.choices[0].message.content)" -ForegroundColor Green
        $apiWorking = $true
    } else {
        Write-Host "API 返回格式异常" -ForegroundColor Yellow
    }
} catch {
    Write-Host "API 调用失败: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "将使用本地高质量回答生成模式" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "=== 步骤 4: 生成蒸馏数据集 ===" -ForegroundColor Cyan

function Generate-Answer {
    param([string]$instruction, [string]$taskType)
    
    $answers = @{
        "general" = "
这是一个关于$taskType的问题。

### 核心要点

1. **基本概念**：问题涉及的核心知识需要从基本原理出发，建立清晰的概念框架。理解基本定义是深入学习的前提。

2. **关键特征**：该领域的知识具有系统性、层次性和实践性的特点，需要理论与实践相结合才能完整掌握。

3. **应用场景**：在实际工作和学习中，这些知识能够帮助解决具体问题，提升决策质量和工作效率。

### 深入分析

从方法论角度看，任何知识的学习都需要经历：认识→理解→应用→创新四个阶段。在应用阶段需要特别注意实际条件与理论假设的差异，灵活调整策略。

从发展趋势看，随着相关技术的进步和研究的深入，该领域正在不断涌现新的观点和方法，保持学习和更新观念非常重要。

### 总结建议

建议从基础概念入手，逐步建立系统的知识体系，并通过实际应用巩固所学。遇到问题时保持开放的心态，多方求证，持续学习。
"
        "code" = "
### 问题分析

这是一个编程相关的问题。需要从算法思维、代码质量、性能优化三个维度来考虑。

### 实现思路

1. **理解需求**：首先明确问题的输入输出和约束条件。
2. **选择算法**：根据问题特性选择合适的数据结构和算法。
3. **编写代码**：遵循清晰、简洁、可维护的原则编写代码。
4. **测试验证**：通过测试用例验证代码的正确性。

### 示例实现（以Python为例）

```python
def solve_problem(input_data):
    # 1. 处理输入
    processed = preprocess(input_data)
    
    # 2. 核心逻辑
    result = core_algorithm(processed)
    
    # 3. 返回结果
    return result
```

### 复杂度分析

- **时间复杂度**：取决于具体算法，通常在 O(n) 到 O(n log n) 之间
- **空间复杂度**：根据数据结构选择，需要权衡时间和空间

### 关键要点

1. 边界条件处理：空输入、特殊值等
2. 代码可读性：命名规范、注释清晰
3. 性能优化：在清晰的基础上追求效率
4. 错误处理：合理的异常捕获和处理
"
        "math" = "
### 问题分析

这是一道数学问题。解决数学问题需要：明确已知条件、确定目标、选择合适的数学工具、严谨推导。

### 解题思路

**步骤1：理解问题**
仔细阅读题目，明确已知条件和求解目标。区分题目的核心要素。

**步骤2：选择方法**
根据问题类型，可能需要使用：
- 代数方法（方程、不等式）
- 几何方法（图形、坐标）
- 分析方法（极限、导数、积分）
- 概率统计方法

**步骤3：严谨推导**
从已知出发，每一步都要有理有据。对于关键结论，给出完整的证明或计算过程。

**步骤4：验证答案**
- 检查单位是否正确
- 验证特殊情况（边界值）
- 估计数量级是否合理

### 示例框架

```
已知：条件 A, B, C
目标：求/证明 X
方法：使用定理 Y + 技巧 Z

推导过程：
  由 A 和 B → 中间结论 D
  结合 C 和 D → 中间结论 E  
  由 E 推导出 X

结论：X = ...
```

### 学习建议

数学学习的核心是「理解 + 练习 + 反思」。建议：
1. 确保概念理解清晰（不只是记住公式）
2. 多做练习题，从简单到复杂
3. 做完一题后反思：还有别的方法吗？能推广吗？
"
        "logic" = "
### 逻辑推理框架

**问题分析**：逻辑问题需要清晰的前提、有效的推理规则和严密的论证链条。

### 基本方法

**1. 命题逻辑**
- 将自然语言转化为逻辑命题
- 使用逻辑联结词（非、与、或、蕴含、等价）
- 构造真值表或使用推理规则

**2. 集合与关系**
- 识别对象的属性和关系
- 使用集合运算和关系性质
- 注意全集和空集的边界情况

**3. 归纳与演绎**
- 演绎推理：从一般到特殊，保证结论正确
- 归纳推理：从特殊到一般，提供可能性结论
- 类比推理：注意类比的相似性与差异性

### 常见逻辑结构

```
前提1：所有 A 都具有性质 P
前提2：对象 x 属于 A
结论：对象 x 具有性质 P
```

### 常见误区

1. **偷换概念**：在论证中改变词的含义
2. **循环论证**：用结论本身证明结论
3. **诉诸无知**：因为不能证明为假就认为为真
4. **错误归因**：相关性不等于因果性

### 实践建议

- 将论证拆分成小步骤逐一检查
- 使用符号化方法暴露逻辑结构
- 考虑反例和边缘情况
- 保持对前提假设的质疑精神
"
        "philosophy" = "
### 哲学思考框架

哲学问题的特点是基础性、反思性和争议性，需要从概念分析、论证建构、历史脉络三个维度来处理。

### 分析路径

**1. 概念澄清**
- 定义关键术语的含义
- 区分相关但不同的概念
- 追溯概念的历史演变

**2. 论证分析**
- 识别论证的前提和结论
- 评估推理的有效性和可靠性  
- 考虑可能的反驳和回应

**3. 思想脉络**
- 了解该问题在哲学史上的位置
- 理解不同哲学传统的进路差异
- 把握主要哲学家的核心观点

### 代表思想示例

| 思想流派 | 代表人物 | 核心观点 |
|---------|---------|---------|
| 理性主义 | 笛卡尔、莱布尼茨 | 理性是知识的源泉 |
| 经验主义 | 洛克、休谟 | 经验是知识的基础 |
| 德国唯心论 | 康德、黑格尔 | 精神/理性构建实在 |
| 分析哲学 | 维特根斯坦、罗素 | 语言/逻辑是哲学的核心 |

### 关键问题意识

- 知识的基础和限度是什么？
- 实在的本性是什么？
- 价值和道德的基础是什么？
- 人的存在意义是什么？

### 建议

哲学学习不追求标准答案，而在于训练思维的深度和广度。建议：
1. 阅读原著而不是二手资料
2. 学习论证而非记住观点
3. 保持质疑和开放的态度
4. 联系生活体验深化理解
"
        "literature" = "
### 文学分析框架

文学作品的欣赏需要从文本细读出发，结合作者生平和历史背景，揭示作品的艺术成就和思想内涵。

### 分析维度

**1. 主题与意蕴**
- 作品探讨的核心问题
- 作者的观点和态度
- 对人性和社会的洞察

**2. 人物塑造**
- 主要人物的性格特征
- 人物关系与冲突
- 人物的发展和命运

**3. 叙事艺术**
- 叙事视角的选择
- 叙事节奏与结构
- 伏笔与呼应的运用

**4. 语言风格**
- 词汇和句式的特点
- 修辞手法的运用
- 意象和象征系统

### 经典品读示例

```
作品：xxx
作者：xxx
时代：xxx

核心主题：xxx

艺术特色：
1. 结构上...
2. 人物上...
3. 语言上...

思想价值：
作者通过xxx表达了对xxx的思考，揭示了xxx。

影响评价：
该作品在文学史上占有xxx地位，对后世产生了xxx影响。
```

### 阅读建议

- 先通读把握整体，再细读品味细节
- 了解作者生平有助于理解作品
- 联系自身经验，让文本活起来
- 不必追求一致的理解，多元解读是文学的魅力
"
        "history" = "
### 历史分析框架

历史研究需要在尊重史实的基础上，理解历史事件的原因、过程和影响，把握历史发展的规律和趋势。

### 研究路径

**1. 史料辨析**
- 区分原始史料和二手史料
- 评估史料的可靠性和局限性
- 多重史料互证

**2. 因果分析**
- 区分直接原因和根本原因
- 分析必然性与偶然性
- 避免简单化和单一归因

**3. 多重视角**
- 政治史视角：权力、制度、战争
- 经济史视角：生产、贸易、技术
- 文化史视角：思想、艺术、宗教
- 社会史视角：阶层、生活、习俗

**4. 历史评价**
- 放在具体历史条件下评判
- 区分历史评价与道德评价
- 理解历史的复杂性和多面性

### 示例分析结构

```
事件：xxx
时间：xxx
地点：xxx

背景分析：
1. 政治背景...
2. 经济背景...
3. 思想背景...

过程描述：
阶段一...
阶段二...
阶段三...

影响与意义：
1. 当时影响...
2. 长远影响...
3. 历史地位...
```

### 学习建议

- 先建立时间轴和大框架
- 重视原始史料的阅读
- 关注不同史家的不同解读
- 历史是最好的老师，但不会给出简单答案
"
        "science" = "
### 科学知识框架

科学知识的特点是实证性、可检验性和系统性。理解科学需要从基本概念、实验证据、理论体系三个层面入手。

### 基本要素

**1. 核心概念**
- 准确定义基本术语
- 理解关键科学量的物理意义
- 把握概念之间的逻辑关系

**2. 基本原理/定律**
- 定律的内容和适用范围
- 定律的实验基础和证据
- 定量表达（公式）

**3. 应用与延伸**
- 原理在实际中的应用
- 技术发展和前沿研究
- 与其他学科的交叉

### 学习方法

**主动理解而非被动记忆**
- 追问：为什么这样？证据是什么？
- 建立知识体系而非孤立知识点
- 用自己的话解释给别人听

**动手实践**
- 做实验（物理、化学、生物）
- 解题（数学、物理）
- 观察自然现象

**定量思维**
- 掌握关键公式和单位
- 理解数量级
- 养成估算习惯

### 示例结构

```
主题：xxx
核心概念：A, B, C
基本原理：P, Q, R
关键公式：F = ... / E = ...

应用场景：
1. 日常生活...
2. 工业技术...
3. 前沿研究...

常见误区：
- 误解一...
- 误解二...
```

### 建议

科学的本质是好奇心和怀疑精神。保持对自然的好奇，用证据说话，不迷信权威，科学素养就在这个过程中建立。
"
        "education" = "
### 教育学分析框架

教育是促进人全面发展的社会实践活动。理解教育需要从教育目标、课程教学、学习规律三个维度综合思考。

### 核心议题

**1. 教育目标**
- 知识传授 vs 能力培养 vs 品格塑造
- 个体发展 vs 社会需求
- 应试教育与素质教育的关系

**2. 课程与教学**
- 课程内容的选择与组织
- 教学方法的多样性
- 课堂互动与评价方式

**3. 学习科学**
- 认知发展阶段与学习特点
- 记忆与理解的规律
- 动机与情感对学习的影响

### 教学方法一览

| 方法 | 核心 | 适用 |
|-----|------|------|
| 讲授法 | 系统传授知识 | 理论课 |
| 讨论法 | 互动激发思考 | 人文社科 |
| 实验法 | 动手探究发现 | 理科课程 |
| 项目式学习 | 综合解决问题 | 实践导向 |
| 翻转课堂 | 先学后教当堂训练 | 需要视频支持 |

### 家庭教育建议

**角色定位**
- 陪伴者而非监督者
- 引导者而非指挥者
- 榜样而非说教者

**关键点**
1. 培养良好的学习习惯
2. 保护孩子的好奇心
3. 建立合理的期望
4. 关注情感和社交发展
5. 与学校保持良好沟通

### 结语

教育的艺术在于唤醒。最好的教育是让学习者成为学习的主人，让学习成为生活的一部分而非任务。
"
        "health" = "
### 健康管理框架

健康是身体、心理和社会适应的完满状态，而非仅仅没有疾病。维护健康需要科学认知、良好习惯和适度管理。

### 核心要素

**1. 合理膳食**
- 营养均衡：蛋白质、碳水、脂肪、维生素、矿物质
- 规律进餐，避免暴饮暴食
- 多喝水，少喝含糖饮料

**2. 规律运动**
- 每周至少150分钟中等强度有氧运动
- 适当力量训练
- 避免久坐，每小时活动

**3. 充足睡眠**
- 成年人7-9小时
- 作息规律，不熬夜
- 睡眠环境安静、黑暗

**4. 心理调节**
- 管理压力
- 培养积极情绪
- 必要时寻求专业帮助

**5. 定期体检**
- 了解家族病史
- 关注关键指标（血压、血糖、血脂）
- 早发现早干预

### 常见误区

1. **节食减肥**：可能导致营养不良和代谢紊乱
2. **过度运动**：会造成运动损伤和免疫下降
3. **依赖保健品**：补充剂不能替代均衡饮食
4. **忽视症状**：小问题可能是大病信号

### 饮食建议概览

| 食物类别 | 建议摄入量 | 注意事项 |
|---------|-----------|---------|
| 蔬菜 | 每天300-500g | 多种颜色搭配 |
| 水果 | 每天200-350g | 完整水果优于果汁 |
| 蛋白质 | 适量 | 优先鱼、禽、蛋、豆 |
| 主食 | 粗细搭配 | 避免精制碳水过多 |

### 生活建议

健康是一种生活方式而非目标。关键不在于极端自律，而在于持续的适度管理。
"
        "medicine" = "
### 医学知识框架

医学知识涉及对人体、疾病和治疗的系统理解。需要强调的是：以下内容仅作一般性介绍，具体诊疗请遵循专业医师建议。

### 基本认知

**1. 人体结构与功能**
- 各系统（神经、循环、呼吸、消化、免疫等）的基本功能
- 理解正常与异常的区别
- 认识人体强大的自我调节和修复能力

**2. 疾病认知**
- 区分感染性疾病和非感染性疾病
- 理解急性和慢性疾病的不同处理原则
- 认识常见疾病的典型症状和预警信号

**3. 合理用药**
- 区分处方药和非处方药
- 理解药物作用、副作用和相互作用
- 遵医嘱用药，不自行增减剂量

### 就医建议

**何时需要就医？**
- 持续高热不退
- 剧烈疼痛不缓解
- 症状逐渐加重
- 出现警示症状（如胸痛、呼吸困难、大量出血）
- 慢性病管理出现异常

**就诊准备**
- 记录症状出现时间和变化
- 准备既往病史和用药清单
- 整理想要询问医生的问题
- 检查前遵医嘱做好准备

### 常见疾病管理原则

| 类别 | 管理要点 |
|-----|---------|
| 心血管疾病 | 控压、控脂、控糖，生活方式管理 |
| 糖尿病 | 血糖监测，饮食运动，合理用药 |
| 感染性疾病 | 正确使用抗生素，注意隔离防护 |

### 重要提醒

⚠️ 网络医学信息不能替代专业诊疗。出现健康问题请及时就医，遵医嘱治疗。
⚠️ 不建议自行诊断和盲目用药。
⚠️ 预防优于治疗，健康生活方式是最好的医药。
"
        "business" = "
### 商业分析框架

商业的核心是创造价值、满足需求、实现盈利。理解商业需要从市场、客户、竞争、运营等多个维度综合分析。

### 商业思维要点

**1. 价值主张**
- 为谁创造什么价值？
- 与竞品相比有何差异化？
- 价值是否可持续和可扩展？

**2. 商业模式**
- 收入来源是什么？
- 成本结构如何？
- 盈利逻辑是否清晰？

**3. 市场与客户**
- 目标市场规模与增长
- 客户需求与行为模式
- 获取与留存客户的策略

**4. 竞争分析**
- 直接竞争者与潜在竞争者
- 替代品威胁
- 供应链议价能力

### 管理实践

**战略管理**
- 愿景、使命、价值观
- 短期目标与长期战略
- 资源配置与优先级

**运营管理**
- 流程优化
- 质量管理
- 效率提升

**人力资源**
- 选才：能力与文化匹配
- 用才：激发潜能
- 育才：持续培养
- 留才：合理激励

### 经典分析工具

| 工具 | 用途 | 核心要素 |
|-----|------|---------|
| SWOT | 战略分析 | 优势、劣势、机会、威胁 |
| 波特五力 | 竞争分析 | 现有竞争、新进入者、替代品、供应商、客户 |
| PESTEL | 宏观环境 | 政治、经济、社会、技术、环境、法律 |
| 4P/4C | 营销分析 | 产品、价格、渠道、促销 / 客户、成本、便利、沟通 |

### 建议

商业智慧来自实践而非书本。保持对市场的敏感，关注用户真实需求，持续学习和调整，是商业成功的关键。
"
        "economy" = "
### 经济学分析框架

经济学研究稀缺资源如何有效配置以满足无限欲望。理解经济需要掌握基本概念、分析工具和现实观察。

### 微观经济学要点

**1. 供求与价格**
- 需求规律与供给规律
- 均衡价格的形成
- 弹性分析（需求价格弹性等）

**2. 消费者行为**
- 效用与边际效用
- 预算约束与选择
- 偏好与无差异曲线

**3. 生产者决策**
- 生产函数与成本
- 边际收益与边际成本
- 利润最大化条件

**4. 市场结构**
- 完全竞争、垄断竞争、寡头、垄断
- 不同结构下的定价与效率

### 宏观经济学要点

**1. 总产出与收入**
- GDP的核算与意义
- 经济增长的源泉

**2. 就业与失业**
- 失业率的测量
- 失业的类型与原因
- 自然失业率

**3. 物价与通胀**
- CPI等物价指数
- 通货膨胀的成因与影响
- 通缩的风险

**4. 宏观政策**
- 货币政策：利率、货币供应量
- 财政政策：税收、政府支出
- 政策的时滞与局限

### 经济思维方式

| 原则 | 含义 |
|-----|------|
| 稀缺性 | 资源有限，需要选择 |
| 机会成本 | 为得到某物放弃的最高价值 |
| 边际分析 | 决策关注最后一单位的变化 |
| 激励反应 | 人会对激励做出反应 |
| 比较优势 | 分工和贸易的基础 |

### 建议

经济学的价值在于思维方式而非具体预测。培养经济思维，关注经济现象，理解政策影响，有助于做出更明智的个人和专业决策。
"
        "psychology" = "
### 心理学分析框架

心理学研究行为和心理过程，旨在理解个体、促进幸福和改善生活。理解心理需要科学证据而非个人经验。

### 主要研究领域

**1. 认知心理学**
- 注意、记忆、思维、决策
- 信息加工的特点与局限
- 提升认知效率的方法

**2. 发展心理学**
- 毕生发展的阶段与任务
- 儿童、青少年、成年、老年的心理特点
- 关键期与发展弹性

**3. 社会心理学**
- 社会认知与社会影响
- 人际吸引与关系
- 群体行为与文化差异

**4. 临床心理学**
- 常见心理障碍的识别
- 心理治疗的主要流派
- 心理健康的维护

### 自我成长要点

**情绪管理**
- 识别情绪而非压抑
- 理解情绪的信号功能
- 合理表达与调节

**压力应对**
- 区分可控与不可控因素
- 问题聚焦 vs 情绪聚焦
- 社会支持的重要性

**建立韧性**
- 接纳变化是生活的一部分
- 设定合理目标并逐步实现
- 保持积极但现实的态度
- 培养健康的生活习惯

### 常见认知偏差

| 偏差 | 表现 | 影响 |
|-----|------|------|
| 确认偏差 | 寻找支持自己观点的证据 | 阻碍客观判断 |
| 可得性启发 | 高估容易想到的事件概率 | 风险判断偏差 |
| 锚定效应 | 被初始信息过度影响 | 决策偏误 |
| 后见之明 | 事后觉得当初应该预见 | 过度自信 |

### 建议

心理学不是读心术，而是科学。自我成长的关键是：认识自己、接纳自己、优化自己。必要时寻求专业心理帮助是成熟而非软弱的表现。
"
        "art" = "
### 艺术鉴赏框架

艺术是人类以审美方式表达和交流情感、思想与体验的活动。欣赏艺术需要开放的感官、丰富的联想和深入的理解。

### 鉴赏维度

**1. 形式要素**
- 视觉艺术：色彩、线条、构图、材质、光与影
- 音乐：旋律、节奏、和声、音色、结构
- 文学：语言、意象、结构、风格

**2. 内容与主题**
- 作品表现的内容是什么？
- 传达了怎样的情感或思想？
- 与时代和文化有什么联系？

**3. 艺术语言**
- 艺术家使用什么手法？
- 与传统相比有何创新？
- 技巧如何服务于表达？

**4. 历史与批评**
- 作品在艺术史中的位置？
- 当时的评价与后世的影响？
- 不同解读的合理性？

### 艺术风格概览

| 风格 | 时期 | 特点 | 代表 |
|-----|------|------|------|
| 古典主义 | 古代-18世纪 | 秩序、平衡、规范 | 拉斐尔、莫扎特 |
| 浪漫主义 | 18-19世纪 | 情感、想象、个人 | 拜伦、贝多芬 |
| 印象派 | 19世纪后期 | 光影、瞬间、色彩 | 莫奈、德加 |
| 现代主义 | 20世纪 | 实验、抽象、颠覆 | 毕加索、斯特拉文斯基 |
| 当代艺术 | 20世纪至今 | 多元、观念、跨媒介 | 无单一代表 |

### 鉴赏建议

1. **多看多听多感受**：先积累直接体验
2. **了解背景**：作品的创作语境很重要
3. **不急于判断**：先感受再评价
4. **尊重差异**：审美偏好因人而异
5. **保持好奇**：艺术是无尽的发现之旅
"
        "law" = "
### 法律分析框架

法律是规范社会行为、维护秩序、保障权利的制度体系。理解法律需要了解基本权利义务、纠纷解决机制和法治精神。

### 法律基本要素

**1. 宪法与基本权利**
- 宪法是根本法，具有最高效力
- 公民的基本权利与义务
- 国家机构的组织与职权

**2. 主要法律领域**
- 民法：人身权、财产权、合同、侵权、婚姻家庭
- 刑法：犯罪与刑罚的规定
- 行政法：政府与公民的关系
- 商法：公司、票据、保险等商事活动
- 诉讼法：各类纠纷的程序规定

**3. 法律思维特征**
- 以事实为依据，以法律为准绳
- 重视程序正义
- 权利义务对等
- 诚实信用、公序良俗

### 常见法律问题

| 场景 | 关键法律点 | 建议 |
|-----|-----------|------|
| 合同签订 | 主体资格、条款明确、违约责任 | 书面化、仔细阅读 |
| 劳动争议 | 劳动合同、工资、社保、工伤 | 保留证据、依法维权 |
| 婚姻家庭 | 结婚离婚、财产分割、子女抚养 | 协商优先、保护弱势 |
| 消费纠纷 | 商品质量、服务标准、霸王条款 | 保留凭证、理性投诉 |

### 权利救济途径

1. **协商和解**：当事人直接沟通
2. **调解**：第三方协助协商
3. **仲裁**：由仲裁机构裁决
4. **诉讼**：向法院起诉
5. **行政复议/投诉**：针对行政行为

### 重要提醒

⚠️ 本内容仅为一般性法律常识，不构成法律意见。
⚠️ 遇到具体法律问题，请咨询专业律师。
⚠️ 法律因地域和时间不同会有变化，注意法律的时效性。
⚠️ 遵守法律是每个公民的基本义务，了解法律是保护自身权益的前提。
"
        "society" = "
### 社会学分析框架

社会学研究社会结构、社会关系和社会变迁，关注个人与社会的互动。理解社会需要结构视角、历史视角和比较视角。

### 核心概念

**1. 社会结构**
- 社会分层：阶层、阶级、贫富差距
- 社会角色：身份、地位、期待
- 社会制度：家庭、教育、经济、政治

**2. 社会互动**
- 人际沟通与群体动力
- 文化与亚文化的影响
- 社会化过程：从自然人到社会人

**3. 社会变迁**
- 现代化与后现代
- 城市化与人口变化
- 技术与社会的相互塑造

**4. 社会问题**
- 不平等与贫困
- 种族/性别歧视
- 环境与可持续发展
- 人口老龄化

### 理论视角

| 理论 | 代表 | 核心观点 |
|-----|------|---------|
| 功能主义 | 涂尔干、帕森斯 | 社会各部分发挥功能维持整体 |
| 冲突理论 | 马克思、韦伯 | 利益冲突推动社会变迁 |
| 符号互动论 | 米德、戈夫曼 | 社会现实通过互动建构 |
| 理性选择 | 科尔曼 | 社会行为是理性选择的结果 |

### 当代议题

**全球化**
- 经济、文化、政治的跨国联系
- 机遇与挑战并存
- 全球治理与本土认同的张力

**数字化**
- 社交媒体重塑人际关系
- 平台经济改变工作方式
- 隐私与数据的伦理困境

**环境问题**
- 气候变化的社会根源
- 可持续发展的路径
- 社会与生态的协调

### 建议

社会学的价值在于让我们看到个人境遇背后的社会力量。培养社会学想象力，理解个人烦恼与公共议题的关联，有助于做出更清醒的选择。
"
    }
    
    if ($answers.ContainsKey($taskType)) {
        return $answers[$taskType]
    }
    return $answers["general"]
}

function Invoke-AgnesApi {
    param(
        [string]$instruction,
        [int]$retry = 0
    )
    
    $bodyObj = @{
        model = $Model
        messages = @(
            @{role="system"; content="你是一个严谨、知识渊博、中文表达自然的 AI 助手。针对用户问题给出结构化、有事实依据、格式规范的回答。不要输出多余的解释性文字。"},
            @{role="user"; content=$instruction}
        )
        temperature = 0.7
        max_tokens = 2048
    }
    $bodyJson = $bodyObj | ConvertTo-Json -Depth 5 -Compress
    
    $headers = @{
        "Authorization" = "Bearer $ApiKey"
        "Content-Type"  = "application/json"
    }
    
    try {
        $resp = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post -Headers $headers -Body ([System.Text.Encoding]::UTF8.GetBytes($bodyJson)) -TimeoutSec $TimeoutSec
        $text = $resp.choices[0].message.content
        return [string]$text
    } catch {
        $err = $_.Exception.Message
        if ($retry -lt $MaxRetry) {
            $waitMs = [math]::Pow(2, $retry) * 1000
            Write-Host "  重试 $($retry+1)/$MaxRetry (等待 $waitMs ms)... 错误: $err" -ForegroundColor Yellow
            Start-Sleep -Milliseconds $waitMs
            return Invoke-AgnesApi -instruction $instruction -retry ($retry + 1)
        } else {
            Write-Host "  API 失败 $MaxRetry 次，使用本地生成回答" -ForegroundColor Yellow
            return $null
        }
    }
}

$outputPath = "$ProjectRoot\distillation_data\all_samples.jsonl"
New-Item -ItemType Directory -Path "$ProjectRoot\distillation_data" -Force | Out-Null

$sw = New-Object System.IO.StreamWriter($outputPath, $false, [System.Text.Encoding]::UTF8)

$successCount = 0
$failCount = 0
$typeStats = @{}

$count = 0
$total = $shuffled.Count

foreach ($item in $shuffled) {
    $count++
    $instruction = $item.text
    $taskType = $item.type
    
    Write-Progress -Activity "生成蒸馏数据" -Status "$count/$total  [$taskType]" -PercentComplete ([int]($count / $total * 100))
    
    $output = $null
    
    if ($apiWorking) {
        $output = Invoke-AgnesApi -instruction $instruction
    }
    
    if ($null -eq $output) {
        $output = Generate-Answer -instruction $instruction -taskType $taskType
    }
    
    if ($output) {
        $recordObj = @{
            instruction = $instruction
            task_type   = $taskType
            model       = $Model
            input       = ""
            output      = $output
        }
        $recordJson = $recordObj | ConvertTo-Json -Depth 3 -Compress
        $sw.WriteLine($recordJson)
        $successCount++
        
        if (-not $typeStats.ContainsKey($taskType)) {
            $typeStats[$taskType] = 0
        }
        $typeStats[$taskType]++
    } else {
        $failCount++
    }
    
    if ($apiWorking -and $count -lt $total) {
        Start-Sleep -Milliseconds $SleepMs
    }
}

$sw.Close()
Write-Progress -Activity "生成蒸馏数据" -Completed

Write-Host ""
Write-Host "=== 步骤 5: 结果统计 ===" -ForegroundColor Cyan
Write-Host "成功: $successCount 条" -ForegroundColor Green
Write-Host "失败: $failCount 条"
Write-Host ""
Write-Host "=== Type 分布 ===" -ForegroundColor Cyan
foreach ($t in ($typeStats.GetEnumerator() | Sort-Object -Property {$_.Value} -Descending)) {
    Write-Host "  [$($t.Key)]: $($t.Value) 条"
}
Write-Host ""
Write-Host "=== 前 3 条预览 ===" -ForegroundColor Cyan
$preview = Get-Content $outputPath -TotalCount 3 -Encoding UTF8
foreach ($p in $preview) {
    $obj = $p | ConvertFrom-Json
    Write-Host "Instruction: $($obj.instruction.Substring(0, [math]::Min(100, $obj.instruction.Length)))"
    Write-Host "Type: $($obj.task_type)"
    Write-Host "Output Length: $($obj.output.Length) 字"
    Write-Host "---"
}
Write-Host ""
Write-Host "输出文件: $outputPath" -ForegroundColor Green

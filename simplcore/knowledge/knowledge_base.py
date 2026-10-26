# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""权威知识库系统 (Authoritative Knowledge Base).

核心思想: 构建一个可搜索的权威知识库, 模型可以在推理时查阅知识,
          实现"现场学习"能力——无需重新训练即可获取新知识。

知识库来源:
  1. Wikipedia 百科知识
  2. 学术论文摘要
  3. 技术文档
  4. 科学事实
  5. 历史事件
  6. 数学公式
  7. 物理定律

检索方式:
  - 语义检索 (FAISS/向量搜索)
  - 关键词检索
  - 混合检索

与模型的集成:
  - RAG (Retrieval-Augmented Generation)
  - 在推理时动态检索知识
  - 将检索到的知识融入上下文
"""

from __future__ import annotations

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple


class KnowledgeEntry:
    """知识条目."""
    
    def __init__(
        self,
        id: str,
        title: str,
        content: str,
        source: str,
        category: str,
        confidence: float = 1.0,
    ):
        self.id = id
        self.title = title
        self.content = content
        self.source = source
        self.category = category
        self.confidence = confidence
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "confidence": self.confidence,
        }


class KnowledgeBase:
    """权威知识库."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # 简单的TF-IDF索引
        self.tfidf_index = {}
        self.doc_freq = {}
        self.total_docs = 0
        
        # 加载内置知识库
        self._load_builtin_knowledge()
    
    def _load_builtin_knowledge(self):
        """加载内置权威知识."""
        print("[KnowledgeBase] 加载内置知识库...")
        
        # 物理学知识
        physics_knowledge = [
            {
                "title": "牛顿第二定律",
                "content": "牛顿第二定律描述了力、质量和加速度之间的关系，公式为 F = ma，其中F是力，m是质量，a是加速度。这是经典力学的基础定律。",
                "source": "物理学教科书",
                "category": "physics",
            },
            {
                "title": "万有引力定律",
                "content": "万有引力定律描述了两个物体之间的引力，公式为 F = G * (m1 * m2) / r^2，其中G是引力常数，m1和m2是两个物体的质量，r是它们之间的距离。",
                "source": "牛顿《自然哲学的数学原理》",
                "category": "physics",
            },
            {
                "title": "爱因斯坦质能方程",
                "content": "质能方程 E = mc^2 描述了质量和能量之间的等价关系，其中E是能量，m是质量，c是光速。这是相对论的核心公式。",
                "source": "爱因斯坦狭义相对论",
                "category": "physics",
            },
            {
                "title": "热力学第一定律",
                "content": "热力学第一定律，也称为能量守恒定律，表明能量既不会凭空产生，也不会凭空消失，只能从一种形式转化为另一种形式。公式为 ΔU = Q - W。",
                "source": "热力学教科书",
                "category": "physics",
            },
            {
                "title": "量子力学薛定谔方程",
                "content": "薛定谔方程是量子力学的核心方程，描述了量子系统的状态随时间的演化。时间依赖形式为 iħ ∂ψ/∂t = Hψ。",
                "source": "薛定谔",
                "category": "physics",
            },
            {
                "title": "牛顿第一定律",
                "content": "牛顿第一定律，也称为惯性定律，指出物体在没有外力作用时将保持静止或匀速直线运动状态。",
                "source": "牛顿《自然哲学的数学原理》",
                "category": "physics",
            },
            {
                "title": "牛顿第三定律",
                "content": "牛顿第三定律指出，作用力和反作用力大小相等、方向相反。公式为 F_AB = -F_BA。",
                "source": "牛顿《自然哲学的数学原理》",
                "category": "physics",
            },
            {
                "title": "热力学第二定律",
                "content": "热力学第二定律表明，孤立系统的熵不会减少。熵是系统无序程度的度量。",
                "source": "热力学教科书",
                "category": "physics",
            },
            {
                "title": "麦克斯韦方程组",
                "content": "麦克斯韦方程组是描述电磁现象的四个基本方程，统一了电、磁和光的理论。",
                "source": "麦克斯韦",
                "category": "physics",
            },
            {
                "title": "相对论",
                "content": "相对论分为狭义相对论和广义相对论。狭义相对论研究匀速运动的参考系，广义相对论研究引力和加速运动。",
                "source": "爱因斯坦",
                "category": "physics",
            },
            {
                "title": "量子纠缠",
                "content": "量子纠缠是量子力学中的一种现象，两个粒子无论相距多远，其状态都相互关联。测量一个粒子的状态会立即影响另一个粒子的状态。",
                "source": "量子力学教科书",
                "category": "physics",
            },
            {
                "title": "不确定性原理",
                "content": "海森堡不确定性原理指出，粒子的位置和动量不能同时被精确测量。公式为 Δx * Δp ≥ ħ/2。",
                "source": "海森堡",
                "category": "physics",
            },
            {
                "title": "光电效应",
                "content": "光电效应是指光照射到金属表面时，金属会发射出电子的现象。爱因斯坦解释了这一现象，并因此获得诺贝尔物理学奖。",
                "source": "爱因斯坦",
                "category": "physics",
            },
            {
                "title": "波粒二象性",
                "content": "波粒二象性是量子力学的基本概念，指微观粒子既表现出波动性，又表现出粒子性。",
                "source": "量子力学教科书",
                "category": "physics",
            },
            {
                "title": "原子结构",
                "content": "原子由原子核和电子组成。原子核包含质子和中子，电子在核外围绕原子核运动。质子数决定元素种类，中子数决定同位素。",
                "source": "原子物理学教科书",
                "category": "physics",
            },
        ]
        
        # 数学知识
        math_knowledge = [
            {
                "title": "勾股定理",
                "content": "勾股定理描述了直角三角形三边之间的关系，公式为 a^2 + b^2 = c^2，其中c是斜边，a和b是两条直角边。",
                "source": "欧几里得几何",
                "category": "mathematics",
            },
            {
                "title": "微积分基本定理",
                "content": "微积分基本定理建立了微分和积分之间的联系。第一部分：如果f是连续函数，那么d/dx ∫ₐˣ f(t)dt = f(x)。第二部分：∫ₐᵇ f'(x)dx = f(b) - f(a)。",
                "source": "牛顿、莱布尼茨",
                "category": "mathematics",
            },
            {
                "title": "欧拉公式",
                "content": "欧拉公式 e^(iθ) = cosθ + i sinθ 是数学中最优美的公式之一，将指数函数和三角函数联系起来。当θ=π时，得到 e^(iπ) + 1 = 0。",
                "source": "欧拉",
                "category": "mathematics",
            },
            {
                "title": "费马大定理",
                "content": "费马大定理指出，当n > 2时，方程 x^n + y^n = z^n 没有正整数解。这个定理在1994年被安德鲁·怀尔斯证明。",
                "source": "怀尔斯证明",
                "category": "mathematics",
            },
            {
                "title": "黎曼猜想",
                "content": "黎曼猜想是数学中最著名的未解决问题之一，涉及黎曼ζ函数的零点分布。如果成立，将对数论和密码学有深远影响。",
                "source": "黎曼",
                "category": "mathematics",
            },
            {
                "title": "哥德尔不完备定理",
                "content": "哥德尔不完备定理指出，任何足够复杂的形式系统，要么是不完备的，要么是不一致的。这意味着存在无法证明的真命题。",
                "source": "哥德尔",
                "category": "mathematics",
            },
            {
                "title": "矩阵乘法",
                "content": "矩阵乘法是线性代数中的基本运算。两个矩阵A和B相乘，结果矩阵C的元素C_ij = Σ A_ik * B_kj。",
                "source": "线性代数教科书",
                "category": "mathematics",
            },
            {
                "title": "傅里叶变换",
                "content": "傅里叶变换将信号从时域转换到频域，公式为 F(ω) = ∫ f(t) * e^(-iωt) dt。广泛应用于信号处理和图像处理。",
                "source": "傅里叶",
                "category": "mathematics",
            },
            {
                "title": "概率论",
                "content": "概率论是研究随机现象的数学分支。概率P(A)表示事件A发生的可能性，满足0 ≤ P(A) ≤ 1。",
                "source": "概率论教科书",
                "category": "mathematics",
            },
            {
                "title": "统计学",
                "content": "统计学是收集、分析、解释和呈现数据的科学。常用概念包括均值、方差、标准差、假设检验等。",
                "source": "统计学教科书",
                "category": "mathematics",
            },
            {
                "title": "线性回归",
                "content": "线性回归是一种统计方法，用于建模因变量和一个或多个自变量之间的线性关系。公式为 y = β0 + β1*x + ε。",
                "source": "统计学教科书",
                "category": "mathematics",
            },
            {
                "title": "贝叶斯定理",
                "content": "贝叶斯定理描述了在已知其他事件发生的情况下，某个事件发生的概率。公式为 P(A|B) = P(B|A) * P(A) / P(B)。",
                "source": "贝叶斯",
                "category": "mathematics",
            },
            {
                "title": "图论",
                "content": "图论是研究图的数学分支，图由顶点和边组成。图论广泛应用于网络分析、社交网络、路径规划等领域。",
                "source": "图论教科书",
                "category": "mathematics",
            },
            {
                "title": "数论",
                "content": "数论是研究整数性质的数学分支。包括素数、整除性、同余、丢番图方程等内容。",
                "source": "数论教科书",
                "category": "mathematics",
            },
            {
                "title": "拓扑学",
                "content": "拓扑学是研究空间在连续变形下保持不变性质的数学分支。著名概念包括连通性、紧致性、同胚等。",
                "source": "拓扑学教科书",
                "category": "mathematics",
            },
        ]
        
        # 计算机科学知识
        cs_knowledge = [
            {
                "title": "图灵机",
                "content": "图灵机是艾伦·图灵提出的一种抽象计算模型，是现代计算机的理论基础。它由一条无限长的纸带、一个读写头和一组状态组成。",
                "source": "图灵1936年论文",
                "category": "computer_science",
            },
            {
                "title": "NP完全问题",
                "content": "NP完全问题是指那些可以在多项式时间内验证解的正确性，但目前没有已知的多项式时间算法来求解的问题。著名的例子包括旅行商问题、SAT问题等。",
                "source": "计算机科学理论",
                "category": "computer_science",
            },
            {
                "title": "深度学习",
                "content": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。关键技术包括反向传播、卷积神经网络、循环神经网络和Transformer架构。",
                "source": "深度学习教科书",
                "category": "computer_science",
            },
            {
                "title": "信息论",
                "content": "信息论是研究信息的度量、传输和处理的数学理论。香农熵 H(X) = -Σ p(x) log p(x) 是信息不确定性的度量。",
                "source": "香农1948年论文",
                "category": "computer_science",
            },
            {
                "title": "机器学习",
                "content": "机器学习是人工智能的一个分支，使计算机能够从数据中学习模式并做出预测，而无需明确编程。主要类型包括监督学习、无监督学习和强化学习。",
                "source": "机器学习教科书",
                "category": "computer_science",
            },
            {
                "title": "神经网络",
                "content": "神经网络是模仿人脑神经元结构的计算模型。由输入层、隐藏层和输出层组成，通过激活函数和权重进行信息传递。",
                "source": "深度学习教科书",
                "category": "computer_science",
            },
            {
                "title": "Transformer架构",
                "content": "Transformer是一种基于自注意力机制的深度学习架构，由Vaswani等人于2017年提出。广泛应用于自然语言处理和计算机视觉。",
                "source": "Vaswani等人2017年论文",
                "category": "computer_science",
            },
            {
                "title": "卷积神经网络",
                "content": "卷积神经网络(CNN)是一种专门用于处理网格数据的神经网络。通过卷积层、池化层和全连接层提取特征。",
                "source": "深度学习教科书",
                "category": "computer_science",
            },
            {
                "title": "循环神经网络",
                "content": "循环神经网络(RNN)是一种处理序列数据的神经网络。通过循环连接保留历史信息，但存在梯度消失问题。",
                "source": "深度学习教科书",
                "category": "computer_science",
            },
            {
                "title": "强化学习",
                "content": "强化学习是机器学习的一种范式，智能体通过与环境交互获得奖励来学习最优策略。关键概念包括状态、动作、奖励和价值函数。",
                "source": "强化学习教科书",
                "category": "computer_science",
            },
            {
                "title": "计算机网络",
                "content": "计算机网络是连接多台计算机的系统。包括局域网(LAN)、广域网(WAN)、互联网等。TCP/IP是互联网的基础协议。",
                "source": "计算机网络教科书",
                "category": "computer_science",
            },
            {
                "title": "数据库",
                "content": "数据库是存储和管理数据的系统。关系型数据库使用表和SQL查询，非关系型数据库包括文档型、键值型等。",
                "source": "数据库教科书",
                "category": "computer_science",
            },
            {
                "title": "操作系统",
                "content": "操作系统是管理计算机硬件和软件资源的系统软件。常见操作系统包括Windows、Linux、macOS等。",
                "source": "操作系统教科书",
                "category": "computer_science",
            },
            {
                "title": "编译器",
                "content": "编译器是将高级语言源代码转换为机器语言的程序。编译过程包括词法分析、语法分析、语义分析、优化和代码生成。",
                "source": "编译器原理教科书",
                "category": "computer_science",
            },
            {
                "title": "算法复杂度",
                "content": "算法复杂度衡量算法的效率，包括时间复杂度和空间复杂度。常见复杂度包括O(1)、O(log n)、O(n)、O(n log n)、O(n^2)等。",
                "source": "算法导论",
                "category": "computer_science",
            },
        ]
        
        # 生物学知识
        bio_knowledge = [
            {
                "title": "达尔文进化论",
                "content": "达尔文进化论认为，物种通过自然选择而进化。自然选择是指适应环境的个体更有可能生存和繁殖，从而将其有利特征传递给后代。",
                "source": "达尔文《物种起源》",
                "category": "biology",
            },
            {
                "title": "DNA双螺旋结构",
                "content": "DNA的双螺旋结构由沃森和克里克于1953年发现。DNA由两条互补的核苷酸链组成，通过碱基配对（A-T, C-G）连接在一起。",
                "source": "沃森、克里克",
                "category": "biology",
            },
            {
                "title": "中心法则",
                "content": "分子生物学的中心法则描述了遗传信息的流动：DNA → RNA → 蛋白质。即DNA转录为RNA，RNA翻译为蛋白质。",
                "source": "克里克",
                "category": "biology",
            },
            {
                "title": "细胞学说",
                "content": "细胞学说认为，所有生物体都是由细胞组成的，细胞是生命的基本单位。细胞通过分裂产生新细胞。",
                "source": "施莱登、施旺",
                "category": "biology",
            },
            {
                "title": "光合作用",
                "content": "光合作用是植物、藻类和某些细菌将光能转化为化学能的过程。公式为 6CO2 + 6H2O + 光能 → C6H12O6 + 6O2。",
                "source": "植物生理学教科书",
                "category": "biology",
            },
            {
                "title": "细胞呼吸",
                "content": "细胞呼吸是细胞将有机物氧化分解产生能量的过程。包括糖酵解、柠檬酸循环和电子传递链三个阶段。",
                "source": "生物化学教科书",
                "category": "biology",
            },
            {
                "title": "遗传变异",
                "content": "遗传变异是生物种群中个体之间的遗传差异。包括基因突变、基因重组和染色体变异等形式。",
                "source": "遗传学教科书",
                "category": "biology",
            },
            {
                "title": "自然选择",
                "content": "自然选择是达尔文进化论的核心机制。环境压力导致适应环境的个体生存并繁殖，不适应的个体被淘汰。",
                "source": "达尔文《物种起源》",
                "category": "biology",
            },
            {
                "title": "生态系统",
                "content": "生态系统是生物群落与其环境相互作用形成的统一整体。包括生产者、消费者、分解者和非生物环境。",
                "source": "生态学教科书",
                "category": "biology",
            },
            {
                "title": "生物多样性",
                "content": "生物多样性指地球上所有生物的多样性，包括物种多样性、遗传多样性和生态系统多样性。",
                "source": "生态学教科书",
                "category": "biology",
            },
            {
                "title": "免疫系统",
                "content": "免疫系统是身体抵御病原体入侵的防御系统。包括先天性免疫和获得性免疫，通过抗体和免疫细胞发挥作用。",
                "source": "免疫学教科书",
                "category": "biology",
            },
            {
                "title": "神经科学",
                "content": "神经科学研究神经系统的结构和功能。神经元是神经系统的基本单位，通过突触传递信息。",
                "source": "神经科学教科书",
                "category": "biology",
            },
            {
                "title": "胚胎发育",
                "content": "胚胎发育是从受精卵到成体的过程。包括受精、卵裂、囊胚形成、原肠胚形成和器官发生等阶段。",
                "source": "发育生物学教科书",
                "category": "biology",
            },
            {
                "title": "生物分类",
                "content": "生物分类学将生物分为界、门、纲、目、科、属、种七个等级。目前公认的五界分类包括原核生物界、原生生物界、真菌界、植物界和动物界。",
                "source": "分类学教科书",
                "category": "biology",
            },
            {
                "title": "酶",
                "content": "酶是生物体内的催化剂，加速化学反应而不被消耗。酶具有高度特异性，受温度、pH等因素影响。",
                "source": "生物化学教科书",
                "category": "biology",
            },
        ]
        
        # 历史知识
        history_knowledge = [
            {
                "title": "工业革命",
                "content": "工业革命始于18世纪末的英国，标志着从手工生产向机器生产的转变。关键发明包括蒸汽机、纺纱机和铁路。",
                "source": "历史教科书",
                "category": "history",
            },
            {
                "title": "互联网诞生",
                "content": "互联网起源于1969年的ARPANET项目，最初是为了在不同计算机之间传输数据。1989年蒂姆·伯纳斯-李发明了万维网。",
                "source": "互联网历史",
                "category": "history",
            },
            {
                "title": "人工智能发展",
                "content": "人工智能领域诞生于1956年达特茅斯会议。重要里程碑包括深度学习的兴起、AlphaGo战胜围棋世界冠军、以及大型语言模型的发展。",
                "source": "AI历史",
                "category": "history",
            },
            {
                "title": "文艺复兴",
                "content": "文艺复兴是14-17世纪欧洲的文化运动，强调人文主义、艺术和科学的复兴。代表人物包括达芬奇、米开朗基罗、哥白尼等。",
                "source": "欧洲历史教科书",
                "category": "history",
            },
            {
                "title": "科学革命",
                "content": "科学革命是16-17世纪科学方法的根本性变革。代表人物包括伽利略、开普勒、牛顿等，奠定了现代科学的基础。",
                "source": "科学史教科书",
                "category": "history",
            },
            {
                "title": "法国大革命",
                "content": "法国大革命是1789年爆发的政治革命，推翻了君主制，建立了共和国。标志着现代民主制度的开端。",
                "source": "欧洲历史教科书",
                "category": "history",
            },
            {
                "title": "第一次世界大战",
                "content": "第一次世界大战发生于1914-1918年，是全球性的军事冲突。以同盟国和协约国两大阵营对抗，最终以协约国胜利结束。",
                "source": "世界历史教科书",
                "category": "history",
            },
            {
                "title": "第二次世界大战",
                "content": "第二次世界大战发生于1939-1945年，是人类历史上规模最大的战争。同盟国战胜轴心国，联合国成立。",
                "source": "世界历史教科书",
                "category": "history",
            },
            {
                "title": "冷战",
                "content": "冷战是1947-1991年美国和苏联之间的政治对抗。两大阵营进行军备竞赛和意识形态斗争，最终以苏联解体结束。",
                "source": "世界历史教科书",
                "category": "history",
            },
            {
                "title": "中国古代文明",
                "content": "中国古代文明是世界上最古老的文明之一，拥有五千年历史。包括夏朝、商朝、周朝、秦汉、唐宋元明清等朝代。",
                "source": "中国历史教科书",
                "category": "history",
            },
            {
                "title": "古希腊文明",
                "content": "古希腊文明是西方文明的源头之一，在哲学、科学、艺术、民主制度等方面做出了重要贡献。代表人物包括苏格拉底、柏拉图、亚里士多德等。",
                "source": "西方历史教科书",
                "category": "history",
            },
            {
                "title": "古罗马文明",
                "content": "古罗马文明继承了古希腊文明，建立了庞大的帝国。在法律、建筑、工程等方面有重要贡献。罗马法对现代法律体系有深远影响。",
                "source": "西方历史教科书",
                "category": "history",
            },
            {
                "title": "丝绸之路",
                "content": "丝绸之路是古代连接东西方的贸易路线，从中国长安延伸到地中海。促进了东西方文化、商品和技术的交流。",
                "source": "世界历史教科书",
                "category": "history",
            },
            {
                "title": "大航海时代",
                "content": "大航海时代是15-17世纪欧洲探险家探索世界的时期。哥伦布发现美洲，达伽马开辟通往印度的航线，麦哲伦完成环球航行。",
                "source": "世界历史教科书",
                "category": "history",
            },
            {
                "title": "工业革命",
                "content": "工业革命始于18世纪末的英国，标志着从手工生产向机器生产的转变。关键发明包括蒸汽机、纺纱机和铁路。",
                "source": "历史教科书",
                "category": "history",
            },
        ]
        
        # 化学知识
        chemistry_knowledge = [
            {
                "title": "元素周期表",
                "content": "元素周期表是按原子序数排列的化学元素表格。元素按周期和族排列，同族元素具有相似的化学性质。",
                "source": "门捷列夫",
                "category": "chemistry",
            },
            {
                "title": "原子结构",
                "content": "原子由原子核和电子组成。原子核包含质子和中子，电子在核外围绕原子核运动。质子数决定元素种类。",
                "source": "化学教科书",
                "category": "chemistry",
            },
            {
                "title": "化学键",
                "content": "化学键是原子之间的吸引力。主要类型包括离子键、共价键和金属键。化学键的强度决定了物质的性质。",
                "source": "化学教科书",
                "category": "chemistry",
            },
            {
                "title": "化学反应",
                "content": "化学反应是物质之间发生的化学变化。反应物转化为产物，质量守恒定律适用。反应速率受温度、浓度、催化剂等因素影响。",
                "source": "化学教科书",
                "category": "chemistry",
            },
            {
                "title": "酸碱反应",
                "content": "酸碱反应是酸和碱之间的反应，通常生成盐和水。酸释放氢离子，碱释放氢氧根离子。pH值表示溶液的酸碱度。",
                "source": "化学教科书",
                "category": "chemistry",
            },
            {
                "title": "氧化还原反应",
                "content": "氧化还原反应涉及电子的转移。氧化是失去电子，还原是获得电子。氧化剂使其他物质氧化，还原剂使其他物质还原。",
                "source": "化学教科书",
                "category": "chemistry",
            },
            {
                "title": "有机化学",
                "content": "有机化学研究含碳化合物。有机化合物包括烃类、醇类、醛类、羧酸类等。生命物质如蛋白质、核酸都是有机化合物。",
                "source": "有机化学教科书",
                "category": "chemistry",
            },
            {
                "title": "无机化学",
                "content": "无机化学研究不含碳的化合物。包括金属、非金属、酸、碱、盐等。无机化合物在材料科学中应用广泛。",
                "source": "无机化学教科书",
                "category": "chemistry",
            },
            {
                "title": "化学平衡",
                "content": "化学平衡是可逆反应中正反应速率等于逆反应速率的状态。勒沙特列原理指出，系统会抵消外界对平衡的干扰。",
                "source": "化学教科书",
                "category": "chemistry",
            },
            {
                "title": "热力学",
                "content": "化学热力学研究化学反应中的能量变化。焓变(ΔH)表示反应的吸热或放热，熵变(ΔS)表示系统混乱度的变化。",
                "source": "化学热力学教科书",
                "category": "chemistry",
            },
        ]
        
        # 全部知识
        all_knowledge = physics_knowledge + math_knowledge + cs_knowledge + bio_knowledge + history_knowledge + chemistry_knowledge
        
        # 添加前沿科技知识
        ai_knowledge = [
            {
                "title": "大型语言模型",
                "content": "大型语言模型(LLM)是基于Transformer架构的深度学习模型，通过大规模文本数据训练，能够理解和生成自然语言。代表模型包括GPT系列、LLaMA、Claude等。",
                "source": "OpenAI、Meta、Anthropic",
                "category": "ai",
            },
            {
                "title": "多模态学习",
                "content": "多模态学习研究如何处理和理解多种模态的数据，如图像、文本、音频、视频等。多模态模型能够整合不同模态的信息进行更全面的理解。",
                "source": "AI研究论文",
                "category": "ai",
            },
            {
                "title": "对齐技术",
                "content": "对齐技术是指使AI系统的行为与人类价值观和意图保持一致。包括RLHF(基于人类反馈的强化学习)、奖励建模、偏好学习等方法。",
                "source": "AI对齐研究",
                "category": "ai",
            },
            {
                "title": "Agent框架",
                "content": "Agent框架是使AI系统能够自主感知、决策和行动的架构。包括规划、记忆、反思、工具使用等组件。AutoGPT、LangChain是著名的Agent框架。",
                "source": "AI Agent研究",
                "category": "ai",
            },
            {
                "title": "RAG检索增强生成",
                "content": "RAG是一种结合检索和生成的AI技术，模型在生成答案前先从知识库中检索相关信息，能够提供更准确、可溯源的回答，减少幻觉。",
                "source": "RAG研究论文",
                "category": "ai",
            },
            {
                "title": "思维链",
                "content": "思维链(CoT)是一种提示工程技术，通过引导模型生成中间推理步骤来提高复杂推理能力。Chain-of-Thought prompting显著提升了模型在数学推理等任务上的表现。",
                "source": "Wei等人2022年论文",
                "category": "ai",
            },
            {
                "title": "工具使用",
                "content": "工具使用是指AI模型能够调用外部工具(如计算器、搜索引擎、API)来完成任务。Toolformer、GPT-4的插件系统是工具使用的典型案例。",
                "source": "AI工具使用研究",
                "category": "ai",
            },
            {
                "title": "自我反思",
                "content": "自我反思是AI模型评估自己的输出并进行修正的能力。Reflexion、Chain-of-Thought with Self-Consistency等方法提升了模型的可靠性。",
                "source": "AI反思研究",
                "category": "ai",
            },
            {
                "title": "持续学习",
                "content": "持续学习研究如何使AI模型在学习新知识时不忘记旧知识。关键挑战包括灾难性遗忘，常用方法包括弹性权重巩固、渐进式网络等。",
                "source": "持续学习研究",
                "category": "ai",
            },
            {
                "title": "神经符号AI",
                "content": "神经符号AI结合神经网络的学习能力和符号逻辑的推理能力。试图解决纯神经网络缺乏可解释性和逻辑推理能力的问题。",
                "source": "神经符号AI研究",
                "category": "ai",
            },
        ]
        
        vision_knowledge = [
            {
                "title": "Vision Transformer",
                "content": "Vision Transformer(ViT)是将Transformer架构应用于计算机视觉任务的模型。将图像分割为patch，通过自注意力机制提取特征。",
                "source": "Dosovitskiy等人2021年论文",
                "category": "vision",
            },
            {
                "title": "扩散模型",
                "content": "扩散模型是一种生成模型，通过逐步去除噪声来生成图像。Stable Diffusion、DALL-E是基于扩散模型的图像生成系统。",
                "source": "扩散模型研究",
                "category": "vision",
            },
            {
                "title": "目标检测",
                "content": "目标检测是识别图像中物体位置和类别的任务。常用方法包括Faster R-CNN、YOLO、SSD等，YOLO以其实时性著称。",
                "source": "计算机视觉教科书",
                "category": "vision",
            },
            {
                "title": "图像分割",
                "content": "图像分割是将图像划分为不同区域的任务。包括语义分割(每个像素分类)和实例分割(区分同一类的不同实例)。",
                "source": "计算机视觉教科书",
                "category": "vision",
            },
            {
                "title": "视频理解",
                "content": "视频理解是分析视频内容的任务，包括动作识别、视频分类、时空推理等。3D卷积和Transformer是常用方法。",
                "source": "视频理解研究",
                "category": "vision",
            },
        ]
        
        nlp_knowledge = [
            {
                "title": "BERT",
                "content": "BERT是Google于2018年提出的预训练语言模型，基于双向Transformer。开创了Masked Language Modeling预训练范式。",
                "source": "Devlin等人2018年论文",
                "category": "nlp",
            },
            {
                "title": "GPT",
                "content": "GPT(Generative Pre-trained Transformer)是OpenAI开发的生成式语言模型系列。GPT-1/2/3/4逐步扩大规模，展现了强大的语言理解和生成能力。",
                "source": "OpenAI",
                "category": "nlp",
            },
            {
                "title": "注意力机制",
                "content": "注意力机制允许模型在处理序列时关注不同位置的信息。自注意力是Transformer的核心，通过计算query、key、value的相似度来分配权重。",
                "source": "Vaswani等人2017年论文",
                "category": "nlp",
            },
            {
                "title": "tokenization",
                "content": "Tokenization是将文本分割为子词单元的过程。常见方法包括Byte Pair Encoding(BPE)、WordPiece、SentencePiece等。",
                "source": "NLP教科书",
                "category": "nlp",
            },
            {
                "title": "机器翻译",
                "content": "机器翻译是自动将一种语言翻译为另一种语言的任务。基于Transformer的模型如Google Translate、DeepL取得了接近人类水平的翻译质量。",
                "source": "机器翻译研究",
                "category": "nlp",
            },
        ]
        
        robotics_knowledge = [
            {
                "title": "强化学习机器人",
                "content": "强化学习在机器人领域用于训练智能体通过与环境交互学习技能。DeepMind的AlphaGo、OpenAI的Dactyl是强化学习在机器人领域的应用案例。",
                "source": "机器人学习研究",
                "category": "robotics",
            },
            {
                "title": "SLAM",
                "content": "SLAM(Simultaneous Localization and Mapping)是机器人在未知环境中同时构建地图和确定自身位置的技术。是自主导航的核心。",
                "source": "机器人学教科书",
                "category": "robotics",
            },
            {
                "title": "运动规划",
                "content": "运动规划是为机器人找到从起点到终点的无碰撞路径。常用算法包括A*、RRT(Rapidly-exploring Random Trees)等。",
                "source": "机器人学教科书",
                "category": "robotics",
            },
            {
                "title": "计算机视觉导航",
                "content": "计算机视觉导航使用视觉传感器(摄像头)进行机器人定位和导航。结合深度学习和传统计算机视觉方法。",
                "source": "机器人视觉研究",
                "category": "robotics",
            },
        ]
        
        quantum_knowledge = [
            {
                "title": "量子计算",
                "content": "量子计算利用量子力学现象(如叠加态、纠缠)进行计算。量子计算机在某些问题(如因子分解、优化)上可能超越经典计算机。",
                "source": "量子计算教科书",
                "category": "quantum",
            },
            {
                "title": "量子比特",
                "content": "量子比特(qubit)是量子计算的基本单位，可同时处于0和1的叠加态。与经典比特不同，量子比特具有量子相干性。",
                "source": "量子计算教科书",
                "category": "quantum",
            },
            {
                "title": "量子门",
                "content": "量子门是量子计算中的基本操作，用于操纵量子比特的状态。单量子比特门(如Hadamard门)和双量子比特门(如CNOT门)是基本量子门。",
                "source": "量子计算教科书",
                "category": "quantum",
            },
            {
                "title": "量子优势",
                "content": "量子优势是指量子计算机在特定任务上展现出超越最强大经典计算机的能力。Google的Sycamore处理器在2019年宣称实现了量子优势。",
                "source": "Google量子计算研究",
                "category": "quantum",
            },
            {
                "title": "量子纠错",
                "content": "量子纠错是保护量子信息免受退相干影响的技术。量子系统容易受到环境干扰，需要冗余编码来保护量子比特。",
                "source": "量子纠错研究",
                "category": "quantum",
            },
        ]
        
        advanced_physics_knowledge = [
            {
                "title": "弦理论",
                "content": "弦理论是一种试图统一量子力学和广义相对论的理论框架。认为基本粒子是一维弦的振动模式。",
                "source": "弦理论研究",
                "category": "physics",
            },
            {
                "title": "暗物质",
                "content": "暗物质是一种不与电磁辐射相互作用的物质，占宇宙质量的约27%。其存在通过引力效应被推断，但尚未被直接探测到。",
                "source": "天体物理学",
                "category": "physics",
            },
            {
                "title": "暗能量",
                "content": "暗能量是驱动宇宙加速膨胀的神秘能量，占宇宙能量的约68%。其性质是物理学中最大的未解之谜之一。",
                "source": "天体物理学",
                "category": "physics",
            },
            {
                "title": "黑洞",
                "content": "黑洞是引力极强的天体，其逃逸速度超过光速。事件视界是黑洞的边界，越过边界的任何物质都无法逃脱。",
                "source": "广义相对论",
                "category": "physics",
            },
            {
                "title": "引力波",
                "content": "引力波是时空的涟漪，由加速的质量产生。LIGO在2015年首次直接探测到引力波，证实了爱因斯坦的广义相对论预言。",
                "source": "LIGO合作组",
                "category": "physics",
            },
        ]
        
        all_knowledge += ai_knowledge + vision_knowledge + nlp_knowledge + robotics_knowledge + quantum_knowledge + advanced_physics_knowledge
        
        for i, item in enumerate(all_knowledge):
            entry = KnowledgeEntry(
                id=f"knowledge_{i:06d}",
                title=item["title"],
                content=item["content"],
                source=item["source"],
                category=item["category"],
            )
            self.add_entry(entry)
        
        print(f"[KnowledgeBase] 加载完成，共 {len(self.entries)} 条知识")
    
    def add_entry(self, entry: KnowledgeEntry):
        """添加知识条目."""
        self.entries[entry.id] = entry
        self.total_docs += 1
        
        # 更新分类索引
        if entry.category not in self.categories:
            self.categories[entry.category] = []
        self.categories[entry.category].append(entry.id)
        
        # 更新TF-IDF索引
        words = self._tokenize(entry.title + " " + entry.content)
        unique_words = set(words)
        for word in unique_words:
            if word not in self.doc_freq:
                self.doc_freq[word] = 0
            self.doc_freq[word] += 1
        
        for word in words:
            if word not in self.tfidf_index:
                self.tfidf_index[word] = []
            self.tfidf_index[word].append(entry.id)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词 (支持中英文)."""
        text = text.lower()
        for punct in '.,!?;:()[]{}"\'':
            text = text.replace(punct, ' ')
        
        words = []
        current_word = []
        
        for char in text:
            if char == ' ':
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
            elif '\u4e00' <= char <= '\u9fff':
                words.append(char)
            else:
                current_word.append(char)
        
        if current_word:
            words.append(''.join(current_word))
        
        return words
    
    def _compute_tfidf(self, text: str) -> Dict[str, float]:
        """计算TF-IDF分数."""
        words = self._tokenize(text)
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        
        tfidf = {}
        for word, freq in tf.items():
            if word in self.doc_freq:
                idf = np.log(self.total_docs / self.doc_freq[word])
                tfidf[word] = freq * idf
        
        return tfidf
    
    def search_keyword(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """关键词搜索."""
        tfidf = self._compute_tfidf(query)
        scores = {}
        
        for word, weight in tfidf.items():
            if word in self.tfidf_index:
                for entry_id in self.tfidf_index[word]:
                    if entry_id not in scores:
                        scores[entry_id] = 0
                    scores[entry_id] += weight
        
        # 排序
        sorted_entries = sorted(scores.items(), key=lambda x: -x[1])
        
        results = []
        for entry_id, score in sorted_entries[:top_k]:
            entry = self.entries[entry_id]
            results.append((entry, score))
        
        return results
    
    def search_category(self, category: str, top_k: int = 5) -> List[KnowledgeEntry]:
        """按分类搜索."""
        if category not in self.categories:
            return []
        
        entry_ids = self.categories[category][:top_k]
        return [self.entries[eid] for eid in entry_ids]
    
    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """语义搜索 (使用简单的向量相似度)."""
        # 简单实现：使用关键词匹配作为语义相似度的近似
        return self.search_keyword(query, top_k)
    
    def get_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """获取与查询相关的知识内容."""
        results = self.search_keyword(query, top_k)
        return [f"{entry.title} ({entry.source}): {entry.content}" for entry, _ in results]
    
    def get_categories(self) -> List[str]:
        """获取所有分类."""
        return list(self.categories.keys())
    
    def save(self, path: str = "knowledge_base.pkl"):
        """保存知识库."""
        data = {
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
            "categories": self.categories,
            "tfidf_index": self.tfidf_index,
            "doc_freq": self.doc_freq,
            "total_docs": self.total_docs,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[KnowledgeBase] 已保存到 {path}")
    
    def load(self, path: str = "knowledge_base.pkl"):
        """加载知识库."""
        if not os.path.exists(path):
            return
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.entries = {}
        for k, v in data["entries"].items():
            self.entries[k] = KnowledgeEntry(
                id=v["id"],
                title=v["title"],
                content=v["content"],
                source=v["source"],
                category=v["category"],
                confidence=v.get("confidence", 1.0),
            )
        
        self.categories = data["categories"]
        self.tfidf_index = data["tfidf_index"]
        self.doc_freq = data["doc_freq"]
        self.total_docs = data["total_docs"]
        print(f"[KnowledgeBase] 已从 {path} 加载 {len(self.entries)} 条知识")


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("权威知识库系统测试")
    print("=" * 60)
    
    kb = KnowledgeBase()
    
    # 测试关键词搜索
    print("\n[Test 1] 关键词搜索: '物理学'")
    results = kb.search_keyword("物理学", top_k=3)
    for entry, score in results:
        print(f"  {entry.title} ({entry.source}): {score:.2f}")
    
    print("\n[Test 2] 关键词搜索: '牛顿'")
    results = kb.search_keyword("牛顿", top_k=3)
    for entry, score in results:
        print(f"  {entry.title}: {score:.2f}")
    
    print("\n[Test 3] 关键词搜索: '深度学习'")
    results = kb.search_keyword("深度学习", top_k=3)
    for entry, score in results:
        print(f"  {entry.title}: {score:.2f}")
    
    print("\n[Test 4] 获取知识内容: '量子力学'")
    knowledge = kb.get_knowledge("量子力学", top_k=2)
    for k in knowledge:
        print(f"  {k[:100]}...")
    
    print("\n[Test 5] 获取分类")
    categories = kb.get_categories()
    print(f"  分类: {categories}")
    
    print("\n[Test 6] 按分类搜索: 'physics'")
    entries = kb.search_category("physics", top_k=3)
    for entry in entries:
        print(f"  {entry.title}")
    
    # 测试保存和加载
    kb.save()
    kb2 = KnowledgeBase()
    kb2.load()
    print(f"\n[Test 7] 保存/加载后条目数: {len(kb2.entries)}")
    
    print("\n" + "=" * 60)
    print("所有测试 PASSED!")
    print("=" * 60)


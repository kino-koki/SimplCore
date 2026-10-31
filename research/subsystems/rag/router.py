# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
查询路由器模块
实现智能查询路由，根据查询类型自动选择最优检索策略
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any


class QueryType(Enum):
    """查询类型枚举"""
    FACTS = "facts"
    KNOWLEDGE = "knowledge"
    CURRENT = "current"
    CONVERSATION = "conversation"
    CREATIVE = "creative"
    CALCULATION = "calculation"
    CODE = "code"
    UNKNOWN = "unknown"


class QueryRouter:
    """查询路由器"""
    
    def __init__(self):
        self.fact_keywords = [
            "是什么", "什么是", "定义", "解释", "说明", "含义", "概念", "原理",
            "谁", "何时", "何地", "多少", "几", "哪些", "哪些是",
        ]
        
        self.knowledge_keywords = [
            "知识", "理论", "定律", "定理", "公式", "方程", "原理",
            "科学", "物理", "数学", "化学", "生物", "历史", "计算机",
        ]
        
        self.current_keywords = [
            "最新", "最近", "今日", "今天", "现在", "当前",
            "2024", "2025", "2026", "最新消息", "新闻", "动态",
        ]
        
        self.conversation_keywords = [
            "你好", "嗨", "在吗", "谢谢", "请", "帮我", "可以",
            "如何", "怎么", "怎样", "能否", "是否",
        ]
        
        self.creative_keywords = [
            "写", "创作", "生成", "设计", "构思", "想象",
            "故事", "诗歌", "文章", "文案", "创意", "想法",
        ]
        
        self.calculation_keywords = [
            "计算", "求解", "求", "等于", "是多少", "结果",
            "加", "减", "乘", "除", "平方", "立方", "根号",
        ]
        
        self.code_keywords = [
            "代码", "编程", "python", "java", "javascript", "c++",
            "实现", "开发", "程序", "算法", "函数", "类",
        ]
    
    def classify(self, query: str) -> QueryType:
        """分类查询类型"""
        query_lower = query.lower()
        
        if self._contains_any(query_lower, self.code_keywords):
            return QueryType.CODE
        
        if self._contains_any(query_lower, self.calculation_keywords):
            return QueryType.CALCULATION
        
        if self._contains_any(query_lower, self.current_keywords):
            return QueryType.CURRENT
        
        if self._contains_any(query_lower, self.fact_keywords):
            return QueryType.FACTS
        
        if self._contains_any(query_lower, self.knowledge_keywords):
            return QueryType.KNOWLEDGE
        
        if self._contains_any(query_lower, self.creative_keywords):
            return QueryType.CREATIVE
        
        if self._contains_any(query_lower, self.conversation_keywords):
            return QueryType.CONVERSATION
        
        return QueryType.UNKNOWN
    
    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        """检查文本是否包含任何关键词"""
        for keyword in keywords:
            if keyword in text:
                return True
        return False
    
    def get_retrieval_strategy(self, query_type: QueryType) -> Dict[str, Any]:
        """获取检索策略"""
        strategies = {
            QueryType.FACTS: {
                'retriever': 'hybrid',
                'top_k': 5,
                'search_weight': 0.3,
                'enable_rag': True,
            },
            QueryType.KNOWLEDGE: {
                'retriever': 'knowledge',
                'top_k': 3,
                'search_weight': 0.0,
                'enable_rag': True,
            },
            QueryType.CURRENT: {
                'retriever': 'search',
                'top_k': 5,
                'search_weight': 1.0,
                'enable_rag': True,
            },
            QueryType.CONVERSATION: {
                'retriever': 'knowledge',
                'top_k': 2,
                'search_weight': 0.1,
                'enable_rag': False,
            },
            QueryType.CREATIVE: {
                'retriever': 'knowledge',
                'top_k': 2,
                'search_weight': 0.1,
                'enable_rag': False,
            },
            QueryType.CALCULATION: {
                'retriever': 'knowledge',
                'top_k': 1,
                'search_weight': 0.0,
                'enable_rag': True,
            },
            QueryType.CODE: {
                'retriever': 'knowledge',
                'top_k': 3,
                'search_weight': 0.2,
                'enable_rag': True,
            },
            QueryType.UNKNOWN: {
                'retriever': 'hybrid',
                'top_k': 3,
                'search_weight': 0.5,
                'enable_rag': True,
            },
        }
        
        return strategies.get(query_type, strategies[QueryType.UNKNOWN])
    
    def route(self, query: str) -> Dict[str, Any]:
        """路由查询"""
        query_type = self.classify(query)
        strategy = self.get_retrieval_strategy(query_type)
        
        return {
            'query': query,
            'query_type': query_type.value,
            'strategy': strategy,
        }

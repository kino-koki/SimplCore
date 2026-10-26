# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) 模块
实现检索增强生成功能，将搜索和知识库能力与AI模型深度集成

核心功能:
- 多源检索: 支持互联网搜索、知识库检索、混合检索
- 上下文管理: 检索结果的整合、排序和去重
- 生成增强: 将检索结果融入模型生成过程
- 智能路由: 根据查询类型自动选择最优检索策略
"""
from __future__ import annotations

from .retriever import Retriever, SearchRetriever, KnowledgeRetriever, HybridRetriever
from .context_manager import ContextManager, ContextItem
from .generator import RAGGenerator
from .router import QueryRouter, QueryType
from .integrator import RAGIntegrator, create_rag_integrator

__all__ = [
    'Retriever',
    'SearchRetriever',
    'KnowledgeRetriever',
    'HybridRetriever',
    'ContextManager',
    'ContextItem',
    'RAGGenerator',
    'QueryRouter',
    'QueryType',
    'RAGIntegrator',
    'create_rag_integrator',
]

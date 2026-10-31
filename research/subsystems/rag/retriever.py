# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
检索器模块
实现多源检索功能
"""
from __future__ import annotations

import abc
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime


class Retriever(abc.ABC):
    """检索器抽象基类"""
    
    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[ContextItem]:
        """执行检索"""
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """获取检索器名称"""
        pass


class ContextItem:
    """上下文项"""
    
    def __init__(
        self,
        content: str,
        source: str,
        title: Optional[str] = None,
        url: Optional[str] = None,
        score: float = 0.0,
        category: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        source_type: str = "unknown",
    ):
        self.content = content
        self.source = source
        self.title = title
        self.url = url
        self.score = score
        self.category = category
        self.timestamp = timestamp or datetime.now()
        self.source_type = source_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source': self.source,
            'title': self.title,
            'url': self.url,
            'score': self.score,
            'category': self.category,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source_type': self.source_type,
        }
    
    def __repr__(self):
        return f"ContextItem(title={self.title}, score={self.score:.2f}, source={self.source_type})"


class SearchRetriever(Retriever):
    """搜索检索器 - 使用互联网搜索获取信息"""
    
    def __init__(self, search_engine=None):
        self.search_engine = search_engine
    
    def _ensure_engine(self):
        if self.search_engine is None:
            from SimplCore.search import SearchEngine
            self.search_engine = SearchEngine(provider="duckduckgo")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[ContextItem]:
        self._ensure_engine()
        
        try:
            results = self.search_engine.search(
                query=query,
                num_results=top_k,
                rank=True,
                save_history=False,
            )
            
            items = []
            for result in results.results:
                item = ContextItem(
                    content=result.description,
                    source=result.source,
                    title=result.title,
                    url=result.url,
                    score=result.relevance_score,
                    timestamp=result.timestamp,
                    source_type="search",
                )
                items.append(item)
            
            return items
        
        except Exception as e:
            print(f"Search retrieval failed: {e}")
            return []
    
    def get_name(self) -> str:
        return "search"


class KnowledgeRetriever(Retriever):
    """知识库检索器 - 使用本地知识库获取信息"""
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
    
    def _ensure_kb(self):
        if self.knowledge_base is None:
            from SimplCore.knowledge import KnowledgeBase
            self.knowledge_base = KnowledgeBase()
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[ContextItem]:
        self._ensure_kb()
        
        try:
            results = self.knowledge_base.search_keyword(query, top_k=top_k)
            
            items = []
            for entry, score in results:
                item = ContextItem(
                    content=entry.content,
                    source=entry.source,
                    title=entry.title,
                    score=score,
                    category=entry.category,
                    source_type="knowledge",
                )
                items.append(item)
            
            return items
        
        except Exception as e:
            print(f"Knowledge retrieval failed: {e}")
            return []
    
    def get_name(self) -> str:
        return "knowledge"


class HybridRetriever(Retriever):
    """混合检索器 - 结合搜索和知识库检索"""
    
    def __init__(self, search_weight: float = 0.5):
        self.search_retriever = SearchRetriever()
        self.knowledge_retriever = KnowledgeRetriever()
        self.search_weight = search_weight
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[ContextItem]:
        search_results = self.search_retriever.retrieve(query, top_k=top_k)
        knowledge_results = self.knowledge_retriever.retrieve(query, top_k=top_k)
        
        all_results = []
        
        for item in search_results:
            item.score = item.score * self.search_weight
            all_results.append(item)
        
        for item in knowledge_results:
            item.score = item.score * (1 - self.search_weight)
            all_results.append(item)
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k]
    
    def get_name(self) -> str:
        return "hybrid"


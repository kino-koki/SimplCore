# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
上下文管理器模块
实现检索结果的整合、排序、去重和摘要
"""
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any
from datetime import datetime


class ContextManager:
    """上下文管理器"""
    
    def __init__(
        self,
        max_context_length: int = 4096,
        max_items: int = 10,
        enable_deduplication: bool = True,
        enable_reranking: bool = True,
    ):
        self.max_context_length = max_context_length
        self.max_items = max_items
        self.enable_deduplication = enable_deduplication
        self.enable_reranking = enable_reranking
    
    def manage(
        self,
        items: List[ContextItem],
        query: str,
    ) -> List[ContextItem]:
        """管理上下文项"""
        if not items:
            return []
        
        items = self._filter_relevant(items, query)
        
        if self.enable_deduplication:
            items = self._deduplicate(items)
        
        if self.enable_reranking:
            items = self._rerank(items, query)
        
        items = self._truncate(items)
        
        return items
    
    def _filter_relevant(self, items: List[ContextItem], query: str) -> List[ContextItem]:
        """过滤相关结果"""
        query_tokens = set(query.lower().split())
        filtered = []
        
        for item in items:
            content_tokens = set(item.content.lower().split())
            if query_tokens & content_tokens:
                filtered.append(item)
        
        return filtered
    
    def _deduplicate(self, items: List[ContextItem]) -> List[ContextItem]:
        """去重"""
        seen = set()
        unique_items = []
        
        for item in items:
            content_hash = hash(item.content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_items.append(item)
        
        return unique_items
    
    def _rerank(self, items: List[ContextItem], query: str) -> List[ContextItem]:
        """重新排序"""
        query_tokens = set(query.lower().split())
        
        for item in items:
            content_tokens = set(item.content.lower().split())
            title_tokens = set((item.title or "").lower().split())
            
            overlap = len(query_tokens & content_tokens) + len(query_tokens & title_tokens)
            relevance = overlap / max(len(query_tokens), 1)
            
            item.score = item.score * 0.6 + relevance * 0.4
        
        return sorted(items, key=lambda x: x.score, reverse=True)
    
    def _truncate(self, items: List[ContextItem]) -> List[ContextItem]:
        """截断到最大长度"""
        if len(items) <= self.max_items:
            return items
        
        return items[:self.max_items]
    
    def build_context_prompt(self, items: List[ContextItem], query: str) -> str:
        """构建上下文提示词"""
        if not items:
            return ""
        
        context_parts = []
        
        for i, item in enumerate(items):
            source_label = {
                "search": "互联网搜索",
                "knowledge": "权威知识库",
                "hybrid": "混合检索",
            }.get(item.source_type, "其他来源")
            
            part = f"""【{i+1}. {source_label}】
标题: {item.title or '无标题'}
来源: {item.source}
内容: {item.content}
"""
            context_parts.append(part)
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""根据以下检索到的信息，回答用户的问题：

【用户问题】
{query}

【参考信息】
{context_text}

【要求】
1. 基于提供的信息回答问题
2. 如果信息中没有相关内容，请说明"根据现有信息无法回答该问题"
3. 回答要简洁准确，引用来源时使用编号标记
"""
        
        return prompt
    
    def summarize_context(self, items: List[ContextItem]) -> str:
        """摘要上下文"""
        if not items:
            return ""
        
        summaries = []
        for item in items:
            summary = f"- {item.title}: {item.content[:100]}..."
            summaries.append(summary)
        
        return "\n".join(summaries)


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

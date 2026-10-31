# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索结果数据模型
定义搜索结果的数据结构和操作方法
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class SearchResult:
    """单个搜索结果"""
    
    title: str
    url: str
    description: str
    source: str = "unknown"
    rank: int = 0
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    snippet: Optional[str] = None
    image_url: Optional[str] = None
    cached_url: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'description': self.description,
            'source': self.source,
            'rank': self.rank,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp.isoformat(),
            'snippet': self.snippet,
            'image_url': self.image_url,
            'cached_url': self.cached_url,
            'category': self.category,
            'language': self.language,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        return cls(
            title=data['title'],
            url=data['url'],
            description=data['description'],
            source=data.get('source', 'unknown'),
            rank=data.get('rank', 0),
            relevance_score=data.get('relevance_score', 0.0),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now(),
            snippet=data.get('snippet'),
            image_url=data.get('image_url'),
            cached_url=data.get('cached_url'),
            category=data.get('category'),
            language=data.get('language'),
        )
    
    def __repr__(self) -> str:
        return f"SearchResult(title='{self.title[:30]}...', url='{self.url[:50]}...', score={self.relevance_score:.2f})"


@dataclass
class SearchResults:
    """搜索结果集合"""
    
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    execution_time: float = 0.0
    provider: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_result(self, result: SearchResult) -> None:
        self.results.append(result)
        self.total_results = len(self.results)
    
    def top_n(self, n: int) -> List[SearchResult]:
        return sorted(self.results, key=lambda r: r.relevance_score, reverse=True)[:n]
    
    def filter_by_language(self, language: str) -> 'SearchResults':
        filtered = [r for r in self.results if r.language == language]
        return SearchResults(
            query=self.query,
            results=filtered,
            total_results=len(filtered),
            execution_time=self.execution_time,
            provider=self.provider,
            timestamp=self.timestamp,
        )
    
    def filter_by_source(self, source: str) -> 'SearchResults':
        filtered = [r for r in self.results if source.lower() in r.source.lower()]
        return SearchResults(
            query=self.query,
            results=filtered,
            total_results=len(filtered),
            execution_time=self.execution_time,
            provider=self.provider,
            timestamp=self.timestamp,
        )
    
    def sort_by_relevance(self) -> None:
        self.results.sort(key=lambda r: r.relevance_score, reverse=True)
    
    def sort_by_date(self) -> None:
        self.results.sort(key=lambda r: r.timestamp, reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'total_results': self.total_results,
            'execution_time': self.execution_time,
            'provider': self.provider,
            'timestamp': self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResults':
        results = [SearchResult.from_dict(r) for r in data.get('results', [])]
        return cls(
            query=data['query'],
            results=results,
            total_results=data.get('total_results', len(results)),
            execution_time=data.get('execution_time', 0.0),
            provider=data.get('provider', 'unknown'),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now(),
        )
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)

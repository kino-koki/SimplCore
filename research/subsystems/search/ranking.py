# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索结果排序与筛选模块
实现搜索结果的智能排序和筛选机制
"""
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta

from .results import SearchResult, SearchResults


class ResultRanker:
    """搜索结果排序器"""
    
    def __init__(self):
        self.ranking_factors = {
            'relevance': 0.4,
            'recency': 0.2,
            'domain_authority': 0.2,
            'content_length': 0.1,
            'keyword_density': 0.1,
        }
        
        self.authoritative_domains = {
            'gov.cn', 'edu.cn', 'org.cn', '.gov', '.edu',
            'wikipedia.org', 'baidu.com', 'sogou.com',
            'news.qq.com', 'news.sina.com.cn', 'news.163.com',
            'github.com', 'stackoverflow.com', 'zhihu.com',
        }
    
    def rank(self, results: SearchResults, query: str) -> SearchResults:
        """对搜索结果进行排序"""
        for result in results.results:
            score = self._calculate_score(result, query)
            result.relevance_score = score
        
        results.sort_by_relevance()
        return results
    
    def _calculate_score(self, result: SearchResult, query: str) -> float:
        """计算综合排序分数"""
        score = 0.0
        
        score += self.ranking_factors['relevance'] * self._score_relevance(result, query)
        score += self.ranking_factors['recency'] * self._score_recency(result)
        score += self.ranking_factors['domain_authority'] * self._score_domain_authority(result)
        score += self.ranking_factors['content_length'] * self._score_content_length(result)
        score += self.ranking_factors['keyword_density'] * self._score_keyword_density(result, query)
        
        return min(score, 1.0)
    
    def _score_relevance(self, result: SearchResult, query: str) -> float:
        """计算相关性分数"""
        query_tokens = set(query.lower().split())
        title_tokens = set(result.title.lower().split())
        desc_tokens = set(result.description.lower().split())
        
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        desc_overlap = len(query_tokens & desc_tokens) / max(len(query_tokens), 1)
        
        return 0.6 * title_overlap + 0.4 * desc_overlap
    
    def _score_recency(self, result: SearchResult) -> float:
        """计算时效性分数"""
        try:
            hours_old = (datetime.now() - result.timestamp).total_seconds() / 3600
            
            if hours_old < 24:
                return 1.0
            elif hours_old < 168:
                return 0.7
            elif hours_old < 720:
                return 0.4
            else:
                return 0.1
        except:
            return 0.5
    
    def _score_domain_authority(self, result: SearchResult) -> float:
        """计算域名权威性分数"""
        domain = result.source.lower()
        
        for auth_domain in self.authoritative_domains:
            if auth_domain in domain:
                return 1.0
        
        if domain.count('.') <= 2:
            return 0.7
        else:
            return 0.3
    
    def _score_content_length(self, result: SearchResult) -> float:
        """计算内容长度分数"""
        length = len(result.description)
        
        if length > 200:
            return 1.0
        elif length > 100:
            return 0.7
        elif length > 50:
            return 0.4
        else:
            return 0.1
    
    def _score_keyword_density(self, result: SearchResult, query: str) -> float:
        """计算关键词密度分数"""
        query_lower = query.lower()
        text = (result.title + " " + result.description).lower()
        
        if not text:
            return 0.0
        
        density = text.count(query_lower) / len(text)
        return min(density * 100, 1.0)


class ResultFilter:
    """搜索结果筛选器"""
    
    def __init__(self):
        self.blocked_domains = set()
        self.allowed_domains = set()
    
    def filter(self, results: SearchResults, **filters) -> SearchResults:
        """应用筛选条件"""
        filtered = results.results
        
        if 'language' in filters:
            filtered = self._filter_by_language(filtered, filters['language'])
        
        if 'domain' in filters:
            filtered = self._filter_by_domain(filtered, filters['domain'])
        
        if 'min_score' in filters:
            filtered = self._filter_by_score(filtered, filters['min_score'])
        
        if 'max_results' in filters:
            filtered = filtered[:filters['max_results']]
        
        if 'exclude' in filters:
            filtered = self._filter_by_exclude(filtered, filters['exclude'])
        
        if 'include' in filters:
            filtered = self._filter_by_include(filtered, filters['include'])
        
        return SearchResults(
            query=results.query,
            results=filtered,
            total_results=len(filtered),
            execution_time=results.execution_time,
            provider=results.provider,
            timestamp=results.timestamp,
        )
    
    def _filter_by_language(self, results: List[SearchResult], language: str) -> List[SearchResult]:
        """按语言筛选"""
        return [r for r in results if r.language == language]
    
    def _filter_by_domain(self, results: List[SearchResult], domain: str) -> List[SearchResult]:
        """按域名筛选"""
        if self.allowed_domains:
            return [r for r in results if any(d in r.source.lower() for d in self.allowed_domains)]
        if domain:
            return [r for r in results if domain.lower() in r.source.lower()]
        return [r for r in results if r.source.lower() not in self.blocked_domains]
    
    def _filter_by_score(self, results: List[SearchResult], min_score: float) -> List[SearchResult]:
        """按最低分数筛选"""
        return [r for r in results if r.relevance_score >= min_score]
    
    def _filter_by_exclude(self, results: List[SearchResult], exclude: str) -> List[SearchResult]:
        """排除包含特定关键词的结果"""
        exclude_tokens = exclude.lower().split()
        return [r for r in results if not any(t in r.title.lower() or t in r.description.lower() for t in exclude_tokens)]
    
    def _filter_by_include(self, results: List[SearchResult], include: str) -> List[SearchResult]:
        """只保留包含特定关键词的结果"""
        include_tokens = include.lower().split()
        return [r for r in results if any(t in r.title.lower() or t in r.description.lower() for t in include_tokens)]
    
    def add_blocked_domain(self, domain: str) -> None:
        """添加屏蔽域名"""
        self.blocked_domains.add(domain.lower())
    
    def add_allowed_domain(self, domain: str) -> None:
        """添加允许域名"""
        self.allowed_domains.add(domain.lower())

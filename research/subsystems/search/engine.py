# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索引擎核心模块
提供AI模型与搜索引擎的对接接口
"""
from __future__ import annotations

import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from .providers import SearchProvider, DuckDuckGoProvider, GoogleProvider, BingProvider, WebScraper
from .results import SearchResult, SearchResults
from .ranking import ResultRanker, ResultFilter
from .history import SearchHistory
from .rate_limiter import RateLimiter, SearchQuotaManager, RetryHandler


class SearchEngine:
    """搜索引擎核心类"""
    
    def __init__(
        self,
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        max_results: int = 10,
        rate_limit: int = 10,
        time_window: int = 60,
        daily_quota: int = 1000,
    ):
        self.provider_name = provider
        self.api_key = api_key
        self.max_results = max_results
        
        self.provider = self._create_provider(provider, api_key)
        self.ranker = ResultRanker()
        self.filter = ResultFilter()
        self.history = SearchHistory()
        self.rate_limiter = RateLimiter(max_requests=rate_limit, time_window=time_window)
        self.quota_manager = SearchQuotaManager(daily_quota=daily_quota)
        self.retry_handler = RetryHandler(max_retries=3)
        self.scraper = WebScraper()
    
    def _create_provider(self, provider: str, api_key: Optional[str]) -> SearchProvider:
        """创建搜索引擎提供商实例"""
        providers = {
            'duckduckgo': DuckDuckGoProvider,
            'google': GoogleProvider,
            'bing': BingProvider,
        }
        
        provider_class = providers.get(provider.lower(), DuckDuckGoProvider)
        return provider_class(api_key=api_key)
    
    def search(
        self,
        query: str,
        num_results: int = None,
        rank: bool = True,
        save_history: bool = True,
        scrape_content: bool = False,
        **filters
    ) -> SearchResults:
        """执行搜索查询"""
        start_time = time.time()
        
        if not self.quota_manager.check_quota():
            print("Daily quota exceeded")
            return SearchResults(query=query, results=[], provider=self.provider_name)
        
        if not self.rate_limiter.acquire():
            print("Rate limit exceeded")
            return SearchResults(query=query, results=[], provider=self.provider_name)
        
        try:
            results = self.retry_handler.retry(
                self.provider.search,
                query=query,
                num_results=num_results or self.max_results,
                **filters
            )
            
            results.execution_time = time.time() - start_time
            
            if rank:
                results = self.ranker.rank(results, query)
            
            if filters:
                results = self.filter.filter(results, **filters)
            
            if scrape_content:
                for result in results.results:
                    content = self.scraper.scrape(result.url)
                    if content:
                        result.snippet = content[:500]
            
            if save_history:
                self.history.add_entry(results)
            
            self.quota_manager.consume_quota()
            
            return results
        
        except Exception as e:
            print(f"Search failed: {e}")
            return SearchResults(query=query, results=[], provider=self.provider_name)
    
    def multi_search(
        self,
        queries: List[str],
        num_results: int = None,
        **filters
    ) -> List[SearchResults]:
        """批量搜索多个查询"""
        results_list = []
        for query in queries:
            results = self.search(query, num_results=num_results, **filters)
            results_list.append(results)
        return results_list
    
    def search_with_context(
        self,
        query: str,
        context: str = "",
        **kwargs
    ) -> SearchResults:
        """带上下文的搜索"""
        enhanced_query = f"{context} {query}" if context else query
        return self.search(enhanced_query, **kwargs)
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取搜索历史"""
        return [entry.to_dict() for entry in self.history.get_history(limit)]
    
    def get_recent_queries(self, limit: int = 10) -> List[str]:
        """获取最近搜索关键词"""
        return self.history.get_recent_queries(limit)
    
    def get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门搜索关键词"""
        return self.history.get_top_queries(limit)
    
    def clear_history(self) -> None:
        """清除搜索历史"""
        self.history.clear_history()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return {
            'history': self.history.get_stats(),
            'remaining_quota': self.quota_manager.get_remaining_quota(),
            'remaining_requests': self.rate_limiter.get_remaining_requests(),
            'provider': self.provider_name,
        }
    
    def switch_provider(self, provider: str, api_key: Optional[str] = None) -> None:
        """切换搜索引擎提供商"""
        self.provider_name = provider
        self.provider = self._create_provider(provider, api_key)
    
    def set_rate_limit(self, max_requests: int, time_window: int) -> None:
        """设置频率限制"""
        self.rate_limiter = RateLimiter(max_requests=max_requests, time_window=time_window)
    
    def set_daily_quota(self, quota: int) -> None:
        """设置每日配额"""
        self.quota_manager = SearchQuotaManager(daily_quota=quota)


_search_engine = None


def search(
    query: str,
    provider: str = "duckduckgo",
    api_key: Optional[str] = None,
    num_results: int = 10,
    rank: bool = True,
    save_history: bool = True,
    **filters
) -> SearchResults:
    """便捷搜索函数"""
    global _search_engine
    
    if _search_engine is None or _search_engine.provider_name != provider:
        _search_engine = SearchEngine(provider=provider, api_key=api_key)
    
    return _search_engine.search(
        query=query,
        num_results=num_results,
        rank=rank,
        save_history=save_history,
        **filters
    )

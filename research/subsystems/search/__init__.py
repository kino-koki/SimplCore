# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore AI搜索模块
提供AI模型与互联网搜索引擎的接口对接，支持关键词检索、信息爬取与解析
"""
from __future__ import annotations

from .engine import SearchEngine, search
from .providers import SearchProvider, DuckDuckGoProvider, GoogleProvider, BingProvider, WebScraper
from .results import SearchResult, SearchResults
from .ranking import ResultRanker, ResultFilter
from .history import SearchHistory, SearchHistoryEntry
from .rate_limiter import RateLimiter, SearchQuotaManager, RetryHandler, SearchError, RateLimitExceededError, QuotaExceededError, SearchTimeoutError, SearchNetworkError
from .ui import SearchUI, SearchAPIServer, run_search_server

__all__ = [
    'SearchEngine',
    'search',
    'SearchProvider',
    'DuckDuckGoProvider',
    'GoogleProvider',
    'BingProvider',
    'WebScraper',
    'SearchResult',
    'SearchResults',
    'ResultRanker',
    'ResultFilter',
    'SearchHistory',
    'SearchHistoryEntry',
    'RateLimiter',
    'SearchQuotaManager',
    'RetryHandler',
    'SearchError',
    'RateLimitExceededError',
    'QuotaExceededError',
    'SearchTimeoutError',
    'SearchNetworkError',
    'SearchUI',
    'SearchAPIServer',
    'run_search_server',
]


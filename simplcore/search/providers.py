# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索引擎提供商接口
支持多搜索引擎对接（DuckDuckGo、Google、Bing等）
"""
from __future__ import annotations

import abc
import json
import time
import urllib.parse
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests
from fake_useragent import UserAgent

from .results import SearchResult, SearchResults


class SearchProvider(abc.ABC):
    """搜索引擎提供商抽象基类"""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': UserAgent().random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
    
    @abc.abstractmethod
    def search(self, query: str, num_results: int = 10, **kwargs) -> SearchResults:
        """执行搜索查询"""
        pass
    
    @abc.abstractmethod
    def name(self) -> str:
        """返回提供商名称"""
        pass
    
    def _safe_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """安全请求，处理异常"""
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """从URL提取域名"""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except:
            return "unknown"


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo搜索引擎提供商"""
    
    BASE_URL = "https://api.duckduckgo.com"
    
    def name(self) -> str:
        return "duckduckgo"
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> SearchResults:
        params = {
            'q': query,
            'format': 'json',
            'pretty': '1',
            'no_html': '1',
            'skip_disambig': '1',
        }
        
        response = self._safe_request(self.BASE_URL, params=params)
        if not response:
            return SearchResults(query=query, results=[], provider=self.name())
        
        try:
            data = response.json()
            results = []
            
            for idx, item in enumerate(data.get('RelatedTopics', [])):
                if idx >= num_results:
                    break
                
                if item.get('Result'):
                    title = item.get('Text', '')
                    url = item.get('FirstURL', '')
                    description = item.get('Text', '')
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        description=description,
                        source=self._extract_domain(url),
                        rank=idx + 1,
                        relevance_score=1.0 - (idx * 0.05),
                    ))
            
            return SearchResults(
                query=query,
                results=results,
                total_results=len(results),
                provider=self.name(),
            )
        except json.JSONDecodeError:
            return SearchResults(query=query, results=[], provider=self.name())


class GoogleProvider(SearchProvider):
    """Google搜索引擎提供商"""
    
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def name(self) -> str:
        return "google"
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> SearchResults:
        if not self.api_key:
            print("Google API key required")
            return SearchResults(query=query, results=[], provider=self.name())
        
        params = {
            'q': query,
            'key': self.api_key,
            'cx': kwargs.get('cx', ''),
            'num': min(num_results, 10),
        }
        
        response = self._safe_request(self.BASE_URL, params=params)
        if not response:
            return SearchResults(query=query, results=[], provider=self.name())
        
        try:
            data = response.json()
            results = []
            
            for idx, item in enumerate(data.get('items', [])):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    description=item.get('snippet', ''),
                    source=self._extract_domain(item.get('link', '')),
                    rank=idx + 1,
                    relevance_score=1.0 - (idx * 0.05),
                ))
            
            return SearchResults(
                query=query,
                results=results,
                total_results=data.get('totalResults', len(results)),
                provider=self.name(),
            )
        except json.JSONDecodeError:
            return SearchResults(query=query, results=[], provider=self.name())


class BingProvider(SearchProvider):
    """Bing搜索引擎提供商"""
    
    BASE_URL = "https://api.bing.microsoft.com/v7.0/search"
    
    def name(self) -> str:
        return "bing"
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> SearchResults:
        if not self.api_key:
            print("Bing API key required")
            return SearchResults(query=query, results=[], provider=self.name())
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
        }
        
        params = {
            'q': query,
            'count': min(num_results, 50),
            'mkt': kwargs.get('mkt', 'zh-CN'),
        }
        
        response = self._safe_request(self.BASE_URL, headers=headers, params=params)
        if not response:
            return SearchResults(query=query, results=[], provider=self.name())
        
        try:
            data = response.json()
            results = []
            
            for idx, item in enumerate(data.get('webPages', {}).get('value', [])):
                results.append(SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    description=item.get('snippet', ''),
                    source=self._extract_domain(item.get('url', '')),
                    rank=idx + 1,
                    relevance_score=1.0 - (idx * 0.05),
                ))
            
            return SearchResults(
                query=query,
                results=results,
                total_results=data.get('webPages', {}).get('totalEstimatedMatches', len(results)),
                provider=self.name(),
            )
        except json.JSONDecodeError:
            return SearchResults(query=query, results=[], provider=self.name())


class WebScraper:
    """网页内容爬取器"""
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': UserAgent().random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'max-age=0',
        })
    
    def scrape(self, url: str, max_chars: int = 5000) -> Optional[str]:
        """爬取网页内容"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(['script', 'style', 'nav', 'header', 'footer']):
                script.decompose()
            
            text = soup.get_text()
            text = ' '.join(text.split())
            
            return text[:max_chars]
        except requests.RequestException as e:
            print(f"Failed to scrape {url}: {e}")
            return None
        except ImportError:
            return response.text[:max_chars] if 'response' in dir() else None
        except Exception as e:
            print(f"Unexpected error scraping {url}: {e}")
            return None

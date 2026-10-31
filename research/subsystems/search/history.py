# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索历史记录管理模块
实现搜索历史的存储与管理功能
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .results import SearchResult, SearchResults


@dataclass
class SearchHistoryEntry:
    """搜索历史条目"""
    
    query: str
    timestamp: datetime
    results: List[SearchResult] = field(default_factory=list)
    provider: str = "unknown"
    execution_time: float = 0.0
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'timestamp': self.timestamp.isoformat(),
            'results': [r.to_dict() for r in self.results],
            'provider': self.provider,
            'execution_time': self.execution_time,
            'user_id': self.user_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchHistoryEntry':
        results = [SearchResult.from_dict(r) for r in data.get('results', [])]
        return cls(
            query=data['query'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            results=results,
            provider=data.get('provider', 'unknown'),
            execution_time=data.get('execution_time', 0.0),
            user_id=data.get('user_id'),
        )


class SearchHistory:
    """搜索历史管理器"""
    
    def __init__(self, storage_path: str = None, max_entries: int = 1000):
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser('~'), '.SimplCore', 'search_history.json'
        )
        self.max_entries = max_entries
        self.entries: List[SearchHistoryEntry] = []
        self._ensure_storage()
        self._load_history()
    
    def _ensure_storage(self) -> None:
        """确保存储目录存在"""
        dir_path = os.path.dirname(self.storage_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    def _load_history(self) -> None:
        """加载历史记录"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entries = [SearchHistoryEntry.from_dict(entry) for entry in data]
            except (json.JSONDecodeError, Exception):
                self.entries = []
    
    def _save_history(self) -> None:
        """保存历史记录"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                data = [entry.to_dict() for entry in self.entries]
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save history: {e}")
    
    def add_entry(self, results: SearchResults, user_id: Optional[str] = None) -> None:
        """添加搜索历史条目"""
        entry = SearchHistoryEntry(
            query=results.query,
            timestamp=results.timestamp,
            results=results.results,
            provider=results.provider,
            execution_time=results.execution_time,
            user_id=user_id,
        )
        
        self.entries.insert(0, entry)
        
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[:self.max_entries]
        
        self._save_history()
    
    def get_history(self, limit: int = 50, user_id: Optional[str] = None) -> List[SearchHistoryEntry]:
        """获取搜索历史"""
        entries = self.entries
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        return entries[:limit]
    
    def search_history(self, query: str, limit: int = 10) -> List[SearchHistoryEntry]:
        """搜索历史记录"""
        query_lower = query.lower()
        return [
            e for e in self.entries
            if query_lower in e.query.lower()
        ][:limit]
    
    def get_recent_queries(self, limit: int = 10) -> List[str]:
        """获取最近的搜索关键词"""
        queries = []
        seen = set()
        for entry in self.entries:
            if entry.query not in seen:
                queries.append(entry.query)
                seen.add(entry.query)
                if len(queries) >= limit:
                    break
        return queries
    
    def get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门搜索关键词"""
        query_counts = {}
        for entry in self.entries:
            query_counts[entry.query] = query_counts.get(entry.query, 0) + 1
        
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'query': q, 'count': c} for q, c in sorted_queries[:limit]]
    
    def clear_history(self, user_id: Optional[str] = None) -> None:
        """清除搜索历史"""
        if user_id:
            self.entries = [e for e in self.entries if e.user_id != user_id]
        else:
            self.entries = []
        self._save_history()
    
    def delete_entry(self, index: int) -> bool:
        """删除指定索引的历史条目"""
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            self._save_history()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        total_searches = len(self.entries)
        unique_queries = len(set(e.query for e in self.entries))
        total_results = sum(len(e.results) for e in self.entries)
        
        if total_searches > 0:
            avg_results = total_results / total_searches
            avg_time = sum(e.execution_time for e in self.entries) / total_searches
        else:
            avg_results = 0
            avg_time = 0
        
        return {
            'total_searches': total_searches,
            'unique_queries': unique_queries,
            'total_results': total_results,
            'avg_results_per_search': avg_results,
            'avg_execution_time': avg_time,
            'providers': list(set(e.provider for e in self.entries)),
        }
    
    def export_history(self, file_path: str) -> bool:
        """导出历史记录"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                data = [entry.to_dict() for entry in self.entries]
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Failed to export history: {e}")
            return False
    
    def import_history(self, file_path: str) -> bool:
        """导入历史记录"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = [SearchHistoryEntry.from_dict(entry) for entry in data]
                self.entries = entries[:self.max_entries]
                self._save_history()
            return True
        except Exception as e:
            print(f"Failed to import history: {e}")
            return False





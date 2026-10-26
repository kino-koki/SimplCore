# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索频率限制器
实现网络请求的频率限制和异常处理
"""
from __future__ import annotations

import time
import threading
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta


class RateLimiter:
    """搜索频率限制器"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = []
        self.lock = threading.Lock()
        self.last_request_time = 0
        self.min_interval = 0.5
    
    def acquire(self) -> bool:
        """获取请求许可"""
        with self.lock:
            now = time.time()
            
            if now - self.last_request_time < self.min_interval:
                time.sleep(self.min_interval - (now - self.last_request_time))
            
            self.request_timestamps = [t for t in self.request_timestamps if now - t < self.time_window]
            
            if len(self.request_timestamps) >= self.max_requests:
                wait_time = self.time_window - (now - self.request_timestamps[0])
                if wait_time > 0:
                    time.sleep(wait_time)
                self.request_timestamps.pop(0)
            
            self.request_timestamps.append(time.time())
            self.last_request_time = time.time()
            
            return True
    
    def get_remaining_requests(self) -> int:
        """获取剩余请求次数"""
        with self.lock:
            now = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if now - t < self.time_window]
            return max(0, self.max_requests - len(self.request_timestamps))
    
    def is_available(self) -> bool:
        """检查是否可以发起请求"""
        with self.lock:
            now = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if now - t < self.time_window]
            return len(self.request_timestamps) < self.max_requests


class SearchQuotaManager:
    """搜索配额管理器"""
    
    def __init__(self, daily_quota: int = 1000):
        self.daily_quota = daily_quota
        self.today_requests = 0
        self.last_reset_date = datetime.now().date()
        self.lock = threading.Lock()
    
    def check_quota(self) -> bool:
        """检查配额是否充足"""
        with self.lock:
            self._reset_if_new_day()
            return self.today_requests < self.daily_quota
    
    def consume_quota(self, amount: int = 1) -> bool:
        """消耗配额"""
        with self.lock:
            self._reset_if_new_day()
            if self.today_requests + amount <= self.daily_quota:
                self.today_requests += amount
                return True
            return False
    
    def get_remaining_quota(self) -> int:
        """获取剩余配额"""
        with self.lock:
            self._reset_if_new_day()
            return max(0, self.daily_quota - self.today_requests)
    
    def _reset_if_new_day(self) -> None:
        """新的一天重置配额"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.today_requests = 0
            self.last_reset_date = today


class RetryHandler:
    """重试处理器"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def retry(self, func: Callable, *args, **kwargs):
        """执行带重试的函数调用"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    raise last_exception
        
        raise last_exception


class SearchError(Exception):
    """搜索错误异常"""
    
    def __init__(self, message: str, code: int = 0, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RateLimitExceededError(SearchError):
    """频率限制超出异常"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, code=429, retryable=True)


class QuotaExceededError(SearchError):
    """配额超出异常"""
    
    def __init__(self, message: str = "Daily quota exceeded"):
        super().__init__(message, code=403, retryable=False)


class SearchTimeoutError(SearchError):
    """搜索超时异常"""
    
    def __init__(self, message: str = "Search timed out"):
        super().__init__(message, code=408, retryable=True)


class SearchNetworkError(SearchError):
    """网络错误异常"""
    
    def __init__(self, message: str = "Network error"):
        super().__init__(message, code=503, retryable=True)

# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
RAG生成器模块
实现检索增强生成逻辑
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime


class RAGGenerator:
    """RAG生成器"""
    
    def __init__(
        self,
        retriever=None,
        context_manager=None,
        model=None,
        enable_rag: bool = True,
    ):
        self.retriever = retriever
        self.context_manager = context_manager
        self.model = model
        self.enable_rag = enable_rag
    
    def _ensure_components(self):
        """确保所有组件已初始化"""
        if self.retriever is None:
            from .retriever import HybridRetriever
            self.retriever = HybridRetriever()
        
        if self.context_manager is None:
            from .context_manager import ContextManager
            self.context_manager = ContextManager()
    
    def generate(
        self,
        query: str,
        top_k: int = 5,
        enable_rag: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行检索增强生成"""
        self._ensure_components()
        
        use_rag = enable_rag if enable_rag is not None else self.enable_rag
        
        context_items = []
        context_prompt = ""
        
        if use_rag:
            context_items = self.retriever.retrieve(query, top_k=top_k)
            context_items = self.context_manager.manage(context_items, query)
            context_prompt = self.context_manager.build_context_prompt(context_items, query)
        
        if self.model is not None:
            final_prompt = context_prompt if context_prompt else query
            response = self._generate_with_model(final_prompt, **kwargs)
        else:
            response = self._generate_fallback(query, context_items)
        
        return {
            'query': query,
            'response': response,
            'context_items': [item.to_dict() for item in context_items],
            'used_rag': use_rag,
            'num_context_items': len(context_items),
            'timestamp': datetime.now().isoformat(),
        }
    
    def _generate_with_model(self, prompt: str, **kwargs) -> str:
        """使用模型生成响应"""
        try:
            if hasattr(self.model, 'generate'):
                return self.model.generate(prompt, **kwargs)
            elif hasattr(self.model, '__call__'):
                result = self.model(prompt, **kwargs)
                if isinstance(result, tuple):
                    return str(result[0])
                return str(result)
            else:
                return f"模型输出: {prompt[:100]}..."
        except Exception as e:
            return f"模型生成失败: {str(e)}"
    
    def _generate_fallback(self, query: str, context_items: List) -> str:
        """降级生成响应"""
        if not context_items:
            return f"这是一个关于'{query}'的问题。要获得准确的答案，请提供一个AI模型或启用搜索功能。"
        
        summaries = []
        for i, item in enumerate(context_items):
            summaries.append(f"【{i+1}】{item.title}: {item.content}")
        
        summary_text = "\n".join(summaries)
        
        return f"""根据检索到的信息，关于"{query}"的答案如下：

{summary_text}

请注意：这是基于检索信息的摘要回答。如需更精确的回答，请提供一个AI模型。"""
    
    def stream_generate(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ):
        """流式生成"""
        self._ensure_components()
        
        context_items = self.retriever.retrieve(query, top_k=top_k)
        context_items = self.context_manager.manage(context_items, query)
        
        yield {
            'type': 'context',
            'items': [item.to_dict() for item in context_items],
            'num_items': len(context_items),
        }
        
        if self.model is not None:
            context_prompt = self.context_manager.build_context_prompt(context_items, query)
            final_prompt = context_prompt if context_prompt else query
            
            try:
                if hasattr(self.model, 'stream'):
                    for chunk in self.model.stream(final_prompt, **kwargs):
                        yield {'type': 'token', 'content': chunk}
                else:
                    response = self._generate_with_model(final_prompt, **kwargs)
                    yield {'type': 'token', 'content': response}
            except Exception as e:
                yield {'type': 'error', 'content': str(e)}
        else:
            response = self._generate_fallback(query, context_items)
            yield {'type': 'token', 'content': response}
        
        yield {'type': 'finish'}


class RAGResponse:
    """RAG响应结果"""
    
    def __init__(
        self,
        query: str,
        response: str,
        context_items: List = None,
        used_rag: bool = True,
        num_context_items: int = 0,
        timestamp: Optional[datetime] = None,
    ):
        self.query = query
        self.response = response
        self.context_items = context_items or []
        self.used_rag = used_rag
        self.num_context_items = num_context_items
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'response': self.response,
            'context_items': [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.context_items],
            'used_rag': self.used_rag,
            'num_context_items': self.num_context_items,
            'timestamp': self.timestamp.isoformat(),
        }
    
    def __repr__(self):
        return f"RAGResponse(query={self.query[:30]}..., response={self.response[:50]}..., used_rag={self.used_rag})"

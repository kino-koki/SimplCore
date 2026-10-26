# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
RAG集成器模块
实现RAG与AI模型的深度集成
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime


class RAGIntegrator:
    """RAG集成器 - 将RAG功能与AI模型深度集成"""
    
    def __init__(
        self,
        model=None,
        search_engine=None,
        knowledge_base=None,
        enable_rag: bool = True,
        auto_route: bool = True,
    ):
        self.model = model
        self.search_engine = search_engine
        self.knowledge_base = knowledge_base
        self.enable_rag = enable_rag
        self.auto_route = auto_route
        
        self.retriever = None
        self.context_manager = None
        self.router = None
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        from .retriever import HybridRetriever, SearchRetriever, KnowledgeRetriever
        from .context_manager import ContextManager
        from .router import QueryRouter
        
        self.retriever = HybridRetriever()
        self.context_manager = ContextManager()
        self.router = QueryRouter()
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        enable_rag: Optional[bool] = None,
        retriever_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行RAG增强查询"""
        use_rag = enable_rag if enable_rag is not None else self.enable_rag
        
        routing_info = None
        if self.auto_route and use_rag:
            routing_info = self.router.route(query)
            strategy = routing_info['strategy']
            top_k = strategy.get('top_k', top_k)
            use_rag = strategy.get('enable_rag', use_rag)
            retriever_type = strategy.get('retriever', retriever_type)
        
        context_items = []
        context_prompt = ""
        
        if use_rag:
            self._select_retriever(retriever_type)
            context_items = self.retriever.retrieve(query, top_k=top_k)
            context_items = self.context_manager.manage(context_items, query)
            context_prompt = self.context_manager.build_context_prompt(context_items, query)
        
        if self.model is not None:
            final_prompt = context_prompt if context_prompt else query
            response = self._generate_response(final_prompt, **kwargs)
        else:
            response = self._generate_fallback(query, context_items)
        
        return {
            'query': query,
            'response': response,
            'context_items': [item.to_dict() for item in context_items],
            'used_rag': use_rag,
            'num_context_items': len(context_items),
            'routing_info': routing_info,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _select_retriever(self, retriever_type: Optional[str]):
        """选择检索器"""
        from .retriever import SearchRetriever, KnowledgeRetriever, HybridRetriever
        
        if retriever_type == 'search':
            self.retriever = SearchRetriever(self.search_engine)
        elif retriever_type == 'knowledge':
            self.retriever = KnowledgeRetriever(self.knowledge_base)
        else:
            self.retriever = HybridRetriever()
    
    def _generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
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
        """降级生成"""
        if not context_items:
            return f"这是一个关于'{query}'的问题。要获得准确的答案，请提供一个AI模型或启用搜索功能。"
        
        summaries = []
        for i, item in enumerate(context_items):
            summaries.append(f"【{i+1}】{item.title}: {item.content}")
        
        summary_text = "\n".join(summaries)
        
        return f"""根据检索到的信息，关于"{query}"的答案如下：

{summary_text}

请注意：这是基于检索信息的摘要回答。如需更精确的回答，请提供一个AI模型。"""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """多轮对话模式"""
        if not messages:
            return {'response': '请提供对话内容'}
        
        last_message = messages[-1]['content']
        
        context_text = ""
        for msg in messages[:-1]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_text += f"{role}: {content}\n"
        
        query = f"{context_text}\n用户: {last_message}"
        
        result = self.query(query, top_k=top_k, **kwargs)
        
        return {
            'response': result['response'],
            'context_items': result['context_items'],
            'used_rag': result['used_rag'],
        }
    
    def set_model(self, model):
        """设置AI模型"""
        self.model = model
    
    def set_search_engine(self, search_engine):
        """设置搜索引擎"""
        self.search_engine = search_engine
        from .retriever import SearchRetriever
        self.retriever = SearchRetriever(search_engine)
    
    def set_knowledge_base(self, knowledge_base):
        """设置知识库"""
        self.knowledge_base = knowledge_base
        from .retriever import KnowledgeRetriever
        self.retriever = KnowledgeRetriever(knowledge_base)
    
    def enable_rag_mode(self):
        """启用RAG模式"""
        self.enable_rag = True
    
    def disable_rag_mode(self):
        """禁用RAG模式"""
        self.enable_rag = False


def create_rag_integrator(model=None, **kwargs) -> RAGIntegrator:
    """便捷创建RAG集成器"""
    return RAGIntegrator(model=model, **kwargs)

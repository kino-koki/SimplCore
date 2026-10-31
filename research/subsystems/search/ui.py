# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
搜索交互界面模块
提供用户友好的搜索交互界面
"""
from __future__ import annotations

import json
from typing import Dict, Any, Optional
from datetime import datetime


class SearchUI:
    """搜索交互界面"""
    
    def __init__(self, search_engine):
        self.search_engine = search_engine
    
    def render_search_page(self, query: str = "", results=None, error: str = "") -> str:
        """渲染搜索页面"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI搜索 - SimplCore</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-top: 40px;
        }}
        .header h1 {{
            color: white;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .header p {{
            color: rgba(255,255,255,0.8);
            font-size: 16px;
        }}
        .search-box {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            margin-bottom: 30px;
        }}
        .search-input-wrapper {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .search-input {{
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s, box-shadow 0.3s;
        }}
        .search-input:focus {{
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        .search-btn {{
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .search-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }}
        .search-options {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .option-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #666;
            font-size: 14px;
        }}
        .option-group select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            outline: none;
        }}
        .option-group select:focus {{
            border-color: #667eea;
        }}
        .results-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
        }}
        .results-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }}
        .results-count {{
            color: #666;
            font-size: 14px;
        }}
        .results-time {{
            color: #999;
            font-size: 14px;
        }}
        .result-item {{
            padding: 20px;
            border-bottom: 1px solid #f5f5f5;
            transition: background-color 0.2s;
        }}
        .result-item:last-child {{
            border-bottom: none;
        }}
        .result-item:hover {{
            background-color: #fafafa;
        }}
        .result-title {{
            color: #1a0dab;
            font-size: 18px;
            font-weight: 600;
            text-decoration: none;
            margin-bottom: 8px;
            display: block;
            transition: color 0.2s;
        }}
        .result-title:hover {{
            color: #1558d6;
            text-decoration: underline;
        }}
        .result-url {{
            color: #006621;
            font-size: 14px;
            margin-bottom: 8px;
            display: block;
        }}
        .result-desc {{
            color: #545454;
            font-size: 14px;
            line-height: 1.6;
        }}
        .result-score {{
            color: #667eea;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
            display: block;
        }}
        .error-message {{
            background: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 8px;
            padding: 15px;
            color: #856404;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .history-panel {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            margin-bottom: 30px;
        }}
        .history-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .history-header h3 {{
            color: #333;
            font-size: 18px;
        }}
        .clear-history-btn {{
            padding: 6px 12px;
            background: #f5f5f5;
            color: #666;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        .clear-history-btn:hover {{
            background: #e0e0e0;
        }}
        .history-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .history-tag {{
            padding: 8px 16px;
            background: #f0f0f0;
            color: #333;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.2s;
        }}
        .history-tag:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-1px);
        }}
        .stats-panel {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 20px;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
        @media (max-width: 600px) {{
            .search-input-wrapper {{
                flex-direction: column;
            }}
            .search-btn {{
                width: 100%;
            }}
            .search-options {{
                flex-direction: column;
            }}
            .header h1 {{
                font-size: 28px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 AI智能搜索</h1>
            <p>基于SimplCore架构的智能搜索引擎，支持多引擎对接与智能排序</p>
        </div>
        
        <div class="search-box">
            <form method="GET" action="/search">
                <div class="search-input-wrapper">
                    <input type="text" class="search-input" name="q" placeholder="输入搜索关键词..." value="{query}" autofocus>
                    <button type="submit" class="search-btn">搜索</button>
                </div>
                <div class="search-options">
                    <div class="option-group">
                        <label>引擎:</label>
                        <select name="provider">
                            <option value="duckduckgo">DuckDuckGo</option>
                            <option value="google">Google</option>
                            <option value="bing">Bing</option>
                        </select>
                    </div>
                    <div class="option-group">
                        <label>数量:</label>
                        <select name="num_results">
                            <option value="5">5</option>
                            <option value="10" selected>10</option>
                            <option value="20">20</option>
                        </select>
                    </div>
                    <div class="option-group">
                        <input type="checkbox" name="rank" checked>
                        <label>智能排序</label>
                    </div>
                    <div class="option-group">
                        <input type="checkbox" name="scrape_content">
                        <label>抓取内容</label>
                    </div>
                </div>
            </form>
        </div>
        
        <div class="history-panel">
            <div class="history-header">
                <h3>📜 搜索历史</h3>
                <button class="clear-history-btn" onclick="clearHistory()">清除历史</button>
            </div>
            <div class="history-tags" id="historyTags">
                {self._render_history_tags()}
            </div>
        </div>
        
        {f'<div class="error-message">{error}</div>' if error else ''}
        
        {self._render_results(results) if results else ''}
        
        <div class="stats-panel">
            <h3 style="margin-bottom: 15px; color: #333;">📊 搜索统计</h3>
            <div class="stats-grid">
                {self._render_stats()}
            </div>
        </div>
    </div>
    
    <script>
        function clearHistory() {{
            if (confirm('确定要清除所有搜索历史吗？')) {{
                fetch('/clear_history', {{ method: 'POST' }}).then(() => {{
                    document.getElementById('historyTags').innerHTML = '<span style="color: #999;">暂无搜索历史</span>';
                }});
            }}
        }}
        
        function searchFromTag(query) {{
            document.querySelector('input[name="q"]').value = query;
            document.querySelector('form').submit();
        }}
    </script>
</body>
</html>
"""
        return html
    
    def _render_history_tags(self) -> str:
        """渲染历史搜索标签"""
        recent = self.search_engine.get_recent_queries(limit=10)
        if not recent:
            return '<span style="color: #999;">暂无搜索历史</span>'
        
        tags = []
        for query in recent:
            tags.append(f'<span class="history-tag" onclick="searchFromTag(\'{query}\')">{query}</span>')
        return ''.join(tags)
    
    def _render_results(self, results) -> str:
        """渲染搜索结果"""
        if not results or not results.results:
            return '<div class="results-container"><div class="results-header"><span class="results-count">未找到结果</span></div></div>'
        
        result_items = []
        for i, result in enumerate(results.results, 1):
            result_items.append(f"""
<div class="result-item">
    <a href="{result.url}" target="_blank" class="result-title">{result.title}</a>
    <span class="result-url">{result.url}</span>
    <p class="result-desc">{result.description}</p>
    <span class="result-score">相关性: {result.relevance_score:.2f}</span>
</div>
""")
        
        return f"""
<div class="results-container">
    <div class="results-header">
        <span class="results-count">找到 {results.total_results} 条结果</span>
        <span class="results-time">用时 {results.execution_time:.2f} 秒</span>
    </div>
    {''.join(result_items)}
</div>
"""
    
    def _render_stats(self) -> str:
        """渲染统计信息"""
        stats = self.search_engine.get_stats()
        history_stats = stats.get('history', {})
        
        stats_html = [
            f'<div class="stat-item"><div class="stat-value">{history_stats.get("total_searches", 0)}</div><div class="stat-label">总搜索次数</div></div>',
            f'<div class="stat-item"><div class="stat-value">{history_stats.get("unique_queries", 0)}</div><div class="stat-label">唯一查询数</div></div>',
            f'<div class="stat-item"><div class="stat-value">{stats.get("remaining_quota", 0)}</div><div class="stat-label">剩余配额</div></div>',
            f'<div class="stat-item"><div class="stat-value">{stats.get("remaining_requests", 0)}</div><div class="stat-label">剩余请求</div></div>',
        ]
        
        return ''.join(stats_html)


class SearchAPIServer:
    """搜索API服务器"""
    
    def __init__(self, search_engine, host: str = "0.0.0.0", port: int = 8000):
        self.search_engine = search_engine
        self.host = host
        self.port = port
    
    def run(self) -> None:
        """启动服务器"""
        try:
            from flask import Flask, request, jsonify, render_template_string
            app = Flask(__name__)
            ui = SearchUI(self.search_engine)
            
            @app.route('/')
            def index():
                return ui.render_search_page()
            
            @app.route('/search')
            def search():
                query = request.args.get('q', '')
                provider = request.args.get('provider', 'duckduckgo')
                num_results = int(request.args.get('num_results', 10))
                rank = request.args.get('rank', 'on') == 'on'
                scrape_content = request.args.get('scrape_content', 'off') == 'on'
                
                if not query:
                    return ui.render_search_page(error="请输入搜索关键词")
                
                try:
                    results = self.search_engine.search(
                        query=query,
                        num_results=num_results,
                        rank=rank,
                        scrape_content=scrape_content,
                    )
                    return ui.render_search_page(query=query, results=results)
                except Exception as e:
                    return ui.render_search_page(query=query, error=str(e))
            
            @app.route('/api/search', methods=['GET', 'POST'])
            def api_search():
                if request.method == 'POST':
                    data = request.get_json()
                    query = data.get('query', '')
                    provider = data.get('provider', 'duckduckgo')
                    num_results = data.get('num_results', 10)
                    rank = data.get('rank', True)
                else:
                    query = request.args.get('q', '')
                    provider = request.args.get('provider', 'duckduckgo')
                    num_results = int(request.args.get('num_results', 10))
                    rank = request.args.get('rank', 'true').lower() == 'true'
                
                if not query:
                    return jsonify({'error': '请输入搜索关键词', 'code': 400}), 400
                
                try:
                    results = self.search_engine.search(
                        query=query,
                        num_results=num_results,
                        rank=rank,
                    )
                    return jsonify(results.to_dict())
                except Exception as e:
                    return jsonify({'error': str(e), 'code': 500}), 500
            
            @app.route('/api/history', methods=['GET'])
            def api_history():
                limit = int(request.args.get('limit', 50))
                history = self.search_engine.get_history(limit=limit)
                return jsonify(history)
            
            @app.route('/api/history/recent', methods=['GET'])
            def api_recent():
                limit = int(request.args.get('limit', 10))
                recent = self.search_engine.get_recent_queries(limit=limit)
                return jsonify({'queries': recent})
            
            @app.route('/api/history/top', methods=['GET'])
            def api_top():
                limit = int(request.args.get('limit', 10))
                top = self.search_engine.get_top_queries(limit=limit)
                return jsonify({'queries': top})
            
            @app.route('/clear_history', methods=['POST'])
            def clear_history():
                self.search_engine.clear_history()
                return jsonify({'success': True})
            
            @app.route('/api/stats', methods=['GET'])
            def api_stats():
                stats = self.search_engine.get_stats()
                return jsonify(stats)
            
            print(f"搜索服务器启动: http://{self.host}:{self.port}")
            app.run(host=self.host, port=self.port, debug=False)
        
        except ImportError:
            print("请先安装Flask: pip install flask")
            raise


def run_search_server(provider: str = "duckduckgo", host: str = "0.0.0.0", port: int = 8000) -> None:
    """便捷启动搜索服务器"""
    from .engine import SearchEngine
    engine = SearchEngine(provider=provider)
    server = SearchAPIServer(engine, host=host, port=port)
    server.run()


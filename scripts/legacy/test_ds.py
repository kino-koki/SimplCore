# -*- coding: utf-8 -*-
"""测试DeepSeek API。"""
import requests
import json

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer ***REMOVED***",
    "Content-Type": "application/json",
}

# 测试 deepseek-chat 模型
print("=== 测试 deepseek-chat ===")
payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "你是青小团，由青小团团队开发的中文多模态大语言模型。"},
        {"role": "user", "content": "你是谁？"},
    ],
    "max_tokens": 100,
}
try:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"回答: {data['choices'][0]['message']['content']}")
    else:
        print(f"Error: {r.text[:300]}")
except Exception as e:
    print(f"错误: {e}")

# 测试 deepseek-v4-flash 模型
print("\n=== 测试 deepseek-v4-flash ===")
payload["model"] = "deepseek-v4-flash"
try:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"回答: {data['choices'][0]['message']['content']}")
    else:
        print(f"Error: {r.text[:300]}")
except Exception as e:
    print(f"错误: {e}")

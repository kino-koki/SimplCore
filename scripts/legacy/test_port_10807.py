# -*- coding: utf-8 -*-
"""测试端口10807。"""
import requests

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"

port = 10807
proxies = {
    "http": f"http://127.0.0.1:{port}",
    "https": f"http://127.0.0.1:{port}"
}

print(f"测试端口 {port} 作为代理...")
print()

try:
    r = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15,
        proxies=proxies,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        models = data.get("data", [])
        print(f"可用模型数: {len(models)}")
        for m in models[:20]:
            mid = m.get("id", "?")
            print(f"  - {mid}")
        print()
        print("✅ 代理可用！")
        
        print("\n测试 LLaMA chat completion ...")
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta/llama-3.1-8b-instruct",
                "messages": [{"role": "user", "content": "你好，用一句话介绍你自己。"}],
                "max_tokens": 64,
                "temperature": 0.7,
            },
            timeout=30,
            proxies=proxies,
        )
        print(f"Status: {r2.status_code}")
        if r2.status_code == 200:
            content = r2.json()["choices"][0]["message"]["content"]
            print(f"回答: {content[:150]}")
        
        print("\n测试 Qwen Coder chat completion ...")
        r3 = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "qwen/qwen2.5-coder-32b-instruct",
                "messages": [{"role": "user", "content": "写一个Python的hello world。"}],
                "max_tokens": 64,
                "temperature": 0.7,
            },
            timeout=30,
            proxies=proxies,
        )
        print(f"Status: {r3.status_code}")
        if r3.status_code == 200:
            content = r3.json()["choices"][0]["message"]["content"]
            print(f"回答: {content[:150]}")
    else:
        print(f"Response: {r.text[:200]}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

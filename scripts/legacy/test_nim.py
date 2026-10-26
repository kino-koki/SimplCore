# -*- coding: utf-8 -*-
"""快速测试 NIM API 连通性 - 多代理端口扫描。"""
import requests
import sys

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"

common_ports = [
    7890, 7891, 7892,  # Clash/V2Ray常见
    1080, 1081, 1082,  # SOCKS常见
    8080, 8081, 8888,  # HTTP代理常见
    38457, 38458, 38459,  # Watt Toolkit常见
    10808, 10809,  # 其他常见
    9090, 9999, 7070,  # 更多端口
]

print("正在扫描常用代理端口...\n")

for port in common_ports:
    proxies = {
        "http": f"http://127.0.0.1:{port}",
        "https": f"http://127.0.0.1:{port}"
    }
    try:
        r = requests.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=8,
            proxies=proxies,
        )
        if r.status_code == 200:
            print(f"✅ 找到可用代理端口: {port}")
            data = r.json()
            models = data.get("data", [])
            print(f"   可用模型数: {len(models)}")
            for m in models[:15]:
                print(f"     - {m.get('id','?')}")
            
            print(f"\n测试 LLaMA chat completion ...")
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
            print(f"   Status: {r2.status_code}")
            if r2.status_code == 200:
                content = r2.json()["choices"][0]["message"]["content"]
                print(f"   回答: {content[:150]}")
            
            print(f"\n测试 Qwen Coder chat completion ...")
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
            print(f"   Status: {r3.status_code}")
            if r3.status_code == 200:
                content = r3.json()["choices"][0]["message"]["content"]
                print(f"   回答: {content[:150]}")
            
            print(f"\n🎉 代理端口 {port} 完全可用！")
            sys.exit(0)
        else:
            print(f"端口 {port}: 连接成功但状态码 {r.status_code}")
    except requests.exceptions.ProxyError:
        print(f"端口 {port}: 无代理服务 (ProxyError)")
    except requests.exceptions.ConnectTimeout:
        print(f"端口 {port}: 连接超时")
    except Exception as e:
        print(f"端口 {port}: {type(e).__name__}: {str(e)[:50]}")

print("\n❌ 未找到可用的代理端口")
print("\n请手动检查Watt Toolkit的代理设置：")
print("1. 打开Watt Toolkit")
print("2. 进入「加速设置」或「代理设置」")
print("3. 查看本地代理端口号")
print("4. 告诉我端口号，我来配置")
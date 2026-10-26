# -*- coding: utf-8 -*-
"""测试0.0.0.0和其他地址的代理。"""
import requests
import socket

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"

addresses = [
    ("127.0.0.1", 10807),
    ("0.0.0.0", 10807),
    ("localhost", 10807),
]

for addr, port in addresses:
    print(f"测试 {addr}:{port} ...")
    proxies = {
        "http": f"http://{addr}:{port}",
        "https": f"http://{addr}:{port}"
    }
    try:
        r = requests.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
            proxies=proxies,
        )
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            print(f"  ✅ 成功！可用模型数: {len(models)}")
            for m in models[:10]:
                mid = m.get("id", "?")
                print(f"    - {mid}")
            break
        else:
            print(f"  Response: {r.text[:100]}")
    except Exception as e:
        print(f"  错误: {type(e).__name__}: {str(e)[:80]}")
    print()

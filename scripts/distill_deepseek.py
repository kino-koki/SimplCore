# -*- coding: utf-8 -*-
"""DeepSeek V4 Flash 蒸馏脚本。

从指令池读取 instruction，调用 DeepSeek V4 Flash 生成回答，
保存为 SFT 训练数据。支持：
  - 并发 API 调用（加速蒸馏）
  - 断点续传（跳过已完成的指令）
  - 实时成本追踪（防止超预算）
  - 增量写入（崩溃不丢数据）

用法:
    python scripts/distill_deepseek.py
    python scripts/distill_deepseek.py --workers 15 --budget 17
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

# =========================================================================== #
# 配置
# =========================================================================== #

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "***REMOVED***")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"

# 定价 (RMB per 1M tokens)
PRICE_INPUT = 1.0    # 缓存未命中
PRICE_OUTPUT = 2.0   # 输出

# 系统提示（固定，可利用缓存命中降低成本）
SYSTEM_PROMPT = (
    "你是一个知识渊博、表达清晰的AI助手。"
    "请针对用户的问题，给出准确、详细、有条理的回答。"
    "回答要有实质内容，避免空话套话，适当使用分点或分段来组织内容。"
    "如果涉及代码，请给出完整可运行的代码示例。"
    "如果涉及数学，请给出清晰的推导过程。"
)

# =========================================================================== #
# API 调用
# =========================================================================== #

def call_deepseek(instruction: str, max_tokens: int = 1024,
                  temperature: float = 0.7, timeout: int = 60) -> Optional[Dict]:
    """调用 DeepSeek V4 Flash，返回 {content, prompt_tokens, completion_tokens}。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(30, 2 ** attempt * 2)
                time.sleep(wait)
                continue
            # 其他错误
            return None
        except Exception:
            time.sleep(min(30, 2 ** attempt * 2))
    return None


# =========================================================================== #
# 蒸馏主流程
# =========================================================================== #

class DistillProgress:
    """线程安全的进度追踪器。"""
    def __init__(self):
        self.lock = threading.Lock()
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_rmb = 0.0
        self.start_time = time.time()

    def add_success(self, in_tok, out_tok):
        with self.lock:
            self.success += 1
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            cost = in_tok / 1e6 * PRICE_INPUT + out_tok / 1e6 * PRICE_OUTPUT
            self.total_cost_rmb += cost

    def add_fail(self):
        with self.lock:
            self.failed += 1

    def add_skip(self):
        with self.lock:
            self.skipped += 1

    def summary(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "success": self.success,
                "failed": self.failed,
                "skipped": self.skipped,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "cost_rmb": round(self.total_cost_rmb, 4),
                "elapsed_sec": round(elapsed, 1),
                "speed": round(self.success / max(1, elapsed), 2),
            }


def load_completed(output_path: str) -> set:
    """加载已完成的指令（用于断点续传）。"""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = hashlib.md5(obj["instruction"].encode()).hexdigest()
                    completed.add(key)
                except:
                    pass
    return completed


def distill_worker(instr_item: Dict, max_tokens: int, temperature: float) -> Optional[Dict]:
    """单条蒸馏任务。"""
    result = call_deepseek(
        instr_item["instruction"],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if result is None or len(result["content"]) < 10:
        return None

    return {
        "instruction": instr_item["instruction"],
        "response": result["content"],
        "task_type": instr_item.get("task_type", "general"),
        "teacher_model": MODEL,
        "response_length": len(result["content"]),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
    }


def main():
    ap = argparse.ArgumentParser(description="DeepSeek V4 Flash 蒸馏")
    ap.add_argument("--input", default="data/distill/instructions/distill_instructions_v2.jsonl")
    ap.add_argument("--output", default="data/distill/sft/deepseek_v4_distill.jsonl")
    ap.add_argument("--workers", type=int, default=10, help="并发数")
    ap.add_argument("--budget", type=float, default=17.0, help="预算上限(RMB)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="每条最大输出token")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    # 加载指令
    instructions = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                instructions.append(json.loads(line))
            except:
                pass
    print(f"[蒸馏] 加载 {len(instructions)} 条指令")

    # 断点续传
    completed = load_completed(args.output)
    print(f"[蒸馏] 已完成 {len(completed)} 条（断点续传）")

    # 过滤未完成
    todo = []
    for instr in instructions:
        key = hashlib.md5(instr["instruction"].encode()).hexdigest()
        if key not in completed:
            todo.append(instr)
    print(f"[蒸馏] 待蒸馏 {len(todo)} 条")

    if not todo:
        print("[蒸馏] 全部已完成，无需继续")
        return

    # 输出文件（追加模式）
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")
    progress = DistillProgress()

    print(f"[蒸馏] 并发数: {args.workers} | 预算: ¥{args.budget} | 模型: {MODEL}")
    print(f"[蒸馏] 开始蒸馏 ...")
    print()

    budget_exhausted = False
    log_counter = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        batch_size = args.workers * 2

        for i in range(0, len(todo), batch_size):
            if budget_exhausted:
                break
            batch = todo[i:i + batch_size]

            for instr in batch:
                future = pool.submit(
                    distill_worker, instr,
                    args.max_tokens, args.temperature,
                )
                futures[future] = instr

            for future in as_completed(futures):
                instr = futures.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = None

                if result:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    progress.add_success(result["prompt_tokens"],
                                        result["completion_tokens"])
                else:
                    progress.add_fail()

                log_counter += 1

                # 预算检查
                if progress.total_cost_rmb >= args.budget:
                    budget_exhausted = True
                    print(f"\n[蒸馏] 预算耗尽 (¥{progress.total_cost_rmb:.2f} >= ¥{args.budget})")
                    break

                # 日志
                if log_counter % args.log_every == 0:
                    s = progress.summary()
                    pct = s["cost_rmb"] / args.budget * 100
                    print(f"  [{s['success']+s['failed']+s['skipped']:>5}/{len(todo)}] "
                          f"成功{s['success']} 失败{s['failed']} | "
                          f"¥{s['cost_rmb']:.2f}/{args.budget} ({pct:.0f}%) | "
                          f"{s['total_tokens']/1e6:.2f}M tokens | "
                          f"{s['speed']:.1f} 条/s", flush=True)

            # 提交下一批前检查预算
            if budget_exhausted:
                break

    out_f.close()

    # 最终统计
    s = progress.summary()
    print()
    print("=" * 60)
    print(f"  蒸馏完成")
    print(f"  成功: {s['success']} 条")
    print(f"  失败: {s['failed']} 条")
    print(f"  跳过: {s['skipped']} 条")
    print(f"  输入token: {s['input_tokens']/1e6:.2f}M")
    print(f"  输出token: {s['output_tokens']/1e6:.2f}M")
    print(f"  总token: {s['total_tokens']/1e6:.2f}M")
    print(f"  花费: ¥{s['cost_rmb']:.2f}")
    print(f"  耗时: {s['elapsed_sec']/60:.1f} 分钟")
    print(f"  速度: {s['speed']:.1f} 条/s")
    print(f"  输出: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
D. Agnes AI 2.0 Flash 蒸馏流水线

  - 输入: distillation_data/_instructions_master.jsonl (步骤 A 生成)
    每条样本: {"instruction": "...", "task_type": "...", "model": "agnes-2.0-flash",
               "input": "", "output": "__TO_BE_DISTILLED__"}
  - 动作: 调用 `https://apihub.agnes-ai.com/v1/chat/completions` 生成 output
  - 输出: distillation_data/all_samples.jsonl (可直接传给 training/sft/train_sft.py)

用法:
    # 1) 先执行步骤 A：
    python scripts/merge_2100_instructions.py

    # 2) 设置 API Key (环境变量或直接编辑 --api_key)
    export AGNES_API_KEY=sk-xxxx

    # 3) 执行蒸馏
    python scripts/distill_with_agnes.py \
        --input distillation_data/_instructions_master.jsonl \
        --output distillation_data/all_samples.jsonl \
        --model agnes-2.0-flash \
        --temperature 0.7 \
        --max_new_tokens 2048 \
        --resume true

也可以替换为 DeepSeek-V4-Pro / Claude-Opus-4.8 等其他教师模型，
只需要改 --model 和对应的 --api_url / --api_key。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover
    print("[依赖缺失] 需要 `requests` 库。请运行: pip install requests")
    sys.exit(1)


DEFAULT_API_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
PLACEHOLDER = "__TO_BE_DISTILLED__"


# --------------------------------------------------------------------------- #
# HTTP 调用
# --------------------------------------------------------------------------- #
def call_teacher_model(
    prompt: str,
    api_key: str,
    api_url: str,
    model: str,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    timeout: int = 120,
) -> Optional[str]:
    """调用外部 LLM API，返回纯文本的 assistant 回复；失败返回 None。"""
    if not api_key:
        raise RuntimeError("需要提供 API Key (--api_key 或环境变量 AGNES_API_KEY)")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "你是一个严谨、知识渊博、中文表达自然的 AI 助手。"
                "针对用户问题给出结构化、有事实依据、格式规范的回答。"
                "不要输出多余的解释性文字。"
            )},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }

    for attempt in range(3):
        try:
            r = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices", [])
                if not choices:
                    return None
                msg = choices[0].get("message", {})
                return (msg.get("content") or "").strip()

            # 限流 / 服务器错误：指数退避
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(60, 2 ** attempt * 3)
                print(f"  ! HTTP {r.status_code}，{sleep_s}s 后重试 ({attempt + 1}/3)")
                time.sleep(sleep_s)
                continue
            # 其他错误（auth 失败等）: 直接失败
            print(f"  ! HTTP {r.status_code}: {r.text[:200]}")
            return None

        except Exception as e:
            sleep_s = min(60, 2 ** attempt * 3)
            print(f"  ! 调用异常: {e}，{sleep_s}s 后重试 ({attempt + 1}/3)")
            time.sleep(sleep_s)

    return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Agnes AI 2.0 Flash 蒸馏流水线")
    parser.add_argument(
        "--input", default=os.path.join("distillation_data", "_instructions_master.jsonl"),
        help="输入 jsonl 路径 (步骤 A 的输出)",
    )
    parser.add_argument(
        "--output", default=os.path.join("distillation_data", "all_samples.jsonl"),
        help="输出 jsonl 路径 (可直接用于 SFT 训练)",
    )
    parser.add_argument("--model", default="agnes-2.0-flash")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--api_key", default=os.environ.get("AGNES_API_KEY", ""),
        help="默认读取 AGNES_API_KEY 环境变量",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--resume", default="true",
        help="true=跳过 output 字段已填充的样本；false=重新蒸馏全部",
    )
    parser.add_argument("--sleep_ms", type=int, default=300, help="每条后睡眠 (ms)")
    args = parser.parse_args()

    resume_flag = str(args.resume).lower() in ("1", "true", "yes", "t")

    print("=" * 62)
    print("  D. Agnes AI 2.0 Flash 蒸馏流水线")
    print("=" * 62)
    print(f"  input      = {args.input}")
    print(f"  output     = {args.output}")
    print(f"  model      = {args.model}")
    print(f"  api_url    = {args.api_url}")
    print(f"  resume     = {resume_flag}")
    print(f"  temp       = {args.temperature}")
    print(f"  max_tokens = {args.max_new_tokens}")

    if not os.path.exists(args.input):
        print(f"  [!] 输入文件不存在: {args.input} （请先运行步骤 A）")
        return 1

    # 读取所有样本
    with open(args.input, "r", encoding="utf-8") as f:
        samples = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception as e:
                print(f"  [skip] 解析失败: {e}")
    print(f"  总样本数: {len(samples)}")

    # 已有的 output (resume 模式下跳过)
    existing_outputs = {}
    if resume_flag and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = (
                        str(obj.get("instruction", ""))
                        + "||"
                        + str(obj.get("task_type", ""))
                    )
                    output = obj.get("output", "")
                    if output and output != PLACEHOLDER:
                        existing_outputs[key] = output
            print(f"  已从 {args.output} 加载 {len(existing_outputs)} 条已蒸馏结果")
        except Exception as e:
            print(f"  [warn] 读取已有输出失败: {e}")
            existing_outputs = {}

    # 写 output 文件（追加 / 覆盖由 resume 决定）
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if (resume_flag and os.path.exists(args.output) and existing_outputs) else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        success = 0
        failed = 0
        skipped = 0

        for i, sample in enumerate(samples, 1):
            instruction = str(sample.get("instruction", "")).strip()
            task_type = str(sample.get("task_type", "general")).strip() or "general"
            key = f"{instruction}||{task_type}"

            # resume / pre-fill
            already = existing_outputs.get(key)
            if resume_flag and already:
                sample["output"] = already
                sample["model"] = args.model
                sample["input"] = sample.get("input", "") or ""
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_f.flush()
                skipped += 1
                if skipped % 20 == 0:
                    print(f"  [resume] 已跳过 {skipped} 条已蒸馏样本")
                continue

            if not instruction:
                skipped += 1
                continue

            # 调用 API
            display = instruction[:50].replace("\n", " ")
            print(f"  [{i}/{len(samples)}] [{task_type}] {display} ... ", end="", flush=True)
            output = call_teacher_model(
                prompt=instruction,
                api_key=args.api_key,
                api_url=args.api_url,
                model=args.model,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            if not output or len(output) < 4:
                failed += 1
                print("FAIL")
                continue

            sample["output"] = output
            sample["model"] = args.model
            sample["input"] = sample.get("input", "") or ""
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out_f.flush()
            success += 1
            print(f"OK ({len(output)} chars)")

            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

        print("")
        print(f"  成功: {success}, 失败: {failed}, 跳过: {skipped}")
        print(f"  输出文件: {args.output}")
        print("")

    # 总结
    total = success + failed + skipped
    print("=" * 62)
    print(f"  完成。成功={success}, 失败={failed}, 跳过={skipped}, 总计={total}")
    print("  下一步: 运行 SFT 训练:")
    print(f"      python training/sft/train_sft.py --model_name_or_path checkpoints2/checkpoint_best.pt --data_path {args.output} --output_dir output_sft --max_length 1048576 --attention_mode standard --enable_progressive_length_warmup")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())

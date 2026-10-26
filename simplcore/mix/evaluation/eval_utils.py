# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""评测工具 — checkpoint 权重加载解析 + few-shot prompt 组装 + 答案提取。

核心：``resolve_state_dict`` 从 .pt checkpoint 解析权重，自动处理：
  - DDP 包装产生的 ``module.`` 前缀
  - state_dict 内嵌（ckpt["model_state_dict"]）
  - 混合精度 fp16 → fp32 转回
  让任何 ckpt 都能可靠加载到 SimplCoreMix 模型做评测。

评测格式对齐行业标准（C-Eval / CMMLU 官方样例）：
  示例 (few-shot) → 问题 → 选项 A/B/C/D → "答案："
  提取：找紧跟在 "答案：" 后的字母，规范化为 ABCD。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import simplcore.tensor as torch


def resolve_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    """从 checkpoint 字典解析出干净的 model state_dict（fp32）。

    兼容:
      - ckpt["model_state_dict"]（训练脚本标准）
      - ckpt["state_dict"]（HF / 通用）
      - 直接是 state_dict（裸权重）
    自动:
      - 去 DDP 前缀 "module."
      - fp16/bf16 → fp32
      - 返回引用（不复制），调用方 load_state_dict(strict=False)
    """
    sd: Dict
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        raise ValueError("checkpoint 里找不到可解析的 state_dict")

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.float16, torch.bfloat16):
                v = v.float()
            out[name] = v
        # 非张量（step/loss 等元数据）跳过
    return out


def normalize_answer(letter: str) -> str:
    """规范化答案字母为 A/B/C/D；非法返回空串。"""
    l = letter.strip().upper()
    if l in ("A", "B", "C", "D"):
        return l
    return ""


def extract_answer(output: str) -> str:
    """从模型输出提取答案字母。

    匹配 "答案：X" / "答案是X" / "X." 前缀 等；输出被截断时也可能只含字母。
    """
    if not output:
        return ""
    out = output.strip()
    # 1) "答案：X" 或 "答案:X"
    m = re.search(r"答案[：:]\s*([A-Da-d])", out)
    if m:
        return normalize_answer(m.group(1))
    # 2) "答案是X"
    m = re.search(r"答案是\s*([A-Da-d])", out)
    if m:
        return normalize_answer(m.group(1))
    # 3) 开头就是字母（clean，如模型只输出 "B"）
    m = re.match(r"^[A-Da-d]\b", out)
    if m:
        return normalize_answer(m.group(0))
    return ""


def build_fewshot_prompt(
    examples: List[Dict],
    question: str,
    choices: List[str],
) -> str:
    """组装 few-shot prompt（对齐 C-Eval/CMMLU 样例格式）。

    examples: [{"question", "choices": [...], "answer"}]
    question / choices: 当前待回答的问题。
    返回文本如:
      "以下是中国关于{subject}的单项选择题，请选出正确答案。\n\n
       问题：...\n选项：A. ...\nB. ...\n答案：B\n\n
       问题：<当前>\n选项：A. ...\n答案："
    """
    lines = ["以下是中国关于人工智能的单项选择题，请根据你的知识选出正确答案。", ""]
    for ex in examples:
        lines.append(f"问题：{ex['question']}")
        for i, c in enumerate(ex["choices"]):
            lines.append(f"{chr(65 + i)}. {c}")
        lines.append(f"答案：{normalize_answer(ex['answer'])}")
        lines.append("")
    lines.append(f"问题：{question}")
    for i, c in enumerate(choices):
        lines.append(f"{chr(65 + i)}. {c}")
    lines.append("答案：")
    return "\n".join(lines)


def run_generation_eval(
    model,
    tokenizer_encode,
    tokenizer_decode,
    *,
    device,
    prompts: List[str],
    max_new_tokens: int = 8,
    temperature: float = 0.0,
) -> List[Dict]:
    """批量生成式评估：对每个 prompt 生成、提取答案。

    返回 [{prompt, output, answer, time_seconds, tokens_per_second}]
    """
    import time

    results = []
    for prompt in prompts:
        ids = tokenizer_encode(prompt)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                n_loops=1,
                temperature=temperature,
                top_k=None if temperature == 0.0 else 50,
                top_p=None if temperature == 0.0 else 0.9,
            )
        elapsed = time.time() - t0
        out_ids = generated[0, len(ids):].tolist()
        out_text = tokenizer_decode(out_ids)
        results.append({
            "prompt": prompt,
            "output": out_text,
            "answer": extract_answer(out_text),
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(len(out_ids) / max(elapsed, 1e-6), 1),
        })
    return results

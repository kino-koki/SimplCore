# -*- coding: utf-8 -*-
"""合并所有蒸馏数据。"""
import json
import os
import random

input_files = [
    "data/distill/sft/qxt_identity_distill.jsonl",
    "data/distill/sft/distill_dataset_final.jsonl",
    "data/distill/sft/distill_dataset.jsonl",
    "data/distill/sft/deepseek_v4_distill.jsonl",
    "data/distill/sft/deepseek_distill_data.jsonl",
    "data/distill/sft/deepseek_v4_batch2.jsonl",
]

output_path = "data/distill/sft/qxt_9b_distill_dataset.jsonl"

all_data = []
seen_instructions = set()

for path in input_files:
    if not os.path.exists(path):
        print(f"跳过: {path} (不存在)")
        continue
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                instr = item.get("instruction", "")
                resp = item.get("response", "")
                if not instr or not resp:
                    continue
                if len(resp) < 20 or len(resp) > 6000:
                    continue
                if instr in seen_instructions:
                    continue
                seen_instructions.add(instr)
                all_data.append(item)
                count += 1
            except:
                pass
    print(f"加载 {path}: {count} 条 (去重后)")

# 打乱
rng = random.Random(42)
rng.shuffle(all_data)

# 保存
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n合并完成: {len(all_data)} 条")
print(f"保存到: {output_path}")

# 统计身份认知数据
identity_count = sum(1 for item in all_data if item.get("category") == "identity")
print(f"身份认知数据: {identity_count} 条")

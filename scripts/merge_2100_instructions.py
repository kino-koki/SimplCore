# -*- coding: utf-8 -*-
"""
A. 合并 19 个 data_*.json → distillation_data/_instructions_master.jsonl

  - 输入格式: {id, type, text}  (不同文件可能字段略有不同
  - 输出统一格式: {"instruction": "...", "task_type": "...", "model": "agnes-2.0-flash", "input": "", "output": "__TO_BE_DISTILLED__"}

  注意：output 字段先用占位符，后续 D 阶段脚本会调用教师模型将其替换为真正的蒸馏文本。
"""
import json
import os
import collections
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST_DIR = os.path.join(ROOT, "distillation_data")
MASTER_FILE = os.path.join(DST_DIR, "_instructions_master.jsonl")

FILES = [
    "data_general.json", "data_code.json", "data_math.json",
    "data_creative.json", "data_logic.json", "data_business.json",
    "data_health.json", "data_philosophy.json", "data_history.json",
    "data_science.json", "data_literature.json",
    "data_art_edu_law_social.json", "data_language_media_general.json",
    "data_economy_politics_culture.json", "data_code_math_general.json",
    "data_mixed_advanced.json", "data_advanced_continued.json",
    "data_final_batch1.json", "data_final_batch2.json",
]

PLACEHOLDER = "__TO_BE_DISTILLED__"
DEFAULT_TEACHER = "agnes-2.0-flash"


def main() -> int:
    os.makedirs(DST_DIR, exist_ok=True)
    print("=" * 62)
    print("  A. 合并 2100 条指令模板")
    print("=" * 62)

    counter = 0
    type_distribution = collections.Counter()

    with open(MASTER_FILE, "w", encoding="utf-8") as out:
        for fname in FILES:
            path = os.path.join(ROOT, fname)
            if not os.path.exists(path):
                print(f"  [SKIP] {fname} (文件不存在)")
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [FAIL] {fname}: {e}")
                continue

            if isinstance(data, dict):
                data = [data]

            written = 0
            for r in data:
                if not isinstance(r, dict):
                    continue
                text = (r.get("text") or r.get("instruction") or "").strip()
                task_type = (r.get("type") or r.get("task_type") or "general").strip() or "general"
                if not text:
                    continue

                obj = {
                    "instruction": text,
                    "task_type": task_type,
                    "model": DEFAULT_TEACHER,
                    "input": "",
                    "output": PLACEHOLDER,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                type_distribution[task_type] += 1
                written += 1

            print(f"  [{ 'OK' if written else '--' }] {fname}: 写入 {written} 条")
            counter += written

    print("")
    print(f"  总计: {counter} 条指令写入 {MASTER_FILE}")
    print("")
    print("  task_type 分布:")
    for k, v in type_distribution.most_common():
        print(f"    - {k}: {v}")
    print("")
    print("✅ A 阶段完成：所有指令已写入 _instructions_master.jsonl")
    print("   output 字段目前为占位符 '__TO_BE_DISTILLED__'，")
    print("   下一步执行 D 阶段蒸馏脚本以填充教师输出")

    return 0


if __name__ == "__main__":
    sys.exit(main())

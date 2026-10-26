import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCLUDED_EXTENSIONS = {'.py', '.md', '.json', '.txt', '.sh', '.yaml', '.yml', '.html', '.css', '.js'}
SKIP_DIRS = {'.git', '__pycache__'}

REPLACEMENTS = [
    ("Qingxiaotuan", "Qingxiaotuan"),
    ("Qingxiaotuan", "Qingxiaotuan"),
    ("青小团", "青小团"),
    ("Qingxiaotuan", "Qingxiaotuan"),
    ("青奈科技", "青奈科技"),
    ("Qingnai", "Qingnai"),
    ("青奈云", "青奈云"),
    ("Qingnai Cloud", "Qingnai Cloud"),
    ("Qingxiaotuan", "Qingxiaotuan"),
    ("qingxiaotuan", "qingxiaotuan"),
    ("qingxiaotuan", "qingxiaotuan"),
    ("qingxiaotuan", "qingxiaotuan"),
]


def replace_license(text):
    license_patterns = [
        (re.compile(r'(?i)(Apache License[^\n]*Version 2\.0[^©]*)', re.DOTALL),
         "青奈科技 ALL RIGHTS RESERVED"),
        (re.compile(r'(?i)(MIT License[^©]*Copyright\s*\([cC]\)\s*\d{4}[^.]*\.)'),
         "MIT License\n\nCopyright (c) 2024 青奈科技 ALL RIGHTS RESERVED."),
        (re.compile(r'(?i)(BSD\s*\d-Clause[^©]*)', re.DOTALL),
         "青奈科技 ALL RIGHTS RESERVED"),
    ]
    for pattern, replacement in license_patterns:
        text = pattern.sub(replacement, text)
    return text


def apply_replacements(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = replace_license(text)
    return text


def should_skip_dir(dir_name):
    return dir_name in SKIP_DIRS


def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        try:
            with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()
        except Exception:
            return False

    new_content = apply_replacements(content)
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(new_content)
        return True
    return False


def main():
    modified_files = []
    total_files = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in INCLUDED_EXTENSIONS:
                continue

            filepath = os.path.join(root, file)
            total_files += 1
            if process_file(filepath):
                relative_path = os.path.relpath(filepath, PROJECT_ROOT)
                modified_files.append(relative_path)
                print(f"[MODIFIED] {relative_path}")

    print(f"\n{'='*60}")
    print(f"扫描文件总数: {total_files}")
    print(f"修改文件数量: {len(modified_files)}")
    print(f"{'='*60}")

    if modified_files:
        print("\n修改的文件列表:")
        for f in modified_files:
            print(f"  - {f}")

    return modified_files


if __name__ == "__main__":
    main()

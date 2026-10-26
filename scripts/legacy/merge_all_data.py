# -*- coding: utf-8 -*-
"""青小团蒸馏训练 v2 - 更大模型 + 更多数据 + 更多步数。"""
import sys
import os
import json
import math
import time
import csv
import random
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from qxt.model.config import QXTConfig
from qxt.model.modeling import QXTForCausalLM
from qxt.distill.tokenizer import DistillTokenizer, build_corpus
from qxt.distill.prepare_data import (
    prepare_dataset, split_dataset, collate_batch, dataset_stats,
    DEFAULT_PROMPT_TEMPLATE, IDENTITY_PREFIX,
)

# ========== 配置 ==========
DATA_FILES = [
    "data/distill/sft/manual_distill.jsonl",
    "data/distill/sft/qxt_identity_distill.jsonl",
    "data/distill/sft/distill_dataset_final.jsonl",
    "data/distill/sft/distill_dataset.jsonl",
]
OUTPUT_DIR = "checkpoints/qxt_distill_v2"
VOCAB_SIZE = 6000
MAX_LEN = 512
BATCH_SIZE = 4
LR = 3e-4
EPOCHS = 3
MAX_STEPS = 200
LOG_EVERY = 20
LIMIT = 1000
SEED = 42

# ========== 日志 ==========
LOG_FILE = os.path.join(PROJECT_ROOT, "train_v2.log")
_log_f = open(LOG_FILE, "w", encoding="utf-8")

def log(msg):
    print(msg)
    _log_f.write(str(msg) + "\n")
    _log_f.flush()

log("=" * 60)
log("  青小团蒸馏训练 v2 (DeepSeek V4 Flash 教师模型)")
log("=" * 60)

torch.manual_seed(SEED)
device = "cpu"

# ========== 数据文件 ==========
data_files = [os.path.join(PROJECT_ROOT, d) for d in DATA_FILES]
data_files = [d for d in data_files if os.path.exists(d)]
log(f"数据文件: {[os.path.basename(d) for d in data_files]}")

# ========== 分词器 ==========
tok_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)
tok_path = os.path.join(tok_dir, "qxt_distill")

old_tok_path = os.path.join(PROJECT_ROOT, "checkpoints/distill_student/tokenizer/qxt_distill")
if os.path.exists(old_tok_path + ".meta.json"):
    log("加载已有分词器...")
    tokenizer = DistillTokenizer.load(old_tok_path)
    for ext in [".meta.json", ".json"]:
        src = old_tok_path + ext
        dst = tok_path + ext
        if os.path.exists(src):
            shutil.copy2(src, dst)
elif os.path.exists(tok_path + ".meta.json"):
    log("加载已有分词器...")
    tokenizer = DistillTokenizer.load(tok_path)
else:
    log("训练新分词器...")
    corpus = os.path.join(tok_dir, "corpus.txt")
    build_corpus(data_files, corpus)
    tokenizer = DistillTokenizer.train(corpus, VOCAB_SIZE, tok_path)

real_vocab = tokenizer.vocab_size
log(f"分词器词表大小: {real_vocab}")

# ========== 数据 ==========
examples = prepare_dataset(data_files, tokenizer, MAX_LEN, limit=LIMIT)
if len(examples) == 0:
    log("错误：未构造出任何样本！")
    sys.exit(1)

train_ex, eval_ex = split_dataset(examples, eval_ratio=0.05, seed=SEED)
stats = dataset_stats(examples)
log(f"样本统计: total={stats['total']}, avg_len={stats['avg_len']:.0f}")
log(f"训练 {len(train_ex)} 条 / 验证 {len(eval_ex)} 条")

# ========== 模型 (small配置 ~27M) ==========
PRESETS = {
    "nano": dict(vocab_size=8000, hidden_size=256, intermediate_size=704,
                 num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                 max_position_embeddings=512),
    "small": dict(vocab_size=8000, hidden_size=512, intermediate_size=1408,
                  num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
                  max_position_embeddings=1024),
    "medium": dict(vocab_size=8000, hidden_size=768, intermediate_size=2112,
                   num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4,
                   max_position_embeddings=1024),
}

PRESET = "small"
cfg_kwargs = dict(PRESETS[PRESET])
cfg_kwargs["vocab_size"] = real_vocab
config = QXTConfig.get_distill_student_config(**cfg_kwargs)
model = QXTForCausalLM(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
log(f"学生模型: {PRESET} 配置, 参数量: {n_params/1e6:.2f}M")

# ========== 训练 ==========
def cosine_lr(step, total_steps, base_lr, warmup=0):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)

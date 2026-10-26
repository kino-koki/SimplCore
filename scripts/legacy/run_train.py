# -*- coding: utf-8 -*-
"""青小团蒸馏训练 - 直接执行版。"""
import sys
import os
import json
import math
import time
import csv
import random

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from qxt.model.config import QXTConfig
from qxt.model.modeling import QXTForCausalLM
from qxt.distill.tokenizer import DistillTokenizer, build_corpus, BOS_ID, EOS_ID, PAD_ID
from qxt.distill.prepare_data import (
    prepare_dataset, split_dataset, collate_batch, dataset_stats,
    DEFAULT_PROMPT_TEMPLATE, IDENTITY_PREFIX, clean_response,
)

# ========== 配置 ==========
DATA_FILES = [
    "data/distill/sft/qxt_identity_v2.jsonl",
    "data/distill/sft/manual_distill.jsonl",
    "data/distill/sft/distill_dataset_final.jsonl",
]
OUTPUT_DIR = "checkpoints/qxt_distill_v2"
PRESET = "nano"  # CPU可用
VOCAB_SIZE = 4000
MAX_LEN = 512
BATCH_SIZE = 4
LR = 3e-4
EPOCHS = 2
MAX_STEPS = 40
LOG_EVERY = 5
LIMIT = 500
SEED = 42

# ========== 开始 ==========
LOG_FILE = os.path.join(PROJECT_ROOT, "train_v2.log")
_log_f = open(LOG_FILE, "w", encoding="utf-8")

def log(msg):
    print(msg)
    _log_f.write(str(msg) + "\n")
    _log_f.flush()

log("=" * 60)
log("  青小团蒸馏训练 (DeepSeek V4 Flash 教师模型)")
log("=" * 60)

torch.manual_seed(SEED)
device = "cpu"
log(f"设备: {device} | 预设: {PRESET}")

# 数据文件
data_files = [os.path.join(PROJECT_ROOT, d) for d in DATA_FILES]
data_files = [d for d in data_files if os.path.exists(d)]
log(f"数据文件: {data_files}")

# 1. 分词器
tok_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)
tok_path = os.path.join(tok_dir, "qxt_distill")

# 检查是否有已有分词器
old_tok_path = os.path.join(PROJECT_ROOT, "checkpoints/distill_student/tokenizer/qxt_distill")
if os.path.exists(old_tok_path + ".meta.json"):
    log("加载已有分词器...")
    tokenizer = DistillTokenizer.load(old_tok_path)
    # 复制到新目录
    import shutil
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

# 2. 数据
examples = prepare_dataset(data_files, tokenizer, MAX_LEN, limit=LIMIT)
if len(examples) == 0:
    log("错误：未构造出任何样本！")
    sys.exit(1)

train_ex, eval_ex = split_dataset(examples, eval_ratio=0.05, seed=SEED)
log(f"样本统计: {dataset_stats(examples)}")
log(f"训练 {len(train_ex)} 条 / 验证 {len(eval_ex)} 条")

# 3. 模型
PRESETS = {
    "nano": dict(vocab_size=8000, hidden_size=256, intermediate_size=704,
                 num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                 max_position_embeddings=512),
    "small": dict(vocab_size=8000, hidden_size=512, intermediate_size=1408,
                  num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2,
                  max_position_embeddings=1024),
}

cfg_kwargs = dict(PRESETS[PRESET])
cfg_kwargs["vocab_size"] = real_vocab
config = QXTConfig.get_distill_student_config(**cfg_kwargs)
model = QXTForCausalLM(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
log(f"学生模型参数量: {n_params/1e6:.2f}M")

# 4. 训练
def cosine_lr(step, total_steps, base_lr, warmup=0):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

def length_sorted_indices(examples, batch_size, seed=42, shuffle=True):
    rng = random.Random(seed)
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches

total_steps = MAX_STEPS
warmup = max(1, int(total_steps * 0.05))
log(f"总步数: {total_steps} | warmup: {warmup} | batch: {BATCH_SIZE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

csv_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, "training_log.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
csv_f = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_f)
writer.writerow(["step", "epoch", "loss", "lr", "elapsed_sec", "tokens_per_sec"])

model.train()
global_step = 0
t0 = time.time()
running_loss = 0.0
running_tokens = 0

for epoch in range(EPOCHS):
    if global_step >= total_steps:
        break
    batches = length_sorted_indices(train_ex, BATCH_SIZE, seed=SEED + epoch)
    for bi, idxs in enumerate(batches):
        if global_step >= total_steps:
            break
        batch = [train_ex[i] for i in idxs]
        input_ids, labels, attn = collate_batch(batch)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attn = attn.to(device)

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]

        loss.backward()

        lr = cosine_lr(global_step, total_steps, LR, warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_tok = int((labels != -100).sum().item())
        running_loss += loss.item()
        running_tokens += n_tok
        global_step += 1

        if global_step % LOG_EVERY == 0 or global_step == 1:
            el = time.time() - t0
            tps = running_tokens / max(1e-6, el)
            avg = running_loss / LOG_EVERY
            writer.writerow([global_step, round(epoch + bi/max(1,len(batches)),3),
                             round(avg, 4), round(lr, 6), round(el, 2), round(tps, 1)])
            csv_f.flush()
            log(f"  step {global_step:>5}/{total_steps} | loss {avg:.4f} | lr {lr:.2e} | {tps:.0f} tok/s")
            running_loss, running_tokens = 0.0, 0
            # 中途保存
            if global_step % 20 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, "student_model.pt"))
                log(f"  [checkpoint saved at step {global_step}]")

csv_f.close()
log(f"\n训练完成，总耗时 {time.time()-t0:.1f}s")

# 5. 验证
model.eval()
if eval_ex:
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(eval_ex), 8):
            batch = eval_ex[i:i+8]
            input_ids, labels, attn = collate_batch(batch)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]
            n_tok = (labels != -100).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
    ev_loss = total_loss / max(1, total_tok)
    log(f"验证集损失: {ev_loss:.4f}")

# 6. 保存
save_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "student_model.pt"))
with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"step": global_step, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "preset": PRESET, "teacher": "DeepSeek V4 Flash",
               "identity": "青小团"}, f, ensure_ascii=False, indent=2)
log(f"Checkpoint 已保存至: {save_dir}")

# 7. 生成示例（验证身份认知）
log("\n" + "=" * 60)
log("  身份认知验证")
log("=" * 60)

test_questions = [
    "你是谁？",
    "你叫什么名字？",
    "你是青小团吗？",
    "请用一句话解释什么是机器学习。",
]

for q in test_questions:
    prompt = IDENTITY_PREFIX + DEFAULT_PROMPT_TEMPLATE.format(instruction=q)
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(80):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids=x)
            logits = out.logits if hasattr(out, "logits") else out[1]
            next_id = int(logits[0, -1].argmax().item())
            if next_id == tokenizer.eos_id:
                break
            ids.append(next_id)
    response = tokenizer.decode(ids)
    log(f"\n[Q] {q}")
    log(f"[A] {response}")

_log_f.close()

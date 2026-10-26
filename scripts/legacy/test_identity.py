# -*- coding: utf-8 -*-
"""测试青小团模型身份认知。"""
import sys
import os
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from qxt.model.config import QXTConfig
from qxt.model.modeling import QXTForCausalLM
from qxt.distill.tokenizer import DistillTokenizer
from qxt.distill.prepare_data import DEFAULT_PROMPT_TEMPLATE, IDENTITY_PREFIX

# 加载模型
ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints/qxt_distill_v2")
tok_path = os.path.join(ckpt_dir, "tokenizer/qxt_distill")

LOG = open(os.path.join(PROJECT_ROOT, "identity_test.log"), "w", encoding="utf-8")

def p(msg):
    print(msg)
    LOG.write(str(msg) + "\n")
    LOG.flush()

p("加载分词器...")
tokenizer = DistillTokenizer.load(tok_path)

p("加载模型配置...")
with open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8") as f:
    cfg_dict = json.load(f)
# 直接用预设配置重建
config = QXTConfig.get_distill_student_config(
    vocab_size=cfg_dict.get("vocab_size", 4000),
    hidden_size=cfg_dict.get("hidden_size", 256),
    intermediate_size=cfg_dict.get("intermediate_size", 704),
    num_hidden_layers=cfg_dict.get("num_hidden_layers", 4),
    num_attention_heads=cfg_dict.get("num_attention_heads", 4),
    num_key_value_heads=cfg_dict.get("num_key_value_heads", 2),
    max_position_embeddings=cfg_dict.get("max_position_embeddings", 512),
)

p("加载模型权重...")
model = QXTForCausalLM(config)
model.load_state_dict(torch.load(os.path.join(ckpt_dir, "student_model.pt"), map_location="cpu"))
model.eval()

p(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
p("")

# 身份认知测试
questions = [
    "你是谁？",
    "你叫什么名字？",
    "你是什么模型？",
    "你是青小团吗？",
    "你是DeepSeek吗？",
    "你是ChatGPT吗？",
    "请用一句话解释什么是机器学习。",
]

p("=" * 60)
p("  青小团身份认知测试")
p("=" * 60)

for q in questions:
    try:
        prompt = IDENTITY_PREFIX + DEFAULT_PROMPT_TEMPLATE.format(instruction=q)
        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        
        with torch.inference_mode():
            for _ in range(30):
                x = torch.tensor([ids], dtype=torch.long)
                out = model(input_ids=x)
                logits = out.logits if hasattr(out, "logits") else out[1]
                next_id = int(logits[0, -1].argmax().item())
                if next_id == tokenizer.eos_id:
                    break
                ids.append(next_id)
        
        response = tokenizer.decode(ids)
        p(f"\n[Q] {q}")
        p(f"[A] {response}")
        p("-" * 40)
    except Exception as e:
        p(f"\n[Q] {q}")
        p(f"[A] 生成失败: {e}")
        p("-" * 40)

LOG.close()

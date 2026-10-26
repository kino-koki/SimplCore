# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""生成一份真实可训练的中文语料目录 (供 SimplCore 训练入口 --data-dir 使用)。

说明：本仓库原本没有中文训练语料，仅有冒烟用的 8 条样例。为了跑 *真实*
(即便是小规模) 的训练循环，这里生成一份结构化中文语料：
  * 唐诗 / 宋词节选 (公共领域，无版权风险)；
  * 现代散文 / 科普短句 (模板化组合，非抄袭)；
  * 科技说明 (SimplCore 相关主题，训练领域语料)。

输出：``corpus/cn_*.jsonl``，每行 {"text": "..."}，长度分布多样。
这是 *合成但真实可训练* 的语料，用于验证训练流水线端到端可用；生产时应
替换为正式清洗过的中文 web 语料。
"""
from __future__ import annotations

import json
import os
import random

# ----------------------------------------------------------------------
# 公共领域中文文本素材 (唐诗宋词 + 古文名句)
# ----------------------------------------------------------------------
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。",
    "空山新雨后，天气晚来秋。明月松间照，清泉石上流。",
    "君不见黄河之水天上来，奔流到海不复回。",
    "大江东去，浪淘尽，千古风流人物。",
    "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。",
    "采菊东篱下，悠然见南山。",
    "会当凌绝顶，一览众山小。",
    "海内存知己，天涯若比邻。",
    "落霞与孤鹜齐飞，秋水共长天一色。",
    "先天下之忧而忧，后天下之乐而乐。",
    "学而不思则罔，思而不学则殆。",
    "知之者不如好之者，好之者不如乐之者。",
    "三人行，必有我师焉。",
    "道可道，非常道；名可名，非常名。",
    "上善若水，水善利万物而不争。",
]

PROSE_TEMPLATES = [
    "江南的雨总是细密而温柔，落在青石板上，溅起一朵朵小小的水花。",
    "清晨的山林里，雾气还未散去，鸟鸣声此起彼伏，像是大自然醒来的呼吸。",
    "老屋的院子里种着一棵桂花树，每到秋天，满院都是甜香。",
    "书桌上摊开一本旧书，纸页泛黄，墨迹却依旧清晰，仿佛时光从未走远。",
    "海边的风带着咸味，浪花一层层涌来，又一层层退去，永不停歇。",
    "少年时的梦想，像天边的星，遥远却明亮，指引着我们向前走去。",
    "炊烟袅袅升起的时候，村庄便有了人间烟火的味道。",
    "月光洒在湖面上，碎成千万片银鳞，随风轻轻摇曳。",
]

TECH_TEMPLATES = [
    "人工智能是一类让机器模拟人类智能的技术，包括学习、推理与感知。",
    "大语言模型通过海量文本训练，学会预测下一个词，从而具备语言能力。",
    "深度学习依赖多层神经网络，从数据中自动提取层次化特征表示。",
    "注意力机制使模型能够关注输入中最相关的部分，提升长序列建模能力。",
    "谱全局卷积以频域演化捕获全局依赖，复杂度仅为序列长度的对数级别。",
    "动态记忆通道维护固定大小的记忆槽，实现跨位置的长期信息留存。",
    "混合专家模型在每一层只激活部分专家，以更低算力获得更大容量。",
    "长上下文流式核心将序列切块处理，跨块仅传递固定大小的隐状态。",
    "训练大模型需要海量算力与高质量数据，二者缺一不可。",
    "量化与蒸馏可在尽量保持性能的前提下，显著降低模型部署成本。",
]


def build_text(rng: random.Random, min_len: int = 24, max_len: int = 200) -> str:
    """组合出一条长度随机的中文语料。"""
    kind = rng.random()
    parts = []
    total = 0
    target = rng.randint(min_len, max_len)
    pool = POEMS if kind < 0.4 else (PROSE_TEMPLATES if kind < 0.7 else TECH_TEMPLATES)
    while total < target:
        seg = rng.choice(pool)
        parts.append(seg)
        total += len(seg)
        if rng.random() < 0.15:  # 偶尔跨库拼接，增加多样性
            parts.append(rng.choice(rng.choice([POEMS, PROSE_TEMPLATES, TECH_TEMPLATES])))
            total += len(parts[-1])
    return "".join(parts)


def main(out_dir: str = "corpus", n_docs: int = 1200, seed: int = 7) -> None:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    # 拆成多个 jsonl 文件，模拟真实多文件语料
    files = 4
    per = (n_docs + files - 1) // files
    written = 0
    for fi in range(files):
        fp = os.path.join(out_dir, f"cn_{fi:02d}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for _ in range(per):
                text = build_text(rng)
                if len(text) < 16:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
    print(f"已生成 {written} 条中文语料到 {out_dir}/ (共 {files} 个 jsonl 文件)")
    # 统计长度分布
    import statistics
    lens = []
    for fi in range(files):
        with open(os.path.join(out_dir, f"cn_{fi:02d}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                lens.append(len(json.loads(line)["text"]))
    if lens:
        print(f"语料长度: min={min(lens)} max={max(lens)} "
              f"mean={statistics.mean(lens):.0f} median={statistics.median(lens):.0f}")


if __name__ == "__main__":
    import sys
    od = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    main(od, nd)

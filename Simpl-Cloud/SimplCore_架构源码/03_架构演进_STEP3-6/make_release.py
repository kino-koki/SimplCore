# -*- coding: utf-8 -*-
"""Build a self-contained release directory: trained weights + architecture +
vocab + a standalone chat/generate script. No training code, no cloud repo,
no torch, no network needed to run it.
"""
from __future__ import annotations

import json, os, shutil, sys, zipfile, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
WS = os.path.dirname(RT)
ARCH = os.path.join(WS, "_cloud_repo", "SimplCore-main", "simplcore",
                    "core", "arch")
RES = os.path.join(RT, "results")
REL = os.path.join(WS, "step7_release")
ZIP = os.path.join(WS, "SimplCore_v19_可说话模型.zip")

CHAT = '''# -*- coding: utf-8 -*-
"""Standalone SimplCore LM inference. Pure NumPy, no torch, no network.

    python chat.py                      # 6 demo prompts
    python chat.py "Q: Who goes there?\\nA:"   # your own prompt
    python chat.py "HAMLET:\\n" --n 300 --temp 0.8
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn
import simplcore_lm as A

D = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(D, "config.json"), encoding="utf8"))
vocab = json.load(open(os.path.join(D, "vocab.json"), encoding="utf8"))
stoi = {c: int(i) for c, i in vocab["stoi"].items()}
itos = {int(i): c for c, i in vocab["stoi"].items()}

m = A.SimplCoreLM(cfg["V"], d=cfg["d"], n_layer=cfg["layers"], W=cfg["W"],
                  rhos=tuple(cfg["rhos"]), lam=cfg["lam"],
                  maxL=cfg["maxL"], seed=cfg["seed"])
z = np.load(os.path.join(D, "model.npz"))
for n, p in m.params().items():
    np.copyto(p.data, z["p:" + n])


def sample(logits, temp, top_p, rng):
    z = logits / temp
    z = z - z.max()
    e = np.exp(np.clip(z, -60, 60))
    p = e / e.sum()
    o = np.argsort(p)[::-1]
    keep = o[:int(np.searchsorted(np.cumsum(p[o]), top_p)) + 1]
    pk = p[keep] / p[keep].sum()
    return int(rng.choice(keep, p=pk))


def chat(prompt, n=220, temp=0.85, top_p=0.92, seed=7):
    rng = np.random.default_rng(seed)
    nn.set_no_grad(True)
    ids = [stoi.get(c, 0) for c in prompt]
    for _ in range(n):
        ctx = np.array([ids[-m.maxL:]], dtype=np.int64)
        logits = m.logits(ctx)[0, -1, :]
        ids.append(sample(logits, temp, top_p, rng))
    nn.set_no_grad(False)
    return "".join(itos[i] for i in ids[len(prompt):])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    prompts = ([a.prompt] if a.prompt else
               ["First Citizen:\\n", "HAMLET:\\n",
                "Q: What is your name?\\nA:", "Q: Shall we proceed?\\nA:",
                "Q: Who goes there?\\nA:", "To be, or not to be,"])
    for pr in prompts:
        print("PROMPT:", repr(pr))
        print(chat(pr, a.n, a.temp, a.top_p, a.seed))
        print("-" * 60)
'''

README = '''# SimplCore v19 — 可独立运行的会说话模型

纯 NumPy，无需 torch / 网络 / 云端仓库。解压后直接：

```powershell
python chat.py
python chat.py "Q: Who goes there?\\nA:" --n 200
python chat.py "HAMLET:\\n" --temp 0.8
```

内容：
- `model.npz` 端到端训练出的权重（不是任何 teacher 的副本）
- `nn.py` / `simplcore_lm.py` 架构与自动微分（ARCH_STEP = 7）
- `vocab.json` / `config.json` 词表与超参
- `chat.py` 独立推理脚本
- `train_record.json` 训练曲线、验收与边界说明

架构：3 层 decoder-only；每层 = 固定窗口内容寻址 + 静态多极点 Toeplitz
(SGC) 通道 + 输入门控 O(d^2) 自进化状态 + 前馈；pre-LN 残差。
状态大小与序列长度无关，无 KV cache，无全局 N×N score。

边界：字符级、202k 参数、1.1 MB 真实英文剧本语料、CPU 训练。
生成的是连贯但含伪词的英文戏剧文本，不是 1B 级中文助手，
不保证事实正确，不与预训练 Transformer 做能力对比。
'''


def main():
    assert os.path.exists(os.path.join(RES, "step7_ckpt.npz")), "no checkpoint"
    z = np.load(os.path.join(RES, "step7_ckpt.npz"))
    cfg = json.loads(str(z["_cfg"]))
    cfg["rhos"] = list(cfg["rhos"])
    from train import build_corpus
    _, _, V, stoi, _ = build_corpus()
    if os.path.isdir(REL):
        shutil.rmtree(REL)
    os.makedirs(REL)
    shutil.copy(os.path.join(ARCH, "nn.py"), os.path.join(REL, "nn.py"))
    shutil.copy(os.path.join(ARCH, "simplcore_lm.py"),
                os.path.join(REL, "simplcore_lm.py"))
    shutil.copy(os.path.join(RES, "step7_ckpt.npz"),
                os.path.join(REL, "model.npz"))
    json.dump(cfg, open(os.path.join(REL, "config.json"), "w"), indent=1)
    json.dump(dict(stoi=stoi), open(os.path.join(REL, "vocab.json"), "w"),
              ensure_ascii=False)
    open(os.path.join(REL, "chat.py"), "w", encoding="utf8").write(CHAT)
    open(os.path.join(REL, "README.md"), "w", encoding="utf8").write(README)
    rec = {}
    for f, key in [("step7_progress.json", "train_curve"),
                   ("step7_verify.json", "whitebox"),
                   ("step7_generation.json", "generation"),
                   ("step7_acceptance.json", "acceptance"),
                   ("step7_train_last.json", "train_final")]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            rec[key] = json.load(open(p))
    json.dump(rec, open(os.path.join(REL, "train_record.json"), "w"),
              ensure_ascii=False, indent=1)

    names = sorted(os.listdir(REL))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for n in names:
            zf.write(os.path.join(REL, n), n)
        mf = {n: dict(bytes=os.path.getsize(os.path.join(REL, n)),
                      sha256=hashlib.sha256(
                          open(os.path.join(REL, n), "rb").read()).hexdigest())
              for n in names}
        zf.writestr("MANIFEST.json", json.dumps(mf, ensure_ascii=False, indent=1))
    zz = zipfile.ZipFile(ZIP)
    print("release dir:", REL, "files", len(names))
    print("zip:", ZIP, os.path.getsize(ZIP), "B, testzip:", zz.testzip())
    print("files:", names)


if __name__ == "__main__":
    main()

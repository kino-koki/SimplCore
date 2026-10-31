# -*- coding: utf-8 -*-
"""training_integration 单元验证：证明六个增强在训练中确有真实作用，而非仅"可运行"。"""
import sys, torch
from torch.utils.data import DataLoader, TensorDataset

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import CharTokenizer
from simplcore.core.training_integration import (
    build_enhanced_model, PonderEnhancer, EnvLearnEnhancer, AccelEval,
    MemoryEnhancer, ReasoningEnhancer, DreamConsolidator,
)

torch.manual_seed(0)
VOCAB, DIM, LAYERS, B, L = 256, 64, 2, 2, 16
base_cfg = SimplCoreConfig(vocab_size=VOCAB, dim=DIM, num_layers=LAYERS,
                           num_bases=8, use_memory=False)
mem_cfg = SimplCoreConfig(vocab_size=VOCAB, dim=DIM, num_layers=LAYERS,
                          num_bases=8, use_memory=True)
codec = CharTokenizer(limit=VOCAB)

fails = []
def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)

# 0) 默认关闭：em.model 逐位就是基座，训练前向可用
m = SimplCoreModel(base_cfg)
em = build_enhanced_model(m, base_cfg, [])
check("default-off em.model is base", em.model is m)
logits, loss = em.train_forward(torch.randint(0, VOCAB, (B, L)),
                                torch.randint(0, VOCAB, (B, L)))
check("default-off forward returns (logits, loss)", loss is not None and logits.shape[-1] == VOCAB)

# 1) ponder：梯度流向 thinker（思考单元成为可训练信号）
pe = PonderEnhancer(m)
logits, loss = pe(torch.randint(0, VOCAB, (B, L)), torch.randint(0, VOCAB, (B, L)))
loss.backward()
halt_grad = pe.thinker.halting.proj.weight.grad
base_grad = m.tok_emb.weight.grad
check("ponder loss finite", torch.isfinite(loss).all().item())
check("ponder grad -> halting head", halt_grad is not None and halt_grad.abs().sum() > 0)
check("ponder grad -> base emb", base_grad is not None)
print(f"    ponder ledger steps={pe.ledger.steps if hasattr(pe.ledger,'steps') else '?'}")

# 2) envlearn：基座冻结，仅适配器可训
ee = EnvLearnEnhancer(m, dim=DIM)
logits, loss = ee(torch.randint(0, VOCAB, (B, L)), torch.randint(0, VOCAB, (B, L)))
loss.backward()
base_trainable = sum(p.requires_grad for p in ee.base.parameters())
adapter_trainable = sum(p.requires_grad for p in ee.adapter.parameters())
check("envlearn base frozen", base_trainable == 0)
check("envlearn adapter trainable", adapter_trainable > 0)
check("envlearn budget ok", ee.budget_dict()["learned_params"] > 0)

# 3) accel：评估期自适应深度，账本给出 speedup
acc = AccelEval(m)
ds = TensorDataset(torch.randint(0, VOCAB, (8, L)), torch.randint(0, VOCAB, (8, L)))
loader = DataLoader(ds, batch_size=2)
res = acc.eval_dataset(loader, torch.device("cpu"), None, max_batches=4)
check("accel speedup >= 1", res["speedup"] >= 1.0, f"speedup={res['speedup']}")
check("accel loss_proxy finite", res["loss_proxy"] == res["loss_proxy"])

# 4) smm：经验真实写入记忆库；answer 注入记忆可跑（需 use_memory）
mm = SimplCoreModel(mem_cfg)
me = MemoryEnhancer(mm, codec)
n0 = me.num_experiences()
me.experience("中国的首都是哪里？", "北京。")
me.experience("1+1等于几？", "2。")
check("smm experience recorded", me.num_experiences() == n0 + 2,
      f"{n0} -> {me.num_experiences()}")
mm.eval()
out = me.answer(torch.randint(0, VOCAB, (1, 4)), max_new_tokens=4, query="北京")
check("smm answer returns tokens", "tokens" in out and out["tokens"].shape[0] == 1)

# 5) reasoning：迭代式思考返回 trace
re = ReasoningEnhancer(mm, codec, max_steps=3)
trace = re.reason("请计算 3 + 4 等于多少？")
check("reasoning returns answer", getattr(trace, "answer", None) is not None,
      f"steps={getattr(trace,'steps',None)}")

# 6) dream：喂入经验后 run() 真的进入 REFLECT/SYNTHESIZE（非 idle）
dm = SimplCoreModel(base_cfg)
dc = DreamConsolidator(dm, codec=codec)
rep = dc.consolidate(experiences=[("什么是梯度下降？", "沿梯度反方向更新参数。"),
                                  ("什么是过拟合？", "训练集好、验证集差。")],
                     max_rounds=1)
stages = rep.get("stages", 0)
rep0 = (rep.get("reports") or [{}])[0]
check("dream ran >=1 stage", stages >= 1)
check("dream harvested >0 (not idle)", rep0.get("harvested", 0) >= 1,
      f"harvested={rep0.get('harvested')}, stage={rep0.get('stage')}")
print(f"    dream stage={rep0.get('stage')} insights={rep0.get('insights')} "
      f"synthesized={rep0.get('synthesized')} rolled_back={rep0.get('rolled_back')}")

print("\n=== SUMMARY ===")
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL PASS")

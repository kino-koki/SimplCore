"""
SGC vs Attention 下游有效性验证 —— 正式运行入口（纯自研 simplcore.tensor 后端）

用法:
  python run_experiment.py            # 跑 attention + sgc 各 STEPS 步
  python run_experiment.py --sgc-only # 只跑 sgc
  python run_experiment.py --attn-only

所有超参在 train.py 顶部。结果写入 results/results.json 与 results/summary.txt。
"""
import sys, os, json, argparse
sys.setrecursionlimit(20000)
import train as T

parser = argparse.ArgumentParser()
parser.add_argument("--sgc-only", action="store_true")
parser.add_argument("--attn-only", action="store_true")
args = parser.parse_args()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)

results = []
if not args.sgc_only:
    results.append(T.run("attn", "Attention-Baseline"))
if not args.attn_only:
    results.append(T.run("sgc", "SGC-Model"))

with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("SGC vs Attention 下游有效性验证 (selective copy 长程依赖任务)\n")
    f.write(f"配置: L={T.L} N_COPY={T.N_COPY} DIM={T.DIM} N_LAYERS={T.N_LAYERS} "
            f"BATCH={T.BATCH} STEPS={T.STEPS} LR={T.LR}\n\n")
    for r in results:
        f.write(f"{r['tag']:20s} params={r['params']:>9,}  "
                f"final_val_acc={r['final_val_acc']*100:5.2f}%  time={r['time_s']:.0f}s\n")
    if len(results) == 2:
        a = results[0]["final_val_acc"]; s = results[1]["final_val_acc"]
        better = "SGC" if s > a else ("Attention" if a > s else "tie")
        f.write(f"\n相对表现: {better}  (Δ={(s-a)*100:+.2f}pp)\n")

print("\n===== SUMMARY =====")
for r in results:
    print(f"  {r['tag']:20s} params={r['params']:>9,}  final_val_acc={r['final_val_acc']*100:5.2f}%  time={r['time_s']:.0f}s")
if len(results) == 2:
    a = results[0]["final_val_acc"]; s = results[1]["final_val_acc"]
    better = "SGC" if s > a else ("Attention" if a > s else "tie")
    print(f"  => 相对表现: {better}  (Δ={(s-a)*100:+.2f}pp)")

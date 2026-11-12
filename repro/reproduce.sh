#!/usr/bin/env bash
# 训练类结果的重跑命令清单
# ⚠️ 说明：模型断点（ckpt/）已清理，本轮「复现验证」只重算了所有结构/解析/统计项
#          （见 verify_all.py，29 PASS / 0 FAIL），以下训练命令本轮【未重跑】。
#          原始结果保存在 results_*.json，可据此核对。
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. 一键轻量复现验证（本轮已跑，29 PASS / 0 FAIL）=="
python verify_all.py
python audit_against_paper.py

echo "== 2. 结构性质全套（定理 1/2/3、命题 5、部署账本）=="
python verify.py

echo "== 3. 能力对照（每个任务需分块续跑，耗时较长）=="
for t in pointing mqar agg65 count markov mqar129 osc osc0 osc05; do
  echo "  python run_chunk.py $t 0 ts  <total> <chunk>"
  echo "  python run_chunk.py $t 0 sgc <total> <chunk>"
done

echo "== 4. 判别性实验 =="
echo "  python sweep_rho.py 0.9999 700 0"
echo "  python sweep_rho.py 0.99   700 0"
echo "  python sel_chunk.py mqar 0 2000 500   # 续跑至 2000"

echo "== 5. 频率网格 / 振荡轴 =="
echo "  python grid_chunk.py osc linear30 1500 250 0   # 分 6 块续跑至 1500"
echo "  python sweep_osc.py curve osc0 ts  1000"
echo "  python sweep_osc.py curve osc0 sgc 1000"
echo "  python ablation_osc.py osc0 300 osc / rand / perm"

echo "== 6. 多 seed（投稿前必补，本轮未跑）=="
echo "  把上述命令的 seed 0 依次改为 1 2 3 4，重算 mean±std 与 CI"

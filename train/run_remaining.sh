#!/usr/bin/env bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /data/workspace/train
# SGC 1000 -> 3000
for i in 1 2 3 4; do
  python run.py --arm sgc --total 3000 --chunk 500 --tag sgc --lr 6e-4 >> run_sgc.out 2>&1
done
# KD 0 -> 3000
for i in 1 2 3 4 5 6; do
  python run.py --arm kd --total 3000 --chunk 500 --tag kd --teacher ts --lr 6e-4 >> run_kd.out 2>&1
done
echo DONE > run_remaining.done

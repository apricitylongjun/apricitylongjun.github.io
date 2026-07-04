#!/usr/bin/env bash
# 第二阶段批量训练：4 方法 × 3 噪声 × 3 seed = 36 实验
# 用法: bash code/run_all_stage2.sh   (在 aa_workspace 下，需已激活 venv)
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/aa_workspace"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

for method in ce sce gce coteaching; do
  for setting in clean noise20 noise40; do
    for seed in 42 2024 2026; do
      echo "=================== $method | $setting | seed=$seed ==================="
      python code/pre_exp_stage2_train.py \
        --method "$method" --setting "$setting" --seed "$seed" \
        --epochs 40 --batch 128 --lr 1e-3
    done
  done
done
echo "ALL_DONE"

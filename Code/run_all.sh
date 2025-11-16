#!/usr/bin/env bash
set -e

export TF_CPP_MIN_LOG_LEVEL=3

ALGS=("PPO" "A2C" "SAC" "TD3" "DDPG")
SEEDS=(7 11)
PS=("0.50" "0.60" "0.80")
NS=(4 9 12 19)

TOTAL_STEPS=300000
TAU=0.03
R_U=10.0
SIGW=1e-3
GAMMA=0.99
HORIZON=512
LOG_ROOT="runs_2"

# device policy: PPO/A2C -> cpu, off-policy -> cuda if available
HAS_CUDA=$(python - <<'PY'
import torch; print(int(torch.cuda.is_available()))
PY
)

for nlinks in "${NS[@]}"; do
  for p in "${PS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for algo in "${ALGS[@]}"; do
        if [[ "$algo" == "PPO" || "$algo" == "A2C" ]]; then
          DEV="cpu"
          NUM_ENVS=1
        else
          DEV=$([[ "$HAS_CUDA" == "1" ]] && echo "cuda" || echo "cpu")
          NUM_ENVS=1
        fi
        echo "[RUN] n=$nlinks p=$p algo=$algo seed=$seed dev=$DEV"
        python train_baselines_arm_gpu.py \
          --algo $algo --total_steps $TOTAL_STEPS \
          --seed $seed --p $p --n_links $nlinks \
          --tau $TAU --r_u $R_U --sigma_omega $SIGW \
          --gamma $GAMMA --horizon $HORIZON \
          --device $DEV --num_envs $NUM_ENVS --normalize \
          --log_root "$LOG_ROOT" \
          --eval_every 10000 \
          --save_checkpoints \
          # remove the line above if you truly want ONLY best; keeping it helps rescue mid-runs if interrupted
      done
    done
  done
done
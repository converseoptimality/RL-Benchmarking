#!/usr/bin/env bash
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=3

# ---------- concurrency knobs (safe on i9 + 3080) ----------
CPU_JOBS=2          # run PPO and A2C pools concurrently on CPU
GPU_JOBS=1          # run exactly one off-policy job (SAC/TD3/DDPG) on the 3080
CPU_ENVS_PER_JOB=6  # envs per PPO/A2C job (keeps CPU from thrashing)
GPU_ENVS_PER_JOB=2  # envs per SAC/TD3/DDPG job

PPO_STEPS=1500000
A2C_STEPS=1500000
OFF_STEPS=700000

# keep numpy/blas from oversubscribing threads:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------- what to run ----------
ALGS_ON=("PPO" "A2C")
ALGS_OFF=("TD3" "DDPG" "SAC")
SEEDS=(7 11)           # <- seed 19 removed
KS=(2)                    # <- ONLY K=2
RVS=(1)
RWS=(1)
DIFFS=("easy" "hard" "insane")  # <- “medium” removed

GAMMA=0.99
HORIZON=700
LOG_ROOT="runs_Nudex_1/runs_Nudex_1resume_k=2"

HAS_CUDA=$(python - <<'PY' 2>/dev/null
import torch; print(int(torch.cuda.is_available()))
PY
)
DEV=$([[ "$HAS_CUDA" == "1" ]] && echo "cuda" || echo "cpu")

mkdir -p "$LOG_ROOT"

# difficulty presets (your current resume values):
# -> (p, alpha, kappa, sigma_pose, sigma_v, sigma_w, rho)
diff_knobs() {
  case "$1" in
    easy)   echo "0.5  0.04 0.20 0.0    1e-4  3e-5  1.001" ;;
    hard)   echo "1.0  0.07 0.27 1e-12  3e-4  2e-5  1.004" ;;
    insane) echo "1.0  0.08 0.30 1e-6   5e-4  5e-5  1.005" ;;
    # medium not used
  esac
}

ON_TXT=$(mktemp)
OFF_TXT=$(mktemp)
trap 'rm -f "$ON_TXT" "$OFF_TXT"' EXIT

for K in "${KS[@]}"; do
  for rv in "${RVS[@]}"; do
    for rw in "${RWS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for diff in "${DIFFS[@]}"; do
          read PVAL ALPHA KAPPA SP SV SW RHO <<<"$(diff_knobs "$diff")"

          # ---------- ON-POLICY (CPU): PPO & A2C ----------
          for algo in "${ALGS_ON[@]}"; do
            if [[ "$algo" == "PPO" ]]; then
              TOTAL=$PPO_STEPS
              EXTRA="--ppo_n_steps 1024 --ppo_batch_size 256 --ppo_n_epochs 10 --ppo_lr 3e-4 --ppo_ent_coef 1e-4 --ppo_clip_range 0.2"
            else
              TOTAL=$A2C_STEPS
              EXTRA="--a2c_n_steps 20 --a2c_lr 3e-4 --a2c_ent_coef 1e-4"
            fi
            echo "echo [ON] $algo K=$K rv=$rv rw=$rw diff=$diff seed=$seed; \
python train_baselines_nudex.py --algo $algo --total_steps $TOTAL --seed $seed \
--K $K --r_v $rv --r_w $rw --p $PVAL --alpha $ALPHA --kappa $KAPPA \
--sigma_pose $SP --sigma_v $SV --sigma_w $SW --rho_target $RHO \
--gamma $GAMMA --horizon $HORIZON --device cpu --num_envs $CPU_ENVS_PER_JOB --normalize \
--fixed_init --log_root $LOG_ROOT --eval_every 10000 $EXTRA" >>"$ON_TXT"
          done

          # ---------- OFF-POLICY (GPU): SAC/TD3/DDPG ----------
          for algo in "${ALGS_OFF[@]}"; do
            EXTRA="--off_batch_size 256 --off_buffer_size 200000 --off_lr 3e-4 --off_tau 0.005 --off_train_freq 1 --off_gradient_steps 1"
            if [[ "$algo" == "SAC" ]]; then
              EXTRA+=" --off_learning_starts 10000 --sac_ent_coef auto"
            else
              EXTRA+=" --off_learning_starts 25000"
              if [[ "$algo" == "TD3" ]]; then
                # IMPORTANT: matches your parser names (not target_policy_noise)
                EXTRA+=" --td3_policy_delay 2 --td3_target_noise 0.2 --td3_target_clip 0.5"
              fi
            fi
            echo "echo [OFF] $algo K=$K rv=$rv rw=$rw diff=$diff seed=$seed; \
CUDA_VISIBLE_DEVICES=0 python train_baselines_nudex.py --algo $algo --total_steps $OFF_STEPS --seed $seed \
--K $K --r_v $rv --r_w $rw --p $PVAL --alpha $ALPHA --kappa $KAPPA \
--sigma_pose $SP --sigma_v $SV --sigma_w $SW --rho_target $RHO \
--gamma $GAMMA --horizon $HORIZON --device $DEV --num_envs $GPU_ENVS_PER_JOB --normalize \
--fixed_init --log_root $LOG_ROOT --eval_every 10000 $EXTRA" >>"$OFF_TXT"
          done

        done
      done
    done
  done
done

echo "[LAUNCH] CPU pool: ${CPU_JOBS} jobs | GPU pool: ${GPU_JOBS} job"
if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "$CPU_JOBS" --lb bash -lc :::: "$ON_TXT" &
  parallel --jobs "$GPU_JOBS" --lb bash -lc :::: "$OFF_TXT" &
  wait
else
  xargs -I{} -P "$CPU_JOBS" bash -lc "{}" < "$ON_TXT" &
  xargs -I{} -P "$GPU_JOBS" bash -lc "{}" < "$OFF_TXT" &
  wait
fi

echo "[DONE] All runs complete."

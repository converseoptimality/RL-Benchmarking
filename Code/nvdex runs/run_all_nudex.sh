#!/usr/bin/env bash
set -e
export TF_CPP_MIN_LOG_LEVEL=3

ALGS=("PPO" "SAC" "A2C" "TD3" "DDPG")
SEEDS=(7 11 19)
KS=(1 2)
RVS=(1)
RWS=(1)
DIFFS=("easy" "medium" "hard" "insane")

GAMMA=0.99
HORIZON=700
LOG_ROOT="runs_Nudex_1"

HAS_CUDA=$(python - <<'PY'
import torch; print(int(torch.cuda.is_available()))
PY
)

for K in "${KS[@]}"; do
  for rv in "${RVS[@]}"; do
    for rw in "${RWS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for algo in "${ALGS[@]}"; do

          # ----- per-algo defaults (wall-clock / fairness by algo, not one-size-fits-all) -----
          TOTAL_STEPS=300000
          NUM_ENVS=1
          DEV=$([[ "$HAS_CUDA" == "1" ]] && echo "cuda" || echo "cpu")

          EXTRA_FLAGS=""
          case "$algo" in
            PPO)
              TOTAL_STEPS=2000000
              NUM_ENVS=8          # large batch via many envs
              DEV="cpu"           # PPO often CPU-bound; keep simple
              EXTRA_FLAGS+=" --ppo_n_steps 1024 --ppo_batch_size 256 --ppo_n_epochs 10 --ppo_lr 3e-4 --ppo_ent_coef 1e-4 --ppo_clip_range 0.2"
              ;;
            A2C)
              TOTAL_STEPS=2000000
              NUM_ENVS=8
              DEV="cpu"
              EXTRA_FLAGS+=" --a2c_n_steps 20 --a2c_lr 7e-4 --a2c_ent_coef 1e-4"
              ;;
            SAC)
              TOTAL_STEPS=1500000
              NUM_ENVS=1
              EXTRA_FLAGS+=" --off_batch_size 256 --off_buffer_size 1000000 --off_learning_starts 10000 --off_train_freq 1 --off_gradient_steps 1 --off_lr 3e-4 --off_tau 0.005 --sac_ent_coef auto"
              ;;
            TD3)
              TOTAL_STEPS=1500000
              NUM_ENVS=2
              EXTRA_FLAGS+=" --off_batch_size 256 --off_buffer_size 1000000 --off_learning_starts 25000 --off_train_freq 1 --off_gradient_steps 1 --off_lr 3e-4 --off_tau 0.005 --td3_policy_delay 2 --td3_target_noise 0.2 --td3_target_clip 0.5"
              ;;
            DDPG)
              TOTAL_STEPS=1500000
              NUM_ENVS=2
              EXTRA_FLAGS+=" --off_batch_size 256 --off_buffer_size 1000000 --off_learning_starts 25000 --off_train_freq 1 --off_gradient_steps 1 --off_lr 3e-4 --off_tau 0.005"
              ;;
          esac

          for diff in "${DIFFS[@]}"; do
            case "$diff" in
              easy)    PVAL=0.5;  ALPHA=0.04; KAPPA=0.2;  SP=0.0; SV=1e-4; SW=3e-5;  RHO=1.001 ;;
              medium)  PVAL=0.8;  ALPHA=0.06; KAPPA=0.23;  SP=0.0; SV=2e-4; SW=1e-5;  RHO=1.002 ;;
              hard)    PVAL=1.0;  ALPHA=0.07; KAPPA=0.27;  SP=1e-12; SV=3e-4; SW=2e-5;  RHO=1.004 ;;
              insane)  PVAL=1.0;  ALPHA=0.08; KAPPA=0.3;  SP=1e-6; SV=5e-4; SW=5e-5;  RHO=1.005 ;;
            esac

            echo "[RUN] K=$K rv=$rv rw=$rw diff=$diff algo=$algo seed=$seed dev=$DEV steps=$TOTAL_STEPS envs=$NUM_ENVS"
            python train_baselines_nudex.py \
              --algo $algo --total_steps $TOTAL_STEPS \
              --seed $seed --K $K --r_v $rv --r_w $rw \
              --p $PVAL --alpha $ALPHA --kappa $KAPPA \
              --sigma_pose $SP --sigma_v $SV --sigma_w $SW \
              --rho_target $RHO --gamma $GAMMA --horizon $HORIZON \
              --device $DEV --num_envs $NUM_ENVS --normalize \
              --fixed_init \
              --log_root "$LOG_ROOT" \
              --eval_every 10000 \
              $EXTRA_FLAGS
          done

        done
      done
    done
  done
done

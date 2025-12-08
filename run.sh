#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-infer}

if [[ "${MODE}" == "train" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  MASTER_ADDR=${MASTER_ADDR:-localhost}
  MASTER_PORT=${MASTER_PORT:-29500}
  NPROC_PER_NODE=${NPROC_PER_NODE:-1}

  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    rl_agents/train_5v5_ppo.py \
    --total-timesteps "${TOTAL_TIMESTEPS:-100000000}" \
    --num-envs "${NUM_ENVS:-32}" \
    --num-steps "${NUM_STEPS:-2048}" \
    --lr "${LR:-1e-05}" \
    --checkpoint-dir "${CHECKPOINT_DIR:-policy}" \
    --logdir "${LOGDIR:-runs/ppo_5v5}" \
    --env-name "${ENV_NAME:-5_vs_5}" \
    --action-set "${ACTION_SET:-full}" \
    --rewards "${REWARDS:-scoring,checkpoints,shaping}" \
    --opponent-difficulty-end "${OPP_DIFFICULTY_END:-0.8}"
else
  STACK_FLAG="--stacked"
  if [[ "${STACKED:-1}" == "0" ]]; then
    STACK_FLAG="--no-stacked"
  fi
  RENDER_FLAG=""
  if [[ "${RENDER:-0}" == "1" ]]; then
    RENDER_FLAG="--render"
  fi
  RECORD_FLAG=""
  if [[ "${RECORD_VIDEO:-0}" == "1" ]]; then
    RECORD_FLAG="--record"
  fi

  python3 -m rl_agents.run_inference \
    --checkpoint "${CHECKPOINT:-policy/best_model.pt}" \
    --env-name "${ENV_NAME:-5_vs_5}" \
    --representation "${REPRESENTATION:-extracted}" \
    --action-set "${ACTION_SET:-full}" \
    --rewards "${REWARDS:-scoring,checkpoints,shaping}" \
    --episodes "${EPISODES:-5}" \
    --difficulty "${DIFFICULTY:-0.6}" \
    --left-players "${LEFT_PLAYERS:-1}" \
    --right-players "${RIGHT_PLAYERS:-0}" \
    --video-format "${VIDEO_FORMAT:-webm}" \
    ${STACK_FLAG} ${RENDER_FLAG} ${RECORD_FLAG}
fi

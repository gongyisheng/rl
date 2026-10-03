#!/usr/bin/env bash
# Usage: nohup bash experiments/lora_lr/run.sh > logs/lora_lr.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_launcher="${script_dir}/run_lora_sm120.sh"
output_dir="${OUTPUT_DIR:-/data/lora/lora_lr}"
learning_rates=(1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4)

for learning_rate in "${learning_rates[@]}"; do
    run_dir="${output_dir}/lr-${learning_rate}"
    mkdir -p "${run_dir}"

    LR="${learning_rate}" \
    SAVE_DIR="${run_dir}/checkpoints" \
    WANDB_PROJECT="rl-lora-lr" \
    WANDB_RUN_NAME="qwen2.5-3B-dapo-gsm8k-lr-${learning_rate}" \
    bash "${base_launcher}" 2>&1 | tee "${run_dir}/train.log"
done

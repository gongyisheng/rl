#!/usr/bin/env bash
# Usage: nohup bash experiments/lora_lr/gsm8k/run.sh > logs/lora_lr_gsm8k.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for learning_rate in 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4; do
    bash "${script_dir}/qwen25_3b_dapo_gsm8k_lora_lr_${learning_rate}.sh"
done

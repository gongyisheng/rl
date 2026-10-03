#!/usr/bin/env bash
# Usage: nohup bash experiments/gsm8k_steps/run.sh > logs/gsm8k_steps.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts=(
    "qwen25_3b_dapo_gsm8k_lora.sh"
    "qwen25_3b_dapo_gsm8k_full.sh"
)

for script in "${scripts[@]}"; do
    bash "${script_dir}/${script}"
done

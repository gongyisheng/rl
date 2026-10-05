#!/usr/bin/env bash
# Usage: nohup bash experiments/lora_batch_size/run.sh > logs/lora_batch_size.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for batch_size in 256 512 1024 2048 4096; do
    bash "${script_dir}/qwen25_3b_dapo_gsm8k_lora_bs_${batch_size}.sh"
done

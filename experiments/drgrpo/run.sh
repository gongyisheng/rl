#!/usr/bin/env bash
# Usage: nohup bash experiments/drgrpo/run.sh > logs/drgrpo.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for algorithm in grpo drgrpo; do
   bash "${script_dir}/qwen25_3b_${algorithm}_gsm8k_lora.sh"
done

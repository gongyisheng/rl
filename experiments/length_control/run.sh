#!/bin/bash
# Usage: nohup bash experiments/length_control/run.sh > logs/length_control.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/qwen25_3b_dapo_gsm8k_lora.sh"

for version in v1 v2; do
   for alpha in 1e-3 1e-2 1e-1; do
      bash "${SCRIPT_DIR}/qwen25_3b_dapo_gsm8k_${version}_lora_alpha_${alpha}.sh"
   done
done

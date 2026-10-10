#!/usr/bin/env bash
# Usage: nohup bash experiments/batch_size/dapo_math_17k/run.sh > logs/batch_size_dapo_math_17k.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for batch_size in 256 512 1024 2048 4096; do
    bash "${script_dir}/qwen25_3b_dapo_dapo_math_17k_bs_${batch_size}.sh"
done

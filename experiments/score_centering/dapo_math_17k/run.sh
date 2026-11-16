#!/usr/bin/env bash
# Usage: nohup bash experiments/score_centering/dapo_math_17k/run.sh > logs/score_centering_dapo_math_17k.log 2>&1 &
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for precision_variant in baseline rollout_fp8_w8a8 train_fp8_w8a16; do
   variant_suffix=""
   if [[ "${precision_variant}" != baseline ]]; then
      variant_suffix="_${precision_variant}"
   fi
   for score_centering_is in none tis mis; do
      bash "${script_dir}/qwen3_4b_base_score_centering_dapo_math_17k${variant_suffix}_is_${score_centering_is}.sh"
   done
done

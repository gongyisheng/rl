#!/usr/bin/env bash
# Usage: nohup bash experiments/score_centering/dapo_math_17k/run.sh > logs/score_centering_dapo_math_17k.log 2>&1 &
export WANDB_CONSOLE=off
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Optional arguments apply only to the first variant, for restarting that run.
first_run_args=("$@")

for precision_variant in baseline rollout_fp8_w8a8 train_fp8_w8a16; do
   variant_suffix=""
   if [[ "${precision_variant}" != baseline ]]; then
      variant_suffix="_${precision_variant}"
   fi
   for score_centering_is in off none tis mis; do
      algorithm=score_centering
      is_suffix="_is_${score_centering_is}"
      if [[ "${score_centering_is}" == off ]]; then
         algorithm=grpo
         is_suffix=""
      fi
      bash "${script_dir}/qwen38_27b_${algorithm}_dapo_math_17k_lora${variant_suffix}${is_suffix}.sh" "${first_run_args[@]}"
      first_run_args=()
   done
done

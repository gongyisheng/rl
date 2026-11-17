#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=2
# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# Use the NCCL bundled with this machine's PyTorch build.
export LD_LIBRARY_PATH="/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# for rerun the task
pkill sglang
ray stop --force
sleep 5 # Wait for processes to terminate gracefully
# Force kill any remaining processes.
# Note: `pkill -9 python` is broad and can be risky.
pkill -9 sglang
pkill -9 ray
pkill -9 python

set -ex

num_steps=100
learning_rate=1e-5
lora_rank=32
lora_alpha="${lora_rank}"
sglang_max_running_requests=128
score_centering_is=mis
run_name="qwen38_27b_score_centering_dapo_math_17k_lora_is_${score_centering_is}"
run_timestamp="$(date -u +%Y%m%dT%H%M%S%N)"
rollout_run_dir="/data/rollouts/score_centering/dapo_math_17k/${run_name}/${run_timestamp}"

ckpt_base_dir="/data/lora/score_centering/dapo_math_17k/${run_name}"
ckpt_run_dir="${ckpt_base_dir}/${run_timestamp}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MILES_ROOT=/root/miles-score-centering
MODEL_ARGS_LINE="$(python3 "${MILES_ROOT}/miles/utils/external_utils/model_args_utils.py" "qwen3.8-27B")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen3.8-27B/
   # --ref-load /root/models/Qwen3.8-27B/
   --megatron-to-hf-mode bridge
   --save "${ckpt_run_dir}"
   --save-interval 10
)

LORA_ARGS=(
   --lora-rank "${lora_rank}"
   --lora-alpha "${lora_alpha}"
   --lora-dropout 0.0
   --target-modules "all-linear"
   # Required by the Bridge hybrid-GDN LoRA recipe.
   --no-gradient-accumulation-fusion
   --update-weight-transfer-mode broadcast
)

ROLLOUT_ARGS=(
   --fully-async
   --save-debug-rollout-data "${rollout_run_dir}/rollout_{rollout_id}.pt"
   --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rollout-seed 42
   --rm-type deepscaler
   --custom-rm-path tasks.dapo_math_17k.rewards.deepscaler
   --reward-key reward
   --num-rollout "${num_steps}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   # Keep the experiment's 32K prompt-plus-response budget (also used by eval).
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --rollout-top-p 1.0
   --rollout-top-k -1
   --score-centering-top-k 128
   --over-sampling-batch-size 32
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 256
)

EVAL_ARGS=(
   --eval-reward-key accuracy
   --eval-interval 10
   --eval-prompt-data aime-2024 /root/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 32768
   --eval-temperature 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # Pack variable-length responses; keep the global batch size unchanged.
   --qkv-format thd
   --use-dynamic-batch-size
   --micro-batch-size 1
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 8192
   # Bound vocabulary-sized temporary tensors for long responses.
   --log-probs-chunk-size 512
   # Activation checkpointing for long responses.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --loss-type score_centering
   --score-centering-is "${score_centering_is}"
   --score-centering-mis-low 0.5
   --score-centering-mis-high 5.0
   --use-rollout-logprobs
   --disable-grpo-std-normalization
   # --use-kl-loss # enable with --ref-load
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --observe-training-entropy
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${learning_rate}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-project rl-score-centering-dapo-math-17k
   --wandb-group "${run_name}"
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # Qwen3.8 uses the Qwen3.5 conditional-generation architecture.
   --sglang-json-model-override-args '{"language_model_only": true}'
   --sglang-context-length 32768
   --sglang-lora-backend triton
   --sglang-max-lora-rank "${lora_rank}"
   --sglang-attention-backend fa3
   --sglang-enable-fp32-lm-head
   --sglang-mem-fraction-static 0.85
   # Avoid five recurrent-state slots per request for prefix caching.
   --sglang-disable-radix-cache
   --sglang-mamba-ssm-dtype float32
   --sglang-max-mamba-cache-size "${sglang_max_running_requests}"
   --sglang-max-running-requests "${sglang_max_running_requests}"
   --sglang-cuda-graph-max-bs-decode "${sglang_max_running_requests}"
   --sglang-chunked-prefill-size 2048
)

MISC_ARGS=(
   --seed 42
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --working-dir "${REPO_ROOT}" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/miles-score-centering:/root/Megatron-LM",
        "WANDB_CONSOLE": "off",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "60",
        "NVTE_FLASH_ATTN_V2": "0",
        "NVTE_FLASH_ATTN_V3": "1",
        "NVTE_FLASH_ATTN_V4": "0"
     }
   }' \
   -- python3 "${MILES_ROOT}/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --num-gpus-per-node $GPUS_PER_NODE \
   --rollout-num-gpus 1 \
   --calculate-per-token-loss \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "$@"

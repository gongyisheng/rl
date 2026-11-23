#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=8
# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

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
response_length=8192
eval_response_length=32768
run_name="qwen38_27b_score_centering_dapo_math_17k_lora_train_fp8_w8a16_is_mis"
run_timestamp="$(date -u +%Y%m%dT%H%M%S%N)"
rollout_run_dir="/data/rollouts/score_centering/dapo_math_17k/${run_name}/${run_timestamp}"

ckpt_base_dir="/data/lora/score_centering/dapo_math_17k/${run_name}"
ckpt_run_dir="${ckpt_base_dir}/${run_timestamp}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MILES_ROOT=/root/miles
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
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "all-linear"
   --update-weight-transfer-mode broadcast
)

ROLLOUT_ARGS=(
   --fully-async
   # Keep three TP2 rollout replicas busy without changing the training batch.
   --async-max-concurrent-samples 384
   --use-session-server v1
   --session-server-workers 8
   --tito-model qwen38small
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path experiments.score_centering.rollout.run_agent
   --custom-agent-function-mode inline
   --save-debug-rollout-data "${rollout_run_dir}/rollout_{rollout_id}.pt"
   --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --balance-data
   --rollout-seed 42
   --rm-type deepscaler
   --custom-rm-path tasks.dapo_math_17k.rewards.deepscaler
   --reward-key reward
   --num-rollout "${num_steps}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len "${response_length}"
   # Leave room for the prompt as well as the 32K evaluation response.
   --max-seq-len 65536
   --rollout-temperature 1
   --rollout-top-p 0.95
   --rollout-top-k 20
   --rollout-top-logprobs-num 128
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 256
)

EVAL_ARGS=(
   --eval-reward-key accuracy
   --eval-interval 10
   --eval-prompt-data aime-2024 /root/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len "${eval_response_length}"
   --eval-temperature 1
)

PERF_ARGS=(
   --train-env-vars '{"PYTORCH_ALLOC_CONF": "expandable_segments:True"}'
   --custom-megatron-init-path experiments.score_centering.train_precision.initialize
   --train-precision-log-dir "${rollout_run_dir}/te_debug"
   --custom-megatron-before-log-prob-hook-path experiments.score_centering.train_precision.before_log_prob
   --custom-megatron-before-train-step-hook-path experiments.score_centering.train_precision.before_train_step
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # Pack variable-length responses; keep the global batch size unchanged.
   --seq-length 65536
   --max-position-embeddings 65536
   --qkv-format thd
   --use-dynamic-batch-size
   --micro-batch-size 1
   --max-tokens-per-gpu 32768
   # Bound vocabulary-sized temporary tensors for long responses.
   --log-probs-chunk-size 512
   # Shared rounded weights leave room to avoid recomputing eight of 64 layers.
   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 56
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --loss-type score_centering
   --score-centering-is mis
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
   --wandb-team osmosis-staging
   --wandb-group "${run_name}"
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --sglang-kv-cache-dtype auto
   --rollout-num-gpus-per-engine 2
   # Qwen3.8 uses the Qwen3.5 conditional-generation architecture.
   --sglang-json-model-override-args '{"language_model_only": true}'
   --sglang-context-length 65536
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
   --sglang-lora-backend triton_v2
   --sglang-max-lora-rank 32
   --sglang-attention-backend fa3
   --sglang-enable-fp32-lm-head
   --sglang-mem-fraction-static 0.85
   # Avoid five recurrent-state slots per request for prefix caching.
   --sglang-disable-radix-cache
   --sglang-mamba-ssm-dtype float32
   --sglang-max-mamba-cache-size 128
   --sglang-max-running-requests 128
   --sglang-cuda-graph-max-bs-decode 128
   --sglang-chunked-prefill-size 8192
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
        "PYTHONPATH": "/root/miles:/root/Megatron-LM",
        "WANDB_CONSOLE": "off",
        "OMP_NUM_THREADS": "4",
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
   --actor-num-gpus-per-node 2 \
   --num-gpus-per-node $GPUS_PER_NODE \
   --rollout-num-gpus 6 \
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

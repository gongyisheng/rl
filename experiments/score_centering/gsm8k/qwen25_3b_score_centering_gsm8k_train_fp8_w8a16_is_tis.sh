#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export GPUS_PER_NODE=8
# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
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

score_centering_is=tis
precision_variant="train_fp8_w8a16"
run_name="qwen25_3b_score_centering_gsm8k_${precision_variant}_is_${score_centering_is}"
rollout_run_dir="/data/rollouts/score_centering/gsm8k/${run_name}/$(date -u +%Y%m%dT%H%M%S%N)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MILES_ROOT=/root/miles
MODEL_ARGS_LINE="$(python3 "${MILES_ROOT}/miles/utils/external_utils/model_args_utils.py" "qwen2.5-3B")" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen2.5-3B/
   # --ref-load /root/models/Qwen2.5-3B/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --fully-async
   --save-debug-rollout-data "${rollout_run_dir}/rollout_{rollout_id}.pt"
   --prompt-data /root/datasets/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 42
   --rm-type math
   --num-rollout 250
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --rollout-top-p 1.0
   --rollout-top-k -1
   --rollout-top-logprobs-num 128
   --over-sampling-batch-size 32
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --global-batch-size 256
)

EVAL_ARGS=(
   # --skip-eval-before-train
   --eval-interval 5
   --eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --custom-megatron-init-path experiments.score_centering.train_precision.initialize
   --custom-megatron-before-log-prob-hook-path experiments.score_centering.train_precision.before_log_prob
   --custom-megatron-before-train-step-hook-path experiments.score_centering.train_precision.before_train_step
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --qkv-format thd
   --use-dynamic-batch-size
   --max-tokens-per-gpu 40960

)

GRPO_ARGS=(
   --advantage-estimator grpo
   --loss-type score_centering
   --score-centering-is "${score_centering_is}"
   --score-centering-tis-clip 2.0
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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.ai/
   --wandb-project rl-score-centering-gsm8k
   --wandb-group "${run_name}"
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-attention-backend fa3
   --sglang-kv-cache-dtype fp8_e4m3
   # --sglang-mem-fraction-static 0.7
   --sglang-mem-fraction-static 0.3
   --sglang-max-running-requests 256
   --sglang-cuda-graph-max-bs-decode 256
   --sglang-chunked-prefill-size 8192

   # --sglang-enable-deterministic-inference
   # --deterministic-mode
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
        "PYTHONPATH": "/root/Megatron-LM",
        "WANDB_CONSOLE": "off",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "60",
        "NVTE_FLASH_ATTN_V2": "0",
        "NVTE_FLASH_ATTN_V3": "1",
        "NVTE_FLASH_ATTN_V4": "0"
     }
   }' \
   -- python3 /root/miles/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --num-gpus-per-node $GPUS_PER_NODE \
   --rollout-num-gpus 6 \
   --calculate-per-token-loss \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}"

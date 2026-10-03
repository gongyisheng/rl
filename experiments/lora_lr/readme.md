# LoRA learning-rate sweep

This experiment uses a [local configuration](run_lora_sm120.sh) copied and adapted from Qwen2.5 GSM8K; follow its [setup instructions](../qwen25_gsm8k/readme.md).

## Setup

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh

# env
TE_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
MAX_JOBS=4 CPATH="${TE_SITE_PACKAGES}/nvidia/cudnn/include:${TE_SITE_PACKAGES}/nvidia/nccl/include${CPATH:+:${CPATH}}" python -m pip install --upgrade --no-cache-dir --no-build-isolation --no-deps "transformer_engine==2.18.0" "transformer_engine_cu13==2.18.0" "transformer_engine_torch==2.18.0"

wandb login
```

## Run

```bash
bash experiments/lora_lr/run.sh
```

The sweep runs sequentially at `1e-6`, `2e-6`, `5e-6`, `1e-5`, `2e-5`, `5e-5`, and `1e-4`. Each run writes checkpoints and `train.log` to `/data/lora/lora_lr/lr-<learning-rate>/`.

Runs are logged to the W&B project `rl-lora-lr` with names `qwen2.5-3B-dapo-gsm8k-lr-<learning-rate>`. Miles uses the W&B group as the run name, with its random suffix disabled for this sweep.

Override the output root when needed:

```bash
OUTPUT_DIR=/data/lora/experiments/lora_lr bash experiments/lora_lr/run.sh
```

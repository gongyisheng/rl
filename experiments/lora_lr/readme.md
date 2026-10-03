# LoRA learning-rate sweep

Each learning-rate script contains the full training configuration copied and adapted from Qwen2.5 GSM8K; follow its [setup instructions](../qwen25_gsm8k/readme.md).

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

The sweep runs seven 250-step scripts sequentially, evaluating every 5 steps with 1 sample per evaluation prompt. Each script can also run independently, for example:

```bash
bash experiments/lora_lr/qwen25_3b_dapo_gsm8k_lr_1e-5.sh
```

Each run writes checkpoints to `/data/lora/lora_lr/lr_<learning-rate>/`.

Runs are logged to the W&B project `rl-lora-lr` with names `qwen2.5-3B-dapo-gsm8k-lr-<learning-rate>`. Miles uses the W&B group as the run name, with its random suffix disabled for this sweep.

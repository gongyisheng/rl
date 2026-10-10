# DAPO Math 17K

DAPO Math 17K training configurations. Set up a [configured container](../../README.md#environment), then run these commands from the repository root inside it.

## Dataset

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 aime-2024.jsonl --local-dir /root/datasets/aime-2024
```

The training dataset must contain `dapo-math-17k.jsonl` with `prompt` and `label` fields. Training applies the chat template, shuffles rollout prompts, balances the data, and uses the `deepscaler` reward model. Evaluation uses `/root/datasets/aime-2024/aime-2024.jsonl`.

## Setup

```bash
# codex
curl -fsSL https://chatgpt.com/codex/install.sh | sh

# env
TE_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
MAX_JOBS=4 CPATH="${TE_SITE_PACKAGES}/nvidia/cudnn/include:${TE_SITE_PACKAGES}/nvidia/nccl/include${CPATH:+:${CPATH}}" python -m pip install --upgrade --no-cache-dir --no-build-isolation --no-deps "transformer_engine==2.18.0" "transformer_engine_cu13==2.18.0" "transformer_engine_torch==2.18.0"

wandb login
```

## Configurations

Both configurations retain the GSM8K training hyperparameters and use rollout seed `42`. They evaluate AIME 2024 every five iterations with 16 samples per prompt and a maximum response length of 16,384; evaluation sampling uses Miles defaults.

### Qwen2.5 3B

```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/models/Qwen2.5-3B
```

LoRA fine-tuning:

```bash
bash tasks/dapo_math_17k/qwen25_3b_dapo_dapo_math_17k_lora.sh
```

Full fine-tuning (checkpoint saving disabled):

```bash
bash tasks/dapo_math_17k/qwen25_3b_dapo_dapo_math_17k.sh
```

LoRA checkpoints are saved under `/data/checkpoints/dapo_math_17k/qwen25_3b_dapo_dapo_math_17k_lora/` every five iterations. Every training rollout is saved under `/data/rollouts/dapo_math_17k/<script-name>/<UTC-timestamp>/rollout_{rollout_id}.pt`, where `<script-name>` excludes `.sh`.

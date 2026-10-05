# GSM8K

GSM8K training configurations. Set up a [configured container](../../README.md#environment), then run these commands from the repository root inside it.

## Dataset

```bash
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k
```

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

All configurations use rollout seed `42`.

### Qwen2.5 3B

```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/models/Qwen2.5-3B
```

LoRA fine-tuning:

```bash
bash tasks/gsm8k/qwen25_3b_dapo_gsm8k_lora.sh
```

Full fine-tuning (checkpoint saving disabled):

```bash
bash tasks/gsm8k/qwen25_3b_dapo_gsm8k.sh
```

Every training rollout is saved under `/data/rollouts/gsm8k/<script-name>/<UTC-timestamp>/rollout_{rollout_id}.pt`, where `<script-name>` excludes `.sh`. This is independent of checkpoint saving.

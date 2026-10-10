# DAPO Math 17K

Train Qwen2.5-3B with DAPO on DAPO Math 17K and evaluate on AIME 2024. Set up the [container](../../README.md#environment), then run these commands from the repository root inside it.

## Dataset

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 aime-2024.jsonl --local-dir /root/datasets/aime-2024
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

## Training

### Qwen2.5 3B

```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/models/Qwen2.5-3B
```

LoRA fine-tuning:

```bash
bash tasks/dapo_math_17k/qwen25_3b_dapo_dapo_math_17k_lora.sh
```

Full fine-tuning:

```bash
bash tasks/dapo_math_17k/qwen25_3b_dapo_dapo_math_17k.sh
```

Reward correct boxed answers.

Reward tests:

```bash
python -m pytest -q tasks/dapo_math_17k/test_rewards.py
```

# DAPO Math 17K

Train Qwen2.5-3B or Qwen3-4B with DAPO on DAPO Math 17K and evaluate on AIME 2024. Set up the [container](../../README.md#environment), then run these commands from the repository root inside it.

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

### Qwen3 4B

Use the downloaded thinking model at `/root/models/Qwen3-4B`:

```bash
bash tasks/dapo_math_17k/qwen3_4b_dapo_dapo_math_17k.sh
bash tasks/dapo_math_17k/qwen3_4b_dapo_dapo_math_17k_lora.sh
```

Run one variant at a time. Both use the model's default thinking chat template and Miles' built-in `deepscaler` reward; no custom reward is needed. Rollout retains the baseline temperature `1` and unrestricted sampling. Evaluation uses the model-recommended temperature `0.6`, top-p `0.95`, and top-k `20`. Response limits are `32768` tokens for rollout and `38912` for evaluation.

The Qwen2.5 baselines supply the remaining DAPO settings: 250 rollouts, 32 prompts × 8 responses per rollout, global batch size 256, and AIME 2024 evaluation every 5 rollouts with 16 responses per problem. Both Qwen3 variants use activation recomputation and 512-token log-probability chunks for long thinking responses. Full fine-tuning uses learning rate `1e-6` and saves no checkpoints. LoRA uses rank/alpha `32`, learning rate `1e-5`, and saves every 5 rollouts under a unique run directory in `/data/lora/dapo_math_17k/qwen3_4b_dapo_dapo_math_17k_lora/`.

Full training parks the paused actor on local NVMe under `/data/offload/dapo_math_17k/`, allowing SGLang to use memory fraction `0.7` for long responses without retaining a full actor copy in system RAM. It also streams the FP32 optimizer state from NVMe, so Adam's moments do not exhaust GPU memory on later updates. These are temporary offload files, not checkpoints. Scratch directories can remain after a run; remove them only after that run's processes stop. LoRA retains the baseline CPU offload and SGLang memory fraction `0.7`.

For a short validation run, retain global batch size 256 and 32 accepted prompt groups × 8 responses. Run two optimizer steps followed by one evaluation, using 2 responses per AIME problem (60 responses total):

```bash
mkdir -p logs/qwen3_4b_validation
WANDB_MODE=offline bash tasks/dapo_math_17k/qwen3_4b_dapo_dapo_math_17k.sh \
  --debug-exit-after-rollout 2 \
  --n-samples-per-eval-prompt 2 --skip-eval-before-train --eval-interval 2 \
  > logs/qwen3_4b_validation/full.log 2>&1

WANDB_MODE=offline bash tasks/dapo_math_17k/qwen3_4b_dapo_dapo_math_17k_lora.sh \
  --debug-exit-after-rollout 2 \
  --n-samples-per-eval-prompt 2 --skip-eval-before-train --eval-interval 2 \
  --save-interval 2 > logs/qwen3_4b_validation/lora.log 2>&1
```

These commands retain the production response limits and sampling settings. Check `rollout/truncated_ratio` and `eval/aime-2024-truncated_ratio` in the logs; each must be below `0.1`. Debug response dumps are saved under `/data/rollouts/dapo_math_17k/` for independent status checks. Both tests evaluate after training; the LoRA test also exercises checkpoint saving.

Validation on one RTX PRO 6000 Blackwell (96 GB), with global batch size 256:

| Variant | Rollout truncation | Post-training eval truncation |
| --- | --- | --- |
| Full, before enabling optimizer streaming | `6/256 = 0.0234375` | `1/60 = 0.0166667` |
| LoRA | `5/256 = 0.01953125` | `1/60 = 0.0166667` |
| Full with optimizer streaming, recorded rollout replayed | `6/256 = 0.0234375` (same recording) | `2/60 = 0.0333333` (fresh eval) |

The live rollout tests completed one valid optimizer step per variant. A separate full-training test then replayed the exact recorded 256-response batch for two optimizer updates, preserving its original rewards and policy versions, and generated a fresh evaluation after the second update. Both updates had finite gradients and logged actual NVMe optimizer streaming; the second backward pass used about 70 GiB of GPU memory. Replay validates optimizer memory and state continuity, not a new rollout sample.

All three jobs exited successfully. Saved response statuses independently matched the logged truncation metrics, and the LoRA adapter checkpoint was saved successfully. Logs and independent metric reports are under `logs/qwen3_4b_validation/`. These are short validation runs, not a complete 250-rollout training run.

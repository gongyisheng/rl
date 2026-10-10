# LoRA batch-size sweeps

Test how batch size, coupled with square-root learning-rate scaling, affects Qwen2.5-3B DAPO LoRA training.

## Tasks

- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/lora_batch_size/gsm8k/run.sh`
- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/lora_batch_size/dapo_math_17k/run.sh`

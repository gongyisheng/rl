# Length-control experiments

Compare the DAPO baseline with absolute and relative length penalties in Qwen2.5-3B LoRA training.

Checkpoints use `${OUTPUT_DIR:-/data/length_control}/<task>`; set `OUTPUT_DIR` to override the experiment root.

## Tasks

- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/length_control/gsm8k/run.sh`
- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/length_control/dapo_math_17k/run.sh`

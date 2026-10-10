# GRPO vs. DrGRPO

Compare Qwen2.5-3B full-parameter GRPO with DrGRPO to measure the effects of response-length and reward standard-deviation normalization.

Both variants use symmetric clipping at `0.2` and omit DAPO dynamic sampling and per-token loss aggregation. DrGRPO additionally disables reward standard-deviation normalization and uses the Miles DrGRPO loss reducer.

## Tasks

- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/drgrpo/gsm8k/run.sh`
- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/drgrpo/dapo_math_17k/run.sh`

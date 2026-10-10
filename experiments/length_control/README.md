# Length-control experiments

Test whether absolute (`v1`) or relative (`v2`) length penalties shorten correct responses while retaining math accuracy in Qwen2.5-3B DAPO LoRA training.

Compare an unpenalized control with both penalty versions at `alpha` values `1e-3`, `1e-2`, and `1e-1`. All runs disable reward standard-deviation normalization, including the control; evaluation reports correctness without the length penalty.

Checkpoints use `/data/length_control/<task>` with a separate directory per variant. Miles creates these directories.

## Tasks

- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/length_control/gsm8k/run.sh`
- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/length_control/dapo_math_17k/run.sh`

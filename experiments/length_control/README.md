# Length-control experiments

Test whether absolute (`v1`) or relative (`v2`) length penalties shorten correct responses while retaining math accuracy in DAPO LoRA training. GSM8K uses Qwen2.5-3B; DAPO Math 17K uses Qwen3-4B-Base with 16384-token rollout responses, a 32768-token context limit, and 32768-token evaluation responses.

Compare an unpenalized control with both penalty versions at `alpha` values `1e-3`, `1e-2`, and `1e-1`. All runs disable reward standard-deviation normalization, including the control; evaluation reports correctness without the length penalty. DAPO Math 17K evaluates every 10 rollouts.

Checkpoints use `/data/length_control/<task>` with a separate directory per variant. DAPO Math 17K saves LoRA checkpoints every 10 rollouts. Miles creates these directories.

## Tasks

- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/length_control/gsm8k/run.sh`
- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/length_control/dapo_math_17k/run.sh`

# Full RL learning-rate sweep

This is a full-RL learning-rate sweep for Qwen2.5 3B on GSM8K. Follow the existing [GSM8K setup instructions](../../tasks/gsm8k/README.md).

## Run

```bash
nohup bash experiments/lr/run.sh > logs/lr.log 2>&1 &
```

The sequential sweep runs `1e-7`, `2e-7`, `5e-7`, `1e-6`, `2e-6`, `5e-6`, and `1e-5`. Each run trains for 250 rollouts, evaluates every 5 steps, uses 1 sample per evaluation prompt, and uses rollout seed `42`. Checkpoint saving is disabled.

Every training rollout is saved under `/data/rollouts/lr/qwen25_3b_dapo_gsm8k_lr_<learning-rate>/<UTC-timestamp>/rollout_{rollout_id}.pt`; evaluation samples use `rollout_eval_<id>.pt` in the same directory. This is independent of checkpoint saving.

Run one learning rate directly:

```bash
bash experiments/lr/qwen25_3b_dapo_gsm8k_lr_1e-6.sh
```

Runs log to W&B project `rl-lr` with group `qwen2.5-3B-dapo-gsm8k-lr-<learning-rate>` and no random suffix.

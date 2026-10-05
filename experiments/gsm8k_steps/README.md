# Qwen2.5 3B DAPO GSM8K runs

Use the [baseline setup instructions](../../tasks/gsm8k/README.md), then launch both runs sequentially (LoRA followed by full fine-tuning):

```bash
bash experiments/gsm8k_steps/run.sh
```

To launch them individually:

```bash
bash experiments/gsm8k_steps/qwen25_3b_dapo_gsm8k_lora.sh
bash experiments/gsm8k_steps/qwen25_3b_dapo_gsm8k.sh
```

Both runs train for 1,000 steps, evaluate every 5 steps with 1 sample per prompt, and use rollout seed `42`. They log to W&B project `rl-gsm8k-steps` with separate groups. The LoRA run uses learning rate `1e-5` and saves checkpoints to `/data/gsm8k_steps/qwen25_3b_dapo_gsm8k_lora/checkpoints`. The full-finetune run uses `1e-6` with checkpoint saving disabled.

Every training rollout is saved under `/data/rollouts/gsm8k_steps/<script-name>/<UTC-timestamp>/rollout_{rollout_id}.pt`, where `<script-name>` excludes `.sh`; evaluation samples use `rollout_eval_<id>.pt` in the same directory. This is independent of checkpoint saving.

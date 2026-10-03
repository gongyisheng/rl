# Qwen2.5 3B DAPO GSM8K runs

Use the [baseline setup instructions](../../tasks/gsm8k/README.md), then launch both runs sequentially (LoRA followed by full fine-tuning):

```bash
bash experiments/gsm8k_steps/run.sh
```

To launch them individually:

```bash
bash experiments/gsm8k_steps/qwen25_3b_dapo_gsm8k_lora.sh
bash experiments/gsm8k_steps/qwen25_3b_dapo_gsm8k_full.sh
```

Both runs train for 1,000 steps, evaluate every 5 steps with 1 sample per prompt, and log to W&B project `rl-gsm8k-steps`. The LoRA run uses learning rate `1e-5`; the full-finetune run uses `1e-6`. They use their respective groups and checkpoint directories: `/data/gsm8k_steps/qwen25_3b_dapo_gsm8k_lora/checkpoints` and `/data/gsm8k_steps/qwen25_3b_dapo_gsm8k_full/checkpoints`.

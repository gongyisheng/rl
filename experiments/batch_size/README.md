# GSM8K full fine-tuning batch-size sweep

Five Qwen2.5-3B DAPO full fine-tuning runs, based on the [1,000-step configuration](../gsm8k_steps/README.md). Follow the [GSM8K setup instructions](../../tasks/gsm8k/README.md), then run inside the configured container:

```bash
bash experiments/batch_size/run.sh
```

Runs execute sequentially in ascending batch size and stop on failure. Each script also runs independently:

```bash
bash experiments/batch_size/qwen25_3b_dapo_gsm8k_bs_1024.sh
```

Learning rates use square-root scaling: `lr = 1e-6 * sqrt(global_batch_size / 256)`.

| Global batch (responses) | Rollout / oversampling batch (prompts) | Learning rate |
| --- | --- | --- |
| 256 | 32 | 1e-6 |
| 512 | 64 | 1.414213562e-6 |
| 1024 | 128 | 2e-6 |
| 2048 | 256 | 2.828427125e-6 |
| 4096 | 512 | 4e-6 |

Each rollout collects eight responses per accepted prompt, giving exactly one optimizer update per rollout. Dynamic reward filtering can require extra generated responses. All runs use 1,000 updates, evaluate every 5 updates, and use one response per evaluation prompt. Dynamic microbatching keeps the 4,096-token GPU limit.

Checkpoint saving is disabled. W&B uses project `rl-batch-size` and groups/run names `qwen25_3b_dapo_gsm8k_bs_<batch>_lr_<lr>`, without random suffixes.

Compare W&B `eval/gsm8k` against `eval/step`. For sample efficiency, compare against completed updates times global batch size. The runs use equal update counts but different accepted-response budgets: 256,000 through 4,096,000. This sweep changes batch size and learning rate together.

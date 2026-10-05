# GSM8K LoRA batch-size sweep

Five Qwen2.5-3B DAPO LoRA runs, based on the [1,000-step configuration](../gsm8k_steps/README.md). Follow the [GSM8K setup instructions](../../tasks/gsm8k/README.md), then run inside the configured container:

```bash
bash experiments/lora_batch_size/run.sh
```

Runs execute sequentially in ascending batch size and stop on failure. Before each run, its script stops SGLang and Ray, waits five seconds, then force-stops remaining SGLang, Ray, and Python processes. The broad Python termination assumes a dedicated training environment. Each script also runs independently:

```bash
bash experiments/lora_batch_size/qwen25_3b_dapo_gsm8k_lora_bs_1024.sh
```

Learning rates use square-root scaling: `lr = 1e-5 * sqrt(global_batch_size / 256)`.

| Global batch (responses) | Rollout / oversampling batch (prompts) | Learning rate |
| --- | --- | --- |
| 256 | 32 | 1e-5 |
| 512 | 64 | 1.414213562e-5 |
| 1024 | 128 | 2e-5 |
| 2048 | 256 | 2.828427125e-5 |
| 4096 | 512 | 4e-5 |

Each rollout collects eight responses per accepted prompt, giving exactly one optimizer update per rollout. Dynamic reward filtering can require extra generated responses. All runs use 1,000 updates, evaluate and save every 5 updates, and use one response per evaluation prompt. LoRA rank/alpha remain 32/32; dynamic microbatching keeps the 4,096-token GPU limit.

Checkpoints go to `/data/lora_batch_size/batch_size_<batch>/checkpoints`; `OUTPUT_DIR` overrides `/data/lora_batch_size`. W&B uses project `rl-lora-batch-size` and groups/run names `qwen25_3b_dapo_gsm8k_lora_bs_<batch>_lr_<lr>`, without random suffixes.

Compare W&B `eval/gsm8k` against `eval/step`. For sample efficiency, compare against completed updates times global batch size. The runs use equal update counts but different accepted-response budgets: 256,000 through 4,096,000. This sweep changes batch size and learning rate together.

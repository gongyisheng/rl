# GSM8K LoRA GRPO vs. DrGRPO

This experiment compares GRPO and DrGRPO on GSM8K with Qwen2.5-3B LoRA. Follow the [GSM8K setup instructions](../../tasks/gsm8k/README.md), then run inside the configured container:

```bash
bash experiments/drgrpo/run.sh
```

The launcher runs GRPO first and DrGRPO second, stopping if either fails. Each script performs the required SGLang/Ray/Python cleanup before it starts, so use a dedicated training environment. The scripts can also run independently:

```bash
bash experiments/drgrpo/qwen25_3b_grpo_gsm8k_lora.sh
bash experiments/drgrpo/qwen25_3b_drgrpo_gsm8k_lora.sh
```

Both runs use Qwen2.5-3B with LoRA rank/alpha 32/32, dropout 0, learning rate `1e-5`, 1,000 rollouts, 32 prompts per rollout batch, and eight responses per prompt for a global batch of 256 responses. They use a 1,024-token response limit, evaluate and save every five updates, and evaluate greedily with one response per prompt. Training seed `1234` and rollout seed `42` are shared to keep initialization and data ordering aligned. Both average across responses, use symmetric PPO clipping of `0.2`, and omit DAPO dynamic reward filtering and oversampling.

There is no `--calculate-per-token-loss`: GRPO divides each token sum by its valid response length, while DrGRPO divides by the fixed constant `1000`. DrGRPO also disables GRPO reward standard-deviation normalization with `--disable-grpo-std-normalization`; it does not enable post-advantage normalization. Its custom reducer is the upstream [`examples/experimental/DrGRPO/custom_reducer.py`](https://github.com/radixark/miles/blob/main/examples/experimental/DrGRPO/custom_reducer.py) function `examples.experimental.DrGRPO.custom_reducer.get_pg_loss_reducer`.

`MILES_ROOT` defaults to `/root/miles` and can be overridden. Ray workers receive `PYTHONPATH=${MILES_ROOT}:/root/Megatron-LM`, making the upstream `examples.experimental.DrGRPO` module importable from any launch directory. Checkpoints are saved under `${OUTPUT_DIR:-/data/drgrpo}/grpo/checkpoints` and `${OUTPUT_DIR:-/data/drgrpo}/drgrpo/checkpoints`. W&B uses project `rl-drgrpo`; the groups/run names are `qwen25_3b_grpo_gsm8k_lora` and `qwen25_3b_drgrpo_gsm8k_lora`.

Every training rollout is saved under `/data/rollouts/drgrpo/qwen25_3b_<grpo-or-drgrpo>_gsm8k_lora/<UTC-timestamp>/samples/rollout_{rollout_id}.pt`; evaluation samples use `rollout_eval_<id>.pt` in the same directory. This is independent of checkpoint saving.

Compare `eval/gsm8k` against `eval/step`: the optimizer, model, data, response budget, sampling setup, and seeds match, while DrGRPO changes only reward standard-deviation normalization and the fixed-divisor policy-gradient reducer.

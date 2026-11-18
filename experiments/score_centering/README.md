# Score centering

Test score centering on **DAPO Math 17K with Qwen3.8-27B**, evaluating on AIME 2024. Start from the [task LoRA baseline](../../tasks/dapo_math_17k/qwen3_4b_base_dapo_dapo_math_17k_lora.sh), with rank = alpha = 32, dropout 0, learning rate `1e-5`, and 100 steps per variant. The runner crosses a GRPO control with score centering disabled and score-centering importance weights `none`, `tis` (upper clip 2), and `mis` (retain ratios in [0.5, 5]) with these precision variants, for **12 sequential runs**:

| Variant | Rollout weights / KV cache | Training |
| --- | --- | --- |
| `lora` | BF16 / BF16 (`auto`) | BF16 base and LoRA |
| `lora_rollout_fp8_w8a8` | Official `Qwen/Qwen3.8-27B-FP8` / FP8 E4M3 | BF16 base and LoRA |
| `lora_train_fp8_w8a16` | BF16 / BF16 (`auto`) | E4M3 fake-quantized base weights, BF16 LoRA and GEMMs |

Use **eight H200 GPUs: two for training (TP2 with sequence parallelism), six for inference (three TP2 replicas)**, with `train_async.py --fully-async`. Training uses packed THD batches and the BF16 checkpoint at `/root/models/Qwen3.8-27B`. Native FP8 rollout loads `/root/models/Qwen3.8-27B-FP8`. All variants use the **Triton v2 LoRA backend**, FlashAttention 3, and disabled prefix caching; GDN recurrent states remain FP32.

Inference admits 128 requests per replica, with 8192-token prefill chunks and 384 in-flight samples across the fully async producer. Training packs up to 32768 tokens per microbatch, with full activation recomputation in four-layer segments. W8A16 trainers use expandable CUDA allocations. Their policy-loss CPU offload remains available for individual sequences above 32K tokens; ordinary batches at the 8K response cap do not trigger it. The global training batch remains 256 responses. Short benchmarks favored 256 serving slots, but the long fully async workload exhausted the KV pool and repeatedly retracted requests; 128 slots also halves recurrent-state allocation, leaving more memory for KV cache.

Each rollout has 32 prompts × 8 responses, matching the global batch size of 256. Sampling uses temperature `1`, top-p `0.95`, and top-k `20`. Score centering records the post-filter sampling support and its log probabilities, with a 128-candidate storage limit, group-mean subtraction, and no standard-deviation normalization. Miles replays this support when scoring the training policy. The GRPO controls use the task's policy loss with clip bounds `0.2` / `0.28`, retaining the same reward centering without standard-deviation normalization. `is_none` still enables score centering; it disables only importance weighting. Training responses are capped at **8192 tokens**, and evaluation responses at **32768 tokens**. This tests whether the model learns to solve problems within a smaller training token budget; initial training truncation above 50% is acceptable and is not a reason to raise the cap. The 65536-token context limit leaves room for evaluation prompts. Track reward, response length, and truncation together to distinguish improved efficiency from incomplete answers. The baseline reward gate is retained. AIME evaluation runs before training and every 10 optimizer updates, with 16 samples per problem; LoRA checkpoints are also saved every 10 updates. Every variant has a distinct W&B group in [`osmosis-staging/rl-score-centering-dapo-math-17k`](https://wandb.ai/osmosis-staging/rl-score-centering-dapo-math-17k) and checkpoint directory under `/data/lora/score_centering/dapo_math_17k/`.

The scripts use `/root/miles` (installed revision `8e419cc739`), which includes score centering and language-only LoRA mappings for attention, GDN, and MLP projections. W8A16 training requires Transformer Engine debug features and `nvidia-dlframework-inspect`; it keeps BF16 weight storage and GEMMs. Benchmark scripts and logs belong under `/tmp/score_centering_tuning/`.

All variants use **TITO v1** through eight Miles session-server workers, the `qwen38small` tokenizer family, and matching `qwen3` reasoning / `qwen3_coder` tool parsers. The single-turn [rollout callback](rollout.py) sends the dataset's messages through the session endpoint; the server owns chat templating, generated token IDs, and sampling probabilities. SGLang automatically caps evaluation completions to the remaining 65536-token context.

From the RL repository root, with checkpoints and datasets under `/root/models` and `/root/datasets`:

```bash
bash experiments/score_centering/dapo_math_17k/run.sh
```

Arguments passed to `run.sh` apply only to its first variant, allowing a restart with an existing `--wandb-run-id` and `--skip-eval-before-train` when no optimizer update has occurred and the initial evaluation is already recorded.

Miles uses W&B shared mode. A reused run with existing history must also have been created in shared mode; writing an initial evaluation in ordinary online mode first makes W&B reject subsequent shared uploads with HTTP 409. A W&B run showing `running` is not sufficient to verify logging: compare its latest `train/step` and `rollout/step` with the local log.

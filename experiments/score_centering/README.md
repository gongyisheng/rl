# Score centering

Test score centering on **DAPO Math 17K with Qwen3.8-27B**, evaluating on AIME 2024. Start from the [task LoRA baseline](../../tasks/dapo_math_17k/qwen3_4b_base_dapo_dapo_math_17k_lora.sh), with rank = alpha = 32, dropout 0, learning rate `1e-5`, and 100 steps per variant. The runner crosses importance weights `none`, `tis` (upper clip 2), and `mis` (retain ratios in [0.5, 5]) with these precision variants:

| Variant | Rollout weights / KV cache | Training |
| --- | --- | --- |
| `lora` | BF16 / BF16 | BF16 base and LoRA |
| `lora_rollout_fp8_w8a8` | Official `Qwen/Qwen3.8-27B-FP8` / FP8 E4M3 | BF16 base and LoRA |
| `lora_train_fp8_w8a16` | BF16 / FP8 E4M3 | E4M3 fake-quantized base weights, BF16 LoRA and GEMMs |

Use **two H200 GPUs: one training GPU and one inference GPU**, with `train_async.py --fully-async`. Training uses packed THD batches, tensor parallelism 1, and the BF16 checkpoint at `/root/models/Qwen3.8-27B`. Native FP8 rollout loads `/root/models/Qwen3.8-27B-FP8`. All variants use the **Triton LoRA backend**, FlashAttention 3, 128 concurrent requests, and disabled prefix caching; GDN recurrent states remain FP32.

Each rollout has 32 prompts × 8 responses, matching the global batch size of 256. Score centering uses rollout log probabilities and top-128 probabilities with group-mean subtraction and no standard-deviation normalization. Training responses are capped at 16384 tokens; evaluation at 32768, with a 32768-token context. The baseline reward gate, evaluation interval, and LoRA checkpoint interval are retained. Every variant has a distinct W&B group and checkpoint directory under `/data/lora/score_centering/dapo_math_17k/`.

The scripts use `/root/miles-score-centering`, combining Miles `2806267d` with the score-centering feature from `286092387`. This retains the current language-only LoRA mappings for attention, GDN, and MLP projections. W8A16 training requires Transformer Engine debug features and `nvidia-dlframework-inspect`; it keeps BF16 weight storage and GEMMs.

From the RL repository root, with checkpoints and datasets under `/root/models` and `/root/datasets`:

```bash
bash experiments/score_centering/dapo_math_17k/run.sh
```

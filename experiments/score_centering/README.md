# Score centering

Test [Miles score centering](https://github.com/radixark/miles/tree/main/examples/infra_features/score_centering) with unweighted (`none`), truncated (`tis`, upper clip 2), and masked (`mis`, retain ratios in [0.5, 5]) importance weights. The loss corrects off-policy score drift using the rollout sampler's top 128 token probabilities.

Start from each task's full-parameter DAPO script: Qwen3-4B-Base for DAPO Math 17K and Qwen2.5-3B for GSM8K. Replace the PPO loss with score centering, use recorded rollout log probabilities, and disable reward standard-deviation normalization while retaining group-mean subtraction. PPO clipping does not apply to this loss. Dynamic sampling, per-token loss aggregation, learning rate, batch sizes, reward functions, and evaluation settings follow the task baselines, including DAPO Math 17K's answer-only training gate.

Each task's runner crosses these precision configurations with `none`, `tis`, and `mis`, executing all 12 runs sequentially:

| Script variant | Rollout | Training |
| --- | --- | --- |
| Original scripts (no precision suffix) | BF16 | BF16 |
| `rollout_fp8_w8a16` | FP8 E4M3 weights, BF16 activations/GEMMs | BF16 |
| `rollout_fp8_w8a8` | FP8 weights and activations | BF16 |
| `train_fp8_w8a16` | BF16 | FP8 E4M3 weight fake quantization, BF16 activations/GEMMs |

The rollout variants use SGLang's `flash_rl` loader to quantize the original BF16 checkpoint and requantize live weight updates. W8A16 uses the local Qwen model extensions to dequantize FP8 weights before BF16 GEMMs; it tests weight-only quantization numerics, not Marlin throughput. The native Marlin packing is incompatible with FlashRL's raw-FP8 update buffers. W8A8 uses native FP8 GEMMs and requires a GPU with native FP8 support, such as Hopper or Blackwell.

Prequantized versions exist: [Qwen3-4B-Base-FP8](https://huggingface.co/Sohailhosseini/Qwen3-4B-Base-FP8) and [Qwen2.5-3B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen2.5-3B-FP8-dynamic). Both configure W8A8 with per-channel FP8 weights and dynamic per-token FP8 activations in `compressed-tensors` format. They do not replace the W8A16 execution hooks, and the installed Miles exporter does not implement live updates for that FP8 format. These runs therefore use the original BF16 checkpoints for training and online rollout quantization.

Training W8A16 uses Transformer Engine's `FakeQuant` feature with per-tensor E4M3 scaling on projection weights in both forward and input-gradient GEMMs. Activations, gradients, master parameters, embeddings, norms, and the output head retain their baseline precision. It is a numerical experiment, not a compressed optimizer or a native W8A16 training kernel. The hooks initialize the feature before TE layers are built and name decoder projections before their first forward pass. This requires Transformer Engine's debug features and `nvidia-dlframework-inspect`.

Runs use separate W&B groups and rollout directories and save no checkpoints. DAPO Math 17K retains disk offload with separate scratch directories per variant. Miles automatically enables the FP32 rollout LM head for score centering. Use Miles/SGLang with the score-centering support in the linked guide and FlashRL's Qwen2/Qwen3 live-reload support.

The scripts use one H200 (140 GiB) per run, with these SGLang settings across all precision and importance-weight variants:

| Task | Static memory fraction | Maximum running requests | Decode graph maximum batch | Prefill chunk tokens |
| --- | --- | --- | --- | --- |
| DAPO Math 17K | 0.8 | 64 | 64 | 8192 |
| GSM8K | 0.3 | 256 | 256 | 8192 |

DAPO Math 17K uses larger prefill chunks and more concurrent requests for its long responses. GSM8K reserves less memory for the rollout server to leave room for its resident trainer. The scripts also select PyTorch's bundled NCCL under `/opt/sglang`; the system NCCL on this machine lacks the `ncclCommResume` symbol required by PyTorch.

Measured BF16 serving throughput on this host, in generated tokens per second:

| Task | Requests × prompt tokens × generated tokens | Before | Tuned |
| --- | --- | --- | --- |
| DAPO Math 17K | 256 × 512 × 128 | 4417 | 5900 |
| DAPO Math 17K | 64 × 8192 × 64 | 367 | 387 |
| GSM8K | 256 × 512 × 128 | 7688 | 7640 |
| GSM8K | 256 × 2048 × 256 | 5656 | 5674 |

These are medians of three repetitions after warmup, using synthetic token inputs, the FP32 LM head, and top-128 log probabilities, including HTTP response transfer. Each server used one GPU and measurements ran sequentially after both servers were ready. GSM8K throughput stayed within 1% while its static memory reservation fell by about 14 GiB. These are inference measurements; full RL runs and maximum-length Math generations have not been tested.

SGLang `3a93821` also needs [this patch](sglang_fp8_activation_dtype.patch), applied on this machine. Its Qwen attention/MLP paths otherwise cast BF16 activations directly to FP8 weight storage without a scale, breaking both rollout FP8 variants. The patch preserves activation dtypes until the projection's quantization method handles them. Apply it to another checkout of that revision with `git -C /path/to/sglang apply /path/to/rl/experiments/score_centering/sglang_fp8_activation_dtype.patch`.

Validation passed for all 26 shell scripts (`bash -n`), six precision tests, and CUDA graph capture plus top-128-log-probability generation for both models under each rollout FP8 configuration. The two native Qwen regression cases also reject the unpatched SGLang code.

The original model checkpoints are cached under `/root/models/{Qwen3-4B-Base,Qwen2.5-3B}`. Training and evaluation data are cached under `/root/datasets/{dapo-math-17k,gsm8k,aime-2024}`. Model tensor indexes, tokenizers, and all dataset rows were checked after downloading.

- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/score_centering/dapo_math_17k/run.sh`
- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/score_centering/gsm8k/run.sh`

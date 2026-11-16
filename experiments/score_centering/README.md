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

Training W8A16 uses Transformer Engine's `FakeQuant` feature with per-tensor E4M3 scaling on projection weights in both forward and input-gradient GEMMs. Activations, gradients, master parameters, embeddings, norms, and the output head retain their baseline precision. It is a numerical experiment, not a compressed optimizer or a native W8A16 training kernel. The hooks initialize the feature before TE layers are built and name decoder projections before their first forward pass. This requires Transformer Engine's debug features and `nvidia-dlframework-inspect`.

Runs use separate W&B groups and rollout directories and save no checkpoints. DAPO Math 17K retains disk offload with separate scratch directories per variant. Miles automatically enables the FP32 rollout LM head for score centering. Use Miles/SGLang with the score-centering support in the linked guide and FlashRL's Qwen2/Qwen3 live-reload support.

- [DAPO Math 17K](../../tasks/dapo_math_17k/README.md): `bash experiments/score_centering/dapo_math_17k/run.sh`
- [GSM8K](../../tasks/gsm8k/README.md): `bash experiments/score_centering/gsm8k/run.sh`

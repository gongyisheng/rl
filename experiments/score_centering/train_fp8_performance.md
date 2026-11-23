# W8A16 training performance

Measured on 2026-11-23 with the full Qwen3.8-27B model on **two NVLink-connected H200 GPUs**, matching the TP2 trainer in the production eight-GPU layout. This measures training separately from the [rollout benchmark](fp8_performance.md).

## Score-centering result

| Configuration | Median optimizer step | Training tokens/s | Peak allocated memory per GPU |
| --- | ---: | ---: | ---: |
| Original W8A16, full recomputation in groups of four | 25.06s | 5,230 | 97.22 GiB |
| Shared rounded weights, recompute 56 of 64 layers | 24.08s | 5,443 | 114.56 GiB |

Training throughput improved **4.1%**. Each step processes the same **16 sequences × 8,192 tokens = 131,072 tokens**, accumulated across four 32K-token microbatches. The three measured step times were 25.09 / 25.04 / 25.06 seconds before tuning and 24.11 / 24.08 / 24.08 seconds after tuning.

The six W8A16 score-centering scripts use the selected settings: `--recompute-method block --recompute-num-layers 56`, retaining TP2, sequence parallelism, and the 32768-token packing budget. Production global batch size stays at **256 responses**. Both train-only and combined training/rollout FP8 variants use the same training optimization.

## GRPO result

| Configuration | Median optimizer step | Training tokens/s | Peak allocated memory per GPU |
| --- | ---: | ---: | ---: |
| Original W8A16, uniform recomputation | 6.40s | 5,121 | 103.80 GiB |
| Shared rounded weights, same recomputation | 6.41s | 5,109 | 81.13 GiB |
| Shared rounded weights, recompute 56 layers | 7.79s | 4,204 | 125.36 GiB |
| Shared rounded weights, recompute 60 layers | 6.25s | 5,239 | 107.68 GiB |

The GRPO control uses **60 recomputed layers**, improving throughput **2.3%** over its original configuration. Retaining activations for eight layers was slower for this loss, so GRPO retains only four. These comparisons use four 8,192-token sequences per optimizer step, packed into one 32K microbatch. The selected configuration took 6.27 / 6.25 / 6.24 seconds across its three timed updates. GRPO's existing BF16 rollout and KV cache are retained.

## What changed

W8A16 here rounds frozen base weights to E4M3 with Transformer Engine's per-tensor scaling, then executes **BF16 GEMMs with BF16 activations and LoRA**. It does not use FP8 activation GEMMs or provide their tensor-core speedup.

Transformer Engine's original `FakeQuant` path creates separate BF16 rounded weight copies for forward and input-gradient GEMMs. `FrozenWeightFakeQuant` shares one rounded copy. It delegates rounding to the original TE feature, retains the original BF16 parameters, and leaves state-dict names and LoRA weights unchanged. The cache checks tensor identity, storage, and version, clears at the precision-step boundary, and rejects trainable base weights.

With the original recomputation settings, sharing weights reduced peak allocated memory from **97.08 to 74.40 GiB per GPU**, saving **22.68 GiB** in the initial 32K-token test. Sharing weights alone left steady-state throughput essentially unchanged: TE already retains weight workspaces between microbatches. The selected score-centering configuration spends that memory saving on retaining activations for eight layers, avoiding their recomputation. Its final peak memory is consequently higher than the original baseline, as shown above.

## Other configurations tested

These screening runs used three measured updates after warm-up. The rows with 16K or 64K packing process different numbers of microbatches; tokens/s normalizes the amount of work.

| Score-centering configuration | Tokens/s | Peak allocated memory per GPU | Decision |
| --- | ---: | ---: | --- |
| Original, 32K packing, uniform recomputation | 5,220 | 97.08 GiB | Baseline |
| Shared weight prototype, same recomputation | 5,239 | 74.40 GiB | Memory saving; no material speed gain alone |
| Shared weights, disable sequence parallelism | 5,298 | 77.21 GiB | Small gain; retain sequence parallelism |
| Shared weights, 64K packing, uniform recomputation | 5,265 | 98.91 GiB | Less than 1% gain; retain 32K packing |
| Shared weights, 16K packing, recompute 40 layers | 4,311 | 120.00 GiB | Slower |
| Shared weights, 32K packing, recompute 48 layers | — | Out of memory | Rejected |
| Shared weights, 32K packing, recompute 56 layers | 5,441 | 114.43 GiB | Selected; confirmed with accumulated steps above |

## Method and validation

The harness loads the real BF16 checkpoint at `/root/models/Qwen3.8-27B`, attaches rank = alpha = 32 LoRA, and calls the installed Miles `train_one_step` implementation. It exercises packed THD batches, GDN and attention, loss computation, backward, gradient accumulation, clipping, and Adam updates. The loss uses fixed synthetic replay data with 256 prompt tokens per sequence, 128 candidate-storage slots, 20 valid candidates, and balanced positive/negative advantages. Score-centering timings use `is_none`; the precision and recomputation changes are shared with `tis` and `mis`.

Each trial excludes model loading and one warm-up update, then measures three optimizer updates. Timings synchronize both GPUs and use the slower rank; reported throughput is total input tokens divided by elapsed time. Memory is the maximum PyTorch allocated memory across ranks, not total reserved/device memory. Profiling is outside the timed trials.

All **13 GPU precision tests** passed, covering all six projection types with and without recomputation, input and LoRA gradients, preservation of original BF16 weights and parameter identities, cache reuse, refresh after weight/storage changes, and rejection of trainable base weights. The full-model baseline and optimized score-centering runs matched on their initial forward-loss statistics and completed all updates with finite gradient norms. This is not a bitwise comparison of optimizer trajectories or a convergence evaluation.

All seven modified training scripts passed `bash -n`, safe shell expansion, Ray environment JSON validation, and the installed Miles/Megatron argument parser. The checks confirm the selected recomputation settings, BF16 training GEMMs, 256-response global batch, and each variant's existing rollout precision without launching the production scripts.

These measurements isolate trainer compute. They exclude rollout generation, reward calculation, adapter transfer, and evaluation; **end-to-end asynchronous RL speedup has not been measured**. The production eight-GPU layout, rollout precisions, sampling settings, and global training batch are retained.

## Local artifacts

The harness, baseline precision YAML, raw timings, logs, CUDA traces, and test output are under `/tmp/score_centering_train_tuning/`. The accumulated-step results are `baseline_16x8192_uniform4.results.json` and `optimized_16x8192_block56.results.json`; GRPO results are `grpo_baseline_4x8192_uniform4.results.json` and `grpo_optimized_4x8192_block60.results.json`. The `train_bench.py`, `final_benchmarks.py`, and `grpo_final_benchmarks.py` helpers reproduce the isolated comparisons without invoking the training scripts' process cleanup.

Environment: Miles `8e419cc739`, PyTorch `2.14.1+cu130`, Transformer Engine `2.17.0+miles`. BF16 checkpoint revision: `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.

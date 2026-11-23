# FP8 rollout performance

Measured on 2026-11-23 on **two NVLink-connected H200 GPUs**, representing one TP2 replica in the production eight-GPU layout (two training GPUs and three TP2 rollout replicas).

| Workload | Original tokens/s | Tuned tokens/s | Throughput gain |
| --- | ---: | ---: | ---: |
| Score centering, 1,024 output tokens | 6,079 | 7,495 | +23.3% |
| GRPO, 1,024 output tokens | 6,123 | 7,507 | +22.6% |
| Score centering, 8,192 output tokens | 5,714 | 6,857 | +20.0% |

The 8K workload generated **2,097,152 output tokens** in 367.04s before tuning and 305.85s after tuning. These measurements cover serving on one replica. Full RL throughput also depends on training, reward calculation, and synchronization.

## Configuration

All four `rollout_fp8_w8a8` variants and all four `train_fp8_w8a16` variants share these rollout settings:

- Explicit `deep_gemm` FP8 W8A8 linear kernels and `fp8_e4m3` KV cache, with `SGLANG_FORCE_FP8_MARLIN=0` and DeepGEMM enabled.
- 256 serving requests, recurrent-state slots, and decode CUDA graph capacity per replica, up from 128. One shell variable controls all three and the production async producer limit of 768 samples across three replicas.
- Full response-budget admission: initial and minimum new-token ratios of 1, with the estimation clip derived from the 32K evaluation response cap. This prevents the default 4K estimate and its decay from overcommitting long responses.
- On-demand DeepGEMM compilation during warm-up (`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`). The timed baseline also used this setting; no startup-speed claim is made.

The 8192-token prefill chunk, 0.85 static memory fraction, FA3 attention, Triton v2 LoRA, FP32 recurrent states, 256-response global training batch, and learning/loss settings are unchanged. The `rollout_fp8_w8a8` variants retain BF16 training; `train_fp8_w8a16` retains its E4M3 weight fake quantization, BF16 LoRA and GEMMs, precision hooks, and expandable training allocations. The official rollout checkpoint retains its unquantized exclusions, including sensitive GDN projections and the output head; LoRA computation remains BF16. W8A8 describes the quantized base linear operations.

The W8A16 variants now reuse the benchmarked FP8 serving configuration. The timings above compare the original and tuned FP8 serving settings; the former BF16 rollout configuration of the W8A16 variants and their full training workload were not benchmarked here.

## Method and validation

The same official `Qwen/Qwen3.8-27B-FP8` checkpoint and rank = alpha = 32 adapter were used before and after tuning. The adapter has random A and zero B weights, matching the initial RL policy while executing the LoRA kernels. Each trial submitted the first 32 DAPO Math 17K prompts with eight responses each: **256 requests and 49,008 prompt tokens**. Temperature was 1, top-p 0.95, and top-k 20. EOS stopping was disabled to fix the output work at exactly 1K or 8K per request.

Measurements use the native SGLang HTTP endpoint with pretokenized prompts, sampled-token log probabilities and sampling masks. Score-centering requests also collect the realized support probabilities; `none`, `tis`, and `mis` share that inference path. Model loading, graph capture, warm-up, and profiling are excluded. The 1K rows aggregate two timed repetitions; the 8K row uses one long trial per configuration. Reported throughput is total output tokens divided by total wall time, including prefill and response transfer. TITO session handling, reward calculation, optimizer updates, and weight synchronization are outside this benchmark.

Both 8K runs completed with finite sampled-token log probabilities and no KV retractions. The tuned engine allocated **5,545,027 FP8 KV tokens**. A separate 16-second probe submitted 256 requests with 32K response budgets: it admitted at most **168** concurrent requests and queued up to **88**, with no reported retractions. The probe requests were then aborted; it was an admission check, not a completed AIME evaluation.

Decode CUDA traces contain Hopper `sm90_fp8_gemm` kernels, FP8 E4M3 activation quantizers, and FA3 `float_e4m3_t` attention kernels, with no Marlin kernels. Every modified shell script passed `bash -n`; all eight expanded launch configurations and Ray environment JSON objects passed the installed Miles/Megatron argument parser without launching training, including the W8A16 precision hooks. The full eight-GPU asynchronous RL workload remains to be measured on that hardware.

The score-centering storage limit remains **128 candidates**. FlashInfer retains ties at the top-k boundary: the long baseline observed support sets as large as 27 despite top-k 20. Reducing storage to 20 would reject valid rollout samples.

## Local artifacts

Scripts, exact launch commands, raw timing JSON, server logs, CUDA traces, validation output, and diffs against the task LoRA baseline are in `/tmp/score_centering_tuning/`. Key files are `baseline_{1024,grpo_1024,8192}.results.json`, `tuned256_{1024,grpo_1024,8192}.results.json`, `eval_admission.results.json`, and `profiles/`. The `serve.py` and `bench.py` helpers reproduce an isolated TP2 benchmark without running the training-script cleanup.

The BF16 and FP8 checkpoints are downloaded under `/root/models/`; DAPO Math 17K and AIME 2024 are under `/root/datasets/`. FP8 checkpoint revision: `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. Environment: Miles `8e419cc739`, SGLang `0.5.24.dev96+gfd01715`, PyTorch `2.14.1+cu130`, FlashInfer `0.7.2`.

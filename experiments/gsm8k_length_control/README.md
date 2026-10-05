# GSM8K length-control experiment

## Run the LoRA sweeps

Use the [GSM8K setup](../../tasks/gsm8k/README.md), then run from the repository root inside the configured container:

```bash
bash experiments/gsm8k_length_control/run.sh
```

The runner completes v1 and then v2, sweeping `alpha=1e-3`, `1e-2`, and `1e-1` for each reward (six sequential runs). Before each run, its script stops SGLang and Ray, waits five seconds, then force-stops remaining SGLang, Ray, and Python processes. The broad Python termination assumes a dedicated training environment. Each script also runs independently, for example:

```bash
bash experiments/gsm8k_length_control/qwen25_3b_dapo_gsm8k_v1_lora_alpha_1e-3.sh
bash experiments/gsm8k_length_control/qwen25_3b_dapo_gsm8k_v2_lora_alpha_1e-3.sh
```

Each run uses Qwen2.5-3B on one GPU, LoRA rank 32 with LoRA alpha 32, learning rate `1e-5`, and 100 rollout steps. A rollout contains 32 prompts with 8 responses each (global batch size 256). Training and evaluation responses are capped at 1,024 tokens. Evaluation and checkpoint saving run every 5 steps, with one evaluation response per prompt.

GRPO standard-deviation normalization is disabled in all six runs so the length-penalty coefficient controls the strength of the reward differences. Mean subtraction remains enabled. Evaluation reports math correctness.

Each script loads its matching `reward_<version>_alpha_<alpha>.yaml` config. V1 uses `max_length=1024`; v2 uses relative lengths among correct responses to the same prompt. The reward alpha is separate from LoRA alpha.

- W&B project: `rl-gsm8k-length-control`.
- W&B groups: `qwen25_3b_dapo_gsm8k_<version>_lora_alpha_<alpha>`, with the random suffix disabled.
- Checkpoints: `/data/lora/gsm8k_length_control/<version>_alpha_<alpha>/`. Set `OUTPUT_DIR` to override the base directory.
- Miles checkout: `/root/miles`. Set `MILES_ROOT` to override it; set `CUDA_VISIBLE_DEVICES` to select the GPU.

The scripts resolve the repository and config paths relative to their own location and pass the repository through Ray's `--working-dir` option. The runtime environment uses a literal JSON block with `PYTHONPATH=/root/Megatron-LM`.

## Reward functions

Both versions reward shorter correct solutions and assign zero to incorrect answers. They use generated response tokens, excluding the prompt, with `alpha=0.2` by default.

| Version | Reward for a correct response |
| --- | --- |
| v1 | `1 - alpha * min(response_length / max_length, 1)`; default `max_length=1024` |
| v2 | `1 - alpha * sigmoid((response_length - mean_correct) / (std_correct + 1e-7))` |

V2 computes the mean and population standard deviation over **correct responses to the same prompt**. A single correct response, or equal correct lengths, receives `1 - alpha / 2`. A group with no correct response receives all zeros. Both functions require `0 <= alpha < 1`.

V2 follows [Training Language Models to Reason Efficiently](https://arxiv.org/html/2502.04463v3) and its [reward implementation](https://github.com/Zanette-Labs/efficient-reasoning/blob/main/reward_server/math_server.py).

## Python usage

Run from the repository root. Each call supplies one prompt's responses in matching order:

```python
from experiments.gsm8k_length_control.rewards import reward_v1, reward_v2

correctness = [1, 0, 1]
response_lengths = [256, 128, 512]

print(reward_v1(correctness, response_lengths))  # [0.95, 0.0, 0.9]
print(reward_v2(correctness, response_lengths))  # approximately [0.9462, 0.0, 0.8538]
```

## Miles integration

Training uses a custom group reward and Miles' standard dynamic-sampling filter:

```bash
--rm-type math \
--custom-rm-path experiments.gsm8k_length_control.rewards.v1 \
--group-rm \
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
--eval-function-path experiments.gsm8k_length_control.evaluation.MathEvalRolloutFn
```

For v2, set `--custom-rm-path` to `experiments.gsm8k_length_control.rewards.v2`.

Each async reward function receives one completed prompt group, grades responses using the same `grade_answer_verl` checker as `--rm-type math`, and returns one length-shaped reward per response. Miles assigns these values to `sample.reward` before the standard filter retains groups with reward standard deviation greater than `1e-8`. All-correct groups can therefore contribute when their shaped rewards differ. Correctness is also stored in `sample.metadata["length_control_accuracy"]`.

The functions use `sample.response_length` without retokenizing. The evaluation adapter uses Miles' class-based rollout API and a copy of the evaluation arguments with group RM disabled and the built-in math scorer selected. Evaluation reports ordinary math accuracy; training keeps the custom group reward. This avoids the standard evaluator's `--group-rm` restriction without changing Miles.

Pass this repository's root through `ray job submit --working-dir` so Ray can load the reward hooks while the literal runtime environment sets `PYTHONPATH` to `/root/Megatron-LM`. Ray [adds the working directory to its Python import path](https://github.com/ray-project/ray/blob/master/python/ray/_private/runtime_env/working_dir.py).

To override defaults, pass a YAML file through Miles' `--custom-config-path`:

```yaml
length_penalty_alpha: 0.1
length_penalty_max_length: 1024  # v1 only
```

The hooks leave advantage normalization to Miles. For a penalty-strength comparison, add `--disable-grpo-std-normalization` to every variant and its `alpha=0` control, and leave `--normalize-advantages` off. With standard GRPO reward standardization, the penalty coefficient cancels in all-correct groups (apart from numerical stabilizers).

## Tests

The tests run with the Python standard library and mock Miles dependencies. Actual training and evaluation require the configured Miles environment:

```bash
python3 -m unittest discover -s experiments/gsm8k_length_control -v
```

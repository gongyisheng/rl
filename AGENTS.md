# Repository Structure

- `experiments/<experiment>/`: experiment variants and parameter sweeps.
- `tasks/<task>/`: one folder per standard task, such as `search-r1` or `retool`.

# Experiment Scripts

- Before creating an experiment script, read the matching `tasks/<task>/` training script and `README.md`. Use the same model, algorithm, and full/LoRA mode as the canonical starting configuration; when the experiment changes one of these, use the closest relevant task baseline. Start from that task script, retaining its argument groups, launch layout, and unchanged settings rather than copying another experiment.
- Change only experiment parameters, required dependent settings, and variant-specific names, output paths, and W&B identities. Define every swept value once and reuse it consistently.
- Diff each new script against its task baseline and document every intentional configuration difference in the experiment `README.md`.
- Validate every new or modified `.sh` script with `bash -n` without launching training.

# Rules

- Name training scripts `<model>_<algo>_<task>[_<variant>...].sh`, using lowercase fields separated by underscores. Full-parameter training is implicit: omit `full` from script and W&B names. Include `lora` in both for LoRA runs and put parameter values last, e.g. `lora_lr_1e-5`.
- In each experiment's `run.sh`, loop over parameter values, using nested loops for multiple sweep axes, construct script names, and run them sequentially. Resolve script paths relative to `run.sh`.
- Add a usage comment after each `run.sh` shebang, e.g. `# Usage: nohup bash experiments/lora_lr/run.sh > logs/lora_lr.log 2>&1 &`.
- Do not save checkpoints for full-parameter RL runs; omit checkpoint-saving arguments and checkpoint output-directory setup.
- Use distinct W&B groups for each variant and distinct checkpoint directories when checkpoint saving is enabled.
- Keep each task or experiment's `README.md` launch commands and configuration summary in sync with its scripts. Each experiment `README.md` must cover its hypothesis, task-baseline script paths, a setup table of changed and swept values, launch commands, and results or notes when available.
- In every training script, after the shebang use this order: environment exports, process cleanup, `set -ex`, argument and path setup, then launch:

  ```bash
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export GPUS_PER_NODE=1
  # will prevent ray from buffering stdout/stderr
  export PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

  # for rerun the task
  pkill sglang
  ray stop --force
  sleep 5 # Wait for processes to terminate gracefully
  # Force kill any remaining processes.
  # Note: `pkill -9 python` is broad and can be risky.
  pkill -9 sglang
  pkill -9 ray
  pkill -9 python

  set -ex
  ```

  After `set -ex`, define experiment parameters such as `learning_rate`; run variables and paths (including `rollout_run_dir`, `SCRIPT_DIR`, `REPO_ROOT`, and `MILES_ROOT` as needed); and `ckpt_base_dir` and `ckpt_run_dir` when checkpoint saving is enabled. Load `MODEL_ARGS_LINE` and `MODEL_ARGS`, and set up the remaining argument arrays. Keep all path resolution, parameter initialization, and directory creation in this section, including `mkdir -p "${ckpt_run_dir}" || exit 1` when checkpoint saving is enabled, using values appropriate to the script.

  Do not enable `-e` until after cleanup because a missing process is expected. Each sequential `run.sh` child performs this cleanup before it begins; `pkill -9 python` assumes a dedicated training environment.

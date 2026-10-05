# Repository Structure

- `experiments/<experiment>/`: experiment variants and parameter sweeps.
- `tasks/<task>/`: one folder per standard task, such as `search-r1` or `retool`.

# Rules

- Name training scripts `<model>_<algo>_<task>[_<variant>...].sh`, using lowercase fields separated by underscores. Put modes and parameter values last, e.g. `lora`, `full`, or `lr_1e-5`.
- In each experiment's `run.sh`, loop over parameter values, construct script names, and run them sequentially. Resolve script paths relative to `run.sh`.
- Add a usage comment after each `run.sh` shebang, e.g. `# Usage: nohup bash experiments/lora_lr/run.sh > logs/lora_lr.log 2>&1 &`.
- Do not save checkpoints for full-parameter RL runs; omit checkpoint-saving arguments and checkpoint output-directory setup.
- Use distinct W&B groups for each variant and distinct checkpoint directories when checkpoint saving is enabled.
- Keep each task or experiment's `README.md` launch commands and configuration summary in sync with its scripts.

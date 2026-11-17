"""Miles hooks for W8A16 base weights with BF16 LoRA adapters and GEMMs."""

from pathlib import Path
import re


CONFIG_PATH = Path(__file__).with_name("train_fp8_w8a16.yaml")


def initialize(args):
    """Enable TE's fake quantization before constructing the Bridge model."""
    import nvdlfw_inspect.api as debug_api
    import transformer_engine

    if getattr(args, "fp8", None) or getattr(args, "fp4", None):
        raise ValueError("W8A16 requires BF16 GEMMs; do not enable an FP8/FP4 GEMM recipe")
    if getattr(args, "fp8_param_gather", False):
        raise ValueError("W8A16 fake quantization requires high-precision base weights")
    debug_api.initialize(
        config_file=str(CONFIG_PATH),
        feature_dirs=[str(Path(transformer_engine.__file__).parent / "debug" / "features")],
        log_dir=str(Path(args.save_debug_rollout_data).parent / "te_debug"),
    )


def prepare_model(model):
    """Name language projections, including TE base layers inside LoRA wrappers."""
    import nvdlfw_inspect.api as debug_api
    import torch
    import yaml
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    config = yaml.safe_load(CONFIG_PATH.read_text())["weight_only_fp8"]
    pattern = re.compile(config["layers"]["layer_name_regex_pattern"])
    for chunk in model if isinstance(model, list) else [model]:
        if getattr(chunk, "_score_centering_precision_ready", False):
            continue
        debug_api.infer_and_assign_layer_names(chunk)
        matched = [
            (name, module)
            for name, module in chunk.named_modules()
            if pattern.search(name) and not hasattr(module, "to_wrap")
        ]
        if not matched:
            raise ValueError("No language projections matched the W8A16 precision config")
        for name, module in matched:
            if not isinstance(module, TransformerEngineBaseModule):
                raise TypeError(f"W8A16 requires a Transformer Engine base projection: {name}")
            if module.weight.dtype != torch.bfloat16:
                raise ValueError(f"W8A16 requires BF16 base weights: {name}")
        chunk._score_centering_precision_ready = True


def before_log_prob(args, model, store_prefix):
    import nvdlfw_inspect.api as debug_api

    prepare_model(model)
    debug_api.step()


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import nvdlfw_inspect.api as debug_api

    prepare_model(model)
    debug_api.step()

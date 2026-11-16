"""Miles hooks for weight-only FP8 numerics with BF16 GEMMs and master weights."""

from pathlib import Path
import re


CONFIG_PATH = Path(__file__).with_name("train_fp8_w8a16.yaml")


def initialize(args):
    """Enable TE's supported fake-quantization API before constructing TE layers."""
    import nvdlfw_inspect.api as debug_api
    import transformer_engine

    if getattr(args, "fp8", None) or getattr(args, "fp4", None):
        raise ValueError("W8A16 requires BF16 GEMMs; do not enable an FP8/FP4 GEMM recipe")
    if getattr(args, "fp8_param_gather", False):
        raise ValueError("W8A16 fake quantization requires high-precision master weights")
    debug_api.initialize(
        config_file=str(CONFIG_PATH),
        feature_dirs=[str(Path(transformer_engine.__file__).parent / "debug" / "features")],
        log_dir=str(Path(args.save_debug_rollout_data).parent / "te_debug"),
    )


def prepare_model(model):
    """Name decoder projections and reject a silently unmatched precision config."""
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
        matched = [module for module in chunk.modules() if pattern.search(getattr(module, "name", ""))]
        if not matched:
            raise ValueError("No decoder projections matched the W8A16 precision config")
        for module in matched:
            if not isinstance(module, TransformerEngineBaseModule):
                raise TypeError(f"W8A16 requires a Transformer Engine projection: {module.name}")
            if module.weight.dtype != torch.bfloat16:
                raise ValueError(f"W8A16 requires BF16 master weights: {module.name}")
        chunk._score_centering_precision_ready = True


def before_log_prob(args, model, store_prefix):
    import nvdlfw_inspect.api as debug_api

    prepare_model(model)
    debug_api.step()


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import nvdlfw_inspect.api as debug_api

    prepare_model(model)
    debug_api.step()

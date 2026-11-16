"""Check published checkpoint bits and partial live updates through native Qwen loaders."""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def sglang_context(tmp_path):
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    if not torch.cuda.is_available():
        pytest.skip("SGLang checkpoint integration requires CUDA")
    if torch.distributed.is_initialized():
        pytest.skip("Requires an isolated distributed process")
    original_dtype = torch.get_default_dtype()
    reset_context()
    torch.set_default_dtype(torch.bfloat16)
    server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
    publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path}/distributed",
    )
    initialize_model_parallel()
    initialize_dp_attention(server)
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        reset_context()
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("model_name", ["Qwen3-4B-Base-FP8", "Qwen2.5-3B-FP8-dynamic"])
def test_checkpoint_bits_and_partial_updates(sglang_context, model_name):
    from safetensors import safe_open
    from transformers import AutoConfig
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.model_loader.loader import DefaultModelLoader
    from experiments.score_centering.sglang_fp8.models import (
        Qwen2W8A16ForCausalLM,
        Qwen3Fp8ForCausalLM,
    )

    checkpoint = Path("/root/models") / model_name
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip(f"Download {model_name} to run this integration check")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config.num_hidden_layers = 1
    quant_config = CompressedTensorsConfig.from_config(config.quantization_config)
    model_class = Qwen3Fp8ForCausalLM if config.model_type == "qwen3" else Qwen2W8A16ForCausalLM
    with torch.device("cuda"):
        model = model_class(config, quant_config)
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cuda") as f:
        weights = {
            name: f.get_tensor(name)
            for name in f.keys()
            if not name.startswith("model.layers.") or name.startswith("model.layers.0.")
        }
    device = torch.device("cuda")
    DefaultModelLoader.load_weights_and_postprocess(model, weights.items(), device)
    qkv = model.model.layers[0].self_attn.qkv_proj
    prefix = "model.layers.0.self_attn."
    projections = ("q_proj", "k_proj", "v_proj")
    expected_weights = torch.cat([weights[prefix + p + ".weight"] for p in projections])
    # SGLang promotes the published BF16 scales to FP32 without changing values.
    expected_scales = torch.cat([weights[prefix + p + ".weight_scale"] for p in projections]).float()
    # Direct checkpoint loading must retain all published bits and scales.
    torch.testing.assert_close(
        qkv.weight.t().contiguous().view(torch.uint8), expected_weights.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(qkv.weight_scale, expected_scales, rtol=0, atol=0)
    pointers = {name: param.data_ptr() for name, param in model.named_parameters()}
    q_rows = weights[prefix + "q_proj.weight"].shape[0]
    for value in (0.125, -0.0625):
        update = torch.full_like(weights[prefix + "q_proj.weight"], value, dtype=torch.bfloat16)
        DefaultModelLoader.restore_weights_before_loading(model, device)
        model.load_weights([(prefix + "q_proj.weight", update)])
        DefaultModelLoader.postprocess_weights(model, device)
        assert {name: param.data_ptr() for name, param in model.named_parameters()} == pointers
        torch.testing.assert_close(
            qkv.weight.t()[:q_rows].float(),
            torch.full_like(update, 448 if value > 0 else -448).float(),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            qkv.weight_scale[:q_rows], torch.full_like(expected_scales[:q_rows], abs(value) / 448),
            rtol=0, atol=0,
        )
        # Updating Q must leave K/V values and scales byte-for-byte unchanged.
        torch.testing.assert_close(
            qkv.weight.t()[q_rows:].contiguous().view(torch.uint8), expected_weights[q_rows:].view(torch.uint8),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(qkv.weight_scale[q_rows:], expected_scales[q_rows:], rtol=0, atol=0)

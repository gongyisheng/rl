"""Numerical checks for rollout and training precision; no training job is launched."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.score_centering import train_precision


@pytest.mark.parametrize("component", ["mlp", "attention"])
def test_native_qwen_preserves_activations_before_fp8_projection(monkeypatch, component):
    from sglang.srt.models import qwen2, qwen3

    # Exercise the native callers, since a direct linear-method test cannot
    # detect an unscaled FP8 cast before the projection receives its input.
    x = torch.tensor([[0.101, -0.303, 1.234, 1234]], dtype=torch.bfloat16)
    seen = []

    class Projection:
        weight = torch.empty(4, 4, dtype=torch.float8_e4m3fn)

        def __call__(self, value, **kwargs):
            seen.append(value)
            return value, None

    monkeypatch.setattr(qwen2, "should_force_bfloat16_dense_tensor_math", lambda: False)
    monkeypatch.setattr(qwen2, "is_true_on_policy_enabled", lambda: False)
    monkeypatch.setattr(qwen3, "should_force_bfloat16_dense_tensor_math", lambda: False)
    if component == "mlp":
        model = SimpleNamespace(
            gate_up_proj=Projection(),
            down_proj=Projection(),
            dual_gemm=SimpleNamespace(can_run=lambda *args: False),
            act_fn=lambda value: value,
        )
        actual = qwen2.Qwen2MLP.forward(model, x)
    else:
        def prepare(positions, hidden_states):
            value, _ = model.qkv_proj(hidden_states)
            return value, value, value

        model = SimpleNamespace(
            qkv_proj=Projection(),
            o_proj=Projection(),
            use_fused_qk_norm_mrope=False,
            forward_prepare_native=prepare,
            attn=lambda q, k, v, batch, **kwargs: q,
        )
        monkeypatch.setattr(qwen3, "_is_npu", False)
        actual = qwen3.Qwen3Attention.forward(model, None, x, None)
    assert seen and all(value.dtype == torch.bfloat16 for value in seen)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_published_fp8_tensors_pass_through_unchanged():
    from experiments.score_centering.sglang_fp8.models import encode_updated_weights

    name = "model.layers.0.self_attn.q_proj.weight"
    weight = torch.tensor([[1.0, -2.0]], dtype=torch.float8_e4m3fn)
    # An arbitrary published scale must be retained rather than recalibrated.
    scale = torch.tensor([[0.0073]], dtype=torch.float32)
    source = [(name, weight), (name.removesuffix(".weight") + ".weight_scale", scale)]
    result = list(encode_updated_weights(source, {name}))
    assert len(result) == len(source)
    assert all(
        n == expected_name and tensor is expected
        for (n, tensor), (expected_name, expected) in zip(result, source)
    )


def test_live_update_scales_are_per_channel_and_zero_rows_stay_finite():
    from experiments.score_centering.sglang_fp8.models import encode_updated_weights

    name = "model.layers.0.mlp.down_proj.weight"
    source = torch.tensor([[0, 0, 0], [1, -2, 3], [100, -200, 300]], dtype=torch.bfloat16)
    bias = torch.zeros(3, dtype=torch.bfloat16)
    result = dict(encode_updated_weights([(name, source), ("bias", bias)], {name}))
    scale = result[name.removesuffix(".weight") + ".weight_scale"]
    assert scale.shape == (3, 1) and scale.dtype == torch.float32
    assert result["bias"] is bias
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert result[name].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(result[name].float() * scale, source.float(), rtol=0.05, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine requires CUDA")
@pytest.mark.parametrize("recompute", [False, True])
def test_training_weight_only_forward_and_backward(tmp_path, recompute):
    import nvdlfw_inspect.api as debug_api
    import transformer_engine.pytorch as te
    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    train_precision.initialize(SimpleNamespace(save_debug_rollout_data=str(tmp_path / "rollout.pt")))
    try:
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.layers = nn.ModuleList([nn.Module()])
        model.decoder.layers[0].self_attention = nn.Module()
        linear = te.Linear(128, 128, params_dtype=torch.bfloat16, device="cuda", bias=False)
        model.decoder.layers[0].self_attention.linear_qkv = linear
        train_precision.before_train_step(None, 0, 0, [model], None, None)
        master = linear.weight.detach().clone()
        torch.manual_seed(43)
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn_like(x)
        actual = checkpoint(linear, x, use_reentrant=True) if recompute else linear(x)
        actual.backward(grad)

        # Independent per-tensor E4M3 quantize/dequantize reference. The input
        # gradient uses the quantized weight; the weight gradient uses BF16 x.
        scale = 448 / master.float().abs().max()
        weight = ((master.float() * scale).to(torch.float8_e4m3fn).float() / scale).to(master.dtype)
        expected = F.linear(x.detach(), weight)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
        torch.testing.assert_close(x.grad, grad @ weight, rtol=0.008, atol=0.001)
        torch.testing.assert_close(linear.weight.grad, grad.t() @ x.detach(), rtol=0.008, atol=0.001)
        torch.testing.assert_close(linear.weight, master, rtol=0, atol=0)
        assert linear.weight.dtype == x.dtype == actual.dtype == torch.bfloat16
    finally:
        debug_api.end_debug()
        TEDebugState._reset()

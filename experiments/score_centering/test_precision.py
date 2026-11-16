"""Numerical checks for the W8A16 adapters; no training job is launched."""

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


@pytest.mark.parametrize("per_channel", [False, True])
def test_rollout_uses_quantized_weights_and_unquantized_activations(per_channel):
    from experiments.score_centering.sglang_w8a16.models import Fp8W8A16LinearMethod

    generator = torch.Generator().manual_seed(8)
    source = torch.randn(32, 16, generator=generator)
    scale = source.abs().amax(dim=0, keepdim=True) / 448 if per_channel else source.abs().max() / 448
    quantized = (source / scale).to(torch.float8_e4m3fn)
    layer = SimpleNamespace(weight=quantized, weight_scale=scale)
    x = torch.randn(7, 32, generator=generator).to(torch.bfloat16)
    bias = torch.randn(16, generator=generator).to(torch.bfloat16)
    method = object.__new__(Fp8W8A16LinearMethod)
    actual = method.apply(layer, x, bias)
    # Evaluate the represented FP8 matrix in double precision, then apply the
    # specified BF16 rounding before GEMM. No activation quantization is allowed.
    represented = (quantized.double() * scale.double()).to(torch.bfloat16)
    expected = F.linear(x, represented.t(), bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="16-bit activations"):
        method.apply(layer, x.float())


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

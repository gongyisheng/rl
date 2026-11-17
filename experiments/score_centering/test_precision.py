"""Check W8A16 base numerics and BF16 LoRA gradients without launching training."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.score_centering import train_precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine requires CUDA")
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize(
    "projection",
    ["self_attention.linear_qkv", "self_attention.linear_proj", "self_attention.in_proj",
     "self_attention.out_proj", "mlp.linear_fc1", "mlp.linear_fc2"],
)
def test_lora_weight_only_forward_and_backward(tmp_path, recompute, projection):
    import nvdlfw_inspect.api as debug_api
    import transformer_engine.pytorch as te
    from megatron.bridge.peft.lora_layers import LoRALinear
    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    train_precision.initialize(SimpleNamespace(train_precision_log_dir=str(tmp_path / "te_debug")))
    try:
        model = nn.Module()
        model.language_model = nn.Module()
        model.language_model.decoder = nn.Module()
        layer = nn.Module()
        model.language_model.decoder.layers = nn.ModuleList([layer])
        parent, leaf = projection.split(".")
        setattr(layer, parent, nn.Module())
        base = te.Linear(128, 128, params_dtype=torch.bfloat16, device="cuda", bias=False)
        base.requires_grad_(False)
        # Rank = alpha = 32, hence the adapter scale is one.
        adapter = nn.Sequential(OrderedDict(
            linear_in=nn.Linear(128, 32, dtype=torch.bfloat16, device="cuda", bias=False),
            linear_out=nn.Linear(32, 128, dtype=torch.bfloat16, device="cuda", bias=False),
        ))
        adapter.dim = adapter.alpha = 32
        linear = LoRALinear(base, adapter)
        setattr(getattr(layer, parent), leaf, linear)
        pointers = {name: param.data_ptr() for name, param in model.named_parameters()}
        state_names = set(model.state_dict())
        train_precision.before_train_step(None, 0, 0, [model], None, None)
        train_precision.before_log_prob(None, [model], None)
        assert {name: param.data_ptr() for name, param in model.named_parameters()} == pointers
        assert set(model.state_dict()) == state_names
        master = base.weight.detach().clone()
        torch.manual_seed(43)
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn_like(x)
        actual = checkpoint(linear, x, use_reentrant=True) if recompute else linear(x)
        actual.backward(grad)

        # Independent E4M3 base + unquantized low-rank branch. Check gradients of
        # both adapter factors as well as the input (which traverses both paths).
        scale = 448 / master.float().abs().max()
        weight = ((master.float() * scale).to(torch.float8_e4m3fn).float() / scale).to(master.dtype)
        ref_x = x.detach().clone().requires_grad_()
        ref_a = adapter.linear_in.weight.detach().clone().requires_grad_()
        ref_b = adapter.linear_out.weight.detach().clone().requires_grad_()
        expected = F.linear(ref_x, weight) + F.linear(F.linear(ref_x, ref_a), ref_b)
        expected.backward(grad)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=0.008, atol=0.001)
        torch.testing.assert_close(adapter.linear_in.weight.grad, ref_a.grad, rtol=0.008, atol=0.001)
        torch.testing.assert_close(adapter.linear_out.weight.grad, ref_b.grad, rtol=0.008, atol=0.001)
        torch.testing.assert_close(base.weight, master, rtol=0, atol=0)
        assert base.weight.grad is None
        assert x.dtype == actual.dtype == base.weight.dtype == torch.bfloat16
        assert all(param.dtype == torch.bfloat16 for param in adapter.parameters())
    finally:
        debug_api.end_debug()
        TEDebugState._reset()

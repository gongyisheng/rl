"""Retain FlashRL's FP8 storage/reload contract and run GEMMs in activation dtype."""

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM


class Fp8W8A16LinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        if self.block_quant or self.is_checkpoint_fp8_serialized:
            raise ValueError("This W8A16 experiment requires online, non-block FP8 quantization")
        # FlashRL updates raw FP8 buffers, not Marlin's packed int32 weights.
        self.use_marlin = False

    def apply(self, layer, x, bias=None):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("W8A16 requires 16-bit activations")
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("W8A16 requires E4M3 weights from the FlashRL loader")
        weight = (layer.weight.float() * layer.weight_scale.float()).to(x.dtype)
        return F.linear(x, weight.t(), bias)


def install_w8a16(model):
    count = 0
    for module in model.model.modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, Fp8LinearMethod):
            module.quant_method = Fp8W8A16LinearMethod(method.quant_config)
            count += 1
    if not count:
        raise ValueError("No FP8 decoder projections found for W8A16 rollout")


class Qwen2W8A16ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        install_w8a16(self)


class Qwen3W8A16ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        install_w8a16(self)


EntryClass = [Qwen2W8A16ForCausalLM, Qwen3W8A16ForCausalLM]

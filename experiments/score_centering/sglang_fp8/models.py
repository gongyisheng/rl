"""Load published per-channel FP8 tensors directly; encode live trainer updates."""

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM


class CheckpointFp8LinearMethod(CompressedTensorsLinearMethod):
    def __init__(self, quant_config, *, weight_only):
        super().__init__(quant_config)
        self.weight_only = weight_only

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_checkpoint_fp8_ready", False):
            return
        # Keep the original Parameter subclasses and loader metadata. A view
        # transpose preserves both the checkpoint bits and CUDA graph addresses.
        layer.weight.data = layer.weight.data.t()
        layer.input_scale = None
        layer._checkpoint_fp8_ready = True

    def restore_weights_before_loading(self, layer):
        if getattr(layer, "_checkpoint_fp8_ready", False):
            layer.weight.data = layer.weight.data.t()
            layer._checkpoint_fp8_ready = False

    def apply(self, layer, x, bias=None):
        if not self.weight_only:
            return super().apply(layer, x, bias)
        if not isinstance(x, torch.Tensor) or x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("W8A16 requires 16-bit activations")
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("W8A16 requires E4M3 checkpoint weights")
        weight = (layer.weight.float() * layer.weight_scale.float().t()).to(x.dtype)
        return F.linear(x, weight.t(), bias)


def encode_updated_weights(weights, quantized_weight_names):
    """Checkpoint FP8 tensors pass through unchanged, including their scales.

    Miles sends BF16 master weights at synchronization (including its initial
    sync). Encode only those new projection weights in the checkpoint's format.
    """
    for name, weight in weights:
        if name not in quantized_weight_names or weight.dtype == torch.float8_e4m3fn:
            yield name, weight
            continue
        if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Unsupported live FP8 weight: {name}, {weight.dtype}, {weight.shape}")
        source = weight.float()
        scale = source.abs().amax(dim=1, keepdim=True) / 448.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = (source / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        yield name, quantized
        yield name.removesuffix(".weight") + ".weight_scale", scale


class Fp8CheckpointModel:
    weight_only = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self._quantized_weight_names = set()
        split_projections = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
        }
        for name, layer in self.named_modules():
            scheme = getattr(layer, "scheme", None)
            if not isinstance(scheme, CompressedTensorsW8A8Fp8):
                continue
            if scheme.strategy != "channel" or scheme.is_static_input_scheme:
                raise ValueError("These experiments require per-channel FP8 weights and dynamic activations")
            layer.quant_method = CheckpointFp8LinearMethod(
                layer.quant_method.quant_config, weight_only=self.weight_only
            )
            self._quantized_weight_names.add(name + ".weight")
            parent, _, leaf = name.rpartition(".")
            for projection in split_projections.get(leaf, ()):
                self._quantized_weight_names.add(f"{parent}.{projection}.weight")
        if not self._quantized_weight_names:
            raise ValueError("No compressed-tensors FP8 projections found in the checkpoint")

    def load_weights(self, weights):
        return super().load_weights(encode_updated_weights(weights, self._quantized_weight_names))


class Qwen2Fp8ForCausalLM(Fp8CheckpointModel, Qwen2ForCausalLM):
    pass


class Qwen3Fp8ForCausalLM(Fp8CheckpointModel, Qwen3ForCausalLM):
    pass


class Qwen2W8A16ForCausalLM(Qwen2Fp8ForCausalLM):
    weight_only = True


class Qwen3W8A16ForCausalLM(Qwen3Fp8ForCausalLM):
    weight_only = True


EntryClass = [Qwen2Fp8ForCausalLM, Qwen3Fp8ForCausalLM, Qwen2W8A16ForCausalLM, Qwen3W8A16ForCausalLM]

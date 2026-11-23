"""Share E4M3-rounded frozen weights across TE's forward and dgrad GEMMs."""

import weakref

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.fake_quant import FakeQuant


@Registry.register_feature(namespace="transformer_engine")
class FrozenWeightFakeQuant(FakeQuant):
    """Keep one BF16 rounded copy per frozen weight during a precision step.

    The Miles hooks advance the debug step before scoring or training. Clearing
    at that boundary also handles checkpoint loads or weight-backup restores
    that write through ``.data`` without advancing a tensor's version counter.
    """

    def __init__(self):
        self._iteration = None
        self._weights = {}

    @api_method
    def modify_tensor(
        self, config, layer_name, gemm, tensor_name, tensor, iteration,
        default_quantizer, out=None, dtype=None,
    ):
        if tensor_name != "weight" or tensor.requires_grad:
            raise ValueError("FrozenWeightFakeQuant requires frozen base weights")
        if iteration != self._iteration:
            self._weights.clear()
            self._iteration = iteration

        key = (layer_name, config["quant_format"], dtype or tensor.dtype)
        signature = (tensor._version, tensor.data_ptr(), tensor.shape, tensor.stride())
        entry = self._weights.get(key)
        if entry is None or entry[0]() is not tensor or entry[1] != signature:
            rounded = super().modify_tensor(
                config, layer_name, gemm, tensor_name, tensor, iteration,
                default_quantizer, dtype=dtype,
            )

            def release(source):
                current = self._weights.get(key)
                if current is not None and current[0] is source:
                    del self._weights[key]

            entry = (weakref.ref(tensor, release), signature, rounded)
            self._weights[key] = entry

        rounded = entry[2]
        if out is not None:
            if out is not rounded:
                out.copy_(rounded)
            return None
        return rounded

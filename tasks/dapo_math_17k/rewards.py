"""DeepScaler answer grading for Qwen2.5's non-thinking responses."""

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


async def deepscaler(args, sample, **kwargs):
    """Accept Miles' single-sample and batched reward callback interfaces."""
    if isinstance(sample, list):
        return [await deepscaler(args, item, **kwargs) for item in sample]

    response = sample.response
    # Upstream requires a reasoning-end marker even for non-thinking models.
    # Prefix only the scorer's input; leave generated text and token IDs intact.
    # Existing thinking responses retain upstream's final-segment semantics.
    if not any(marker in response for marker in ("</think>", "###Response", "<|open|>response<|sep|>", "<think>")):
        response = "</think>" + response
    return get_deepscaler_rule_based_reward(response, sample.label)

"""DeepScaler correctness scoring with an answer-only training gate."""

import re

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


async def deepscaler(args, sample, **kwargs):
    """Return gated `reward` for training and raw `accuracy` for evaluation."""
    if isinstance(sample, list):
        return [await deepscaler(args, item, **kwargs) for item in sample]

    response = sample.response
    # Upstream requires a reasoning-end marker even for non-thinking models.
    # Prefix only the scorer's input; leave generated text and token IDs intact.
    # Existing thinking responses retain upstream's final-segment semantics.
    if not any(marker in response for marker in ("</think>", "###Response", "<|open|>response<|sep|>", "<think>")):
        response = "</think>" + response
    accuracy = get_deepscaler_rule_based_reward(response, sample.label)
    answer_only = is_answer_only(sample.response)
    return {
        "reward": 0 if answer_only else accuracy,
        "accuracy": accuracy,
        "answer_only": int(answer_only),
    }


def is_answer_only(response: str) -> bool:
    """Detect boxed answers with only labels/formatting outside the boxes.

    This is a narrow format heuristic, not a reasoning-quality verifier.
    Inspect the whole response so genuine work inside thinking tags counts.
    """
    outside = []
    cursor = 0
    found_box = False
    while match := re.search(r"\\boxed\s*\{", response[cursor:]):
        start = cursor + match.start()
        content_start = cursor + match.end()
        outside.append(response[cursor:start])
        # Balance nested LaTeX braces, ignoring escaped braces such as \{.
        depth = 1
        for brace in re.finditer(r"\\.|[{}]", response[content_start:]):
            if brace.group() == "{":
                depth += 1
            elif brace.group() == "}":
                depth -= 1
            if depth == 0:
                cursor = content_start + brace.end()
                break
        else:
            return False
        found_box = True

    if not found_box:
        return False
    outside.append(response[cursor:])
    remainder = "".join(outside)
    for marker in ("<think>", "</think>", "###Response", "<|open|>response<|sep|>"):
        remainder = remainder.replace(marker, "")
    remainder = re.sub(r"\b(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\b", "", remainder, flags=re.IGNORECASE)
    return all(char.isspace() or char in r".,:;!?$\()[]{}*_`#" for char in remainder)

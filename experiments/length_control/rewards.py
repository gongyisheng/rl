"""Length-aware reward functions for math rollouts."""

import math
import statistics
from collections.abc import Sequence


def _validate_inputs(
    correctness: Sequence[int],
    response_lengths: Sequence[float],
    alpha: float,
) -> None:
    if len(correctness) != len(response_lengths):
        raise ValueError("correctness and response_lengths must have the same length")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be finite and satisfy 0 <= alpha < 1")

    for is_correct in correctness:
        if is_correct not in (0, 1):
            raise ValueError("correctness values must be 0 or 1")

    for length in response_lengths:
        if not isinstance(length, (int, float)) or not math.isfinite(length) or length < 0:
            raise ValueError("response lengths must be finite and nonnegative")


def reward_v1(
    correctness: Sequence[int],
    response_lengths: Sequence[float],
    alpha: float = 0.2,
    max_length: float = 1024,
) -> list[float]:
    """Reward correct responses, with a capped absolute-length penalty."""
    _validate_inputs(correctness, response_lengths, alpha)
    if not isinstance(max_length, (int, float)) or not math.isfinite(max_length) or max_length <= 0:
        raise ValueError("max_length must be finite and greater than zero")

    return [
        1.0 - alpha * min(length / max_length, 1.0) if is_correct else 0
        for is_correct, length in zip(correctness, response_lengths)
    ]


def reward_v2(
    correctness: Sequence[int],
    response_lengths: Sequence[float],
    alpha: float = 0.2,
) -> list[float]:
    """Reward correct responses relative to other correct responses in a group."""
    _validate_inputs(correctness, response_lengths, alpha)
    correct_lengths = [
        length for is_correct, length in zip(correctness, response_lengths) if is_correct
    ]
    if not correct_lengths:
        return [0] * len(correctness)

    mean_length = statistics.mean(correct_lengths)
    standard_deviation = statistics.pstdev(correct_lengths)
    denominator = standard_deviation + 1e-7
    rewards = []
    for is_correct, length in zip(correctness, response_lengths):
        if not is_correct:
            rewards.append(0)
            continue

        standardized_length = (length - mean_length) / denominator
        if standardized_length >= 0:
            sigmoid = 1.0 / (1.0 + math.exp(-standardized_length))
        else:
            exponential = math.exp(standardized_length)
            sigmoid = exponential / (1.0 + exponential)
        rewards.append(1.0 - alpha * sigmoid)
    return rewards


async def _apply_reward(args, samples, version) -> list[float]:
    if getattr(args, "reward_key", None):
        raise ValueError("Length control expects scalar rewards")
    if len({sample.group_index for sample in samples}) > 1:
        raise ValueError("Length rewards must be computed separately for each prompt group")

    if getattr(args, "rm_type", None) == "deepscaler":
        from tasks.dapo_math_17k.rewards import deepscaler

        correctness = [int(await deepscaler(args, sample)) for sample in samples]
    else:
        from miles.rollout.rm_hub.math_utils import grade_answer_verl

        correctness = [
            int(grade_answer_verl(sample.response, sample.label)) for sample in samples
        ]
    response_lengths = [sample.response_length for sample in samples]
    alpha = getattr(args, "length_penalty_alpha", 0.2)
    if version == "v1":
        rewards = reward_v1(correctness, response_lengths, alpha, args.rollout_max_response_len)
    else:
        rewards = reward_v2(correctness, response_lengths, alpha)

    for sample, is_correct in zip(samples, correctness):
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["length_control_accuracy"] = is_correct

    return rewards


async def v1(args, samples) -> list[float]:
    """Miles custom reward callback for the absolute-length reward."""
    return await _apply_reward(args, samples, "v1")


async def v2(args, samples) -> list[float]:
    """Miles custom reward callback for the relative sigmoid reward."""
    return await _apply_reward(args, samples, "v2")

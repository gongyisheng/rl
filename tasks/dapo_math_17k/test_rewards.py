"""Test correctness and answer-only gating against the installed Miles scorer."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from miles.utils.types import Sample


ARGS = SimpleNamespace(
    rm_type="deepscaler",
    custom_rm_path="tasks.dapo_math_17k.rewards.deepscaler",
    reward_key="reward",
    eval_reward_key="accuracy",
)


@pytest.mark.parametrize(
    "response,label,expected",
    [
        (r"Answer: \boxed{42}", "42", 1),
        (r"Answer: \boxed{41}", "42", 0),
        (r"Answer: \boxed{\frac{1}{2}}", "0.5", 1),
        (r"Answer: \boxed{-3}", "-3", 1),
        ("The answer is 42.", "42", 0),
        ("", "42", 0),
        (r"Answer: \boxed{42}", "", 0),
        (r"First \boxed{41}, finally \boxed{42}", "42", 1),
        (r"<think>Maybe \boxed{42}</think>Answer: \boxed{41}", "42", 0),
        (r"<think>Work</think>Answer: \boxed{42}", "42", 1),
        (r"<think>Unfinished reasoning: \boxed{42}", "42", 0),
        (r"Work ###Response Answer: \boxed{42}", "42", 1),
        (r"Work <|open|>response<|sep|> Answer: \boxed{42}", "42", 1),
    ],
)
def test_reward(response, label, expected):
    sample = Sample(response=response, label=label)
    assert asyncio.run(async_rm(ARGS, sample))["accuracy"] == expected
    assert sample.response == response


def test_batch_reward():
    samples = [Sample(response=r"\boxed{42}", label=label) for label in ("42", "41")]
    expected = [
        {"reward": 0, "accuracy": 1, "answer_only": 1},
        {"reward": 0, "accuracy": 0, "answer_only": 1},
    ]
    assert asyncio.run(batched_async_rm(ARGS, samples)) == expected
    asyncio.run(batched_async_rm(ARGS, samples, inplace_set_reward_field=True))
    assert [sample.reward for sample in samples] == expected


@pytest.mark.parametrize(
    "response,label,accuracy,answer_only",
    [
        (r"Answer: \boxed{42}", "42", 1, 1),
        (r"\boxed{42}", "42", 1, 1),
        (r"Answer \boxed{41}", "42", 0, 1),
        (r"The final answer is $\boxed{\frac{1}{2}}$.", "0.5", 1, 1),
        (r"**Answer:** \[\boxed{-3}\]", "-3", 1, 1),
        ("<think> \n </think>Answer: \\boxed{42}", "42", 1, 1),
        (r"</think>Answer: \boxed{42}", "42", 1, 1),
        (r"###Response Answer: \boxed{42}", "42", 1, 1),
        (r"<|open|>response<|sep|> Answer: \boxed{42}", "42", 1, 1),
        (r"\boxed{41}, \boxed{42}", "42", 1, 1),
        (r"6 * 7 = 42. Answer: \boxed{42}", "42", 1, 0),
        (r"\boxed{42} because 6 * 7 = 42.", "42", 1, 0),
        (r"<think>6 * 7 = 42.</think>Answer: \boxed{42}", "42", 1, 0),
        (r"6 * 7 = 42. ###Response Answer: \boxed{42}", "42", 1, 0),
        (r"6 * 7 = 42. Answer: \boxed{41}", "42", 0, 0),
        (r"<think>6 * 7 = 42. \boxed{42}", "42", 0, 0),
        (r"Answer: \boxed{42", "42", 0, 0),
        ("", "42", 0, 0),
        ("The answer is 42.", "42", 0, 0),
    ],
)
def test_answer_gate(response, label, accuracy, answer_only):
    sample = Sample(response=response, label=label, tokens=[11, 12], metadata={"source": "test"})
    result = asyncio.run(async_rm(ARGS, sample))
    assert result == {
        "reward": 0 if answer_only else accuracy,
        "accuracy": accuracy,
        "answer_only": answer_only,
    }
    assert sample.response == response
    assert sample.tokens == [11, 12]
    assert sample.metadata == {"source": "test"}


def test_answer_gate_batch_and_training_reward_selection():
    samples = [
        Sample(response=r"Answer: \boxed{42}", label="42"),
        Sample(response=r"6 * 7 = 42. Answer: \boxed{42}", label="42"),
        Sample(response=r"6 * 7 = 41. Answer: \boxed{41}", label="42"),
    ]
    expected = [
        {"reward": 0, "accuracy": 1, "answer_only": 1},
        {"reward": 1, "accuracy": 1, "answer_only": 0},
        {"reward": 0, "accuracy": 0, "answer_only": 0},
    ]
    assert asyncio.run(batched_async_rm(ARGS, samples)) == expected
    asyncio.run(batched_async_rm(ARGS, samples, inplace_set_reward_field=True))
    assert [sample.get_reward_value(ARGS) for sample in samples] == [0, 1, 0]
    assert [sample.reward[ARGS.eval_reward_key] for sample in samples] == [1, 1, 0]
    assert check_reward_nonzero_std(ARGS, samples[:2]).keep
    assert not check_reward_nonzero_std(ARGS, [samples[0], samples[2]]).keep


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_length_control_uses_gate_and_preserves_accuracy(version):
    from experiments.length_control import rewards

    args = SimpleNamespace(rm_type="deepscaler", reward_key=None, rollout_max_response_len=8192)
    samples = [
        Sample(response=r"Answer: \boxed{42}", label="42", response_length=10),
        Sample(response=r"6 * 7 = 42. Answer: \boxed{42}", label="42", response_length=4096),
        Sample(response=r"6 * 7 = 41. Answer: \boxed{41}", label="42", response_length=4096),
    ]
    assert asyncio.run(getattr(rewards, version)(args, samples)) == [0, 0.9, 0]
    assert [sample.metadata["length_control_accuracy"] for sample in samples] == [1, 1, 0]

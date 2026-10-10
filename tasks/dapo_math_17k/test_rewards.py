"""Test non-thinking rewards against the installed Miles scorer."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.types import Sample


ARGS = SimpleNamespace(
    rm_type="deepscaler",
    custom_rm_path="tasks.dapo_math_17k.rewards.deepscaler",
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
    assert asyncio.run(async_rm(ARGS, sample)) == expected
    assert sample.response == response


def test_batch_reward():
    samples = [Sample(response=r"\boxed{42}", label=label) for label in ("42", "41")]
    assert asyncio.run(batched_async_rm(ARGS, samples)) == [1, 0]
    asyncio.run(batched_async_rm(ARGS, samples, inplace_set_reward_field=True))
    assert [sample.reward for sample in samples] == [1, 0]

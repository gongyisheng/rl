"""Unit tests for GSM8K length-control rewards."""

import asyncio
from dataclasses import dataclass
import importlib
import math
import sys
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest import mock

from experiments.gsm8k_length_control import rewards
from experiments.gsm8k_length_control.rewards import reward_v1, reward_v2


def make_sample(
    response: str,
    label: str,
    response_length: float,
    group_index: int = 0,
    metadata: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        group_index=group_index,
        metadata=metadata,
        response_length=response_length,
        response=response,
        label=label,
        reward=None,
    )


class RewardV1Tests(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(reward_v1([], []), [])

    def test_absolute_length_penalty_and_clipping(self) -> None:
        rewards = reward_v1([1, 1, 1, 0], [0, 512, 4096, 3], alpha=0.2)

        self.assertEqual(rewards, [1.0, 0.9, 0.8, 0.0])

    def test_zero_alpha_is_correctness_reward(self) -> None:
        self.assertEqual(reward_v1([1, 0, 1], [10, 20, 30], alpha=0), [1.0, 0.0, 1.0])


class RewardV2Tests(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(reward_v2([], []), [])

    def test_fixed_population_standard_deviation_reference(self) -> None:
        rewards = reward_v2([1, 1], [100, 200])

        self.assertAlmostEqual(rewards[0], 0.9462117157, places=9)
        self.assertAlmostEqual(rewards[1], 0.8537882843, places=9)

    def test_wrong_responses_do_not_affect_correct_length_statistics(self) -> None:
        without_wrong = reward_v2([1, 1], [100, 200])
        with_wrong = reward_v2([1, 0, 1], [100, 1_000_000, 200])

        self.assertEqual(with_wrong[0], without_wrong[0])
        self.assertEqual(with_wrong[2], without_wrong[1])
        self.assertEqual(with_wrong[1], 0.0)

    def test_singleton_and_equal_correct_lengths_receive_half_penalty(self) -> None:
        expected = 1.0 - 0.2 / 2
        self.assertEqual(reward_v2([1], [100]), [expected])
        self.assertEqual(reward_v2([1, 0, 1], [100, 500, 100]), [expected, 0.0, expected])

    def test_rewards_follow_their_responses_under_permutation(self) -> None:
        original = reward_v2([1, 1, 0, 1], [100, 200, 50, 300])
        permutation = [3, 0, 2, 1]
        permuted = reward_v2(
            [1, 1, 0, 1],
            [300, 100, 50, 200],
        )

        self.assertEqual(permuted, [original[index] for index in permutation])

    def test_no_correct_responses_are_zero(self) -> None:
        self.assertEqual(reward_v2([0, 0], [10, 20]), [0.0, 0.0])

    def test_zero_alpha_is_correctness_reward(self) -> None:
        self.assertEqual(reward_v2([1, 0, 1], [10, 20, 30], alpha=0), [1.0, 0.0, 1.0])


class InputValidationTests(unittest.TestCase):
    def test_invalid_inputs_raise_value_error(self) -> None:
        invalid_calls = [
            lambda: reward_v1([1], [1, 2]),
            lambda: reward_v2([2], [1]),
            lambda: reward_v1([1], [-1]),
            lambda: reward_v2([1], [math.inf]),
            lambda: reward_v1([1], [1], alpha=1),
            lambda: reward_v2([1], [1], alpha=math.nan),
            lambda: reward_v1([1], [1], max_length=0),
        ]

        for invalid_call in invalid_calls:
            with self.assertRaises(ValueError):
                invalid_call()


class MilesRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        math_utils = ModuleType("miles.rollout.rm_hub.math_utils")

        def grade_answer_verl(response: str, label: str) -> bool:
            return response == label

        math_utils.grade_answer_verl = grade_answer_verl
        miles = ModuleType("miles")
        miles.__path__ = []
        rollout = ModuleType("miles.rollout")
        rollout.__path__ = []
        rm_hub = ModuleType("miles.rollout.rm_hub")
        rm_hub.__path__ = []
        self.modules = mock.patch.dict(
            sys.modules,
            {
                "miles": miles,
                "miles.rollout": rollout,
                "miles.rollout.rm_hub": rm_hub,
                "miles.rollout.rm_hub.math_utils": math_utils,
            },
        )
        self.modules.start()

    def tearDown(self) -> None:
        self.modules.stop()

    def test_v1_returns_absolute_length_rewards_without_mutating_samples(self) -> None:
        args = SimpleNamespace(reward_key=None)
        samples = [
            make_sample("answer", "answer", 0),
            make_sample("answer", "answer", 512),
            make_sample("wrong", "answer", 10),
        ]

        returned_rewards = asyncio.run(rewards.v1(args, samples))

        self.assertEqual(returned_rewards, [1.0, 0.9, 0.0])
        self.assertEqual([sample.reward for sample in samples], [None, None, None])
        self.assertEqual(
            [sample.metadata["length_control_accuracy"] for sample in samples], [1, 1, 0]
        )

    def test_v2_scores_unscored_all_correct_group(self) -> None:
        args = SimpleNamespace(reward_key=None)
        samples = [make_sample("answer", "answer", 100), make_sample("answer", "answer", 200)]

        returned_rewards = asyncio.run(rewards.v2(args, samples))

        self.assertAlmostEqual(returned_rewards[0], 0.9462117157, places=9)
        self.assertAlmostEqual(returned_rewards[1], 0.8537882843, places=9)
        self.assertEqual([sample.reward for sample in samples], [None, None])

    def test_all_wrong_tied_and_singleton_groups_return_expected_rewards(self) -> None:
        args = SimpleNamespace(reward_key=None)
        all_wrong = [make_sample("wrong", "answer", 100), make_sample("wrong", "answer", 200)]
        tied_correct = [make_sample("answer", "answer", 100), make_sample("answer", "answer", 100)]
        singleton = [make_sample("answer", "answer", 100)]

        self.assertEqual(asyncio.run(rewards.v1(args, all_wrong)), [0.0, 0.0])
        self.assertEqual(asyncio.run(rewards.v2(args, tied_correct)), [0.9, 0.9])
        self.assertEqual(asyncio.run(rewards.v2(args, singleton)), [0.9])

    def test_repeated_calls_are_idempotent_and_preserve_metadata(self) -> None:
        args = SimpleNamespace(reward_key=None)
        samples = [
            make_sample("answer", "answer", 256, metadata={"source": "first"}),
            make_sample("wrong", "answer", 128, metadata={"source": "second"}),
        ]

        first_rewards = asyncio.run(rewards.v1(args, samples))
        second_rewards = asyncio.run(rewards.v1(args, samples))

        self.assertEqual(second_rewards, first_rewards)
        self.assertEqual([sample.reward for sample in samples], [None, None])
        self.assertEqual([sample.metadata["source"] for sample in samples], ["first", "second"])
        self.assertEqual(
            [sample.metadata["length_control_accuracy"] for sample in samples], [1, 0]
        )

    def test_mixed_group_index_raises(self) -> None:
        args = SimpleNamespace(reward_key=None)
        samples = [
            make_sample("answer", "answer", 100, group_index=0),
            make_sample("answer", "answer", 200, group_index=1),
        ]

        with self.assertRaises(ValueError):
            asyncio.run(rewards.v1(args, samples))

    def test_custom_alpha_and_max_length_are_used(self) -> None:
        args = SimpleNamespace(
            reward_key=None,
            length_penalty_alpha=0.4,
            length_penalty_max_length=100,
        )
        samples = [make_sample("answer", "answer", 50), make_sample("answer", "answer", 200)]

        self.assertEqual(asyncio.run(rewards.v1(args, samples)), [0.8, 0.6])

    def test_custom_alpha_is_used_by_v2(self) -> None:
        args = SimpleNamespace(reward_key=None, length_penalty_alpha=0.4)
        samples = [make_sample("answer", "answer", 100), make_sample("answer", "answer", 200)]

        returned_rewards = asyncio.run(rewards.v2(args, samples))
        self.assertAlmostEqual(returned_rewards[0], 0.8924234315, places=9)
        self.assertAlmostEqual(returned_rewards[1], 0.7075765685, places=9)


@dataclass
class EvaluationInput:
    generate_state: object | None
    request_id: str
    payload: str


class EvaluationAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        inference_module = ModuleType(
            "miles.rollout.inference_rollout.inference_rollout_common"
        )

        class InferenceRolloutFn:
            async def _call_eval(self, input: EvaluationInput) -> dict[str, object]:
                self.delegated_input = input
                return {"request_id": input.request_id, "payload": input.payload}

        inference_module.InferenceRolloutFn = InferenceRolloutFn
        miles = ModuleType("miles")
        miles.__path__ = []
        rollout = ModuleType("miles.rollout")
        rollout.__path__ = []
        inference_rollout = ModuleType("miles.rollout.inference_rollout")
        inference_rollout.__path__ = []
        self.modules = mock.patch.dict(
            sys.modules,
            {
                "miles": miles,
                "miles.rollout": rollout,
                "miles.rollout.inference_rollout": inference_rollout,
                "miles.rollout.inference_rollout.inference_rollout_common": inference_module,
            },
        )
        self.modules.start()
        sys.modules.pop("experiments.gsm8k_length_control.evaluation", None)
        self.evaluation = importlib.import_module("experiments.gsm8k_length_control.evaluation")

    def tearDown(self) -> None:
        sys.modules.pop("experiments.gsm8k_length_control.evaluation", None)
        self.modules.stop()

    def test_uses_self_state_without_mutating_training_arguments(self) -> None:
        training_args = SimpleNamespace(group_rm=True, custom_rm_path="rewards.v1", rm_type="math")
        training_state = SimpleNamespace(args=training_args, cache="train-cache")
        adapter = self.evaluation.MathEvalRolloutFn.__new__(self.evaluation.MathEvalRolloutFn)
        adapter.state = training_state
        input = EvaluationInput(generate_state=None, request_id="self", payload="keep")

        result = asyncio.run(adapter._call_eval(input))

        delegated_input = adapter.delegated_input
        self.assertEqual(result, {"request_id": "self", "payload": "keep"})
        self.assertEqual((delegated_input.request_id, delegated_input.payload), ("self", "keep"))
        self.assertEqual(delegated_input.generate_state.cache, "train-cache")
        self.assertFalse(delegated_input.generate_state.args.group_rm)
        self.assertIsNone(delegated_input.generate_state.args.custom_rm_path)
        self.assertEqual(delegated_input.generate_state.args.rm_type, "math")
        self.assertTrue(training_args.group_rm)
        self.assertEqual(training_args.custom_rm_path, "rewards.v1")
        self.assertEqual(training_args.rm_type, "math")
        self.assertIsNone(input.generate_state)

    def test_uses_supplied_state_without_mutating_it(self) -> None:
        self_args = SimpleNamespace(group_rm=True, custom_rm_path="rewards.v1", rm_type="math")
        supplied_args = SimpleNamespace(group_rm=True, custom_rm_path="rewards.v2", rm_type="dapo")
        adapter = self.evaluation.MathEvalRolloutFn.__new__(self.evaluation.MathEvalRolloutFn)
        adapter.state = SimpleNamespace(args=self_args, cache="self-cache")
        supplied_state = SimpleNamespace(args=supplied_args, cache="supplied-cache")
        input = EvaluationInput(generate_state=supplied_state, request_id="provided", payload="keep")

        result = asyncio.run(adapter._call_eval(input))

        delegated_input = adapter.delegated_input
        self.assertEqual(result, {"request_id": "provided", "payload": "keep"})
        self.assertIs(delegated_input.generate_state.args.group_rm, False)
        self.assertIsNone(delegated_input.generate_state.args.custom_rm_path)
        self.assertEqual(delegated_input.generate_state.args.rm_type, "math")
        self.assertEqual(delegated_input.generate_state.cache, "supplied-cache")
        self.assertIs(input.generate_state, supplied_state)
        self.assertTrue(supplied_args.group_rm)
        self.assertEqual(supplied_args.custom_rm_path, "rewards.v2")
        self.assertEqual(supplied_args.rm_type, "dapo")
        self.assertTrue(self_args.group_rm)


if __name__ == "__main__":
    unittest.main()

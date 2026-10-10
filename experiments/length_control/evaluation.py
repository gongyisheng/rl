"""Math-accuracy evaluation for group-reward training."""

from copy import copy
from dataclasses import replace

from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn


class MathEvalRolloutFn(InferenceRolloutFn):
    async def _call_eval(self, input):
        state = copy(input.generate_state or self.state)
        state.args = copy(state.args)
        state.args.group_rm = False
        if state.args.rm_type == "deepscaler":
            state.args.custom_rm_path = "tasks.dapo_math_17k.rewards.deepscaler"
        else:
            state.args.rm_type = "math"
            state.args.custom_rm_path = None
        return await super()._call_eval(replace(input, generate_state=state))

"""Check task selection and actual replanning boundaries without GPU dependencies."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'policy'))
from xr1_submission import deploy_policy as policy


class Action(np.ndarray):
    is_cuda = False


class Env:
    def __init__(self, success_at):
        self.unwrapped = self
        self._current_instruction = 'Transfer the item.'
        self.steps = 0
        self.success_at = success_at

    def step(self, action):
        self.steps += 1
        return {}, 0, False, False, {}

    def get_wrapper_attr(self, name):
        assert name == 'is_task_success'
        return lambda: np.array([self.steps >= self.success_at])


class TaskExecutionTest(unittest.TestCase):
    def test_direct_entrypoint_uses_task_mapping_and_rejects_mismatch(self):
        for task, k in [('items_handover', 20), ('handle_basket', 29), ('water_pouring', 29)]:
            self.assertEqual(policy.get_model({'task_name': task, 'xr1_port': 1}).k, k)
            self.assertEqual(policy.get_model({'task_name': task, 'xr1_port': 1, 'xr1_execute_steps': None}).k, k)
            with self.assertRaises(ValueError):
                policy.get_model({'task_name': task, 'xr1_port': 1, 'xr1_execute_steps': 29 if k == 20 else 20})

    def test_replan_at_selected_k_and_stop_mid_chunk_on_success(self):
        for task, k in [('items_handover', 20), ('handle_basket', 29)]:
            model = policy.get_model({'task_name': task, 'xr1_port': 1})
            requests = []
            def infer(payload):
                requests.append(payload)
                return {'target_joint_action': np.zeros((30, 14), dtype=np.float32),
                        'policy_seed': 20260908, 'prefix_length': 0}
            model.infer = infer
            env = Env(k + 3)
            with patch.object(policy, 'extract_observation', return_value=({}, np.zeros(14))), \
                 patch.object(policy, '_direct_joint_action', side_effect=lambda *args: np.zeros(14).view(Action)), \
                 patch.dict(sys.modules, {'torch': types.SimpleNamespace()}):
                policy.eval(env, model, {})
                self.assertEqual(env.steps, k)
                policy.eval(env, model, {})
                self.assertEqual(env.steps, k + 3)
            self.assertEqual([int(r['chunk_index']) for r in requests], [0, 1])
            self.assertTrue(all(r['prefix_joint_action'].shape == (0, 14) for r in requests))
            policy.reset_model(model)
            self.assertEqual((model.chunk, model.steps), (0, 0))


if __name__ == '__main__':
    unittest.main()

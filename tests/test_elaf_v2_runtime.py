import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from elaf.v0_training import load_checkpoint, rng_state, seed_all
from elaf.v2_runtime import StabilityGuard, action_only_backward, save_v2
from elaf.v2_schedule import StageSchedule


def schedule_config():
    return {'max_steps': 200000, 'stages': {
        'freeze_step': 3000, 'unfreeze_step': 10000,
        'stage1_warmup_steps': 200, 'stage3_lora_warmup_steps': 1000,
        'cosine_min_ratio': .1,
        'stage1': {'lora_lr': 1e-5, 'resampler_lr': 5e-5, 'new_modules_lr': 1e-4},
        'stage3': {'lora_lr': 2e-6, 'resampler_lr': 2e-5, 'new_modules_lr': 5e-5},
        'stage3_min': {'lora_lr': 2e-7, 'resampler_lr': 2e-6, 'new_modules_lr': 5e-6}}}


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.tensor([.25, -.5]))
        self.other = nn.Parameter(torch.tensor([.75]))

    def forward(self, x):
        return (x * self.lora_A).sum() + self.other.sum()


def make_schedule(policy):
    optimizer = torch.optim.AdamW([
        {'params': [policy.lora_A], 'group_name': 'lora'},
        {'params': [policy.other], 'group_name': 'resampler'}], lr=1e-4)
    return optimizer, StageSchedule(optimizer, schedule_config())


class RuntimeTests(unittest.TestCase):
    def test_guard_requires_persistent_joint_gradient_and_loss_growth(self):
        guard = StabilityGuard({'window': 2, 'baseline_start': 2,
            'gradient_ratio': 10., 'gradient_absolute_floor': 50.,
            'loss_ratio': 1.5, 'loss_absolute_increase': .05,
            'consecutive_windows': 3})

        def record(step, grad, action, delta):
            return {'step': step, 'train/grad/global_preclip': grad,
                    'train/action_loss': action, 'train/delta_loss': delta}

        for step, grad, action, delta in [(1, 5., .2, .2), (2, 6., .2, .2),
                                           (3, 100., .5, .2), (4, 5., .2, .2),
                                           (5, 120., .6, .2), (6, 130., .6, .2),
                                           (7, 140., .6, .2), (8, 150., .6, .2)]:
            stopped, _ = guard.update(record(step, grad, action, delta))
        self.assertFalse(stopped, 'isolated bad windows must not trigger a pause')
        for step in (9, 10, 11, 12, 13, 14):
            stopped, reason = guard.update(record(step, 100., .6, .6))
        self.assertTrue(stopped)
        self.assertEqual(reason, 'sustained_gradient_and_loss_growth')

    def test_nonfinite_guard_stops_immediately(self):
        guard = StabilityGuard({'window': 10, 'baseline_start': 100,
            'gradient_ratio': 10., 'gradient_absolute_floor': 50.,
            'loss_ratio': 1.5, 'loss_absolute_increase': .05,
            'consecutive_windows': 3})
        stopped, reason = guard.update({'step': 1, 'train/grad/global_preclip': float('inf'),
            'train/action_loss': .2, 'train/delta_loss': .2})
        self.assertTrue(stopped)
        self.assertEqual(reason, 'nonfinite_metrics')

    def test_action_only_routes_lora_and_keeps_both_losses_for_other_modules(self):
        policy = ToyPolicy()
        x = torch.tensor([2., -1.])
        action = (policy.lora_A * x).sum() + policy.other.square().sum()
        delta = (policy.lora_A.square().sum() - policy.other).square().sum()
        total = action + delta
        expected_lora = torch.autograd.grad(action, policy.lora_A, retain_graph=True)[0]
        expected_other = torch.autograd.grad(total, policy.other, retain_graph=True)[0]
        action_only_backward(policy, action, total)
        torch.testing.assert_close(policy.lora_A.grad, expected_lora)
        torch.testing.assert_close(policy.other.grad, expected_other)

    def test_frozen_stage_boundary_checkpoint_restores_weights_optimizer_and_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            checkpoints = root / 'checkpoints'
            policy = ToyPolicy()
            optimizer, schedule = make_schedule(policy)
            schedule.apply(policy, 2999)
            optimizer.zero_grad(set_to_none=True)
            policy(torch.tensor([.4, -.2])).square().backward()
            optimizer.step()
            lora_before = policy.lora_A.detach().clone()
            non_lora_state = copy.deepcopy(optimizer.state[policy.other])

            # The stage boundary checkpoint records all learned LoRA tensors even
            # when the resumed Stage 2 process will freeze them immediately.
            progress = {'step': 3000, 'micro_step': 24000, 'best_loss': .2,
                        'window_loss': 0., 'window_count': 0, 'is_best': True,
                        'wandb_run_id': 'test-run', 'stage': 1}
            contract = {'config': {'preserve_steps': [3000], 'full_diagnostic_steps': []}}
            seed_all(123)
            expected_rng = torch.rand(4)
            seed_all(123)
            save_v2(checkpoints, policy, optimizer, schedule, progress, contract, rank=0)
            checkpoint = checkpoints / 'latest'
            saved = torch.load(checkpoint / 'training.pt', map_location='cpu', weights_only=False)
            self.assertIn('lora_A', saved['model'])
            self.assertTrue(torch.equal(saved['model']['lora_A'], lora_before))

            resumed = ToyPolicy()
            opt2, schedule2 = make_schedule(resumed)
            loaded = load_checkpoint(checkpoint, resumed, opt2, schedule2, contract, rank=0)
            self.assertEqual(loaded['step'], 3000)
            self.assertEqual(loaded['micro_step'], 24000)
            self.assertTrue(torch.equal(resumed.lora_A, lora_before))
            schedule2.apply(resumed, loaded['step'])
            self.assertFalse(resumed.lora_A.requires_grad)
            self.assertTrue(torch.equal(resumed.lora_A, lora_before))
            for key, value in non_lora_state.items():
                self.assertTrue(torch.equal(opt2.state[resumed.other][key], value))
            torch.testing.assert_close(torch.rand(4), expected_rng)


if __name__ == '__main__':
    unittest.main()

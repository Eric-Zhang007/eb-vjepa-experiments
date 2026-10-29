import copy
import unittest

import torch

from elaf.v2_schedule import StageSchedule


def config():
    return {'max_steps': 200000, 'stages': {
        'freeze_step': 3000, 'unfreeze_step': 10000,
        'stage1_warmup_steps': 200, 'stage3_lora_warmup_steps': 1000,
        'cosine_min_ratio': 0.1,
        'stage1': {'lora_lr': 1e-5, 'resampler_lr': 5e-5, 'new_modules_lr': 1e-4},
        'stage3': {'lora_lr': 2e-6, 'resampler_lr': 2e-5, 'new_modules_lr': 5e-5},
        'stage3_min': {'lora_lr': 2e-7, 'resampler_lr': 2e-6, 'new_modules_lr': 5e-6}}}


def make(cfg=None):
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.1)
    optimizer = torch.optim.AdamW([
        {'params': [model.weight], 'group_name': 'lora'},
        {'params': [model.bias], 'group_name': 'resampler'}], lr=1e-4)
    return model, optimizer, StageSchedule(optimizer, cfg or config())


def update(model, optimizer, scheduler, step):
    scheduler.apply(model, step)
    optimizer.zero_grad(set_to_none=True)
    model(torch.tensor([[0.3, -0.7]], dtype=torch.float64)).square().sum().backward()
    optimizer.step()


class ScheduleTests(unittest.TestCase):
    def test_absolute_boundaries_and_learning_rates(self):
        model, opt, schedule = make()
        schedule.apply(model, 0)
        self.assertEqual(opt.param_groups[0]['lr'], 1e-5 / 200)
        schedule.apply(model, 2999)
        self.assertEqual(opt.param_groups[0]['lr'], 1e-5)
        self.assertTrue(schedule.apply(model, 3000))
        self.assertFalse(model.weight.requires_grad)
        self.assertEqual(opt.param_groups[0]['lr'], 0)
        self.assertTrue(schedule.apply(model, 10000))
        self.assertTrue(model.weight.requires_grad)
        self.assertEqual(opt.param_groups[0]['lr'], 0)
        schedule.apply(model, 10500)
        self.assertEqual(opt.param_groups[0]['lr'], 1e-6)
        schedule.apply(model, 11000)
        self.assertEqual(opt.param_groups[0]['lr'], 2e-6)
        schedule.apply(model, 200000)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 2e-7)
        self.assertAlmostEqual(opt.param_groups[1]['lr'], 2e-6)

    def test_frozen_weights_and_moments_unchanged_and_nonlora_preserved(self):
        model, opt, schedule = make()
        update(model, opt, schedule, 2999)
        weight = model.weight.detach().clone()
        moments = copy.deepcopy(opt.state[model.weight])
        for step in (3000, 3001, 9999):
            update(model, opt, schedule, step)
        self.assertTrue(torch.equal(model.weight, weight))
        for key, value in moments.items():
            self.assertTrue(torch.equal(opt.state[model.weight][key], value))
        other_state = copy.deepcopy(opt.state[model.bias])
        schedule.apply(model, 10000)
        self.assertNotIn(model.weight, opt.state)
        for key, value in other_state.items():
            self.assertTrue(torch.equal(opt.state[model.bias][key], value))

    def test_resume_equivalence_all_stages_and_boundaries(self):
        # Short schedules exercise exactly the same transitions with affordable
        # full uninterrupted runs; every optimizer state tensor is compared.
        cfg = config()
        cfg['max_steps'] = 20
        cfg['stages'].update(freeze_step=3, unfreeze_step=10,
                             stage1_warmup_steps=2, stage3_lora_warmup_steps=2)
        for checkpoint_step in (2, 3, 7, 10, 14):
            with self.subTest(checkpoint_step=checkpoint_step):
                model, opt, schedule = make(cfg)
                for step in range(checkpoint_step):
                    update(model, opt, schedule, step)
                saved = copy.deepcopy((model.state_dict(), opt.state_dict(), schedule.state_dict()))
                for step in range(checkpoint_step, 20):
                    update(model, opt, schedule, step)
                resumed, opt2, schedule2 = make(cfg)
                resumed.load_state_dict(saved[0])
                opt2.load_state_dict(saved[1])
                schedule2.load_state_dict(saved[2])
                for step in range(checkpoint_step, 20):
                    update(resumed, opt2, schedule2, step)
                for p, q in zip(model.parameters(), resumed.parameters()):
                    self.assertTrue(torch.equal(p, q))
                    for key, value in opt.state[p].items():
                        self.assertTrue(torch.equal(value, opt2.state[q][key]))
                self.assertEqual(schedule.state_dict(), schedule2.state_dict())

    def test_stage3_resume_does_not_clear_lora_moments(self):
        model, opt, schedule = make()
        update(model, opt, schedule, 11000)
        saved = copy.deepcopy(schedule.state_dict())
        state = copy.deepcopy(opt.state[model.weight])
        resumed = StageSchedule(opt, config())
        resumed.load_state_dict(saved)
        resumed.apply(model, 11001)
        for key, value in state.items():
            self.assertTrue(torch.equal(opt.state[model.weight][key], value))


if __name__ == '__main__':
    unittest.main()

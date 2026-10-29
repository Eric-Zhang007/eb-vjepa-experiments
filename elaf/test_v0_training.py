"""Resume and sampling invariants, independent of the full native model."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from elaf.v0_training import (seed_all, sample_indices, save_checkpoint, load_checkpoint,
                              parameter_groups, cosine_multiplier, build_scheduler,
                              clip_grad_groups, load_fork_checkpoint)


class TrainingTests(unittest.TestCase):
    @staticmethod
    def flow_fixture():
        model = torch.nn.Module()
        for name in ('backbone', 'resampler', 'action_in', 'action_out', 'action_time',
                     'delta_in', 'delta_out', 'delta_time', 'current_in', 'proprio_in'):
            setattr(model, name, torch.nn.Linear(2, 2))
        for name in ('action_position', 'delta_position', 'current_position'):
            model.register_parameter(name, torch.nn.Parameter(torch.zeros(1, 2, 2)))
        model.register_parameter('frozen', torch.nn.Parameter(torch.zeros(2), requires_grad=False))
        config = dict(weight_decay=.01, lora_lr=2e-5, resampler_lr=5e-5,
                      new_modules_lr=1e-4, clip_partition='flow')
        return model, config

    def test_flow_partition_is_complete_disjoint_and_preserves_learning_rates(self):
        model, config = self.flow_fixture()
        groups = parameter_groups(model, config)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        ownership = {id(p): group for group in groups for p in group['params']}
        self.assertEqual({g['group_name'] for g in groups},
                         {'lora', 'resampler', 'action', 'delta', 'current'})
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            expected = ('lora' if name.startswith('backbone.') else
                        'resampler' if name.startswith('resampler.') else
                        'current' if name.startswith(('current_', 'proprio_')) else
                        name.split('_', 1)[0])
            group = ownership[id(parameter)]
            self.assertEqual(group['group_name'], expected)
            rate = expected if expected in ('lora', 'resampler') else 'new_modules'
            self.assertEqual(group['lr'], config[rate + '_lr'])
            if 'position' in name or parameter.ndim < 2:
                self.assertEqual(group['weight_decay'], 0)
        default = parameter_groups(model, {k: v for k, v in config.items() if k != 'clip_partition'})
        self.assertEqual({g['group_name'] for g in default}, {'lora', 'resampler', 'new_modules'})

    def test_flow_partition_rejects_unassigned_parameters(self):
        model, config = self.flow_fixture()
        model.extra = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, 'Unassigned.*extra.weight'):
            parameter_groups(model, config)
        with self.assertRaisesRegex(ValueError, 'Unknown clip_partition'):
            parameter_groups(model, dict(config, clip_partition='typo'))

    def test_action_explosion_does_not_clip_delta_in_flow_partition(self):
        model, config = self.flow_fixture()
        optimizer = torch.optim.AdamW(parameter_groups(model, config))
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        model.action_out.weight.grad.fill_(100000.)
        model.delta_out.weight.grad.fill_(.2)
        expected = model.delta_out.weight.grad.clone()
        norms = clip_grad_groups(optimizer)
        self.assertTrue(torch.equal(model.delta_out.weight.grad, expected))
        self.assertAlmostEqual(norms['groups']['delta']['postclip'], .4, places=6)
        self.assertAlmostEqual(norms['groups']['action']['postclip'], 1., places=5)

    def test_recovery_warmup_counts_new_updates(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=2e-5)
        scheduler = build_scheduler(optimizer, dict(optimizer=dict(
            scheduler='warmup_constant', warmup_steps=200)))
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2e-5 / 200)
        for _ in range(199):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2e-5)
        for _ in range(5):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2e-5)

    def test_independent_clipping_combines_decay_buckets(self):
        lora = torch.nn.Parameter(torch.zeros(1))
        resampler = torch.nn.Parameter(torch.zeros(1))
        resampler_bias = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([
            dict(params=[lora], group_name='lora'),
            dict(params=[resampler], group_name='resampler'),
            dict(params=[resampler_bias], group_name='resampler', weight_decay=0)])
        lora.grad = torch.tensor([100000.])
        resampler.grad = torch.tensor([.3])
        resampler_bias.grad = torch.tensor([.4])
        norms = clip_grad_groups(optimizer)
        self.assertAlmostEqual(float(lora.grad), 1., places=5)
        self.assertAlmostEqual(float(resampler.grad), .3, places=6)
        self.assertAlmostEqual(norms['groups']['resampler']['preclip'], .5)
        self.assertAlmostEqual(norms['global_preclip'], (1e10 + .25) ** .5)
        resampler.grad.fill_(3)
        resampler_bias.grad.fill_(4)
        norms = clip_grad_groups(optimizer)
        self.assertAlmostEqual(norms['groups']['resampler']['postclip'], 1., places=5)

    def test_new_fork_restores_model_rng_and_cursor_with_new_optimizer(self):
        seed_all(239)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        model(torch.randn(2, 3)).sum().backward(); optimizer.step()
        expected = copy.deepcopy(model.state_dict())
        contract = dict(world_size=1, manifest_sha256='data', base_config_sha256='base',
                        base_receipt_sha256='receipt', config=dict(lora_rank=32))
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(directory, model, optimizer, scheduler,
                dict(step=5000, micro_step=5000, is_best=True, wandb_run_id='old'), contract, 0)
            expected_random = torch.rand(4)
            with torch.no_grad():
                model.weight.zero_()
            optimizer.param_groups[0]['lr'] = .002
            progress = load_fork_checkpoint(path, model, optimizer, 0, contract=contract)
            self.assertEqual(progress['micro_step'], 5000)
            self.assertTrue(torch.equal(torch.rand(4), expected_random))
            self.assertFalse(optimizer.state)
            self.assertEqual(optimizer.param_groups[0]['lr'], .002)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))
            load_fork_checkpoint(path, model, optimizer, 0, False, contract=contract)
            self.assertTrue(optimizer.state)
            self.assertEqual(optimizer.param_groups[0]['lr'], .002)
            with self.assertRaisesRegex(ValueError, 'world_size'):
                load_fork_checkpoint(path, model, optimizer, 0,
                                     contract=dict(contract, world_size=2))

    def test_baseline_multistep_boundary_and_resume(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        config = dict(optimizer=dict(scheduler='multistep', warmup_ratio=0,
                                     milestones=[100000], gamma=.1))
        scheduler = build_scheduler(optimizer, config)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        scheduler.last_epoch = 99998
        optimizer.step(); scheduler.step()
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-5)
        replacement = build_scheduler(optimizer, config)
        replacement.load_state_dict(scheduler.state_dict())
        optimizer.step(); replacement.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-5)

    def test_sampler_cursor_rank_and_suite_balance(self):
        suites = {str(i): list(range(i * 100, i * 100 + i + 2)) for i in range(4)}
        def draw(step, rank=0):
            return sample_indices(suites, seed=239, micro_step=step, rank=rank, world_size=2, batch_size=8)
        self.assertEqual(draw(100), draw(100))
        self.assertNotEqual(draw(100), draw(100, 1))
        counts = torch.bincount(torch.tensor([i // 100 for step in range(1000) for i in draw(step)]))
        self.assertTrue(((counts.float() / counts.sum() - .25).abs() < .025).all())

    def test_full_resume_matches_next_update_and_rotation(self):
        seed_all(239)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
        model.register_buffer('counter', torch.zeros(()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_multiplier(s, 8, 2))
        def update():
            optimizer.zero_grad()
            loss = (model(torch.randn(5, 3)) - torch.randn(5, 2)).square().mean()
            loss.backward(); optimizer.step(); scheduler.step(); model.counter.add_(1)
            return loss.detach()
        update()
        with tempfile.TemporaryDirectory() as directory:
            progress = dict(step=1, micro_step=10, is_best=True)
            path = save_checkpoint(directory, model, optimizer, scheduler, progress, {'test': 1}, 0)
            expected_loss = update()
            expected = copy.deepcopy(model.state_dict())
            restored = load_checkpoint(path, model, optimizer, scheduler, {'test': 1}, 0)
            self.assertEqual(restored, progress)
            self.assertTrue(torch.equal(update(), expected_loss))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, scheduler, {'test': 2}, 0)
            for step in (2, 3):
                save_checkpoint(directory, model, optimizer, scheduler,
                    dict(step=step, micro_step=step * 10, is_best=False), {'test': 1}, 0)
            self.assertEqual(len(list(Path(directory).glob('generation_*'))), 2)
            self.assertEqual((Path(directory) / 'best').resolve(), path)

    def test_optimizer_groups_have_no_duplicates_and_norm_decay_zero(self):
        model = torch.nn.Module()
        model.resampler = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
        model.backbone = torch.nn.Linear(3, 4)
        model.head = torch.nn.Linear(4, 2)
        groups = parameter_groups(model, dict(weight_decay=.01, lora_lr=5e-5, resampler_lr=1e-4, new_modules_lr=2e-4))
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        for group in groups:
            for parameter in group['params']:
                if parameter.ndim < 2:
                    self.assertEqual(group['weight_decay'], 0.)


if __name__ == '__main__':
    unittest.main()

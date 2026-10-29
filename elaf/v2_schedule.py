"""Absolute optimizer-step schedule for E-LAF v2.

Call ``apply`` before each optimizer update. The caller rebuilds DDP on a
transition; optimizer parameter groups remain stable across all three stages.
"""
from __future__ import annotations

import copy
import math


class StageSchedule:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = copy.deepcopy(config['stages'])
        self.total_steps = int(config['max_steps'])
        self.freeze_step = int(self.config.get('freeze_step', 3000))
        self.unfreeze_step = int(self.config.get('unfreeze_step', 10000))
        self.stage1_warmup = int(self.config.get('stage1_warmup_steps', 200))
        self.stage3_warmup = int(self.config.get('stage3_lora_warmup_steps', 1000))
        self.min_ratio = float(self.config['cosine_min_ratio'])
        self.stage3_min = copy.deepcopy(self.config.get('stage3_min'))
        if not 0 < self.freeze_step < self.unfreeze_step < self.total_steps:
            raise ValueError('Expected 0 < freeze_step < unfreeze_step < max_steps')
        if not 0 <= self.min_ratio <= 1:
            raise ValueError('cosine_min_ratio must lie in [0, 1]')
        if self.stage3_min is not None:
            for name in ('lora_lr', 'resampler_lr', 'new_modules_lr'):
                base = float(self.config['stage3'][name])
                floor = float(self.stage3_min[name])
                if not 0 <= floor <= base or not math.isclose(floor, base*self.min_ratio, rel_tol=1e-8, abs_tol=1e-15):
                    raise ValueError(f'stage3_min.{name} must equal stage3.{name} * cosine_min_ratio')
        if not 0 < self.stage1_warmup <= self.freeze_step:
            raise ValueError('Stage 1 warmup must fit Stage 1')
        if not 0 < self.stage3_warmup < self.total_steps - self.unfreeze_step:
            raise ValueError('Stage 3 LoRA warmup must fit Stage 3')
        self.stage = None
        self.completed_step = None
        self.lora_parameters = [p for g in optimizer.param_groups
                                if g['group_name'] == 'lora' for p in g['params']]
        if not self.lora_parameters:
            raise ValueError('Stable optimizer groups must contain LoRA parameters')

    def stage_for(self, completed_step):
        if completed_step < 0:
            raise ValueError('completed_step must be nonnegative')
        return 1 if completed_step < self.freeze_step else 2 if completed_step < self.unfreeze_step else 3

    def _cosine(self, step, start):
        progress = min(1.0, max(0.0, (step - start) / (self.total_steps - start)))
        return self.min_ratio + (1 - self.min_ratio) * (1 + math.cos(math.pi * progress)) / 2

    def apply(self, policy, completed_step):
        """Set update LR/trainability; return whether an established stage changed.

        The policy argument is accepted to keep the training integration explicit;
        parameter identity comes from the optimizer, never parameter-name guesses.
        LoRA moment state is dropped exactly once when entering Stage 3. Loading
        a Stage 3 scheduler checkpoint prevents a second reset on resume.
        """
        step = int(completed_step)
        if step != completed_step:
            raise ValueError('completed_step must be an integer')
        new_stage = self.stage_for(step)
        if self.completed_step is not None and step < self.completed_step:
            raise ValueError('Cannot rewind a live schedule; load its checkpoint first')
        transition = self.stage is not None and self.stage != new_stage
        if new_stage == 3 and self.stage != 3:
            for parameter in self.lora_parameters:
                self.optimizer.state.pop(parameter, None)
        for parameter in self.lora_parameters:
            parameter.requires_grad_(new_stage != 2)
            if new_stage == 2:
                # LR=0 alone still advances Adam moments; grad=None prevents both
                # moment updates and AdamW weight decay for frozen parameters.
                parameter.grad = None
        for group in self.optimizer.param_groups:
            kind = group['group_name']
            lr_kind = kind if kind in ('lora', 'resampler') else 'new_modules'
            if new_stage in (1, 2):
                lr = float(self.config['stage1'][lr_kind + '_lr'])
                if new_stage == 1:
                    lr *= min(1.0, (step + 1) / self.stage1_warmup)
                elif kind == 'lora':
                    lr = 0.0
            else:
                base_lr = float(self.config['stage3'][lr_kind + '_lr'])
                min_lr = (float(self.stage3_min[lr_kind + '_lr']) if self.stage3_min is not None
                          else base_lr*self.min_ratio)
                if kind == 'lora':
                    elapsed = step - self.unfreeze_step
                    if elapsed < self.stage3_warmup:
                        lr = base_lr * elapsed / self.stage3_warmup
                    else:
                        progress = min(1., (step-self.unfreeze_step-self.stage3_warmup)/
                                       (self.total_steps-self.unfreeze_step-self.stage3_warmup))
                        lr = min_lr + (base_lr-min_lr)*(1+math.cos(math.pi*progress))/2
                else:
                    progress = min(1., (step-self.unfreeze_step)/(self.total_steps-self.unfreeze_step))
                    lr = min_lr + (base_lr-min_lr)*(1+math.cos(math.pi*progress))/2
            group['lr'] = lr
        self.stage = new_stage
        self.completed_step = step
        return transition

    def state_dict(self):
        return dict(version=1, config=copy.deepcopy(self.config), total_steps=self.total_steps,
                    stage=self.stage, completed_step=self.completed_step)

    def load_state_dict(self, state):
        if state['version'] != 1 or state['config'] != self.config or state['total_steps'] != self.total_steps:
            raise ValueError('E-LAF v2 stage schedule contract differs from checkpoint')
        step, stage = state['completed_step'], state['stage']
        if (step is None) != (stage is None) or (step is not None and self.stage_for(step) != stage):
            raise ValueError('Inconsistent checkpoint stage and completed_step')
        self.stage, self.completed_step = stage, step

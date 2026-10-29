"""E-LAF v2 closed-loop adapter: raw grid11k history + joint Euler flow."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from elaf.evaluation.policy import (
    _result, dream_eval_image, initial_noise, sha256, validate_observations,
)
from elaf.v0_cache import PREPROCESS, load_teacher, normalization


def history_clips(observations):
    clips = []
    for obs in observations:
        history = obs['history']
        if not history:
            raise ValueError('Every-control-step image history is required')
        if not np.array_equal(history[-1]['image'], obs['image']):
            raise ValueError('History must end at the current observation')
        clips.append(np.stack([history[max(0, len(history)-4+i)]['image'] for i in range(4)]))
    return np.stack(clips)


def unnormalize_action(values, stats):
    lo = torch.as_tensor(stats['q01'], device=values.device, dtype=torch.float32)
    hi = torch.as_tensor(stats['q99'], device=values.device, dtype=torch.float32)
    mask = torch.as_tensor(stats.get('mask', [True]*7), device=values.device)
    raw = torch.where(mask, .5*(values.float()+1)*(hi-lo+1e-8)+lo, values.float())
    constant = torch.as_tensor(np.asarray(stats['min']) == np.asarray(stats['max']), device=values.device)
    return torch.where(constant, torch.as_tensor(stats['min'], device=values.device), raw)


class ELAFV2InferencePolicy:
    def __init__(self, config):
        from transformers import AutoProcessor
        from elaf.v2_train import build_backbone, model_config
        from elaf.v2_model import ELAFV2Policy
        from elaf.v0_training import load_trainable_state

        self.device = torch.device(config.get('device', 'cuda:0'))
        self.steps = int(config.get('num_steps', 4))
        if self.steps != 4:
            raise ValueError('This evaluation protocol uses four joint Euler steps')
        checkpoint = Path(config['checkpoint']).resolve()
        metadata = json.loads((checkpoint/'progress.json').read_text())
        cfg = metadata['contract']['config']
        if not cfg['version'].startswith('elaf_v2'):
            raise ValueError('Expected E-LAF v2 checkpoint')
        self.model = ELAFV2Policy(build_backbone(cfg), model_config(cfg))
        state = torch.load(checkpoint/'training.pt', map_location='cpu', weights_only=False)
        load_trainable_state(self.model, state['model'])
        del state
        self.model.to(self.device).eval().requires_grad_(False)
        self.teacher = load_teacher(config['root'], cfg['future_encoder']['checkpoint'], self.device)
        self.processor = AutoProcessor.from_pretrained(
            cfg['checkpoint'], trust_remote_code=True, local_files_only=True, use_fast=False)
        self.processor.tokenizer.padding_side = 'left'
        manifest_path = Path(cfg['manifest'])
        manifest = json.loads(manifest_path.read_text())
        stats_path = manifest_path.parent/manifest['stats_path']
        if sha256(stats_path) != manifest['stats_sha256']:
            raise ValueError('Training normalization statistics changed')
        self.stats = json.loads(stats_path.read_text())
        self.center_crop = config.get('center_crop', True)
        self.encoder_batch = int(config.get('encoder_batch_size', 8))
        self.identity = dict(type='elaf_v2', policy_family='E-LAF v2 spatial joint flow',
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint/'training.pt'),
            training_step=metadata['progress']['step'],
            encoder_sha256=cfg['future_encoder']['sha256'], stats_sha256=manifest['stats_sha256'],
            action_dim=7, proprio_dim=8, action_horizon=8, recommended_execute_steps=8,
            history_length=4, action_space='libero_delta_ee',
            action_coordinates='raw_dataset_gripper_0close_1open', num_steps=self.steps,
            noise='per_observation_seed_independent_action_delta', center_crop=self.center_crop,
            encoder_preprocessing=PREPROCESS, joint_time='u=s=tau', extra_tokens=72)

    def fingerprint(self):
        return self.identity.copy()

    @torch.inference_mode()
    def encode(self, observations):
        images = history_clips(observations)
        features = []
        for start in range(0, len(images), self.encoder_batch):
            clip = torch.from_numpy(images[start:start+self.encoder_batch].copy()).to(self.device)
            clip = clip.permute(0, 1, 4, 2, 3).float()/255
            b, t, c, h, w = clip.shape
            clip = F.interpolate(clip.reshape(b*t, c, h, w), (384, 384),
                                 mode='bilinear', align_corners=False, antialias=True)
            mean = clip.new_tensor(PREPROCESS['mean'])[None, :, None, None]
            std = clip.new_tensor(PREPROCESS['std'])[None, :, None, None]
            clip = ((clip-mean)/std).reshape(b, t, c, 384, 384).permute(0, 2, 1, 3, 4)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = self.teacher(clip)
            if tuple(value.shape[1:]) != (1152, 1024):
                raise ValueError(f'Unexpected raw grid11k shape: {value.shape}')
            features.append(value.to(torch.bfloat16))
        return torch.cat(features)

    def prepare(self, observations):
        validate_observations(observations)
        texts, images, states, keys = [], [], [], []
        for obs in observations:
            key = obs['suite'] + '_no_noops'
            if key not in self.stats:
                raise ValueError(f'Unknown suite normalization: {key}')
            keys.append(key)
            images.extend([dream_eval_image(obs['image'], self.center_crop),
                           dream_eval_image(obs['wrist_image'], self.center_crop)])
            conversation = [{'role':'user', 'content':[
                {'type':'image'}, {'type':'image'},
                {'type':'text', 'text':f"What action should the robot take to {obs['instruction']}?"}]}]
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
            states.append(normalization(obs['proprio'], self.stats[key]['proprio']))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors='pt')
        batch = {k:v.to(self.device) for k,v in inputs.items()}
        batch['proprio'] = torch.as_tensor(np.stack(states), device=self.device)
        batch['current_features'] = self.encode(observations)
        return batch, keys

    @torch.inference_mode()
    def predict(self, observations):
        batch, keys = self.prepare(observations)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            condition, mask = self.model._condition(batch)
            current = self.model._resample(batch['current_features'])
            delta = initial_noise(observations, (32, 1024), self.device, torch.float32, stream=1)
            actions = initial_noise(observations, (8, 7), self.device, condition.dtype)
            for step in range(self.steps):
                tau = torch.full((len(observations),), step/self.steps, device=self.device)
                va, vd = self.model.velocity(condition, mask, current, delta, actions, tau, tau)
                actions = actions + va/self.steps
                delta = delta + vd/self.steps
        raw = torch.stack([unnormalize_action(a, self.stats[k]['action']) for a,k in zip(actions, keys)])
        return _result(raw, len(observations))
